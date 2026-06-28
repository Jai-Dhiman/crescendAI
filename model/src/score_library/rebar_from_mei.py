"""Re-bar a score JSON's bar grid to match the displayed MEI's written measures.

Why: the score JSON bar grid (`build_bar_grid` in parse.py) is arithmetic
(total_ticks / ticks_per_bar) and drifts from the engraved MEI two ways --
repeat-expanded source MIDI inflates it, and irregular bar lengths mis-segment it
(verified: ~12% of renderable pieces had |MEI measures - total_bars| > 2). Live
score-highlighting maps a teacher-chosen bar number K onto Verovio's K-th ordinal
measure, so the JSON bar grid MUST equal the MEI measure grid or highlights mis-point.

How: keep every note and pedal event byte-identical (so the piece-ID fingerprint,
which is a flat note-set chroma + chord-event index, is unchanged) and only
re-group them into the MEI's measure boundaries. Verovio renders MIDI at ppq=120,
so a note's onset_tick/120 shares the timemap qstamp (quarter) axis.

Only single-pass pieces can be re-barred this way: repeat-expanded pieces have
notes beyond the written-measure span (their second pass plays past the last MEI
bar), so re-barring would dump them into the final bar. Those are detected and
skipped (`fits=False`) for separate handling.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import verovio

from .parse import ticks_to_seconds

PPQ = 120  # default: Verovio MIDI render resolution (kern-derived score MIDIs).


def _derive_ppq(data: dict) -> int:
    """Ticks-per-quarter of THIS score JSON's bar grid. parse.py inherits the
    source MIDI resolution (kern=120, GiantMIDI=384, PDMX=480), so a rebar that
    matches the JSON's tick axis to a Verovio quarter timemap MUST use the JSON's
    own PPQ, not a fixed constant. Derived from bar lengths: a bar of N/D spans
    N*4/D quarters, so ppq = bar_ticks / (N*4/D). Median over bars is robust to
    pickups and mid-piece meter changes."""
    bars = data.get("bars") or []
    if len(bars) < 2:
        return PPQ
    ratios = []
    for a, b in zip(bars, bars[1:]):
        span = b["start_tick"] - a["start_tick"]
        if span <= 0:
            continue
        try:
            num, den = (int(x) for x in a["time_signature"].split("/"))
        except (KeyError, ValueError):
            continue
        quarters = num * 4 / den
        if quarters > 0:
            ratios.append(span / quarters)
    if not ratios:
        return PPQ
    ratios.sort()
    return int(round(ratios[len(ratios) // 2]))


def _measure_start_quarters(
    tk: verovio.toolkit, mei_path: Path
) -> tuple[list[float], float]:
    """Ordered measure-start quarter positions, and the max qstamp seen."""
    tk.loadFile(str(mei_path))
    tmap = tk.renderToTimemap({"includeMeasures": True})
    if isinstance(tmap, str):
        tmap = json.loads(tmap)
    seen: set[float] = set()
    starts: list[float] = []
    max_q = 0.0
    for e in tmap:
        max_q = max(max_q, e.get("qstamp", 0))
        if e.get("measureOn") is not None and e["qstamp"] not in seen:
            seen.add(e["qstamp"])
            starts.append(e["qstamp"])
    starts.sort()
    return starts, max_q


def _active_time_sig(time_sigs: list[dict], tick: int) -> str:
    cur = "4/4"
    for ts in time_sigs:
        if ts["tick"] <= tick:
            cur = f'{ts["numerator"]}/{ts["denominator"]}'
        else:
            break
    return cur


def _seconds_to_ticks(seconds: float, tempo_map: list[dict], ppq: int) -> int:
    """Inverse of parse.ticks_to_seconds: real seconds -> absolute JSON tick."""
    if not tempo_map:
        tempo_map = [{"tick": 0, "tempo": 500_000}]
    acc = 0.0
    prev_tick = 0
    cur_tempo = tempo_map[0]["tempo"]
    for tm in tempo_map:
        if tm["tick"] <= prev_tick:
            cur_tempo = tm["tempo"]
            continue
        seg = (tm["tick"] - prev_tick) * cur_tempo / (ppq * 1_000_000)
        if acc + seg >= seconds:
            break
        acc += seg
        prev_tick = tm["tick"]
        cur_tempo = tm["tempo"]
    return prev_tick + round((seconds - acc) * ppq * 1_000_000 / cur_tempo)


def _measure_start_seconds(tk: verovio.toolkit, path: Path) -> list[float]:
    """Ordered measure-start real-time positions (seconds) from the engraving's
    Verovio timemap. Re-bars MIDI-rendered .mxl whose NOTATED quarter axis diverges
    from the JSON arithmetic bar grid (MuseScore meter inference != parse.py) but
    whose REAL TIME is preserved (MuseScore HumanPerformance keeps absolute onsets)."""
    tk.loadFile(str(path))
    tmap = tk.renderToTimemap({"includeMeasures": True})
    if isinstance(tmap, str):
        tmap = json.loads(tmap)
    seen: set[float] = set()
    starts: list[float] = []
    for e in tmap:
        if e.get("measureOn") is not None and e.get("tstamp") not in seen:
            seen.add(e.get("tstamp"))
            starts.append(e.get("tstamp", 0) / 1000.0)
    starts.sort()
    return starts


def rebar_score(
    data: dict, mei_path: Path, tk: verovio.toolkit
) -> tuple[dict, bool, dict]:
    """Return (new_score_dict, fits, stats).

    `fits` is False when the existing notes overrun the engraving's written-measure
    span by both the quarter AND the seconds axis (true repeat-expansion). When
    True, every note is preserved and re-grouped to the engraving's measures.

    Alignment axis: the quarter-tick axis works for Verovio-rendered .mei and clean
    score-MIDI .mxl. For performance-MIDI .mxl (GiantMIDI) MuseScore's meter
    inference assigns a different quarter count than parse.py's arithmetic grid, so
    the quarter axis overruns -- we fall back to the REAL-TIME (seconds) axis, which
    MuseScore preserves, mapping measure-start seconds back to JSON ticks.
    """
    ppq = _derive_ppq(data)
    notes = [n for b in data["bars"] for n in b["notes"]]
    pedals = [p for b in data["bars"] for p in b["pedal_events"]]
    starts_q, max_q = _measure_start_quarters(tk, mei_path)
    stats = {
        "mei_measures": len(starts_q),
        "old_total_bars": data.get("total_bars"),
        "n_notes": len(notes),
    }
    if not starts_q:
        return data, False, {**stats, "reason": "no_measures"}

    tempo_map = [
        {"tick": t["tick"], "tempo": t["tempo_usec"]} for t in data["tempo_markings"]
    ]
    time_sigs = data["time_signatures"]

    last_note_q = max((n["onset_tick"] / ppq for n in notes), default=0.0)
    # Overflow tolerance: one written measure past the last measure start covers a
    # final-bar onset; anything well beyond is a repeat second pass.
    span_q = (starts_q[-1] - starts_q[-2]) if len(starts_q) >= 2 else max_q
    stats["last_note_q"] = round(last_note_q, 3)
    stats["max_q"] = round(max_q, 3)

    if last_note_q <= max_q + max(span_q, 1.0) + 1e-6:
        # Quarter axis aligns: Verovio-rendered .mei or clean score-MIDI .mxl.
        start_ticks = [round(q * ppq) for q in starts_q]
        stats["align"] = "quarters"
    else:
        # Quarter axis overruns -> performance-MIDI .mxl. Re-bar on the preserved
        # real-time axis: map the engraving's measure-start seconds to JSON ticks.
        starts_s = _measure_start_seconds(tk, mei_path)
        if not starts_s:
            return data, False, {**stats, "reason": "repeat_overflow"}
        start_ticks = [_seconds_to_ticks(s, tempo_map, ppq) for s in starts_s]
        last_note_tick = max((n["onset_tick"] for n in notes), default=0)
        span_t = (start_ticks[-1] - start_ticks[-2]) if len(start_ticks) >= 2 else ppq
        if last_note_tick > start_ticks[-1] + max(span_t, ppq) * 2:
            return data, False, {**stats, "reason": "seconds_overflow"}
        stats["align"] = "seconds"
        stats["mei_measures"] = len(start_ticks)

    nbars = len(start_ticks)
    note_buckets: list[list[dict]] = [[] for _ in range(nbars)]
    for n in notes:
        i = bisect.bisect_right(start_ticks, n["onset_tick"]) - 1
        note_buckets[max(i, 0)].append(n)
    ped_buckets: list[list[dict]] = [[] for _ in range(nbars)]
    for p in pedals:
        i = bisect.bisect_right(start_ticks, p["tick"]) - 1
        ped_buckets[max(i, 0)].append(p)

    bars = []
    for bi in range(nbars):
        st = start_ticks[bi]
        bnotes = sorted(note_buckets[bi], key=lambda x: (x["onset_tick"], x["pitch"]))
        pitches = [n["pitch"] for n in bnotes]
        vels = [n["velocity"] for n in bnotes]
        bars.append(
            {
                "bar_number": bi + 1,
                "start_tick": st,
                "start_seconds": round(ticks_to_seconds(st, tempo_map, ppq), 6),
                "time_signature": _active_time_sig(time_sigs, st),
                "notes": bnotes,
                "pedal_events": sorted(ped_buckets[bi], key=lambda x: x["tick"]),
                "note_count": len(bnotes),
                "pitch_range": [min(pitches), max(pitches)] if pitches else [],
                "mean_velocity": round(sum(vels) / len(vels)) if vels else 0,
            }
        )

    out = dict(data)
    out["bars"] = bars
    out["total_bars"] = nbars
    stats["new_total_bars"] = nbars
    return out, True, stats


def main() -> None:
    """Re-bar every score JSON that has a paired engraved MEI so its bar grid
    matches the displayed measures. Notes are preserved exactly (fingerprint
    unchanged); only the bar grid + per-bar summaries are rewritten."""
    import argparse
    import shutil

    ap = argparse.ArgumentParser(description="Re-bar score JSONs to MEI measures.")
    ap.add_argument("--scores-dir", required=True, type=Path, help="dir of <id>.json")
    ap.add_argument("--mei-dir", required=True, type=Path,
                    help="dir of <id>.mei / <id>.mxl engravings (both are re-barred)")
    ap.add_argument("--backup-dir", type=Path, help="if set, copy originals here first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verovio.enableLog(verovio.LOG_OFF)
    tk = verovio.toolkit()
    meis = sorted(
        [*args.mei_dir.glob("*.mei"), *args.mei_dir.glob("*.mxl")]
    )
    changed = skipped = unfit = missing = 0
    deltas: list[int] = []
    if args.backup_dir and not args.dry_run:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
    for mei in meis:
        pid = mei.stem
        jp = args.scores_dir / f"{pid}.json"
        if not jp.exists():
            missing += 1
            continue
        data = json.load(open(jp))
        new, fits, stats = rebar_score(data, mei, tk)
        if not fits:
            unfit += 1
            print(f"UNFIT  {pid}: {stats.get('reason')}")
            continue
        delta = stats["new_total_bars"] - stats["old_total_bars"]
        if delta == 0:
            skipped += 1
            continue
        deltas.append(delta)
        changed += 1
        if not args.dry_run:
            if args.backup_dir:
                shutil.copy2(jp, args.backup_dir / f"{pid}.json")
            with open(jp, "w") as f:
                json.dump(new, f, indent=2)
    print(
        f"\nrebar: {changed} rewritten, {skipped} already-aligned, "
        f"{unfit} unfit (repeat-overflow), {missing} mei-without-json"
        + (" [DRY-RUN]" if args.dry_run else "")
    )
    if deltas:
        import statistics

        print(
            f"  bar-count deltas applied: min={min(deltas)} max={max(deltas)} "
            f"median={statistics.median(deltas):.0f} n={len(deltas)}"
        )


if __name__ == "__main__":
    main()
