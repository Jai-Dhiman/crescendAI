# model/src/follower_eval/asap_eval.py
"""Track A of the real-audio follower eval (#133): AUTOMATIC accuracy against
ASAP's human-verified beat alignment -- no tapping, no labeling.

ASAP ships, per real performance, a beat-level alignment between the performance
timeline and the score timeline (`performance_beats[i]` in perf-seconds pairs
with `midi_score_beats[i]` in score-seconds). That pairing IS the ground-truth
answer key the amateur corpus lacks: at performance-time `p_i`, the true score
position is `s_i`. We run the follower on the ASAP performance, read its decoded
score-position at each `p_i`, and compare to `s_i` -> a real bar/beat-localization
error, over 1000+ real pianists across the rep idiom, for free.

TWO EVAL MODES:
  * full          -- follow the whole performance from the start; error at every
                     beat. The "does it track a full run-through" number.
  * random-start  -- cold-start the follower at random mid-performance offsets
                     (feed only notes from t_start on) and check it still
                     localizes. This is the real interactive case: the user is
                     somewhere in the middle, the follower has no lead-in. The
                     hardest and most production-relevant test, and only possible
                     because ASAP gives per-beat truth everywhere.

INPUT NOTE: this runs on ASAP performance MIDI (clean, on-disk). Swapping the
perf-note source for MAESTRO-audio -> Transkun measures the SAME follower through
the real transcription stage; the truth (ASAP beats) is unchanged. Clean-MIDI
here is the low-transcription-noise end / harness validation; audio is the next
step.

RUNNING (from the PRIMARY checkout so data/raw/asap resolves):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.asap_eval \
    --pieces "Bach/Prelude/bwv_846/Shi05M.mid" --random-starts 8 --window-sec 20
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import partitura as pa

from follower_bench.asap_alignment import (
    AsapAlignmentMissingError,
    load_alignment,
)
from follower_bench.follower import bar_boundary_columns
from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm
from follower_bench.score_notes import ScoreNote, load_score_notes_from_midi
from follower_bench.segments import PerfNote

from follower_eval.accuracy import _pctl, decode_at

# A decoded beat is "localized" if within this many beats of its true score
# position. One beat = the follower is on the right beat; half a beat is the
# sub-beat bound. Tempo-invariant (error is divided by the local score-beat
# spacing), so a Bach prelude and a Liszt etude pool into one number.
TOL_BEATS_LENIENT = 1.0
TOL_BEATS_STRICT = 0.5


class AsapEvalError(RuntimeError):
    """Raised when a requested ASAP piece cannot be evaluated -- loud, never a
    silent skip that would inflate the corpus with empty cells."""


@dataclass(frozen=True)
class BeatTruth:
    """The answer key for one performance: paired perf-time / score-second beat
    anchors, plus the local score-beat spacing at each (for beat-unit errors)."""
    perf_times: tuple[float, ...]
    score_secs: tuple[float, ...]
    beat_spacing: tuple[float, ...]


@dataclass(frozen=True)
class WindowResult:
    """One follow run's localization error over its evaluated beats."""
    label: str                 # "full" or "start@<t>s"
    t_start: float
    n_beats_eval: int
    median_abs_err_sec: float | None
    p90_abs_err_sec: float | None
    median_abs_err_beats: float | None
    p90_abs_err_beats: float | None
    within_1beat_frac: float | None
    within_half_beat_frac: float | None
    transpose_semitones: int


@dataclass(frozen=True)
class AsapClipResult:
    """A performance's full-follow result and its cold-start windows."""
    asap_piece: str
    n_perf_notes: int
    n_score_notes: int
    n_beats: int
    full: WindowResult
    random_starts: tuple[WindowResult, ...]


def _load_perf_notes(path: Path) -> list[PerfNote]:
    """ASAP performance MIDI -> PerfNote list (partitura, sorted by onset)."""
    ppart = pa.load_performance_midi(str(path))
    na = ppart.note_array()
    notes = [
        PerfNote(onset=float(r["onset_sec"]), offset=float(r["onset_sec"] + r["duration_sec"]),
                 pitch=int(r["pitch"]), velocity=int(r["velocity"]))
        for r in na
    ]
    notes.sort(key=lambda n: n.onset)
    return notes


def load_asap_clip(asap_piece: str, asap_root: Path
                   ) -> tuple[list[PerfNote], list[ScoreNote], tuple[int, ...], BeatTruth]:
    """Load one ASAP performance into everything the follower + metric need:
    (perf_notes, score_notes, bar_boundaries, beat_truth).

    ``asap_root`` is passed explicitly (not the module-anchored default) because
    this code runs from a worktree whose ``data/`` is absent -- the real ASAP
    tree lives in the PRIMARY checkout. Same reason realaudio.py takes explicit
    roots.

    Raises:
        AsapEvalError: the piece has no usable ASAP alignment or its MIDI is
            missing (wraps the underlying loader errors loudly).
    """
    try:
        al = load_alignment(asap_piece, asap_root=asap_root,
                            annotations_path=asap_root / "asap_annotations.json")
    except (AsapAlignmentMissingError, FileNotFoundError) as exc:
        raise AsapEvalError(f"{asap_piece}: {type(exc).__name__}: {exc}") from exc

    perf_notes = _load_perf_notes(al.performance_midi_path)
    score_notes = load_score_notes_from_midi(al.score_midi_path)
    if not perf_notes or not score_notes:
        raise AsapEvalError(f"{asap_piece}: empty perf ({len(perf_notes)}) or score ({len(score_notes)}) notes")

    downbeats = sorted(al.midi_score_downbeats) or sorted(al.midi_score_beats)
    bar_boundaries = bar_boundary_columns([n.position for n in score_notes], downbeats)

    ps = list(al.performance_beats)
    ss = list(al.midi_score_beats)
    # local score-beat spacing (median-filled at the ends) for beat-unit errors
    spacing: list[float] = []
    for i in range(len(ss)):
        if i + 1 < len(ss):
            spacing.append(ss[i + 1] - ss[i])
        elif i > 0:
            spacing.append(ss[i] - ss[i - 1])
        else:
            spacing.append(0.0)
    truth = BeatTruth(perf_times=tuple(ps), score_secs=tuple(ss), beat_spacing=tuple(spacing))
    return perf_notes, score_notes, bar_boundaries, truth


def _beat_errors(matches, truth: BeatTruth,
                 window: tuple[float, float] | None) -> list[tuple[float, float]]:
    """(abs_err_sec, abs_err_beats) at each truth beat inside `window` (or all).
    Uses the follower's decoded score-position interpolated at the beat's
    perf-time."""
    ms = sorted(matches, key=lambda m: m.perf_time)
    pt = [m.perf_time for m in ms]
    sp = [m.score_position for m in ms]
    out: list[tuple[float, float]] = []
    for p_time, s_true, spc in zip(truth.perf_times, truth.score_secs, truth.beat_spacing):
        if window is not None and not (window[0] <= p_time <= window[1]):
            continue
        decoded = decode_at(pt, sp, p_time)
        if decoded is None:
            continue
        abs_err = abs(decoded - s_true)
        out.append((abs_err, abs_err / spc if spc > 0 else 0.0))
    return out


def _summarize(label: str, t_start: float, errs: list[tuple[float, float]],
               transpose: int) -> WindowResult:
    """Errors -> a WindowResult (median/p90 in sec & beats, within-tol rates)."""
    secs = [e[0] for e in errs]
    beats = [e[1] for e in errs]
    return WindowResult(
        label=label,
        t_start=round(t_start, 2),
        n_beats_eval=len(errs),
        median_abs_err_sec=round(statistics.median(secs), 3) if secs else None,
        p90_abs_err_sec=round(_pctl(secs, 0.9), 3) if secs else None,
        median_abs_err_beats=round(statistics.median(beats), 3) if beats else None,
        p90_abs_err_beats=round(_pctl(beats, 0.9), 3) if beats else None,
        within_1beat_frac=round(sum(b <= TOL_BEATS_LENIENT for b in beats) / len(beats), 4) if beats else None,
        within_half_beat_frac=round(sum(b <= TOL_BEATS_STRICT for b in beats) / len(beats), 4) if beats else None,
        transpose_semitones=transpose,
    )


def follow_window(perf_notes: list[PerfNote], score_notes: list[ScoreNote],
                  bar_boundaries: tuple[int, ...], truth: BeatTruth,
                  t_start: float, window_sec: float | None, label: str) -> WindowResult:
    """Cold-start the follower at `t_start` (feed only notes with onset >=
    t_start) and score the beats in [t_start, t_start+window_sec]. window_sec
    None => to the end (the 'full' run uses t_start=0, window None)."""
    sub = [n for n in perf_notes if n.onset >= t_start]
    if window_sec is not None:
        sub = [n for n in sub if n.onset <= t_start + window_sec]
    if len(sub) < 4:
        return _summarize(label, t_start, [], 0)
    est = follow_hmm(sub, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries)
    win = None if window_sec is None else (t_start, t_start + window_sec)
    errs = _beat_errors(est.matches, truth, win)
    return _summarize(label, t_start, errs, est.transpose_semitones)


def _rng_starts(truth: BeatTruth, n: int, window_sec: float, seed: int) -> list[float]:
    """n reproducible random start-times, each leaving room for a full window
    before the last beat. Uses random.Random(seed) so runs are deterministic."""
    import random
    lo = truth.perf_times[0]
    hi = truth.perf_times[-1] - window_sec
    if hi <= lo:
        return []
    rng = random.Random(seed)
    return sorted(rng.uniform(lo, hi) for _ in range(n))


def evaluate_clip(asap_piece: str, asap_root: Path, random_starts: int = 8,
                  window_sec: float = 20.0, seed: int = 0) -> AsapClipResult:
    """Full-follow + `random_starts` cold-start windows for one ASAP performance."""
    perf_notes, score_notes, bar_boundaries, truth = load_asap_clip(asap_piece, asap_root)
    full = follow_window(perf_notes, score_notes, bar_boundaries, truth,
                         t_start=0.0, window_sec=None, label="full")
    starts = _rng_starts(truth, random_starts, window_sec, seed)
    windows = tuple(
        follow_window(perf_notes, score_notes, bar_boundaries, truth,
                      t_start=ts, window_sec=window_sec, label=f"start@{ts:.0f}s")
        for ts in starts
    )
    return AsapClipResult(
        asap_piece=asap_piece,
        n_perf_notes=len(perf_notes),
        n_score_notes=len(score_notes),
        n_beats=len(truth.perf_times),
        full=full,
        random_starts=windows,
    )


def aligned_pieces(annotations_path: Path, limit: int | None = None) -> list[str]:
    """ASAP piece keys with a usable score<->performance alignment (the same
    filter load_alignment enforces), sorted. Used to run the whole corpus."""
    data = json.loads(annotations_path.read_text())
    keys = [k for k, v in data.items()
            if v.get("score_and_performance_aligned")
            and len(v.get("performance_beats") or []) >= 4
            and len(v.get("performance_beats") or []) == len(v.get("midi_score_beats") or [])]
    keys.sort()
    return keys[:limit] if limit else keys


def _pool_random_starts(results: list[AsapClipResult]) -> dict | None:
    """Pool cold-start localization across every random window of every clip."""
    beats_err: list[float] = []
    w1 = w1_tot = 0
    for r in results:
        for w in r.random_starts:
            if w.median_abs_err_beats is None:
                continue
            beats_err.append(w.median_abs_err_beats)
            if w.within_1beat_frac is not None:
                w1 += w.within_1beat_frac * w.n_beats_eval
                w1_tot += w.n_beats_eval
    if not beats_err:
        return None
    return {
        "n_windows": len(beats_err),
        "median_window_median_err_beats": round(statistics.median(beats_err), 3),
        "p90_window_median_err_beats": round(_pctl(beats_err, 0.9), 3),
        "pooled_within_1beat_frac": round(w1 / w1_tot, 4) if w1_tot else None,
    }


def _format(results: list[AsapClipResult]) -> str:
    L = ["=" * 92,
         "REAL-AUDIO FOLLOWER EVAL (#133) -- TRACK A -- ASAP beat-alignment ground truth",
         "=" * 92,
         f"performances: {len(results)}"]
    hdr = (f"{'piece':<46}{'beats':>6}{'full_eb':>8}{'full<=1':>8}"
           f"{'rs_eb':>7}{'rs<=1':>7}")
    L.append("")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in sorted(results, key=lambda r: r.asap_piece):
        feb = f"{r.full.median_abs_err_beats:.2f}" if r.full.median_abs_err_beats is not None else " n/a"
        fw1 = f"{r.full.within_1beat_frac:.2f}" if r.full.within_1beat_frac is not None else " n/a"
        rs = [w for w in r.random_starts if w.median_abs_err_beats is not None]
        reb_vals = [w.median_abs_err_beats for w in rs if w.median_abs_err_beats is not None]
        rw1_vals = [w.within_1beat_frac for w in rs if w.within_1beat_frac is not None]
        reb = f"{statistics.median(reb_vals):.2f}" if reb_vals else " n/a"
        rw1 = f"{statistics.median(rw1_vals):.2f}" if rw1_vals else " n/a"
        L.append(f"{r.asap_piece[:45]:<46}{r.n_beats:>6}{feb:>8}{fw1:>8}{reb:>7}{rw1:>7}")
    L.append("-" * len(hdr))
    full_eb = [r.full.median_abs_err_beats for r in results if r.full.median_abs_err_beats is not None]
    if full_eb:
        L.append(f"FULL-FOLLOW  median clip err = {statistics.median(full_eb):.3f} beats "
                 f"over {len(full_eb)} clips")
    rs_pool = _pool_random_starts(results)
    if rs_pool:
        L.append(f"COLD-START   median window err = {rs_pool['median_window_median_err_beats']:.3f} beats, "
                 f"pooled within-1-beat = {rs_pool['pooled_within_1beat_frac']} "
                 f"({rs_pool['n_windows']} windows)")
    return "\n".join(L)


def _to_jsonable(results: list[AsapClipResult]) -> dict:
    return {
        "n_performances": len(results),
        "clips": [
            {**{k: v for k, v in asdict(r).items() if k not in ("full", "random_starts")},
             "full": asdict(r.full),
             "random_starts": [asdict(w) for w in r.random_starts]}
            for r in results
        ],
        "random_start_pool": _pool_random_starts(results),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-audio follower eval -- Track A (ASAP ground truth, #133)")
    ap.add_argument("--asap-root", type=Path, default=Path("data/raw/asap-dataset"),
                    help="ASAP tree (CWD-relative; run from the PRIMARY checkout where data/ lives)")
    ap.add_argument("--pieces", nargs="+", default=None,
                    help="ASAP piece keys (default: all aligned pieces, capped by --limit)")
    ap.add_argument("--limit", type=int, default=None, help="cap #pieces when running the corpus")
    ap.add_argument("--random-starts", type=int, default=8, help="cold-start windows per clip")
    ap.add_argument("--window-sec", type=float, default=20.0, help="cold-start window length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    pieces = args.pieces or aligned_pieces(args.asap_root / "asap_annotations.json", limit=args.limit)
    results: list[AsapClipResult] = []
    failures: list[dict] = []
    for p in pieces:
        try:
            results.append(evaluate_clip(p, args.asap_root, random_starts=args.random_starts,
                                         window_sec=args.window_sec, seed=args.seed))
        except AsapEvalError as exc:
            failures.append({"piece": p, "error": str(exc)})

    print(_format(results))
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:20]:
            print(f"  {f['piece']} -> {f['error']}")
    if args.out:
        payload = _to_jsonable(results)
        payload["failures"] = failures
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
