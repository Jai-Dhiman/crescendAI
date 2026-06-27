"""Dedup + self-consistency ingest of 4 new KernScores collections.

Collections and piece_id schemes:
  beethoven-piano-sonatas  -> prefix "beethoven.sonatas"  e.g. sonata01-1 -> beethoven.sonatas.sonata01_1
  mozart-piano-sonatas     -> prefix "mozart.sonatas"     e.g. sonata01-1 -> mozart.sonatas.sonata01_1
  haydn-keyboard-sonatas   -> prefix "haydn.sonatas"      e.g. sonata12-1 -> haydn.sonatas.sonata12_1
  chopin-preludes          -> prefix "chopin.preludes"    e.g. prelude28-01 -> chopin.preludes.prelude28_01

Dedup uses the exact same machinery as mutopia_dedup.py:
  NoteChromaMatcher top-5 recall -> ElasticGate pitch-only DTW (w_time=0).
  DUP if best_cost <= 0.2885, else NET-NEW.

Ingest uses the self-consistency gate from kernscores_bulk.py:
  expected_key = estimate_key(score), expected_bars = score.total_bars,
  then validate_score. Gate failures are printed and excluded; their JSON
  is NOT written. Piece-ID collisions with existing catalog HALT (should
  not occur given the dedup step; a collision means namespace pollution).

Run:  cd model && uv run python -m score_library.kernscores_expand
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note, load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.key_estimate import estimate_key
from score_library.parse import parse_score_midi
from score_library.validate import ExpectedMeta, validate_score

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_STAGE = Path(os.environ.get("KERNSCORES_STAGING_DIR", str(Path.home() / "crescendai_corpus_staging")))
_MIDI_BASE = _STAGE / "kernscores_midi"

TOP_K = 5
W_PITCH = 1.0
W_TIME = 0.0
DUP_THRESHOLD = 0.2885


# ---------------------------------------------------------------------------
# Piece-ID helpers (canonical; imported by render_kern_assets.py)
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _beethoven_sonata_piece_id(stem: str) -> str:
    """e.g. 'sonata01-1' -> 'beethoven.sonatas.sonata01_1'."""
    return f"beethoven.sonatas.{_sanitize(stem)}"


def _mozart_sonata_piece_id(stem: str) -> str:
    """e.g. 'sonata01-1' -> 'mozart.sonatas.sonata01_1'."""
    return f"mozart.sonatas.{_sanitize(stem)}"


def _haydn_sonata_piece_id(stem: str) -> str:
    """e.g. 'sonata12-1' -> 'haydn.sonatas.sonata12_1'."""
    return f"haydn.sonatas.{_sanitize(stem)}"


def _chopin_prelude_piece_id(stem: str) -> str:
    """e.g. 'prelude28-01' -> 'chopin.preludes.prelude28_01'."""
    return f"chopin.preludes.{_sanitize(stem)}"


def _hummel_prelude_piece_id(stem: str) -> str:
    """e.g. 'prelude67-07' -> 'hummel.preludes.prelude67_07'."""
    return f"hummel.preludes.{_sanitize(stem)}"


def _artfugue_piece_id(stem: str) -> str:
    """e.g. 'artfugue-014' -> 'bach.art_of_fugue.artfugue_014'."""
    return f"bach.art_of_fugue.{_sanitize(stem)}"


def _scriabin_piece_id(stem: str) -> str:
    """e.g. 'scriabin-op14_no02' -> 'scriabin.op14_no02'."""
    body = re.sub(r"^scriabin[-_]", "", stem, flags=re.IGNORECASE)
    return f"scriabin.{_sanitize(body)}"


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

def _score_to_notes(score_data) -> list[Note]:
    notes: list[Note] = []
    for bar in score_data.bars:
        for n in bar.notes:
            notes.append(Note(
                onset=n.onset_seconds,
                offset=n.onset_seconds + n.duration_seconds,
                pitch=n.pitch,
                velocity=n.velocity,
            ))
    notes.sort(key=lambda n: n.onset)
    return notes


def _load_catalog() -> dict[str, list[Note]]:
    skip = {"titles.json", "seed.sql"}
    catalog: dict[str, list[Note]] = {}
    for path in sorted(f for f in _SCORES_DIR.glob("*.json") if f.name not in skip):
        catalog[path.stem] = load_score_notes(path)
    return catalog


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(sorted(values))
    return {
        "min": float(arr[0]),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _check_gap(all_costs: list[float]) -> str:
    """Report whether 0.2885 sits in a genuine gap, or warn if not."""
    if not all_costs:
        return "  (no costs to assess)"
    arr = sorted(all_costs)
    below = [c for c in arr if c <= DUP_THRESHOLD]
    above = [c for c in arr if c > DUP_THRESHOLD]
    if not below or not above:
        return f"  WARNING: all costs {'below' if not above else 'above'} threshold -- no bimodal structure visible"
    gap_lo = max(below)
    gap_hi = min(above)
    gap_size = gap_hi - gap_lo
    return f"  gap at threshold: [{gap_lo:.4f} (last DUP), {gap_hi:.4f} (first NET-NEW)], gap_size={gap_size:.4f}"


# ---------------------------------------------------------------------------
# Core per-collection pipeline
# ---------------------------------------------------------------------------

def _dedup_and_ingest_collection(
    collection: str,
    midi_dir: Path,
    composer: str,
    piece_id_fn,
    chroma: NoteChromaMatcher,
    gate: ElasticGate,
) -> dict:
    """Dedup MIDIs vs catalog, then ingest net-new via self-consistency gate.

    Returns a result dict with counts and cost distribution.
    Raises SystemExit on hard errors (zero MIDIs, zero net-new ingested when
    there were candidates, catastrophic parse failure).
    """
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        raise SystemExit(f"ABORT: no .mid files in {midi_dir}")

    n_candidates = len(midi_files)
    dedup_costs: list[float] = []
    dups: list[tuple[str, float, str]] = []     # (stem, cost, match_id)
    parse_failures: list[tuple[str, str]] = []  # (stem, reason)
    net_new: list[tuple[str, float]] = []       # (stem, best_cost)

    for midi in midi_files:
        stem = midi.stem
        try:
            score_data = parse_score_midi(midi, stem, composer, stem)
            notes = _score_to_notes(score_data)
            if len(notes) < 2:
                raise ValueError(f"too few notes: {len(notes)}")
            top = [r.piece_id for r in chroma.rank(notes)[:TOP_K]]
            q_pc, q_li = _notes_to_events(notes)
            if q_pc.shape[0] < 2:
                raise ValueError("too few chord-events for DTW")
            costs = []
            for cid in top:
                c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
                if c is not None and np.isfinite(c):
                    costs.append((cid, float(c)))
            if not costs:
                raise ValueError("no finite DTW costs vs top-K catalog entries")
        except Exception as e:
            parse_failures.append((stem, f"{type(e).__name__}: {e}"))
            continue

        costs.sort(key=lambda x: x[1])
        best_id, best_cost = costs[0]
        dedup_costs.append(best_cost)

        if best_cost <= DUP_THRESHOLD:
            dups.append((stem, best_cost, best_id))
        else:
            net_new.append((stem, best_cost))

    # Ingest net-new via self-consistency gate
    gate_passed: list[str] = []
    gate_failures: list[tuple[str, str]] = []
    collisions: list[str] = []

    for stem, _cost in net_new:
        midi = midi_dir / f"{stem}.mid"
        pid = piece_id_fn(stem)
        dest = _SCORES_DIR / f"{pid}.json"

        if dest.exists():
            collisions.append(pid)
            continue

        try:
            score = parse_score_midi(midi, pid, composer, stem.replace("-", " ").replace("_", " ").title())
            expected = ExpectedMeta(
                piece_id=pid,
                expected_key=estimate_key(score),
                expected_bars=score.total_bars,
            )
            violations = validate_score(score, expected)
        except Exception as e:
            gate_failures.append((stem, f"{type(e).__name__}: {e}"))
            continue

        if violations:
            gate_failures.append((stem, "; ".join(f"{v.check}: {v.detail}" for v in violations)))
            continue

        with open(dest, "w") as fh:
            json.dump(score.model_dump(), fh, indent=2)
        gate_passed.append(pid)

    if collisions:
        raise SystemExit(
            f"ABORT: {len(collisions)} piece_id collisions in {collection} -- "
            f"namespace pollution: {collisions[:5]}"
        )

    return {
        "collection": collection,
        "candidates": n_candidates,
        "parse_failures": parse_failures,
        "dups": dups,
        "net_new_total": len(net_new),
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
        "dedup_costs": dedup_costs,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COLLECTIONS = [
    ("beethoven-piano-sonatas", "Beethoven", _beethoven_sonata_piece_id),
    ("mozart-piano-sonatas", "Mozart", _mozart_sonata_piece_id),
    ("haydn-keyboard-sonatas", "Haydn", _haydn_sonata_piece_id),
    ("chopin-preludes", "Chopin", _chopin_prelude_piece_id),
    ("hummel-preludes", "Hummel", _hummel_prelude_piece_id),
    ("art-of-the-fugue", "Bach", _artfugue_piece_id),
    ("scriabin", "Scriabin", _scriabin_piece_id),
]


def main() -> None:
    t0 = time.time()

    print("Loading catalog...", flush=True)
    catalog = _load_catalog()
    print(f"  catalog: {len(catalog)} pieces", flush=True)

    chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    print(f"  index built ({time.time()-t0:.1f}s)\n", flush=True)

    all_costs_global: list[float] = []
    results: list[dict] = []

    for collection, composer, pid_fn in _COLLECTIONS:
        midi_dir = _MIDI_BASE / collection
        if not midi_dir.exists():
            raise SystemExit(f"ABORT: MIDI dir missing: {midi_dir}")

        print(f"=== {collection} ({composer}) ===", flush=True)
        t1 = time.time()
        r = _dedup_and_ingest_collection(collection, midi_dir, composer, pid_fn, chroma, gate)
        elapsed = time.time() - t1

        all_costs_global.extend(r["dedup_costs"])
        results.append(r)

        # Per-collection report
        pct = _percentiles(r["dedup_costs"])
        print(f"  candidates:     {r['candidates']}")
        print(f"  parse failures: {len(r['parse_failures'])}")
        print(f"  DUPs (<= {DUP_THRESHOLD}): {len(r['dups'])}")
        print(f"  NET-NEW:        {r['net_new_total']}")
        print(f"  gate passed:    {len(r['gate_passed'])}")
        print(f"  gate failures:  {len(r['gate_failures'])}")
        if pct:
            print(f"  best_cost dist: min={pct['min']:.4f} p10={pct['p10']:.4f} p25={pct['p25']:.4f} "
                  f"median={pct['median']:.4f} p75={pct['p75']:.4f}")
        else:
            print("  best_cost dist: (no valid costs)")

        if r["parse_failures"]:
            print("  -- parse failures (fail-loud) --")
            for stem, reason in r["parse_failures"]:
                print(f"    {stem}: {reason}")

        if r["gate_failures"]:
            print(f"  -- gate failures ({len(r['gate_failures'])}) --")
            failure_reasons = Counter(f.split(":")[0] for _, f in r["gate_failures"])
            for k, n in failure_reasons.most_common():
                print(f"    {n:4d}  {k}")
            for stem, reason in r["gate_failures"][:20]:
                print(f"      {stem}: {reason}")

        print(f"  ({elapsed:.1f}s)", flush=True)
        print()

    # Global gap check
    print("=== GLOBAL COST DISTRIBUTION (all 4 collections) ===")
    global_pct = _percentiles(all_costs_global)
    if global_pct:
        print(f"  min={global_pct['min']:.4f} p10={global_pct['p10']:.4f} p25={global_pct['p25']:.4f} "
              f"median={global_pct['median']:.4f} p75={global_pct['p75']:.4f}")
    print(_check_gap(all_costs_global))

    # Any parse failures = fatal (fail-loud policy)
    any_parse_failures = sum(len(r["parse_failures"]) for r in results)
    if any_parse_failures:
        raise SystemExit(f"ABORT: {any_parse_failures} total parse failures -- see above")

    total_ingested = sum(len(r["gate_passed"]) for r in results)
    catalog_count = len(list(_SCORES_DIR.glob("*.json"))) - (1 if (_SCORES_DIR / "titles.json").exists() else 0)
    print(f"\n=== FINAL ===")
    print(f"  total net-new ingested: {total_ingested}")
    print(f"  catalog size (*.json minus titles.json): {catalog_count}")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
