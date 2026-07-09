# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "pretty_midi>=0.2.10",
# ]
# ///
"""Phase 3a -- does the difficulty SIGNAL exist? (MIREX 2026, #104)

Before building any ranker: compute each difficulty feature on all 7,901 PSyllabus
whole-piece MIDIs and measure its Kendall tau-c against the ground-truth grade (ps
0..10), alongside the trivial baselines we must beat (note-count, duration, mean-pitch).

This is the deferred RubricNet-Table-2 check and the real preview of the #104
commit-gate. If pitch-entropy / LZ / polyphony do not individually rank difficulty on
clean whole-piece MIDI, no downstream ranker will -- that is a kill signal. If they do,
Phase 3b (trained composer-disjoint ranker) is worth building.

tau-c is computed both over the full set (max-N signal estimate) and on a held-out
composer-disjoint TEST split (honest generalization preview). Whole-piece features (no
30s crop): difficulty is a whole-piece property and the eventual audio submission chunks
to cover the whole piece; aria-amt's 30s window is an inference detail, not modeling.

Run (data read from the primary checkout, result written there):
    uv run --script phase3_feature_signal.py [--limit N]
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_features import extract_difficulty_features  # noqa: E402
from psyllabus import (  # noqa: E402
    composer_disjoint_split,
    load_records,
    notes_from_midi_bytes,
)

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
OUT = PRIMARY / "model/data/results/mirex_phase3_feature_signal.json"

FEATURES = ["pitch_entropy", "pitch_lz_complexity", "polyphony", "pitch_range"]
BASELINES = ["note_count", "duration", "mean_pitch"]


def tau_c(x, y) -> float | None:
    """scipy Kendall tau-c (the official MIREX scorer). None if degenerate."""
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else round(float(t), 4)


def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    records, report = load_records(LABELS, MID_ZIP)
    if limit:
        # stride across the whole set: the label JSON is grade-grouped, so a prefix
        # would be single-grade (no signal). Striding preserves grade variation.
        records = records[:: max(1, len(records) // limit)][:limit]
    print(f"loaded {report}", flush=True)
    print(f"processing {len(records)} records\n", flush=True)

    rows = []          # per-piece: {grade, duration, features..., baselines...}
    parse_fail = 0
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, rec in enumerate(records):
            try:
                notes = notes_from_midi_bytes(zf.read(rec.midi_name))
                if len(notes) < 5:
                    raise ValueError(f"too few notes ({len(notes)})")
                feats = extract_difficulty_features(notes)
                pitches = [n["pitch"] for n in notes]
                rows.append({
                    "grade": rec.grade, "composer": rec.composer,
                    **feats,
                    "note_count": len(notes),
                    "duration": rec.duration,
                    "mean_pitch": float(np.mean(pitches)),
                })
            except Exception as exc:  # noqa: BLE001 - surface, count, keep going
                parse_fail += 1
                if parse_fail <= 10:
                    print(f"  parse fail [{rec.key[:40]}]: {exc!r}", flush=True)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(records)} ({parse_fail} fails)", flush=True)

    print(f"\nusable pieces: {len(rows)}  parse failures: {parse_fail}", flush=True)
    grades = np.array([r["grade"] for r in rows])

    # composer-disjoint test split (indices via a parallel record list)
    idx_records = [type("R", (), {"composer": r["composer"], "i": i})()
                   for i, r in enumerate(rows)]
    _train_r, test_r = composer_disjoint_split(idx_records, test_frac=0.2, seed=2026)
    test_idx = np.array([r.i for r in test_r])
    tgrades = grades[test_idx]

    def col(name):
        return np.array([r[name] for r in rows])

    table = {}
    for name in FEATURES + BASELINES:
        full = tau_c(col(name), grades)
        held = tau_c(col(name)[test_idx], tgrades)
        table[name] = {"tau_c_full": full, "tau_c_test": held,
                       "kind": "feature" if name in FEATURES else "baseline"}

    best_feature = max(
        (n for n in FEATURES if table[n]["tau_c_full"] is not None),
        key=lambda n: abs(table[n]["tau_c_full"]), default=None)
    best_baseline = max(
        (n for n in BASELINES if table[n]["tau_c_full"] is not None),
        key=lambda n: abs(table[n]["tau_c_full"]), default=None)
    beats = (best_feature is not None and best_baseline is not None and
             abs(table[best_feature]["tau_c_full"]) > abs(table[best_baseline]["tau_c_full"]))

    summary = {
        "n_usable": len(rows), "n_parse_fail": parse_fail,
        "n_test_composer_disjoint": len(test_idx),
        "table": table, "best_feature": best_feature, "best_baseline": best_baseline,
        "best_feature_beats_best_baseline": beats,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== PHASE 3a: DIFFICULTY FEATURE SIGNAL (tau-c vs grade) ===")
    print(f"  usable={len(rows)}  held-out(composer-disjoint) test={len(test_idx)}")
    print(f"  {'name':22s} {'kind':9s} {'tau_c(full)':>12s} {'tau_c(test)':>12s}")
    for name in FEATURES + BASELINES:
        t = table[name]
        print(f"  {name:22s} {t['kind']:9s} {str(t['tau_c_full']):>12s} {str(t['tau_c_test']):>12s}")
    print(f"\n  best feature={best_feature} ({table[best_feature]['tau_c_full'] if best_feature else None}) "
          f"vs best baseline={best_baseline} ({table[best_baseline]['tau_c_full'] if best_baseline else None})")
    print(f"  best feature beats best trivial baseline: {beats}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
