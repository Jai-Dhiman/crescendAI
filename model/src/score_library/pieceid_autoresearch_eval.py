"""Fast single-metric piece-ID eval -- the VERIFY COMMAND for autoresearch.

Drives the experimental knob-board (pieceid_experimental) over the held-out ASAP
performances vs the 11K catalog and emits ONE scalar the autoresearch loop
hill-climbs:

  AUTORESEARCH_METRIC = 0.5*recog@0.0 + 0.3*recog@0.25 + 0.2*recog@0.5
                        - 1.5*max(0, fa_genuine - 0.05)

i.e. reward opening + mid-piece recognition (the axis-B gap is weighted in so a
windowed-shortlist win shows up), and PENALIZE genuine open-set false-accepts
above the 5% certification bar so the loop cannot game recognition by simply
loosening the threshold. The penalty is the guard; a run that raises recognition
but pushes fa_genuine over 5% loses score.

Speed (so the loop iterates in minutes, not the ~8min full eval):
  * parsed-catalog pickle cache (shared with the comprehensive eval).
  * LABEL cache: ground truth is computed ONCE with the BASELINE gate and frozen
    to JSON -- both a speedup AND a leakage guard (the experiment cannot move its
    own target). Keyed by the catalog stat-hash + note_cap.
  * one elastic-gate build; the shortlist index is rebuilt per run (cheap) because
    a windowed-shortlist experiment changes it.

Honesty: ground truth is gate-independent (derive_piece_id deterministic join +
baseline score-oracle), identical to the comprehensive harness. fa_genuine masks
the true id (leave-one-out) and excludes locks onto a residual catalog DUPLICATE
of the true piece (cost <= DUP_THRESHOLD) -- only genuinely-different-piece locks
count, so the guard tracks real open-set failures, not catalog cleanliness.

Run:  PYTHONPATH=src uv run python -m score_library.pieceid_autoresearch_eval \
          --scores-dir <11K scores> --catalog-cache <pkl> --label-cache <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate
from score_library import pieceid_experimental as X
from score_library.bulk_ingest import DUP_THRESHOLD, W_PITCH, W_TIME, _RunningChromaIndex
from score_library.pieceid_comprehensive_eval import load_catalog, start_at_fraction
from score_library.pieceid_crossperf_verify import (
    _ASAP_DIR,
    _SCORES_DIR,
    _load_perf_midi,
    label_works,
)

_START_FRACS = [0.0, 0.25, 0.5]
_FRAC_WEIGHTS = {0.0: 0.5, 0.25: 0.3, 0.5: 0.2}
_FA_PENALTY = 1.5
_FA_BAR = 0.05


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Deterministic train/test split (overfit guard for the hill-climb)
# ---------------------------------------------------------------------------
# A win on the n=242 metric is only REAL if it also holds on works the search
# never saw. The split is keyed on the WORK FOLDER (so every performance of a
# work stays on one side -- train and test share no repertoire) and uses a fixed
# sha1 hash (NOT the salted builtin hash()) so it is byte-identical across runs
# and sessions. bucket(folder) % 10 < 7 => train (~70%), else test (~30%).

def _works_bucket(folder: str) -> int:
    # sha256 (not the salted builtin hash()) => deterministic across runs/sessions.
    h = hashlib.sha256(folder.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10


def _in_split(folder: str, split: str) -> bool:
    if split == "all":
        return True
    train = _works_bucket(folder) < 7
    return train if split == "train" else (not train)


# ---------------------------------------------------------------------------
# Label cache (ground truth frozen at the baseline gate)
# ---------------------------------------------------------------------------

def load_or_build_labels(catalog, note_cap, scores_dir, label_cache: str | None) -> dict:
    """work_folder -> true_id, computed once with the BASELINE gate and cached."""
    from score_library.pieceid_comprehensive_eval import _catalog_key
    key = _catalog_key(Path(scores_dir), note_cap)
    if label_cache:
        cp = Path(label_cache)
        if cp.exists():
            blob = json.loads(cp.read_text())
            if blob.get("key") == key:
                _log(f"  label cache HIT ({cp.name}, {len(blob['labels'])} works)")
                return blob["labels"]
    # Build with the baseline gate (gate-independent det labels + baseline oracle).
    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    labels_full, _ = label_works(catalog, chroma, gate, note_cap)
    labels = {f: v["true_id"] for f, v in labels_full.items()}
    if label_cache:
        Path(label_cache).write_text(json.dumps({"key": key, "labels": labels}))
        _log(f"  labels cached -> {label_cache}")
    return labels


# ---------------------------------------------------------------------------
# Per-query evaluation through the EXPERIMENTAL gate
# ---------------------------------------------------------------------------

def _dup_of_true(gate: ElasticGate, true_id: str, locked_id: str) -> bool:
    ev = gate._events.get(true_id)
    if ev is None:
        return False
    c = gate.cost(ev[0], ev[1], locked_id, W_PITCH, W_TIME)
    return c is not None and np.isfinite(c) and c <= DUP_THRESHOLD


def eval_recognition(query: list[Note], true_id: str, index, gate) -> bool:
    """recognized = the experimental gate locks AND the locked id is the true id."""
    if len(query) < 2:
        return False
    cands = index.shortlist(query, X.TOP_K)
    ranked = X.ranked_costs(query, cands, gate)
    best_id, _, locked = X.accept(ranked)
    return bool(locked and best_id == true_id)


def eval_loo_fa(query: list[Note], true_id: str, index, gate) -> bool | None:
    """Leave-one-out genuine false-accept: mask the true id; a lock onto a
    genuinely-different piece (not a residual duplicate of true) is a real
    open-set failure. None if the query is too short."""
    if len(query) < 2:
        return None
    cands = index.shortlist(query, X.TOP_K + 1)
    cands = [c for c in cands if c != true_id][: X.TOP_K]
    ranked = X.ranked_costs(query, cands, gate)
    best_id, _, locked = X.accept(ranked)
    if not locked or best_id is None:
        return False
    return not _dup_of_true(gate, true_id, best_id)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", type=str, default=str(_SCORES_DIR))
    ap.add_argument("--catalog-cache", type=str, default="")
    ap.add_argument("--label-cache", type=str, default="")
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--limit-works", type=int, default=0,
                    help="cap works for a faster (noisier) loop; 0 = all 242")
    ap.add_argument("--works-split", choices=["train", "test", "all"], default="all",
                    help="deterministic folder-hash split: train (~70%), test (~30%), "
                         "all. Hill-climb on train, confirm the win on held-out test.")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    _log(f"levers: {X.levers()}")
    catalog = load_catalog(Path(args.scores_dir), args.note_cap,
                           args.catalog_cache or None, exclude=None)
    _log(f"  catalog: {len(catalog)} pieces  [{time.time()-t0:.1f}s]")

    labels = load_or_build_labels(catalog, args.note_cap, args.scores_dir,
                                  args.label_cache or None)
    gate = ElasticGate(catalog)
    index = X.build_shortlist_index(catalog)
    _log(f"  built gate + shortlist index ({X.SHORTLIST_MODE})  [{time.time()-t0:.1f}s]")

    folders = [f for f in sorted(labels) if _in_split(f, args.works_split)]
    _log(f"  works-split={args.works_split}: {len(folders)}/{len(labels)} works")
    if args.limit_works:
        folders = folders[: args.limit_works]

    recog: dict[float, list[bool]] = {f: [] for f in _START_FRACS}
    fa: list[bool] = []
    n_queries = 0
    for i, folder in enumerate(folders):
        true_id = labels[folder]
        if true_id not in catalog:
            continue
        wdir = _ASAP_DIR / folder
        perfs = sorted(p for p in wdir.glob("*.mid") if p.name != "midi_score.mid")
        if not perfs:
            continue
        notes = _load_perf_midi(perfs[0])[: args.note_cap]
        if len(notes) < 2:
            continue
        n_queries += 1
        for f in _START_FRACS:
            q = start_at_fraction(notes, f, args.note_cap)
            recog[f].append(eval_recognition(q, true_id, index, gate))
        v = eval_loo_fa(notes[: args.note_cap], true_id, index, gate)
        if v is not None:
            fa.append(v)
        if (i + 1) % 50 == 0:
            _log(f"  [{i+1}/{len(folders)}] works  [{time.time()-t0:.1f}s]")

    recog_rates = {f: (sum(recog[f]) / len(recog[f]) if recog[f] else 0.0) for f in _START_FRACS}
    fa_genuine = sum(fa) / len(fa) if fa else 0.0
    metric = (sum(_FRAC_WEIGHTS[f] * recog_rates[f] for f in _START_FRACS)
              - _FA_PENALTY * max(0.0, fa_genuine - _FA_BAR))

    result = {
        "metric": round(metric, 5),
        "works_split": args.works_split,
        "n_queries": n_queries,
        "recognized": {f"{f:.2f}": round(recog_rates[f], 4) for f in _START_FRACS},
        "fa_genuine": round(fa_genuine, 4),
        "fa_bar": _FA_BAR,
        "levers": X.levers(),
        "catalog_size": len(catalog),
        "seconds": round(time.time() - t0, 1),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nrecognized: open={recog_rates[0.0]*100:.1f}% "
          f"mid25={recog_rates[0.25]*100:.1f}% mid50={recog_rates[0.5]*100:.1f}% "
          f"| fa_genuine={fa_genuine*100:.1f}% (bar {_FA_BAR*100:.0f}%)")
    # The line autoresearch greps:
    print(f"AUTORESEARCH_METRIC: {metric:.5f}")


if __name__ == "__main__":
    main()
