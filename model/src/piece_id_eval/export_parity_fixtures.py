# model/src/piece_id_eval/export_parity_fixtures.py
"""Freeze the FROZEN Phase-0 gate (Stage-0c/0f, pitch-only chord-Jaccard elastic
margin) into a golden fixture the Rust/WASM port is asserted against.

For each in-catalog recording (full piece) and a deterministic OOD sample, emit:
  - query_events: u16 12-bit pitch-class-set masks (onsets collapsed within 50ms)
  - candidates: chroma top-5, each with its u16 events + Python elastic cost
  - expected_best_piece_id / expected_margin / expected_locked at threshold 0.0935

Requires ASAP + MAESTRO present (data/raw); raises if missing (no silent fallback).
Run: cd model && uv run python -m piece_id_eval.export_parity_fixtures
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.stage0c_elastic_dtwgate import (
    _TOP_K,
    _W_PITCH,
    ElasticGate,
    _notes_to_events,
    load_data,
)
from piece_id_eval.stage0f_hard_ood_certify import (
    _MAESTRO_DIR,
    _MAX_OOD_EVENTS,
    _candidates,
    _load_perf_midi,
)

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_OUT = _MODEL_ROOT / "data/evals/piece_id/parity_fixtures.json"
_THRESHOLD = 0.0935
_N_OOD = 12
_SEED = 42


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pc_mat_to_masks(pc_mat: np.ndarray) -> list[int]:
    """Each event row (E,12) binary -> a 12-bit pitch-class-set mask (bit pc set)."""
    masks: list[int] = []
    for i in range(pc_mat.shape[0]):
        m = 0
        for pc in range(12):
            if pc_mat[i, pc] > 0:
                m |= 1 << pc
        masks.append(m)
    return masks


def _query_record(query_id: str, in_catalog: bool, notes, full_chroma: NoteChromaMatcher, gate: ElasticGate) -> dict | None:
    q_pc, q_li = _notes_to_events(notes)
    if q_pc.shape[0] < 2:
        return None
    topk = [r.piece_id for r in full_chroma.rank(notes)[:_TOP_K]]
    cands: list[dict] = []
    for cid in topk:
        c = gate.cost(q_pc, q_li, cid, _W_PITCH, 0.0)  # pitch-only (w_time=0)
        if c is None or not np.isfinite(c):
            continue
        r_pc, _ = gate._events[cid]
        cands.append({"piece_id": cid, "events": _pc_mat_to_masks(r_pc), "expected_cost": round(float(c), 6)})
    if len(cands) < 2:
        return None
    by_cost = sorted(cands, key=lambda x: x["expected_cost"])
    margin = by_cost[1]["expected_cost"] - by_cost[0]["expected_cost"]
    return {
        "query_id": query_id,
        "in_catalog": in_catalog,
        "query_events": _pc_mat_to_masks(q_pc),
        "candidates": cands,  # chroma-rank order — matches the order the WASM receives
        "expected_best_piece_id": by_cost[0]["piece_id"],
        "expected_margin": round(float(margin), 6),
        "expected_locked": bool(margin >= _THRESHOLD),
    }


def main() -> None:
    catalog, recordings = load_data()
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    queries: list[dict] = []
    for true_id, notes in recordings.items():
        rec = _query_record(f"in:{true_id}", True, notes, full_chroma, gate)
        if rec is None:
            raise RuntimeError(f"in-catalog query produced no record: {true_id}")
        queries.append(rec)

    if not _MAESTRO_DIR.exists():
        raise FileNotFoundError(f"MAESTRO required for OOD fixtures, missing: {_MAESTRO_DIR}")
    cands = _candidates()
    rng = random.Random(_SEED)
    rng.shuffle(cands)
    n_ood = 0
    for c in cands:
        if n_ood >= _N_OOD:
            break
        midi = _MAESTRO_DIR / c["row"]["midi_filename"]
        if not midi.exists():
            continue
        notes = _load_perf_midi(midi, _MAX_OOD_EVENTS)
        rec = _query_record(f"ood:{c['row']['canonical_title']}", False, notes, full_chroma, gate)
        if rec is None:
            continue
        queries.append(rec)
        n_ood += 1
    if n_ood < 10:
        raise RuntimeError(f"only {n_ood} OOD queries materialized; need >=10")

    out = {"margin_threshold": _THRESHOLD, "onset_tol_ms": 50, "top_k": _TOP_K, "queries": queries}
    _OUT.write_text(json.dumps(out))
    _log(f"Wrote {_OUT}: {len(queries)} queries ({n_ood} OOD)")


if __name__ == "__main__":
    main()
