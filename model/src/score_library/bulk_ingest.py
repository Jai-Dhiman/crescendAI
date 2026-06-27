"""Scalable, source-agnostic bulk ingest for the piece-ID catalog.

The per-collection kernscores_expand pipeline dedups each candidate against the
EXISTING catalog only (its chroma+gate index is built once at start), so it
cannot catch duplicates WITHIN a large import batch -- fine for a few hundred
craigsapp pieces, wrong for thousands of GiantMIDI/PDMX entries where the same
work recurs many times. This module fixes both issues for scale:

  * Vectorized chroma recall: the catalog's 12-bin chroma vectors are stacked
    into one (N, 12) matrix, so top-K recall is a single matmul instead of an
    O(N) Python loop -- the difference between minutes and hours at 10k+.
  * Incremental intra-batch dedup: every ACCEPTED piece is appended to the
    chroma matrix AND the ElasticGate, so the next candidate is deduped against
    the existing catalog PLUS everything accepted so far this batch.

Dedup semantics are identical to kernscores_expand/mutopia_dedup: chroma top-K
-> pitch-only elastic DTW (w_time=0); DUP if best_cost <= DUP_THRESHOLD (0.2885,
the calibrated bimodal-gap threshold). Net-new pieces pass the self-consistency
gate (Krumhansl key + total_bars + validate_score) before their JSON is written;
gate failures are excluded and surfaced, never silently dropped.

Callers supply an iterator of Candidate(midi_path, piece_id, composer, title);
this module owns dedup, gating, JSON writing, and the running index. piece_id
collisions with the on-disk catalog HALT (namespace pollution).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note, load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.key_estimate import estimate_key
from score_library.parse import parse_score_midi
from score_library.validate import ExpectedMeta, validate_score

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"

TOP_K = 5
W_PITCH = 1.0
W_TIME = 0.0
DUP_THRESHOLD = 0.2885


@dataclass
class Candidate:
    midi_path: Path
    piece_id: str
    composer: str
    title: str


@dataclass
class IngestResult:
    ingested: list[str]
    dups: list[tuple[str, float, str]]      # (piece_id, best_cost, match_id)
    gate_failures: list[tuple[str, str]]    # (piece_id, reason)
    parse_failures: list[tuple[str, str]]   # (piece_id, reason)
    collisions: list[str]


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


class _RunningChromaIndex:
    """Vectorized top-K chroma recall with incremental append."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        self._ids: list[str] = []
        vecs: list[np.ndarray] = []
        for pid, notes in catalog.items():
            if notes:
                self._ids.append(pid)
                vecs.append(chroma_vector(notes))
        self._matrix = np.vstack(vecs) if vecs else np.zeros((0, 12), dtype=float)

    def top_k(self, query_vec: np.ndarray, k: int) -> list[str]:
        if self._matrix.shape[0] == 0:
            return []
        sims = self._matrix @ query_vec
        k = min(k, sims.shape[0])
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [self._ids[i] for i in idx]

    def add(self, pid: str, vec: np.ndarray) -> None:
        self._ids.append(pid)
        self._matrix = np.vstack([self._matrix, vec[None, :]])


def _load_catalog_notes() -> dict[str, list[Note]]:
    skip = {"titles.json", "seed.sql"}
    catalog: dict[str, list[Note]] = {}
    for path in sorted(f for f in _SCORES_DIR.glob("*.json") if f.name not in skip):
        catalog[path.stem] = load_score_notes(path)
    return catalog


def bulk_ingest(
    candidates: Iterable[Candidate],
    *,
    progress_every: int = 250,
    on_progress: Callable[[int, IngestResult], None] | None = None,
) -> IngestResult:
    """Dedup + self-consistency ingest a candidate stream against the catalog
    plus everything accepted earlier in the same stream. Returns counts."""
    t0 = time.time()
    catalog = _load_catalog_notes()
    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    existing_ids = set(catalog)

    result = IngestResult([], [], [], [], [])
    seen = 0
    for cand in candidates:
        seen += 1
        try:
            score = parse_score_midi(cand.midi_path, cand.piece_id, cand.composer, cand.title)
            notes = _score_to_notes(score)
            if len(notes) < 2:
                raise ValueError(f"too few notes: {len(notes)}")
            q_vec = chroma_vector(notes)
            q_pc, q_li = _notes_to_events(notes)
            if q_pc.shape[0] < 2:
                raise ValueError("too few chord-events for DTW")
        except Exception as e:  # noqa: BLE001 -- surface every parse failure
            result.parse_failures.append((cand.piece_id, f"{type(e).__name__}: {e}"))
            continue

        best_cost, best_id = np.inf, ""
        for cid in chroma.top_k(q_vec, TOP_K):
            c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
            if c is not None and np.isfinite(c) and c < best_cost:
                best_cost, best_id = float(c), cid

        if best_id and best_cost <= DUP_THRESHOLD:
            result.dups.append((cand.piece_id, best_cost, best_id))
            continue

        if cand.piece_id in existing_ids:
            result.collisions.append(cand.piece_id)
            continue

        # Net-new: self-consistency gate (reuses the score parsed above), then
        # write + index incrementally.
        try:
            expected = ExpectedMeta(
                piece_id=cand.piece_id,
                expected_key=estimate_key(score),
                expected_bars=score.total_bars,
            )
            violations = validate_score(score, expected)
        except Exception as e:  # noqa: BLE001
            result.gate_failures.append((cand.piece_id, f"{type(e).__name__}: {e}"))
            continue
        if violations:
            result.gate_failures.append(
                (cand.piece_id, "; ".join(f"{v.check}: {v.detail}" for v in violations))
            )
            continue

        (_SCORES_DIR / f"{cand.piece_id}.json").write_text(json.dumps(score.model_dump(), indent=2))
        result.ingested.append(cand.piece_id)
        existing_ids.add(cand.piece_id)
        chroma.add(cand.piece_id, q_vec)
        gate._events[cand.piece_id] = (q_pc, q_li)

        if on_progress and seen % progress_every == 0:
            on_progress(seen, result)

    if result.collisions:
        raise SystemExit(
            f"ABORT: {len(result.collisions)} piece_id collisions (namespace pollution): "
            f"{result.collisions[:5]}"
        )

    print(
        f"\n=== bulk_ingest: seen={seen} ingested={len(result.ingested)} "
        f"dups={len(result.dups)} gate_fail={len(result.gate_failures)} "
        f"parse_fail={len(result.parse_failures)} ({time.time()-t0:.1f}s) ===",
        flush=True,
    )
    return result
