"""Position-stratified chunk sampler.

Given a list of pieces with durations, produces a deterministic manifest of
chunk start times stratified into 5 position buckets: intro (0-5%), early
(5-25%), middle (25-65%), late (65-90%), cadence (90-100%).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("intro", 0.0, 0.05),
    ("early", 0.05, 0.25),
    ("middle", 0.25, 0.65),
    ("late", 0.65, 0.90),
    ("cadence", 0.90, 1.0),
)


@dataclass(frozen=True)
class PieceSpec:
    piece_id: str
    duration_s: float


@dataclass(frozen=True)
class Chunk:
    piece_id: str
    start_s: float
    chunk_len_s: float
    piece_duration_s: float
    position_bucket: str


def sample_chunks(
    pieces: list[PieceSpec],
    n_per_piece: int,
    chunk_len_s: float,
    seed: int,
) -> list[Chunk]:
    if n_per_piece < len(BUCKETS):
        raise ValueError(f"n_per_piece={n_per_piece} < {len(BUCKETS)} buckets")
    rng = random.Random(seed)
    out: list[Chunk] = []
    per_bucket_base = n_per_piece // len(BUCKETS)
    remainder = n_per_piece - per_bucket_base * len(BUCKETS)
    counts = [per_bucket_base + (1 if i < remainder else 0) for i in range(len(BUCKETS))]
    for piece in pieces:
        if piece.duration_s <= chunk_len_s:
            raise ValueError(f"piece {piece.piece_id} duration {piece.duration_s} <= chunk_len {chunk_len_s}")
        for (name, lo, hi), count in zip(BUCKETS, counts):
            lo_s = lo * piece.duration_s
            hi_s = max(lo_s + 1e-3, hi * piece.duration_s - chunk_len_s)
            for _ in range(count):
                start = rng.uniform(lo_s, hi_s)
                out.append(Chunk(piece.piece_id, start, chunk_len_s, piece.duration_s, name))
    return out
