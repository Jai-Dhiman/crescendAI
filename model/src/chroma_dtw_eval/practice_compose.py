"""Synthetic practice composition for guard G4.

Stitches a source chroma matrix into 15s chunks following four patterns that
mimic real rehearsal behaviour: repeat (play same bar twice), restart (start
over partway through), jump (skip forward mid-chunk), partial (play first half,
silence second half).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


PATTERNS = ("repeat", "restart", "jump", "partial")


@dataclass
class ComposedSequence:
    pattern: str
    chroma: np.ndarray  # (12, chunk_len_frames)
    stitch_points: list[tuple[int, int]] = field(default_factory=list)


def _take(source: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(source.shape[1], start + length)
    seg = source[:, start:end]
    if seg.shape[1] < length:
        pad = np.zeros((12, length - seg.shape[1]), dtype=np.float32)
        pad[:, :] = 1.0 / np.sqrt(12.0)
        seg = np.concatenate([seg, pad], axis=1)
    return seg


def compose_practice_sequence(
    source: np.ndarray, pattern: str, chunk_len_frames: int, rng: random.Random
) -> ComposedSequence:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}")
    if source.shape[0] != 12:
        raise ValueError(f"source must be 12-row, got shape {source.shape}")
    half = chunk_len_frames // 2
    max_start = max(0, source.shape[1] - chunk_len_frames - 1)
    s0 = rng.randint(0, max_start) if max_start > 0 else 0

    out = np.zeros((12, chunk_len_frames), dtype=np.float32)
    stitches: list[tuple[int, int]] = [(0, s0)]
    if pattern == "repeat":
        first = _take(source, s0, half)
        out[:, :half] = first
        out[:, half:half * 2] = first
        if chunk_len_frames > half * 2:
            out[:, half * 2:] = _take(source, s0 + half, chunk_len_frames - half * 2)
        stitches.append((half, s0))
    elif pattern == "restart":
        out[:, :half] = _take(source, s0, half)
        out[:, half:] = _take(source, s0, chunk_len_frames - half)
        stitches.append((half, s0))
    elif pattern == "jump":
        out[:, :half] = _take(source, s0, half)
        jump_to = min(source.shape[1] - 1, s0 + half * 4)
        out[:, half:] = _take(source, jump_to, chunk_len_frames - half)
        stitches.append((half, jump_to))
    elif pattern == "partial":
        out[:, :half] = _take(source, s0, half)
        out[:, half:] = 1.0 / np.sqrt(12.0)
    return ComposedSequence(pattern, out, stitches)


def compose_batch(
    source: np.ndarray, n_per_pattern: int, chunk_len_frames: int, seed: int
) -> list[ComposedSequence]:
    rng = random.Random(seed)
    out: list[ComposedSequence] = []
    for pattern in PATTERNS:
        for _ in range(n_per_pattern):
            out.append(compose_practice_sequence(source, pattern, chunk_len_frames, rng))
    return out
