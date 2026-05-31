"""Synthetic silence chunk generator for guard G3."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SilenceChunk:
    waveform: np.ndarray  # float32 mono
    sr: int
    kind: str  # "zero" or "low_noise"


def generate_silence_chunks(n: int, sr: int, chunk_len_s: float, seed: int) -> list[SilenceChunk]:
    if n < 2:
        raise ValueError("need at least 2 chunks to cover both kinds")
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    n_samples = int(sr * chunk_len_s)
    chunks: list[SilenceChunk] = []
    for i in range(n):
        if i == 0:
            wav = np.zeros(n_samples, dtype=np.float32)
            kind = "zero"
        elif i == 1:
            wav = (np_rng.randn(n_samples).astype(np.float32) * 0.005)
            kind = "low_noise"
        else:
            if rng.random() < 0.5:
                wav = np.zeros(n_samples, dtype=np.float32)
                kind = "zero"
            else:
                wav = (np_rng.randn(n_samples).astype(np.float32) * 0.005)
                kind = "low_noise"
        chunks.append(SilenceChunk(wav, sr, kind))
    return chunks
