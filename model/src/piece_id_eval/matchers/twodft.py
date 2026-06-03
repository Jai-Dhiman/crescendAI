"""2D-FFT-magnitude embedding + cosine similarity matcher.

Each catalog chroma fingerprint is embedded as a concatenation of:
  1. Mean chroma vector (12,) — captures pitch-class distribution
  2. Magnitude of the 1D FFT of the temporal mean chroma (time-shift invariant)
  3. Low-frequency block of the 2D FFT magnitude (texture)

The combined embedding is L2-normalized, yielding a fixed-size cosine-comparable
vector that is sensitive to harmonic content differences between pieces.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked

_FREQ_COLS = 50


def _embed(chroma: np.ndarray) -> np.ndarray:
    """Embed a (12, N) chroma as a fixed-size L2-normalized 1D vector."""
    c = chroma.astype(np.float32)

    # Part 1: mean pitch-class profile (12,)
    mean_pc = c.mean(axis=1)

    # Part 2: 1D FFT magnitude of temporal mean (time-shift invariant, 12-dim)
    temporal_mean = c.mean(axis=0)
    fft1d = np.abs(np.fft.rfft(temporal_mean))[:12]

    # Part 3: 2D FFT magnitude low-frequency block
    n_cols = max(c.shape[1], _FREQ_COLS * 2)
    if c.shape[1] < n_cols:
        c_padded = np.pad(c, ((0, 0), (0, n_cols - c.shape[1])))
    else:
        c_padded = c
    mag2d = np.abs(np.fft.rfft2(c_padded))
    block = mag2d[:12, :_FREQ_COLS]
    flat2d = block.flatten()

    flat = np.concatenate([mean_pc, fft1d, flat2d])
    norm = np.linalg.norm(flat) + 1e-9
    return flat / norm


class TwoDFTMatcher:
    """2D-FFT embedding + cosine similarity matcher."""

    def __init__(self, catalog: dict[str, np.ndarray], oti: bool = False) -> None:
        self._oti = oti
        self._embeddings: dict[str, np.ndarray] = {}
        for piece_id, chroma in catalog.items():
            c = self._oti_canonicalize(chroma) if oti else chroma
            self._embeddings[piece_id] = _embed(c)

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"twodft{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        q = self._oti_canonicalize(query) if self._oti else query
        q_emb = _embed(q)
        results = [
            Ranked(piece_id=pid, score=float(np.dot(q_emb, ref_emb)))
            for pid, ref_emb in self._embeddings.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _oti_canonicalize(self, chroma: np.ndarray) -> np.ndarray:
        best_rot = min(range(12), key=lambda k: float(np.roll(chroma, k, axis=0)[0].sum()))
        return np.roll(chroma, best_rot, axis=0)
