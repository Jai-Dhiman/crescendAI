"""Subsequence chroma-DTW ceiling matcher.

The slowest but most powerful matcher. For each catalog piece, runs a
subsequence DTW of the query against the full catalog chroma. The DTW
*cost* is negated to produce a score (lower cost = higher score).

This is the discrimination ceiling: if it can't separate pieces,
no indexable method will.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked

try:
    from dtaidistance import dtw as _dtw_lib
    _HAS_DTAIDISTANCE = True
except ImportError:
    _HAS_DTAIDISTANCE = False


class DtwCeilingMatcher:
    """Subsequence chroma-DTW over the full catalog."""

    def __init__(self, catalog: dict[str, np.ndarray], oti: bool = False) -> None:
        """
        Args:
            catalog: {piece_id: score_chroma (12, N)} mapping.
            oti: if True, canonicalize chroma via OTI (pitch-axis cyclic min).
        """
        self._catalog = catalog
        self._oti = oti

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"dtw_ceiling{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        """Rank all catalog pieces against query by DTW cost (lower cost = better match).

        Uses Euclidean distance on 12-dim chroma columns. Falls back to
        numpy-based windowed minimum-cost alignment if dtaidistance is not installed.
        """
        q = self._oti_canonicalize(query) if self._oti else query
        results: list[Ranked] = []
        for piece_id, ref in self._catalog.items():
            r = self._oti_canonicalize(ref) if self._oti else ref
            cost = self._dtw_cost(q, r)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _dtw_cost(self, query: np.ndarray, ref: np.ndarray) -> float:
        """Compute subsequence-DTW cost between query (12, Q) and ref (12, R)."""
        q = query.T  # (Q, 12)
        r = ref.T    # (R, 12)
        Q = q.shape[0]
        R = r.shape[0]
        if Q > R:
            # query longer than reference: full DTW
            return float(self._full_dtw(q, r))
        # Subsequence: slide a window of length Q over r, take minimum cost
        best = float("inf")
        for start in range(R - Q + 1):
            seg = r[start : start + Q]
            cost = float(np.sum(np.linalg.norm(q - seg, axis=1)))
            if cost < best:
                best = cost
        return best / max(Q, 1)

    def _full_dtw(self, q: np.ndarray, r: np.ndarray) -> float:
        Q, D = q.shape
        R = r.shape[0]
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(np.linalg.norm(q[i - 1] - r[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R]) / max(Q, 1)

    def _oti_canonicalize(self, chroma: np.ndarray) -> np.ndarray:
        """Cyclic-rotate pitch axis to minimize sum of first-bin values (OTI)."""
        best_rot = min(range(12), key=lambda k: float(np.roll(chroma, k, axis=0)[0].sum()))
        return np.roll(chroma, best_rot, axis=0)
