# model/src/piece_id_eval/matchers/dtw_ceiling.py
"""C3: Subsequence onset-ordered pitch DTW ceiling matcher (note-based).

For each catalog piece, extracts the pitch sequence (sorted by onset) and
runs a subsequence DTW of the query pitch sequence against it. DTW cost is
normalized by query length and negated to produce a score (lower cost = higher
score).

This is the discrimination ceiling: if note-to-note DTW cannot separate
pieces, no indexable method will.

Hides: pitch extraction, subsequence DTW, query-length normalization.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.notes import Note


class DtwCeilingMatcher:
    """Subsequence onset-ordered pitch DTW over note sequences (C3)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        """Pre-extract pitch sequences for all catalog pieces.

        Args:
            catalog: {piece_id: list[Note]} for all catalog entries.
        """
        self._pitches: dict[str, np.ndarray] = {
            pid: np.array([n.pitch for n in sorted(notes, key=lambda n: n.onset)], dtype=np.float32)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "dtw_ceiling"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by subsequence pitch DTW cost (lower = better).

        Score = -normalized_cost (higher = better match).

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by score.

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_pitches = np.array(
            [n.pitch for n in sorted(query, key=lambda n: n.onset)], dtype=np.float32
        )
        results: list[Ranked] = []
        for piece_id, ref_pitches in self._pitches.items():
            cost = self._subseq_dtw_cost(q_pitches, ref_pitches)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _subseq_dtw_cost(self, query: np.ndarray, ref: np.ndarray) -> float:
        """Subsequence DTW: slide query over ref; return min normalized path cost.

        If query is longer than ref, falls back to full DTW.
        Cost is normalized by query length.
        """
        Q = len(query)
        R = len(ref)
        if Q == 0:
            return 0.0
        if Q > R:
            return self._full_dtw(query, ref) / Q

        best = float("inf")
        for start in range(R - Q + 1):
            seg = ref[start : start + Q]
            cost = float(np.sum(np.abs(query - seg)))
            if cost < best:
                best = cost
        return best / Q

    def _full_dtw(self, query: np.ndarray, ref: np.ndarray) -> float:
        Q = len(query)
        R = len(ref)
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(abs(query[i - 1] - ref[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R])
