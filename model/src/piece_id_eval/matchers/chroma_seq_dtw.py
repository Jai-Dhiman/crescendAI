# model/src/piece_id_eval/matchers/chroma_seq_dtw.py
"""C4: Subsequence DTW over note-derived chroma sequences.

Computes chroma_sequence (12, T) for both query and each catalog piece
at frame_seconds=0.5, then runs the same subsequence sliding-window DTW
as C3 but on 12-dim chroma columns instead of scalar pitch.
Cost normalized by query frame count; negated to score.

Hides: chroma_sequence computation, column-wise Euclidean DTW, normalization.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.note_chroma import chroma_sequence
from piece_id_eval.notes import Note

_DEFAULT_FRAME_SECONDS = 0.5


class ChromaSeqDtwMatcher:
    """Subsequence chroma-sequence DTW matcher (C4)."""

    def __init__(
        self,
        catalog: dict[str, list[Note]],
        frame_seconds: float = _DEFAULT_FRAME_SECONDS,
    ) -> None:
        self._frame_seconds = frame_seconds
        self._catalog_seq: dict[str, np.ndarray] = {
            pid: chroma_sequence(notes, frame_seconds)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "chroma_seq_dtw"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by subsequence chroma-sequence DTW.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by score (-normalized cost).

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_seq = chroma_sequence(query, self._frame_seconds)  # (12, Q)
        Q = q_seq.shape[1]
        results: list[Ranked] = []
        for piece_id, ref_seq in self._catalog_seq.items():
            cost = self._subseq_dtw_cost(q_seq, ref_seq, Q)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _subseq_dtw_cost(
        self, query: np.ndarray, ref: np.ndarray, Q: int
    ) -> float:
        """Subsequence DTW of query (12, Q) against ref (12, R).

        Slides a window of Q frames over ref; returns minimum total
        column-wise Euclidean cost / Q.
        If Q > R, falls back to full DTW.
        """
        R = ref.shape[1]
        if Q == 0:
            return 0.0
        if Q > R:
            return self._full_dtw(query.T, ref.T) / Q

        best = float("inf")
        for start in range(R - Q + 1):
            seg = ref[:, start : start + Q]  # (12, Q)
            cost = float(np.sum(np.linalg.norm(query - seg, axis=0)))
            if cost < best:
                best = cost
        return best / Q

    def _full_dtw(self, q: np.ndarray, r: np.ndarray) -> float:
        """Full DTW on (Q, 12) and (R, 12) arrays."""
        Q, R = q.shape[0], r.shape[0]
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(np.linalg.norm(q[i - 1] - r[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R])
