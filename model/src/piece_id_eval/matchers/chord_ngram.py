"""Chord-token n-gram inverted index matcher.

Each chroma column is quantized to a 12-bit pitch-class mask (binarized at
column mean). N-gram tokens formed over consecutive columns. The catalog is
indexed; query n-grams are looked up and hit counts summed per piece.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from piece_id_eval.matchers.base import Ranked


def _chroma_to_tokens(chroma: np.ndarray, oti: bool) -> list[int]:
    """Convert (12, N) chroma to list of 12-bit integer tokens."""
    tokens: list[int] = []
    for col in chroma.T:
        threshold = float(col.mean())
        bits = int(sum((1 << i) for i, v in enumerate(col) if v >= threshold))
        if oti:
            # OTI: rotate to minimum integer representation
            bits = min(
                int(((bits >> k) | ((bits << (12 - k)) & 0xFFF)) & 0xFFF)
                for k in range(12)
            )
        tokens.append(bits)
    return tokens


def _make_ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class ChordNgramMatcher:
    """Inverted chord-token n-gram index."""

    def __init__(
        self, catalog: dict[str, np.ndarray], oti: bool = False, n: int = 3
    ) -> None:
        self._oti = oti
        self._n = n
        self._index: dict[tuple[int, ...], list[str]] = defaultdict(list)
        self._pieces = list(catalog.keys())
        for piece_id, chroma in catalog.items():
            tokens = _chroma_to_tokens(chroma, oti)
            for gram in _make_ngrams(tokens, n):
                self._index[gram].append(piece_id)

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"chord_ngram_n{self._n}{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        tokens = _chroma_to_tokens(query, self._oti)
        hit_counts: dict[str, int] = defaultdict(int)
        for piece_id in self._pieces:
            hit_counts[piece_id] = 0
        for gram in _make_ngrams(tokens, self._n):
            for piece_id in self._index.get(gram, []):
                hit_counts[piece_id] += 1
        total = max(sum(hit_counts.values()), 1)
        results = [
            Ranked(piece_id=pid, score=count / total)
            for pid, count in hit_counts.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results
