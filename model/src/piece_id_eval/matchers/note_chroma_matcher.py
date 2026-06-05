# model/src/piece_id_eval/matchers/note_chroma_matcher.py
"""C1: Note-chroma cosine matcher.

Computes a single 12-bin key-dependent chroma vector per piece and query,
then ranks by cosine similarity. Fast O(N_catalog) search.

Hides: chroma_vector computation, cosine similarity, catalog indexing.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note


class NoteChromaMatcher:
    """Cosine similarity over key-dependent note-chroma vectors (C1)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        """Pre-compute chroma vectors for all catalog pieces.

        Args:
            catalog: {piece_id: list[Note]} for all catalog entries.
        """
        self._index: dict[str, np.ndarray] = {
            pid: chroma_vector(notes)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "note_chroma_cosine"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by cosine similarity to query chroma vector.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by cosine similarity.

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_vec = chroma_vector(query)
        results: list[Ranked] = []
        for piece_id, ref_vec in self._index.items():
            similarity = float(np.dot(q_vec, ref_vec))
            results.append(Ranked(piece_id=piece_id, score=similarity))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
