# model/src/piece_id_eval/matchers/base.py
"""Matcher protocol and Ranked result type (note-based)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable, NamedTuple

from piece_id_eval.notes import Note


class Ranked(NamedTuple):
    """A single (piece_id, score) result. Higher score = better match."""
    piece_id: str
    score: float


@runtime_checkable
class Matcher(Protocol):
    """Protocol for note-based piece-ID matchers."""

    @property
    def name(self) -> str:
        """Short identifier for this matcher (used in report tables)."""
        ...

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces against a query note list.

        Returns list of Ranked sorted descending by score (highest first).
        """
        ...
