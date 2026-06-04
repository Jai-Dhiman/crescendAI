"""Matcher protocol and Ranked result type."""
from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class Ranked(NamedTuple):
    """A single (piece_id, score) result. Higher score = better match."""
    piece_id: str
    score: float


@runtime_checkable
class Matcher(Protocol):
    """Protocol for piece-ID recall matchers."""

    @property
    def name(self) -> str:
        """Short identifier for this matcher (used in report tables)."""
        ...

    def rank(self, query: np.ndarray) -> list[Ranked]:
        """Rank catalog pieces against a query chroma window (12, N).

        Returns list of Ranked sorted descending by score (highest first).
        """
        ...
