"""ClipScout: discovers candidate piano clips from YouTube + TikTok."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    url: str
    source_type: str
    duration_sec: float
    title: str


@dataclass(frozen=True)
class SourceCriteria:
    source_types: list[str]
    max_duration_sec: float
    weights: dict[str, float]


class _Backend(Protocol):
    def search(self, query: str, max_results: int) -> list[Candidate]: ...


class ClipScout:
    def __init__(self, youtube_backend: _Backend | None, tiktok_backend: _Backend | None):
        self._yt = youtube_backend
        self._tt = tiktok_backend

    def search(self, criteria: SourceCriteria, count: int) -> list[Candidate]:
        raw: list[Candidate] = []
        if self._yt is not None:
            raw.extend(self._yt.search(query="piano performance", max_results=count))
        if self._tt is not None:
            raw.extend(self._tt.search(query="piano performance", max_results=count))

        filtered = [
            c for c in raw
            if c.source_type in criteria.source_types
            and c.duration_sec <= criteria.max_duration_sec
        ]
        ranked = sorted(
            filtered,
            key=lambda c: criteria.weights.get(c.source_type, 0.0),
            reverse=True,
        )
        return ranked[:count]
