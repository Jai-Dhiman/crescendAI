"""Moment parsing and audio segment extraction for masterclass experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Moment:
    """A single teaching moment from a masterclass video."""

    moment_id: str
    video_id: str
    teacher: str
    stop_timestamp: float
    playing_before_start: float
    playing_before_end: float
    feedback_start: float
    feedback_end: float
    feedback_summary: str
    musical_dimension: str
    severity: str
    piece: str
    confidence: float


def load_moments(jsonl_path: Path) -> list[Moment]:
    """Parse moments JSONL file, sorted by video_id then stop_timestamp."""
    moments = []
    with open(jsonl_path) as f:
        for line in f:
            raw = json.loads(line)
            moments.append(
                Moment(
                    moment_id=raw["moment_id"],
                    video_id=raw["video_id"],
                    teacher=raw["teacher"],
                    stop_timestamp=raw["stop_timestamp"],
                    playing_before_start=raw["playing_before_start"],
                    playing_before_end=raw["playing_before_end"],
                    feedback_start=raw["feedback_start"],
                    feedback_end=raw["feedback_end"],
                    feedback_summary=raw["feedback_summary"],
                    musical_dimension=raw["musical_dimension"],
                    severity=raw["severity"],
                    piece=raw["piece"],
                    confidence=raw["confidence"],
                )
            )
    moments.sort(key=lambda m: (m.video_id, m.stop_timestamp))
    return moments
