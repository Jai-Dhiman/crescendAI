"""Moment parsing and audio segment extraction for masterclass experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import groupby
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


@dataclass
class Segment:
    """An audio segment labeled as STOP or CONTINUE."""

    segment_id: str
    video_id: str
    label: str  # "stop" or "continue"
    start_time: float  # seconds into the WAV
    end_time: float
    moment_id: str | None = None  # linked moment for STOP segments


def identify_segments(
    moments: list[Moment],
    min_continue_duration: float = 5.0,
) -> list[Segment]:
    """Identify STOP and CONTINUE segments from moments.

    STOP segments: playing window before each teacher intervention.
    CONTINUE segments: gaps between consecutive moments in the same video
    where the student was playing but the teacher did not stop.
    """
    segments: list[Segment] = []
    seq = 0

    for video_id, group in groupby(moments, key=lambda m: m.video_id):
        video_moments = list(group)

        for i, m in enumerate(video_moments):
            segments.append(
                Segment(
                    segment_id=f"stop_{seq:04d}",
                    video_id=m.video_id,
                    label="stop",
                    start_time=m.playing_before_start,
                    end_time=m.playing_before_end,
                    moment_id=m.moment_id,
                )
            )
            seq += 1

            if i < len(video_moments) - 1:
                next_m = video_moments[i + 1]
                gap_start = m.feedback_end
                gap_end = next_m.playing_before_start

                if gap_end - gap_start >= min_continue_duration:
                    segments.append(
                        Segment(
                            segment_id=f"cont_{seq:04d}",
                            video_id=m.video_id,
                            label="continue",
                            start_time=gap_start,
                            end_time=gap_end,
                        )
                    )
                    seq += 1

    return segments
