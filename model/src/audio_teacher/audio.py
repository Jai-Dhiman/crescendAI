"""WAV header validation for probe clips."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


class MalformedClipError(Exception):
    """A probe clip failed WAV validation. The message always names the file."""


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    num_frames: int
    duration_seconds: float


def validate_wav(path: Path | str, expected_sample_rate: int) -> WavInfo:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        nframes = wf.getnframes()
    return WavInfo(
        path=Path(path),
        sample_rate=rate,
        num_frames=nframes,
        duration_seconds=nframes / rate,
    )
