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
    """Parse and validate a probe clip: readable RIFF/WAVE, mono, expected
    sample rate, non-empty, not truncated (declared frames all present)."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            nframes = wf.getnframes()
            sampwidth = wf.getsampwidth()
            payload = wf.readframes(nframes)
    except (wave.Error, EOFError) as exc:
        raise MalformedClipError(f"{path}: not a readable WAV file ({exc})") from exc

    if channels != 1:
        raise MalformedClipError(f"{path}: expected mono, got {channels} channels")
    if rate != expected_sample_rate:
        raise MalformedClipError(
            f"{path}: expected sample rate {expected_sample_rate}, got {rate}"
        )
    if nframes == 0:
        raise MalformedClipError(f"{path}: zero-length audio")
    expected_bytes = nframes * channels * sampwidth
    if len(payload) != expected_bytes:
        raise MalformedClipError(
            f"{path}: truncated audio data "
            f"(header declares {expected_bytes} bytes, read {len(payload)})"
        )
    return WavInfo(
        path=path,
        sample_rate=rate,
        num_frames=nframes,
        duration_seconds=nframes / rate,
    )
