"""Shared audio I/O and segmentation utilities.

Uses Pedalboard for fast audio I/O (4x faster than librosa at scale).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono, resample to target_sr.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, etc.).
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (audio as 1D float32 ndarray, sample rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    from pedalboard.io import AudioFile

    with AudioFile(str(path)).resampled_to(target_sr) as f:
        audio = f.read(f.frames)  # shape: (channels, samples)

    # Convert to mono if stereo/multi-channel
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(axis=0)
    elif audio.ndim == 2:
        audio = audio[0]

    return audio.astype(np.float32), target_sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = 30.0,
    min_duration: float = 5.0,
) -> list[dict]:
    """Split audio into fixed-duration segments.

    Args:
        audio: 1D audio array.
        sr: Sample rate in Hz.
        segment_duration: Duration of each segment in seconds.
        min_duration: Minimum duration for the last segment. Shorter tails
            are dropped.

    Returns:
        List of dicts with keys: audio (np.ndarray), start_sec (float),
        end_sec (float).
    """
    total_samples = len(audio)
    segment_samples = int(segment_duration * sr)
    min_samples = int(min_duration * sr)

    segments = []
    offset = 0

    while offset < total_samples:
        end = min(offset + segment_samples, total_samples)
        chunk = audio[offset:end]

        if len(chunk) < min_samples:
            break  # drop runt tail

        start_sec = offset / sr
        end_sec = end / sr

        segments.append({
            "audio": chunk,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        })

        offset = end

    return segments


def save_audio(audio: np.ndarray, path: Path, sr: int = 24000) -> None:
    """Write mono audio to WAV file.

    Args:
        audio: 1D float32 audio array.
        path: Output file path.
        sr: Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from pedalboard.io import AudioFile

    audio_2d = audio.reshape(1, -1).astype(np.float32)
    with AudioFile(str(path), "w", samplerate=sr, num_channels=1) as f:
        f.write(audio_2d)
