"""Split audio files into 15-second chunks for eval inference."""

from __future__ import annotations

import numpy as np
from preprocessing.audio import preprocess_audio_from_bytes

CHUNK_DURATION_S = 15
SAMPLE_RATE = 24000
CHUNK_SAMPLES = CHUNK_DURATION_S * SAMPLE_RATE


def chunk_audio_file(file_path: str, max_duration: int = 300) -> list[np.ndarray]:
    """Load audio file and split into 15s chunks.

    Args:
        file_path: Path to audio file (any format ffmpeg supports).
        max_duration: Maximum audio duration in seconds.

    Returns:
        List of numpy arrays, each 15s of 24kHz mono float32.
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio, _duration = preprocess_audio_from_bytes(audio_bytes, max_duration=max_duration)

    chunks = []
    total_samples = len(audio)
    for start in range(0, total_samples, CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk) >= SAMPLE_RATE:  # skip chunks < 1s
            chunks.append(chunk)

    return chunks
