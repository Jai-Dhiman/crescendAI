# apps/inference/muq/chroma.py
"""
chroma_feature: compute 12-row chroma at ~50 Hz from a mono float32 waveform.

Used by handler.py to attach chroma data to MuQ inference responses.
"""
from __future__ import annotations

import struct

import librosa
import numpy as np


def chroma_feature(y: np.ndarray, sr: int) -> tuple[bytes, int, float]:
    """Compute L2-normalized chroma and serialize as raw float32 bytes.

    The hop size is derived from `sr` to target ~50 Hz frame rate.
    The actual achieved frame rate (sr / hop) is returned as the third element.

    Args:
        y: mono float32 waveform at `sr` Hz.
        sr: sample rate in Hz (must match `y`).

    Returns:
        (raw_bytes, n_frames, frame_rate_hz) where raw_bytes is row-major
        float32 LE, shape (12, n_frames); n_frames is the number of chroma
        columns; and frame_rate_hz is the actual frame rate in Hz.

    Raises:
        ValueError: if `y` is empty or `sr` is zero or negative.
    """
    if len(y) == 0:
        raise ValueError("chroma_feature: waveform is empty")
    if sr <= 0:
        raise ValueError(f"chroma_feature: invalid sample rate {sr}")

    hop = max(1, round(sr / 50))
    frame_rate_hz = sr / hop

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm

    n_frames = chroma.shape[1]
    # Row-major: chroma[0, :], chroma[1, :], ..., chroma[11, :]
    flat = chroma.flatten()
    raw_bytes = struct.pack(f"<{12 * n_frames}f", *flat.tolist())
    return raw_bytes, n_frames, frame_rate_hz
