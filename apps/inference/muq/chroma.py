# apps/inference/muq/chroma.py
"""
chroma_feature: compute 12-row chroma at 50 Hz from a mono float32 waveform.

Used by handler.py to attach chroma data to MuQ inference responses.
"""
from __future__ import annotations

import struct

import librosa
import numpy as np

_HOP = 441  # 50 Hz at 22050 Hz


def chroma_feature(y: np.ndarray, sr: int) -> tuple[bytes, int]:
    """Compute L2-normalized chroma and serialize as raw float32 bytes.

    Args:
        y: mono float32 waveform at `sr` Hz.
        sr: sample rate in Hz (must match `y`).

    Returns:
        (raw_bytes, n_frames) where raw_bytes is row-major float32 LE,
        shape (12, n_frames), and n_frames is the number of chroma columns.

    Raises:
        ValueError: if `y` is empty or `sr` is zero.
    """
    if len(y) == 0:
        raise ValueError("chroma_feature: waveform is empty")
    if sr <= 0:
        raise ValueError(f"chroma_feature: invalid sample rate {sr}")

    hop = _HOP if sr == 22050 else max(1, sr // 50)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm

    n_frames = chroma.shape[1]
    # Row-major: chroma[0, :], chroma[1, :], ..., chroma[11, :]
    flat = chroma.flatten()
    raw_bytes = struct.pack(f"<{12 * n_frames}f", *flat.tolist())
    return raw_bytes, n_frames
