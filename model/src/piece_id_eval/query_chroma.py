"""Production-identical audio chroma extraction and windowing.

Uses the same chroma_cqt + 1e-3 floor + L2-normalization recipe as
apps/inference/muq/chroma.py::chroma_feature. Paths are __file__-anchored.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def audio_to_chroma(wav_path: Path) -> tuple[np.ndarray, float]:
    """Load a WAV file and return (chroma, frame_rate_hz).

    The chroma is computed identically to the production MuQ endpoint:
    chroma_cqt at target ~50 Hz, 1e-3 floor, L2-normalized columns.

    Returns:
        chroma: np.ndarray shape (12, N), dtype float32, L2-normed columns
        frame_rate_hz: actual frame rate (sr / hop)

    Raises:
        FileNotFoundError: if wav_path does not exist.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"audio not found: {wav_path}")

    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    hop = max(1, round(sr / 50))
    frame_rate_hz = float(sr / hop)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).astype(np.float32)
    chroma += 1e-3
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norms
    return chroma, frame_rate_hz


def window_chroma(
    chroma: np.ndarray,
    frame_rate_hz: float,
    window_seconds: float,
    hop_seconds: float,
) -> list[np.ndarray]:
    """Slice a chroma array into fixed-length overlapping windows.

    Windows that would extend beyond the chroma array are discarded (no padding).

    Returns:
        List of (12, window_frames) arrays, each a view into chroma.

    Raises:
        ValueError: if window_seconds <= 0 or hop_seconds <= 0.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

    window_frames = int(window_seconds * frame_rate_hz)
    hop_frames = int(hop_seconds * frame_rate_hz)
    n_frames = chroma.shape[1]
    windows: list[np.ndarray] = []
    start = 0
    while start + window_frames <= n_frames:
        windows.append(chroma[:, start : start + window_frames])
        start += hop_frames
    return windows
