#!/usr/bin/env python3
"""
Centralized audio I/O and mel-dB preprocessing utilities.

Contract (from IMPROVEMENT_PLAN):
- Input audio: mono float32 PCM @ 22050 Hz
- Mel: n_fft=2048, hop=512, n_mels=128
- dB: librosa.power_to_db(..., ref=np.max) clipped to [-80, 0]
- Output shape: [time=128, mel=128]
- Orientation: [time, mel]
- Explicit exceptions for invalid inputs (no silent fallbacks)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple

import numpy as np

try:
    import librosa
except Exception as e:  # explicit, per user preference
    raise ImportError("librosa is required for audio preprocessing") from e


TargetShape = Tuple[int, int]


def load_audio_mono_22050(path: Union[str, Path], *, target_sr: int = 22050) -> np.ndarray:
    """
    Load audio as mono float32 at 22050 Hz.

    Args:
        path: Path to an audio file readable by librosa
        target_sr: Target sample rate (default 22050)

    Returns:
        1D numpy array (float32) of mono audio at target_sr

    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if audio is empty or cannot be loaded
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    try:
        y, sr = librosa.load(str(p), sr=target_sr, mono=True)
    except Exception as e:  # explicit exception
        raise ValueError(f"Failed to load audio: {p} ({type(e).__name__}: {e})") from e

    if y is None or y.size == 0:
        raise ValueError(f"Loaded empty audio: {p}")

    # Ensure dtype float32 (librosa returns float32 by default, but be explicit)
    y = np.asarray(y, dtype=np.float32)
    return y


def mel_db_128x128(
    y: np.ndarray,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    target_shape: TargetShape = (128, 128),
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Compute mel spectrogram, convert to dB with ref to per-example max, clip to [-80, 0],
    and return a [time=128, mel=128] window (center crop or pad with -80).

    Args:
        y: mono audio waveform (float32)
        sr: sample rate (expected 22050)
        n_fft: FFT window size
        hop_length: hop size
        n_mels: number of mel bins (expected 128)
        target_shape: (time, mel) target shape (default (128, 128))
        top_db: dynamic range for power_to_db (default 80 dB)

    Returns:
        np.ndarray of shape [time=128, mel=128], dtype float32, values in [-80, 0]

    Raises:
        ValueError: on invalid inputs
    """
    if y is None or y.ndim != 1 or y.size == 0:
        raise ValueError("Input audio must be a non-empty 1D array")
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    # Mel spectrogram (power)
    try:
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
    except Exception as e:
        raise ValueError(f"Failed to compute mel spectrogram: {type(e).__name__}: {e}") from e

    # Convert to dB relative to max power in this clip
    # librosa.power_to_db with ref=np.max and top_db clamps to [-top_db, 0]
    try:
        S_db = librosa.power_to_db(S, ref=np.max, top_db=top_db)
    except Exception as e:
        raise ValueError(f"Failed to convert power to dB: {type(e).__name__}: {e}") from e

    # Orient as [time, mel]
    S_db = S_db.T  # [time, mel]

    # Clip explicitly to [-80, 0]
    S_db = np.clip(S_db, -top_db, 0.0)

    # Ensure target shape via center crop or pad with -80
    t_target, f_target = target_shape
    t_cur, f_cur = S_db.shape

    # Frequency crop/pad first (should normally already be 128)
    if f_cur > f_target:
        # center crop
        start = (f_cur - f_target) // 2
        S_db = S_db[:, start : start + f_target]
    elif f_cur < f_target:
        pad_f = f_target - f_cur
        S_db = np.pad(S_db, ((0, 0), (0, pad_f)), mode="constant", constant_values=-top_db)

    # Time crop/pad to window ~3.0s (128 frames at hop 512, sr 22050)
    t_cur = S_db.shape[0]
    if t_cur > t_target:
        start = (t_cur - t_target) // 2
        S_db = S_db[start : start + t_target, :]
    elif t_cur < t_target:
        pad_t = t_target - t_cur
        S_db = np.pad(S_db, ((0, pad_t), (0, 0)), mode="constant", constant_values=-top_db)

    if S_db.shape != (t_target, f_target):
        raise ValueError(
            f"Post-process shape {S_db.shape} != target {(t_target, f_target)}"
        )

    return S_db.astype(np.float32)


def mel_db_time_major(
    y: np.ndarray,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Compute full-length mel spectrogram in dB, time-major [time, mel], clipped to [-80, 0].
    No time cropping; intended for dataset loaders that will segment into windows.

    Args:
        y: mono audio waveform (float32)
        sr: sample rate (expected 22050)
        n_fft: FFT window size
        hop_length: hop size
        n_mels: number of mel bins (expected 128)
        top_db: dynamic range for power_to_db (default 80 dB)

    Returns:
        np.ndarray of shape [time, mel], dtype float32, values in [-80, 0]
    """
    if y is None or y.ndim != 1 or y.size == 0:
        raise ValueError("Input audio must be a non-empty 1D array")
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    try:
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
    except Exception as e:
        raise ValueError(f"Failed to compute mel spectrogram: {type(e).__name__}: {e}") from e

    try:
        S_db = librosa.power_to_db(S, ref=np.max, top_db=top_db)
    except Exception as e:
        raise ValueError(f"Failed to convert power to dB: {type(e).__name__}: {e}") from e

    S_db = np.clip(S_db, -top_db, 0.0)
    S_db = S_db.T.astype(np.float32)  # [time, mel]
    return S_db


def mel_db_128x128_from_file(
    path: Union[str, Path], *, target_sr: int = 22050, **kwargs
) -> np.ndarray:
    """
    Convenience wrapper: load audio and compute a single [128,128] mel-dB window.

    Args:
        path: path to audio file
        target_sr: sample rate for loading (default 22050)
        **kwargs: forwarded to mel_db_128x128

    Returns:
        [128,128] float32 dB array in [-80, 0]
    """
    y = load_audio_mono_22050(path, target_sr=target_sr)
    return mel_db_128x128(y, sr=target_sr, **kwargs)
