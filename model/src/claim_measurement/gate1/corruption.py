"""Construction-known audio corruption for GATE 1 localization-robustness testing.

Each corruption degrades a clean cached clip in a way whose induced clean->corrupt
time mapping is known *by construction*. For time-warps the map (`warp_map`) is
derived from the actual produced sample counts, so it describes the audio exactly;
identity corruptions (noise, dropout, pitch) preserve duration and so carry the
identity map. Localization error is then measured against this known map without
any hand-annotated ground truth.

The truth label / verifier never runs here -- this module only manufactures the
error-rich audio and the ground-truth time map that GATE 1 measures against.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class WarpMap:
    """Piecewise-linear clean_sec -> corrupt_sec map. Breakpoints are exact
    (cumulative produced sample counts), so interpolation between them is exact
    for the constant-rate pieces that produced them."""

    clean_sec: tuple[float, ...]
    corrupt_sec: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"clean_sec": list(self.clean_sec), "corrupt_sec": list(self.corrupt_sec)}

    @classmethod
    def from_dict(cls, d: dict) -> "WarpMap":
        return cls(tuple(d["clean_sec"]), tuple(d["corrupt_sec"]))

    @classmethod
    def identity(cls, duration_sec: float) -> "WarpMap":
        return cls((0.0, duration_sec), (0.0, duration_sec))


def warp_time(warp_map: WarpMap | dict, t: float) -> float:
    """Map a clean-time second to its corrupt-time second under `warp_map`."""
    wm = warp_map if isinstance(warp_map, WarpMap) else WarpMap.from_dict(warp_map)
    return float(np.interp(t, wm.clean_sec, wm.corrupt_sec))


def _normalize_segments(
    segments: list[tuple[float, float, float]], total_sec: float
) -> list[tuple[float, float, float]]:
    """Fill the gaps between requested warp segments with rate-1.0 passthrough,
    yielding a contiguous cover of [0, total_sec]. Fails loud on overlap."""
    segs = sorted(segments, key=lambda s: s[0])
    pieces: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end, rate in segs:
        if start < cursor - 1e-9:
            raise ValueError(f"overlapping/out-of-order warp segments near {start}")
        if rate <= 0:
            raise ValueError(f"warp rate must be > 0, got {rate}")
        if start > cursor:
            pieces.append((cursor, start, 1.0))
        pieces.append((start, end, rate))
        cursor = end
    if cursor < total_sec - 1e-9:
        pieces.append((cursor, total_sec, 1.0))
    return pieces


def apply_piecewise_time_warp(
    audio: np.ndarray, sr: int, segments: list[tuple[float, float, float]]
) -> tuple[np.ndarray, WarpMap]:
    """Piecewise constant-rate time-warp. `segments` = [(start_sec, end_sec, rate)],
    rate > 1 speeds up (compresses), rate < 1 slows down. Untouched spans pass
    through unchanged. Returns (corrupted_audio, warp_map) where warp_map is built
    from the actual produced sample counts -> exact ground truth."""
    total_sec = len(audio) / sr
    pieces = _normalize_segments(segments, total_sec)

    out_parts: list[np.ndarray] = []
    clean_breaks = [0.0]
    corrupt_breaks = [0.0]
    cum_clean = 0
    cum_corrupt = 0
    for start, end, rate in pieces:
        a = int(round(start * sr))
        b = int(round(end * sr))
        seg = audio[a:b]
        if seg.size == 0:
            continue
        if rate == 1.0:
            seg_out = seg
        else:
            seg_out = librosa.effects.time_stretch(seg, rate=rate).astype(np.float32)
        out_parts.append(seg_out)
        cum_clean += seg.size
        cum_corrupt += seg_out.size
        clean_breaks.append(cum_clean / sr)
        corrupt_breaks.append(cum_corrupt / sr)

    corrupted = np.concatenate(out_parts).astype(np.float32) if out_parts else audio.copy()
    return corrupted, WarpMap(tuple(clean_breaks), tuple(corrupt_breaks))


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB). W = identity (no time shift)."""
    sig_power = float(np.mean(audio.astype(np.float64) ** 2))
    if sig_power <= 0:
        raise ValueError("cannot set SNR on a silent signal")
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(audio.shape) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def silence_region(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> np.ndarray:
    """Zero a time window (dropped/missing notes). W = identity (duration preserved)."""
    out = audio.copy()
    out[int(round(start_sec * sr)):int(round(end_sec * sr))] = 0.0
    return out


def pitch_shift_region(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float, semitones: float
) -> np.ndarray:
    """Pitch-shift a time window (wrong notes). W = identity (duration preserved)."""
    out = audio.copy()
    a = int(round(start_sec * sr))
    b = int(round(end_sec * sr))
    out[a:b] = librosa.effects.pitch_shift(
        audio[a:b], sr=sr, n_steps=semitones
    ).astype(np.float32)[: b - a]
    return out
