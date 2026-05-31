"""Hash-keyed on-disk chroma cache.

Computes 12-bin chroma at ~target_frame_rate_hz from a mono audio file using
the same recipe as apps/inference/muq/chroma.py (chroma_cqt + 1e-3 floor +
L2 column normalization). Idempotent -- second call with same audio+params
returns the cached array without recomputing.

Raises explicitly on missing audio, unreadable files, or sample-rate mismatch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ChromaParams:
    target_frame_rate_hz: float
    sr: int


@dataclass
class CachedChroma:
    data: np.ndarray  # shape (12, n_frames), float32, L2-normed
    frame_rate_hz: float
    audio_path: Path


def _hash_audio(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(cache_root: Path, audio_hash: str, params: ChromaParams) -> Path:
    key = f"{audio_hash}_sr{params.sr}_fr{params.target_frame_rate_hz:.1f}.bin"
    return cache_root / key


def get_chroma(audio_path: Path, params: ChromaParams, cache_root: Path) -> CachedChroma:
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    audio_hash = _hash_audio(audio_path)
    cache_file = _cache_path(cache_root, audio_hash, params)
    meta_file = cache_file.with_suffix(".meta")

    if cache_file.exists() and meta_file.exists():
        meta = meta_file.read_text().strip().split(",")
        n_frames = int(meta[0])
        frame_rate_hz = float(meta[1])
        raw = np.fromfile(cache_file, dtype=np.float32)
        if raw.size != 12 * n_frames:
            raise RuntimeError(
                f"chroma cache corrupt: {cache_file} size {raw.size} != 12*{n_frames}"
            )
        return CachedChroma(raw.reshape(12, n_frames), frame_rate_hz, audio_path)

    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if sr != params.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=params.sr)
        sr = params.sr
    if y.ndim == 2:
        y = y.mean(axis=1)

    hop = max(1, round(sr / params.target_frame_rate_hz))
    frame_rate_hz = sr / hop
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).astype(np.float32)
    chroma += 1e-3
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9

    tmp = cache_file.with_suffix(".bin.tmp")
    chroma.flatten().astype(np.float32).tofile(tmp)
    tmp.replace(cache_file)
    meta_file.write_text(f"{chroma.shape[1]},{frame_rate_hz}")
    return CachedChroma(chroma, frame_rate_hz, audio_path)
