#!/usr/bin/env python3
"""
ASAP Dataset Integration

Lightweight dataset loader that scans an ASAP root directory for audio files and
provides an iterator of [B, T=segment_length, F=n_mels] spectrogram windows with
optional unique_per_file sampling to reduce false negatives in contrastive SSL.

Preprocessing strictly uses src.data.audio_io (mono 22050 Hz; mel dB in [-80, 0]).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import pandas as pd
import pickle
import logging

from src.data.audio_io import load_audio_mono_22050, mel_db_time_major

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".aac"}


class ASAPDataset:
    """
    ASAP dataset loader for proxy tasks and auxiliary training.
    Scans a root directory for audio files and yields mel-dB windows.
    """

    def __init__(
        self,
        asap_root: str,
        *,
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_length: int = 128,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.asap_root = Path(asap_root)
        if not self.asap_root.exists():
            raise FileNotFoundError(f"ASAP root not found: {self.asap_root}")

        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.metadata = self._scan_files()
        print(
            "🎼 ASAP Dataset initialized:",
            f"files={len(self.metadata)} sr={target_sr} n_mels={n_mels} seg_len={segment_length}",
        )

    def _scan_files(self) -> pd.DataFrame:
        records: List[dict] = []
        for p in self.asap_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                # Heuristics for composer/piece by path parts
                parts = [q for q in p.parts[-5:]]  # last few path components
                composer = next((q for q in parts if q.lower() in (
                    "bach","mozart","beethoven","chopin","schubert","liszt","schumann","debussy","rachmaninoff"
                )), "Unknown")
                records.append({
                    "audio_path": p,
                    "composer": composer,
                    "relpath": str(p.relative_to(self.asap_root)) if p.is_relative_to(self.asap_root) else p.name,
                })
        if not records:
            raise FileNotFoundError(f"No audio files found under {self.asap_root}")
        return pd.DataFrame(records)

    def _cache_path(self, audio_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        name = f"{audio_path.stem}_{self.target_sr}_{self.n_mels}_{self.segment_length}.pkl"
        return self.cache_dir / name

    def _load_or_create_spectrogram(self, audio_path: Path) -> Optional[np.ndarray]:
        cache_path = self._cache_path(audio_path)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed for {cache_path}: {e}")

        try:
            y = load_audio_mono_22050(audio_path, target_sr=self.target_sr)
            log_mel_t = mel_db_time_major(
                y,
                sr=self.target_sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        except Exception as e:
            logger.error(f"Failed to preprocess {audio_path}: {e}")
            return None

        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(log_mel_t, f)
            except Exception as e:
                logger.warning(f"Cache save failed for {cache_path}: {e}")
        return log_mel_t

    def _segment(self, spectrogram: np.ndarray) -> List[np.ndarray]:
        t, f = spectrogram.shape
        seg_len = self.segment_length
        if t <= 0 or f != self.n_mels:
            return []
        if t <= seg_len:
            # Pad up to one segment
            pad = seg_len - t
            seg = np.pad(spectrogram, ((0, pad), (0, 0)), mode="constant", constant_values=-80.0)
            return [seg]
        hop = seg_len // 2  # 50% overlap
        out: List[np.ndarray] = []
        for start in range(0, t - seg_len + 1, hop):
            out.append(spectrogram[start : start + seg_len])
        return out

    def get_data_iterator(
        self,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        infinite: bool = True,
        unique_per_file: bool = False,
    ) -> Iterator[jnp.ndarray]:
        """Yield batches [B, T, F]."""
        rng = np.random.default_rng()

        def gen():
            while True:
                indices = np.arange(len(self.metadata))
                if shuffle:
                    rng.shuffle(indices)
                batch: List[np.ndarray] = []
                for idx in indices:
                    row = self.metadata.iloc[idx]
                    spec = self._load_or_create_spectrogram(row["audio_path"])
                    if spec is None:
                        continue
                    segments = self._segment(spec)
                    if not segments:
                        continue
                    if unique_per_file:
                        sidx = int(rng.integers(0, len(segments)))
                        batch.append(segments[sidx])
                    else:
                        batch.extend(segments)

                    while len(batch) >= batch_size:
                        out = jnp.array(batch[:batch_size])
                        yield out
                        batch = batch[batch_size:]

                if batch:
                    # pad
                    while len(batch) < batch_size:
                        batch.append(batch[0])
                    yield jnp.array(batch[:batch_size])
                if not infinite:
                    break
        return gen()
