"""Hand-crafted audio statistics extraction."""

from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm


def extract_audio_statistics(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Extract 49-dim hand-crafted audio features.

    Features:
    - Energy (3): RMS mean, std, max
    - Spectral (8): centroid, bandwidth, rolloff, ZCR (mean, std each)
    - MFCCs (26): 13 coefficients x mean/std
    - Chroma (12): 12 bins x mean

    Total: 49 dimensions
    """
    features = []

    # Energy features (3)
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([rms.mean(), rms.std(), rms.max()])

    # Spectral features (8)
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([
        cent.mean(), cent.std(),
        bw.mean(), bw.std(),
        rolloff.mean(), rolloff.std(),
        zcr.mean(), zcr.std(),
    ])

    # MFCCs (26 = 13 coeffs x mean/std)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(mfcc.mean(axis=1).tolist())
    features.extend(mfcc.std(axis=1).tolist())

    # Chroma (12 = 12 bins x mean)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(chroma.mean(axis=1).tolist())

    return np.array(features, dtype=np.float32)  # Shape: (49,)


def extract_statistics_for_all(
    audio_dir: Path,
    cache_dir: Path,
    keys: List[str],
) -> int:
    """Extract audio statistics for all keys.

    Returns count of newly extracted statistics.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {p.stem for p in cache_dir.glob("*.pt")}
    to_extract = [k for k in keys if k not in cached]

    if not to_extract:
        print(f"All {len(keys)} statistics already cached.")
        return 0

    print(f"Extracting {len(to_extract)} audio statistics...")

    for key in tqdm(to_extract, desc="Stats extraction"):
        audio_path = Path(audio_dir) / f"{key}.wav"
        if audio_path.exists():
            audio, sr = librosa.load(audio_path, sr=24000, mono=True)
            stats_arr = extract_audio_statistics(audio, sr)
            torch.save(torch.from_numpy(stats_arr), cache_dir / f"{key}.pt")

    return len(to_extract)
