"""Multi-feature approach: flatness + centroid + kurtosis + ZCR + chroma energy.

Tests whether additional features improve over the two-feature gate.
If this doesn't beat the two-feature approach significantly, the extra
complexity isn't worth it.
"""

from __future__ import annotations

import numpy as np
import librosa


def compute_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute a rich feature set for audio classification."""
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)

    # Spectral bandwidth (spread around centroid)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)

    # Zero-crossing rate (higher for noisy/speech, lower for tonal)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)

    # Chroma energy (pitch class distribution -- strong for piano, weak for noise)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
    chroma_energy = np.sum(chroma, axis=0)  # total chroma energy per frame

    # RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)

    # Spectral rolloff (frequency below which 85% of energy is concentrated)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=2048, hop_length=512)

    return {
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "chroma_energy_mean": float(np.mean(chroma_energy)),
        "chroma_peak_ratio": float(np.max(np.mean(chroma, axis=1)) / (np.mean(chroma) + 1e-10)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rolloff_mean": float(np.mean(rolloff)),
    }


def classify(
    features: dict[str, float],
    flatness_threshold: float = 0.12,
    centroid_min: float = 200.0,
    centroid_max: float = 4500.0,
    zcr_threshold: float = 0.15,
    chroma_peak_threshold: float = 1.5,
) -> tuple[str, float]:
    """Classify using multi-feature decision tree.

    Piano indicators:
    - Low flatness (tonal)
    - Centroid in piano range
    - Low ZCR (pitched, not noisy)
    - High chroma peak ratio (strong pitch content)
    """
    flatness = features["flatness_mean"]
    centroid = features["centroid_mean"]
    zcr = features["zcr_mean"]
    chroma_peak = features["chroma_peak_ratio"]

    score = 0.0
    # Tonality check (most important)
    if flatness < flatness_threshold:
        score += 0.35
    # Frequency range check
    if centroid_min < centroid < centroid_max:
        score += 0.25
    # ZCR check (low = pitched)
    if zcr < zcr_threshold:
        score += 0.20
    # Chroma peak check (strong pitch = piano-like)
    if chroma_peak > chroma_peak_threshold:
        score += 0.20

    if score >= 0.55:
        return "piano", score
    else:
        return "not_piano", 1.0 - score
