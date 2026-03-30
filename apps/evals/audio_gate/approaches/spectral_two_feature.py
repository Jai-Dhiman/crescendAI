"""Approach A production candidate: two-feature gate (flatness + centroid).

Combines spectral flatness (tonality) with spectral centroid (brightness).
Piano: low flatness + centroid in piano range (~200-4000 Hz).
Speech: moderate flatness + centroid in voice range (~300-3000 Hz but less tonal).
Noise: high flatness + spread centroid.
"""

from __future__ import annotations

import numpy as np
import librosa


def compute_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute spectral flatness and centroid averaged over all frames."""
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)

    return {
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        # Normalized centroid (0-1 range relative to Nyquist)
        "centroid_norm": float(np.mean(centroid) / (sr / 2)),
    }


def classify(
    features: dict[str, float],
    flatness_threshold: float = 0.12,
    centroid_max: float = 2000.0,
) -> tuple[str, float]:
    """Classify using both flatness and centroid.

    Piano-like: flatness below threshold OR centroid below max.
    Key insight from data: speech has higher centroid (1600-3200 Hz) than
    piano (300-2400 Hz). Fan/AC has very high centroid (5000+ Hz).
    The gate should be generous (OR logic) to maintain high recall.
    """
    flatness = features["flatness_mean"]
    centroid = features["centroid_mean"]

    is_tonal = flatness < flatness_threshold
    is_low_centroid = centroid < centroid_max

    # OR logic: either signal suggests piano -> pass it through
    # This favors recall over precision (the right trade-off for a gate)
    if is_tonal or is_low_centroid:
        confidence = 0.5
        if is_tonal:
            confidence += 0.25 * (1.0 - flatness / flatness_threshold)
        if is_low_centroid:
            confidence += 0.25 * (1.0 - centroid / centroid_max)
        return "piano", min(confidence, 1.0)
    else:
        # Both signals say not piano -- high confidence rejection
        confidence = 0.5
        confidence += 0.25 * min((flatness - flatness_threshold) / 0.3, 1.0)
        confidence += 0.25 * min((centroid - centroid_max) / 3000.0, 1.0)
        return "not_piano", min(confidence, 1.0)


def sweep_thresholds(
    all_features: list[dict],
    thresholds: list[dict] | None = None,
) -> list[dict]:
    """Sweep flatness x centroid_max threshold combinations."""
    if thresholds is None:
        flatness_range = np.arange(0.02, 0.60, 0.02)
        centroid_max_range = [1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2500.0, 3000.0]
        thresholds = [
            {"flatness": float(f), "centroid_max": cx}
            for f in flatness_range
            for cx in centroid_max_range
        ]

    results = []
    for t in thresholds:
        preds = [
            classify(f, t["flatness"], t["centroid_max"])[0]
            for f in all_features
        ]
        results.append({"params": t, "predictions": preds})
    return results
