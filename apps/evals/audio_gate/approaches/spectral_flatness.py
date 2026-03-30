"""Approach A baseline: single-feature spectral flatness gate.

Spectral flatness = geometric mean / arithmetic mean of power spectrum.
Piano (tonal): low flatness (~0.0-0.1)
Speech (moderate): mid flatness (~0.1-0.3)
Noise (broadband): high flatness (~0.3-1.0)
"""

from __future__ import annotations

import numpy as np
import librosa


def compute_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute spectral flatness averaged over all frames."""
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)
    return {
        "flatness_mean": float(np.mean(flatness)),
        "flatness_median": float(np.median(flatness)),
        "flatness_std": float(np.std(flatness)),
    }


def classify(
    features: dict[str, float],
    threshold: float = 0.15,
) -> tuple[str, float]:
    """Classify as piano or not_piano based on spectral flatness.

    Returns (predicted_class, confidence).
    Lower flatness = more tonal = more likely piano.
    """
    flatness = features["flatness_mean"]
    if flatness < threshold:
        confidence = 1.0 - (flatness / threshold)
        return "piano", confidence
    else:
        confidence = min((flatness - threshold) / (1.0 - threshold), 1.0)
        return "not_piano", confidence


def sweep_thresholds(
    all_features: list[dict],
    labels: list[str],
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Sweep flatness thresholds and return metrics for each."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.50, 0.01)

    results = []
    for t in thresholds:
        preds = [classify(f, threshold=t)[0] for f in all_features]
        results.append({
            "threshold": float(t),
            "predictions": preds,
        })
    return results
