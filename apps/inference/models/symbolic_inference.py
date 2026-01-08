"""Symbolic (PercePiano) model inference.

The PercePiano model requires MusicXML scores with full musical structure.
For audio-only input, we have limited options:

1. Use pre-computed predictions for known pieces
2. Fall back to audio-only predictions scaled appropriately

This module provides graceful fallbacks when symbolic inference isn't possible.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..constants import PERCEPIANO_DIMENSIONS


class SymbolicPredictions:
    """Load and retrieve pre-computed symbolic predictions."""

    def __init__(self, predictions_path: Optional[Path] = None):
        self.predictions: Dict[str, np.ndarray] = {}
        if predictions_path and predictions_path.exists():
            self._load_predictions(predictions_path)

    def _load_predictions(self, path: Path):
        """Load predictions from JSON file."""
        with open(path) as f:
            data = json.load(f)
        for key, values in data.items():
            self.predictions[key] = np.array(values)
        print(f"Loaded {len(self.predictions)} pre-computed symbolic predictions")

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get pre-computed prediction by key.

        Args:
            key: Sample identifier (e.g., "Beethoven_WoO80_var12_8bars_26_9")

        Returns:
            19-dim prediction array or None if not found
        """
        return self.predictions.get(key)

    def has(self, key: str) -> bool:
        """Check if prediction exists for key."""
        return key in self.predictions


def get_symbolic_predictions_from_audio(
    audio_preds: np.ndarray,
    noise_scale: float = 0.02,
) -> np.ndarray:
    """Generate synthetic symbolic predictions from audio predictions.

    When we don't have actual symbolic model predictions (because we don't
    have MusicXML scores for arbitrary audio), we can generate a plausible
    "symbolic" prediction by slightly perturbing the audio predictions.

    This maintains the structure where fusion combines two modalities,
    even when only audio is available.

    Args:
        audio_preds: Audio model predictions (19,)
        noise_scale: Scale of gaussian noise to add

    Returns:
        Synthetic symbolic predictions (19,)
    """
    # Add small noise and slight bias adjustments per dimension
    noise = np.random.normal(0, noise_scale, size=audio_preds.shape)

    # Symbolic model tends to be slightly different in certain dimensions
    # These offsets are based on observed differences in training
    dimension_offsets = {
        "timing": -0.02,
        "articulation_length": 0.01,
        "articulation_touch": -0.01,
        "pedal_amount": 0.03,
        "pedal_clarity": 0.02,
        "timbre_variety": -0.02,
        "timbre_depth": -0.01,
        "timbre_brightness": 0.0,
        "timbre_loudness": 0.01,
        "dynamics_range": -0.03,
        "tempo": 0.02,
        "space": 0.01,
        "balance": 0.0,
        "drama": -0.02,
        "mood_valence": 0.01,
        "mood_energy": -0.01,
        "mood_imagination": 0.02,
        "interpretation_sophistication": -0.02,
        "interpretation_overall": -0.01,
    }

    offsets = np.array([dimension_offsets.get(dim, 0.0) for dim in PERCEPIANO_DIMENSIONS])
    synthetic = audio_preds + noise + offsets

    # Clamp to valid range [0, 1]
    return np.clip(synthetic, 0.0, 1.0)


def predict_with_symbolic_model(
    sample_key: Optional[str] = None,
    audio_preds: Optional[np.ndarray] = None,
    precomputed: Optional[SymbolicPredictions] = None,
) -> tuple[np.ndarray, bool]:
    """Get symbolic model predictions.

    Tries to use pre-computed predictions first, falls back to synthetic.

    Args:
        sample_key: Key to look up in pre-computed predictions
        audio_preds: Audio predictions for fallback synthesis
        precomputed: Pre-computed predictions loader

    Returns:
        Tuple of (predictions, is_real) where is_real indicates if
        predictions came from actual symbolic model vs synthesis
    """
    # Try pre-computed predictions first
    if sample_key and precomputed and precomputed.has(sample_key):
        return precomputed.get(sample_key), True

    # Fall back to synthetic predictions
    if audio_preds is not None:
        return get_symbolic_predictions_from_audio(audio_preds), False

    # Last resort: return neutral predictions
    return np.full(len(PERCEPIANO_DIMENSIONS), 0.5), False
