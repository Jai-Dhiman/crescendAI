"""Late fusion of audio and symbolic predictions."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..constants import PERCEPIANO_DIMENSIONS


# Default fusion weights (audio weight per dimension)
# Based on cross-validation analysis showing where each modality excels
DEFAULT_FUSION_WEIGHTS = {
    "timing": 0.55,
    "articulation_length": 0.45,
    "articulation_touch": 0.50,
    "pedal_amount": 0.40,
    "pedal_clarity": 0.45,
    "timbre_variety": 0.60,
    "timbre_depth": 0.55,
    "timbre_brightness": 0.65,
    "timbre_loudness": 0.70,
    "dynamics_range": 0.60,
    "tempo": 0.50,
    "space": 0.55,
    "balance": 0.50,
    "drama": 0.55,
    "mood_valence": 0.50,
    "mood_energy": 0.55,
    "mood_imagination": 0.50,
    "interpretation_sophistication": 0.45,
    "interpretation_overall": 0.50,
}


def load_fusion_weights(weights_path: Path) -> Dict[str, float]:
    """Load fusion weights from JSON file.

    Args:
        weights_path: Path to weights JSON file

    Returns:
        Dict mapping dimension name to audio weight [0, 1]
    """
    if not weights_path.exists():
        return DEFAULT_FUSION_WEIGHTS.copy()

    with open(weights_path) as f:
        weights = json.load(f)

    # Validate and fill in defaults
    result = DEFAULT_FUSION_WEIGHTS.copy()
    for dim in PERCEPIANO_DIMENSIONS:
        if dim in weights:
            result[dim] = float(weights[dim])

    return result


def late_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Apply per-dimension weighted late fusion.

    Fusion formula: fused[i] = w[i] * audio[i] + (1 - w[i]) * symbolic[i]

    Args:
        audio_preds: Audio model predictions (19,)
        symbolic_preds: Symbolic model predictions (19,)
        weights: Dict mapping dimension name to audio weight

    Returns:
        Fused predictions (19,)
    """
    if weights is None:
        weights = DEFAULT_FUSION_WEIGHTS

    fused = np.zeros(len(PERCEPIANO_DIMENSIONS))

    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        w_audio = weights.get(dim, 0.5)
        fused[i] = w_audio * audio_preds[i] + (1 - w_audio) * symbolic_preds[i]

    return fused


def simple_average_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
) -> np.ndarray:
    """Simple average of audio and symbolic predictions.

    Args:
        audio_preds: Audio model predictions (19,)
        symbolic_preds: Symbolic model predictions (19,)

    Returns:
        Averaged predictions (19,)
    """
    return (audio_preds + symbolic_preds) / 2


def confidence_weighted_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    audio_confidence: float = 0.55,
) -> np.ndarray:
    """Confidence-weighted fusion with global weights.

    Args:
        audio_preds: Audio model predictions (19,)
        symbolic_preds: Symbolic model predictions (19,)
        audio_confidence: Global confidence in audio model [0, 1]

    Returns:
        Fused predictions (19,)
    """
    symbolic_confidence = 1.0 - audio_confidence
    return audio_confidence * audio_preds + symbolic_confidence * symbolic_preds
