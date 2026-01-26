"""MAESTRO-based calibration for performance predictions.

Normalizes raw model predictions relative to professional MAESTRO recordings,
making scores more interpretable for end users.
"""

import numpy as np
from typing import Dict

from constants import MAESTRO_CALIBRATION, PERCEPIANO_DIMENSIONS


def calibrate_predictions(
    raw_predictions: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """Calibrate raw predictions using MAESTRO professional benchmarks.

    Args:
        raw_predictions: Raw model outputs [19] in range ~[0, 1]
        method: Calibration method:
            - "percentile": Scale to [0, 1] where 0 = MAESTRO 5th percentile,
              1 = MAESTRO 95th percentile. Scores can exceed [0, 1] for
              exceptional or below-average performances.
            - "zscore": Convert to z-scores relative to MAESTRO distribution.

    Returns:
        Calibrated predictions [19]. For "percentile" method, ~0.5 means
        comparable to average MAESTRO professional performance.
    """
    calibrated = np.zeros_like(raw_predictions)

    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        raw_score = raw_predictions[i]

        # Get calibration stats - keys match PERCEPIANO_DIMENSIONS exactly
        dim_key = dim
        if dim_key not in MAESTRO_CALIBRATION:
            # Fallback: use raw score (this shouldn't happen with properly configured data)
            calibrated[i] = raw_score
            continue

        stats = MAESTRO_CALIBRATION[dim_key]

        if method == "percentile":
            # Scale so MAESTRO 5th percentile = 0, 95th percentile = 1
            # This means ~0.5 = average professional performance
            p5 = stats["p5"]
            p95 = stats["p95"]
            range_width = p95 - p5

            if range_width > 0:
                calibrated[i] = (raw_score - p5) / range_width
            else:
                calibrated[i] = 0.5

        elif method == "zscore":
            # Convert to z-score relative to MAESTRO mean/std
            mean = stats["mean"]
            std = stats["std"]
            if std > 0:
                calibrated[i] = (raw_score - mean) / std
            else:
                calibrated[i] = 0.0

        else:
            calibrated[i] = raw_score

    return calibrated


def predictions_to_calibrated_dict(
    raw_predictions: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Convert raw predictions to a dict with both raw and calibrated scores.

    Args:
        raw_predictions: Raw model outputs [19]

    Returns:
        Dict with structure:
        {
            "timing": {"raw": 0.65, "calibrated": 0.42, "percentile_rank": 42},
            ...
        }
    """
    calibrated = calibrate_predictions(raw_predictions, method="percentile")
    result = {}

    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        raw_score = float(raw_predictions[i])
        cal_score = float(calibrated[i])

        # Clamp percentile rank to [0, 100] for display
        percentile_rank = int(max(0, min(100, cal_score * 100)))

        result[dim] = {
            "raw": round(raw_score, 4),
            "calibrated": round(max(0.0, min(1.0, cal_score)), 4),
            "percentile_rank": percentile_rank,
        }

    return result


def get_calibration_context() -> str:
    """Get a text description of the calibration for LLM context.

    Returns:
        String describing how to interpret calibrated scores.
    """
    return """Score Interpretation (calibrated relative to 500 professional MAESTRO recordings):
- 0.0 = Performance at the 5th percentile of professionals (lower end)
- 0.5 = Performance at the 50th percentile of professionals (average professional level)
- 1.0 = Performance at the 95th percentile of professionals (exceptional)
- Scores can exceed [0, 1] for truly exceptional or below-average performances

Note: These scores compare against competition-level professional pianists.
A calibrated score of 0.5 represents professional-level competency."""
