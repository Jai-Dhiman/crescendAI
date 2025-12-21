"""
Evaluation metrics for piano performance assessment.

Implements all metrics from PercePiano paper:
- R-squared (coefficient of determination)
- MSE (mean squared error)
- MAE (mean absolute error)
- Pearson correlation
- Spearman correlation
- Std-Score (range accuracy accounting for annotator disagreement)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


@dataclass
class MetricResult:
    """Container for a single metric result."""
    value: float
    per_dimension: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    std: Optional[float] = None


def compute_r2(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute R-squared (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot

    This is the PRIMARY metric used in PercePiano for model comparison.

    Args:
        predictions: [n_samples, n_dims] or [n_samples]
        targets: [n_samples, n_dims] or [n_samples]
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension R^2
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Handle 1D case
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_dims = predictions.shape[1]

    # Per-dimension R^2
    r2_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        pred = predictions[:, i]
        targ = targets[:, i]

        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)

        # Avoid division by zero
        if ss_tot < 1e-10:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        r2_scores.append(r2)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = r2

    # Overall R^2 (mean across dimensions, matching PercePiano)
    overall_r2 = np.mean(r2_scores)

    return MetricResult(
        value=overall_r2,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(r2_scores),
    )


def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute Mean Squared Error.

    Args:
        predictions: [n_samples, n_dims] or [n_samples]
        targets: [n_samples, n_dims] or [n_samples]
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension MSE
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_dims = predictions.shape[1]

    mse_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
        mse_scores.append(mse)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = mse

    overall_mse = np.mean(mse_scores)

    return MetricResult(
        value=overall_mse,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(mse_scores),
    )


def compute_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: [n_samples, n_dims] or [n_samples]
        targets: [n_samples, n_dims] or [n_samples]
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension MAE
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_dims = predictions.shape[1]

    mae_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        mae_scores.append(mae)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = mae

    overall_mae = np.mean(mae_scores)

    return MetricResult(
        value=overall_mae,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(mae_scores),
    )


def compute_pearson_r(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute Pearson correlation coefficient.

    Args:
        predictions: [n_samples, n_dims] or [n_samples]
        targets: [n_samples, n_dims] or [n_samples]
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension Pearson r
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_dims = predictions.shape[1]

    r_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        pred = predictions[:, i]
        targ = targets[:, i]

        # Handle constant arrays
        if np.std(pred) < 1e-10 or np.std(targ) < 1e-10:
            r = 0.0
        else:
            r, _ = stats.pearsonr(pred, targ)
            if np.isnan(r):
                r = 0.0

        r_scores.append(r)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = r

    overall_r = np.mean(r_scores)

    return MetricResult(
        value=overall_r,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(r_scores),
    )


def compute_spearman_rho(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute Spearman rank correlation coefficient.

    More robust to outliers than Pearson.

    Args:
        predictions: [n_samples, n_dims] or [n_samples]
        targets: [n_samples, n_dims] or [n_samples]
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension Spearman rho
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_dims = predictions.shape[1]

    rho_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        pred = predictions[:, i]
        targ = targets[:, i]

        if np.std(pred) < 1e-10 or np.std(targ) < 1e-10:
            rho = 0.0
        else:
            rho, _ = stats.spearmanr(pred, targ)
            if np.isnan(rho):
                rho = 0.0

        rho_scores.append(rho)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = rho

    overall_rho = np.mean(rho_scores)

    return MetricResult(
        value=overall_rho,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(rho_scores),
    )


def compute_std_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_stds: np.ndarray,
    alpha: float = 1.0,
    per_dimension: bool = True,
    dimension_names: Optional[List[str]] = None,
) -> MetricResult:
    """
    Compute Std-Score (Range Accuracy) metric from PercePiano.

    This metric accounts for annotator disagreement. A prediction is
    considered correct if it falls within alpha * std of the target.

    Std-Score = percentage of predictions where |pred - target| <= alpha * std

    Standard thresholds:
    - alpha=1.0: Std-Score@1 (allows 1 std deviation)
    - alpha=0.5: Std-Score@0.5 (allows 0.5 std)
    - alpha=0.1: Std-Score@0.1 (strict threshold)

    Args:
        predictions: [n_samples, n_dims]
        targets: [n_samples, n_dims] mean target values
        target_stds: [n_dims] standard deviation of annotations per dimension
        alpha: Multiplier for acceptable range
        per_dimension: Whether to compute per-dimension scores
        dimension_names: Names for each dimension

    Returns:
        MetricResult with overall and per-dimension Std-Score
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    target_stds = np.asarray(target_stds)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_samples, n_dims = predictions.shape

    if target_stds.ndim == 0:
        target_stds = np.full(n_dims, target_stds)

    std_scores = []
    per_dim_results = {}

    for i in range(n_dims):
        errors = np.abs(predictions[:, i] - targets[:, i])
        threshold = alpha * target_stds[i] if i < len(target_stds) else alpha * 0.1

        # Avoid zero threshold
        if threshold < 1e-10:
            threshold = 0.01

        within_range = errors <= threshold
        score = np.mean(within_range) * 100  # As percentage

        std_scores.append(score)

        if dimension_names and i < len(dimension_names):
            per_dim_results[dimension_names[i]] = score

    overall_score = np.mean(std_scores)

    return MetricResult(
        value=overall_score,
        per_dimension=per_dim_results if per_dimension else None,
        std=np.std(std_scores),
    )


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_names: Optional[List[str]] = None,
    target_stds: Optional[np.ndarray] = None,
) -> Dict[str, MetricResult]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: [n_samples, n_dims]
        targets: [n_samples, n_dims]
        dimension_names: Names for each dimension
        target_stds: Standard deviations for Std-Score (optional)

    Returns:
        Dictionary of metric name -> MetricResult
    """
    results = {
        "r2": compute_r2(predictions, targets, True, dimension_names),
        "mse": compute_mse(predictions, targets, True, dimension_names),
        "mae": compute_mae(predictions, targets, True, dimension_names),
        "pearson_r": compute_pearson_r(predictions, targets, True, dimension_names),
        "spearman_rho": compute_spearman_rho(predictions, targets, True, dimension_names),
    }

    # Add Std-Score if target_stds provided
    if target_stds is not None:
        for alpha in [1.0, 0.5, 0.1]:
            key = f"std_score_{alpha}".replace(".", "_")
            results[key] = compute_std_score(
                predictions, targets, target_stds, alpha, True, dimension_names
            )

    return results


# Dimension categories for grouped analysis
DIMENSION_CATEGORIES = {
    "timing": ["timing", "tempo"],
    "articulation": ["articulation_length", "articulation_touch"],
    "pedal": ["pedal_amount", "pedal_clarity"],
    "timbre": ["timbre_variety", "timbre_depth", "timbre_brightness", "timbre_loudness"],
    "dynamics": ["dynamic_range", "sophistication"],
    "musical": ["space", "balance", "drama"],
    "emotion": ["mood_valence", "mood_energy", "mood_imagination"],
    "interpretation": ["interpretation"],
}


def compute_category_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics grouped by dimension category.

    Args:
        predictions: [n_samples, n_dims]
        targets: [n_samples, n_dims]
        dimension_names: Names for each dimension

    Returns:
        Dict mapping category -> {metric_name: value}
    """
    dim_to_idx = {name: i for i, name in enumerate(dimension_names)}

    category_results = {}

    for category, dims in DIMENSION_CATEGORIES.items():
        # Get indices for dimensions in this category
        indices = [dim_to_idx[d] for d in dims if d in dim_to_idx]

        if not indices:
            continue

        cat_preds = predictions[:, indices]
        cat_targets = targets[:, indices]

        r2 = compute_r2(cat_preds, cat_targets, per_dimension=False)
        mae = compute_mae(cat_preds, cat_targets, per_dimension=False)

        category_results[category] = {
            "r2": r2.value,
            "mae": mae.value,
            "n_dims": len(indices),
        }

    return category_results


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    n_samples, n_dims = 100, 19

    targets = np.random.rand(n_samples, n_dims)
    predictions = targets + np.random.randn(n_samples, n_dims) * 0.1

    dims = [f"dim_{i}" for i in range(n_dims)]

    results = compute_all_metrics(predictions, targets, dims)

    print("Metric Results:")
    for name, result in results.items():
        print(f"  {name}: {result.value:.4f} (+/- {result.std:.4f})")
