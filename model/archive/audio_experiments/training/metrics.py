"""Metrics and evaluation utilities."""

from typing import Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..constants import PERCEPIANO_DIMENSIONS, SEED


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """Compute bootstrap 95% CI for R2.

    Returns (lower_ci, median, upper_ci).
    """
    np.random.seed(SEED)
    n_samples = len(y_true)
    r2_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        r2_scores.append(r2_score(y_true[idx], y_pred[idx]))
    return tuple(np.percentile(r2_scores, [2.5, 50, 97.5]))


def compute_comprehensive_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> Dict:
    """Compute all metrics for experiment results.

    Returns dict with:
    - overall_r2, r2_ci_95, overall_mae, overall_rmse
    - dispersion_ratio (pred_std / label_std)
    - per_dimension metrics
    """
    overall_r2 = r2_score(all_labels, all_preds)
    overall_mae = mean_absolute_error(all_labels, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

    # Bootstrap CI
    ci = bootstrap_r2(all_labels, all_preds)

    # Per-dimension metrics
    per_dim = {}
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        y_true, y_pred = all_labels[:, i], all_preds[:, i]
        pearson, p_val = stats.pearsonr(y_true, y_pred)
        per_dim[dim] = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "pearson": float(pearson),
            "p_value": float(p_val),
            "label_mean": float(y_true.mean()),
            "label_std": float(y_true.std()),
            "pred_mean": float(y_pred.mean()),
            "pred_std": float(y_pred.std()),
        }

    # Dispersion ratio (measure of mean regression)
    avg_label_std = np.mean([all_labels[:, i].std() for i in range(19)])
    avg_pred_std = np.mean([all_preds[:, i].std() for i in range(19)])
    dispersion_ratio = avg_pred_std / avg_label_std if avg_label_std > 0 else 0

    return {
        "overall_r2": float(overall_r2),
        "r2_ci_95": [float(ci[0]), float(ci[2])],
        "overall_mae": float(overall_mae),
        "overall_rmse": float(overall_rmse),
        "dispersion_ratio": float(dispersion_ratio),
        "per_dimension": per_dim,
    }
