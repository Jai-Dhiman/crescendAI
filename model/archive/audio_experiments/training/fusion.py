"""Fusion utilities for multimodal prediction combination and statistical testing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from ..constants import DIMENSION_CATEGORIES, PERCEPIANO_DIMENSIONS, SEED


# ============================================================================
# Experiment Runner Functions
# ============================================================================


def run_bootstrap_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run bootstrap CI experiment for audio and symbolic models."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Bootstrap 95% CIs (n={n_bootstrap})")
    print(f"{'='*60}")

    audio_bootstrap = bootstrap_r2_extended(labels, audio_preds, n_bootstrap)
    print(f"Audio R2: {audio_bootstrap['overall']['r2']:.4f} "
          f"[{audio_bootstrap['overall']['ci_lower']:.4f}, {audio_bootstrap['overall']['ci_upper']:.4f}]")

    symbolic_bootstrap = bootstrap_r2_extended(labels, symbolic_preds, n_bootstrap)
    print(f"Symbolic R2: {symbolic_bootstrap['overall']['r2']:.4f} "
          f"[{symbolic_bootstrap['overall']['ci_lower']:.4f}, {symbolic_bootstrap['overall']['ci_upper']:.4f}]")

    return {
        "exp_id": exp_id,
        "audio": audio_bootstrap,
        "symbolic": symbolic_bootstrap,
        "n_bootstrap": n_bootstrap,
    }


def run_paired_tests_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Run paired statistical tests between audio and symbolic."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Paired Statistical Tests")
    print(f"{'='*60}")

    ttest = paired_ttest_per_sample(labels, audio_preds, symbolic_preds)
    wilcox = wilcoxon_test(labels, audio_preds, symbolic_preds)
    effect = cohens_d(labels, audio_preds, symbolic_preds)

    print(f"Paired t-test: t={ttest['t_stat']:.4f}, p={ttest['p_value']:.2e}")
    print(f"Wilcoxon: stat={wilcox['stat']:.4f}, p={wilcox['p_value']:.2e}")
    print(f"Cohen's d: {effect:.4f}")
    print(f"Winner: {'Audio' if ttest['a_better'] else 'Symbolic'} (significant: {ttest['significant']})")

    # Per-dimension tests
    per_dim = {}
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        per_dim[dim] = paired_ttest_per_sample(
            labels[:, i:i+1], audio_preds[:, i:i+1], symbolic_preds[:, i:i+1]
        )

    return {
        "exp_id": exp_id,
        "audio_vs_symbolic": {"ttest": ttest, "wilcoxon": wilcox, "cohens_d": effect},
        "per_dimension": per_dim,
    }


def run_multiple_correction_experiment(
    exp_id: str,
    paired_tests_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Run multiple comparison corrections on per-dimension tests."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Multiple Comparison Corrections")
    print(f"{'='*60}")

    p_values = np.array([
        paired_tests_result["per_dimension"][dim]["p_value"]
        for dim in PERCEPIANO_DIMENSIONS
    ])

    bonf_corrected, bonf_sig = bonferroni_correction(p_values)
    fdr_corrected, fdr_sig = fdr_correction(p_values)

    print(f"Bonferroni: {sum(bonf_sig)}/{len(p_values)} significant")
    print(f"FDR (BH): {sum(fdr_sig)}/{len(p_values)} significant")

    return {
        "exp_id": exp_id,
        "bonferroni": {"corrected_p": bonf_corrected.tolist(), "significant": bonf_sig.tolist()},
        "fdr": {"corrected_p": fdr_corrected.tolist(), "significant": fdr_sig.tolist()},
    }


def run_simple_fusion_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run simple average fusion experiment."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Simple Average Fusion")
    print(f"{'='*60}")

    fused = simple_average_fusion(audio_preds, symbolic_preds)
    r2 = r2_score(labels, fused)
    bootstrap = bootstrap_r2_extended(labels, fused, n_bootstrap)

    print(f"Simple Fusion R2: {r2:.4f} [{bootstrap['overall']['ci_lower']:.4f}, {bootstrap['overall']['ci_upper']:.4f}]")

    return {
        "exp_id": exp_id,
        "overall_r2": r2,
        "bootstrap": bootstrap,
        "predictions": fused,
    }


def run_weighted_fusion_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run per-dimension weighted fusion with CV."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Per-Dimension Weighted Fusion (CV)")
    print(f"{'='*60}")

    fused, weights, fold_weights = weighted_fusion_grid_search(
        audio_preds, symbolic_preds, labels, fold_assignments, sample_keys
    )
    r2 = r2_score(labels, fused)
    bootstrap = bootstrap_r2_extended(labels, fused, n_bootstrap)

    print(f"Weighted Fusion R2: {r2:.4f} [{bootstrap['overall']['ci_lower']:.4f}, {bootstrap['overall']['ci_upper']:.4f}]")

    return {
        "exp_id": exp_id,
        "overall_r2": r2,
        "bootstrap": bootstrap,
        "optimal_weights": weights,
        "fold_weights": fold_weights,
        "predictions": fused,
    }


def run_ridge_fusion_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run ridge regression stacking fusion."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Ridge Regression Stacking")
    print(f"{'='*60}")

    fused, coefficients = ridge_stacking_fusion(
        audio_preds, symbolic_preds, labels, fold_assignments, sample_keys
    )
    r2 = r2_score(labels, fused)
    bootstrap = bootstrap_r2_extended(labels, fused, n_bootstrap)

    print(f"Ridge Fusion R2: {r2:.4f} [{bootstrap['overall']['ci_lower']:.4f}, {bootstrap['overall']['ci_upper']:.4f}]")

    return {
        "exp_id": exp_id,
        "overall_r2": r2,
        "bootstrap": bootstrap,
        "coefficients": coefficients,
        "predictions": fused,
    }


def run_confidence_fusion_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Run confidence-weighted fusion."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Confidence-Weighted Fusion")
    print(f"{'='*60}")

    # Compute per-dimension R2 as confidence
    n_dims = labels.shape[1]
    audio_r2 = np.array([r2_score(labels[:, i], audio_preds[:, i]) for i in range(n_dims)])
    symbolic_r2 = np.array([r2_score(labels[:, i], symbolic_preds[:, i]) for i in range(n_dims)])

    fused, weights = confidence_weighted_fusion(audio_preds, symbolic_preds, audio_r2, symbolic_r2)
    r2 = r2_score(labels, fused)
    bootstrap = bootstrap_r2_extended(labels, fused, n_bootstrap)

    print(f"Confidence Fusion R2: {r2:.4f} [{bootstrap['overall']['ci_lower']:.4f}, {bootstrap['overall']['ci_upper']:.4f}]")

    return {
        "exp_id": exp_id,
        "overall_r2": r2,
        "bootstrap": bootstrap,
        "weights": weights,
        "predictions": fused,
    }


def run_weight_stability_experiment(
    exp_id: str,
    fold_weights: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Run weight stability analysis."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Weight Stability Analysis")
    print(f"{'='*60}")

    stability = compute_weight_stability(fold_weights)
    print(f"Mean CV: {stability['_overall']['mean_cv']:.4f}")
    print(f"Stable: {stability['_overall']['stable']}")

    return {"exp_id": exp_id, **stability}


def run_category_fusion_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
) -> Dict[str, Any]:
    """Run per-category fusion optimization."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Per-Category Fusion")
    print(f"{'='*60}")

    fused, cat_weights = category_weighted_fusion(
        audio_preds, symbolic_preds, labels, fold_assignments, sample_keys
    )
    r2 = r2_score(labels, fused)

    print(f"Category Fusion R2: {r2:.4f}")
    for cat, w in cat_weights.items():
        print(f"  {cat}: {w:.2f}")

    return {
        "exp_id": exp_id,
        "overall_r2": r2,
        "category_weights": cat_weights,
        "predictions": fused,
    }


def run_error_correlation_experiment(
    exp_id: str,
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Run error correlation analysis."""
    print(f"\n{'='*60}")
    print(f"Running {exp_id}: Error Correlation Analysis")
    print(f"{'='*60}")

    result = compute_error_correlation(audio_preds, symbolic_preds, labels)
    print(f"Overall correlation: {result['overall']['correlation']:.4f}")
    print(f"Interpretation: {result['interpretation']}")

    return {"exp_id": exp_id, **result}


def save_fusion_experiment(
    exp_id: str,
    result: Dict[str, Any],
    results_dir: Path,
    all_results: Dict[str, Any],
) -> None:
    """Save experiment result to disk."""
    # Remove numpy arrays before saving (keep only serializable data)
    save_result = {k: v for k, v in result.items() if k != "predictions"}
    all_results[exp_id] = save_result

    # Save individual result
    result_file = results_dir / f"{exp_id}.json"
    with open(result_file, "w") as f:
        json.dump(save_result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

    print(f"Saved: {result_file}")


# ============================================================================
# Statistical Testing Functions
# ============================================================================


def bootstrap_r2_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Extended bootstrap analysis with overall and per-dimension CIs.

    Args:
        y_true: Ground truth labels (n_samples, n_dims) or (n_samples,)
        y_pred: Predictions (n_samples, n_dims) or (n_samples,)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with 'overall' CI and 'per_dimension' CIs if multi-dimensional
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    # Overall R2 bootstrap
    overall_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        overall_scores.append(r2_score(y_true[idx], y_pred[idx]))

    result = {
        "overall": {
            "r2": float(r2_score(y_true, y_pred)),
            "ci_lower": float(np.percentile(overall_scores, 2.5)),
            "ci_median": float(np.percentile(overall_scores, 50)),
            "ci_upper": float(np.percentile(overall_scores, 97.5)),
        }
    }

    # Per-dimension if multi-dimensional
    if y_true.ndim == 2:
        per_dim = {}
        for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
            dim_scores = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_samples, n_samples, replace=True)
                dim_scores.append(r2_score(y_true[idx, i], y_pred[idx, i]))
            per_dim[dim] = {
                "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
                "ci_lower": float(np.percentile(dim_scores, 2.5)),
                "ci_upper": float(np.percentile(dim_scores, 97.5)),
            }
        result["per_dimension"] = per_dim

    return result


def bootstrap_r2_comparison(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Bootstrap CI for difference in R2 between two models.

    Tests if model A is significantly better than model B.

    Args:
        y_true: Ground truth labels
        pred_a: Predictions from model A
        pred_b: Predictions from model B
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Dict with R2 difference, CI, and whether difference is significant
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    r2_a = r2_score(y_true, pred_a)
    r2_b = r2_score(y_true, pred_b)
    diff = r2_a - r2_b

    # Bootstrap the difference
    diff_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        r2_a_boot = r2_score(y_true[idx], pred_a[idx])
        r2_b_boot = r2_score(y_true[idx], pred_b[idx])
        diff_scores.append(r2_a_boot - r2_b_boot)

    ci_lower = float(np.percentile(diff_scores, 2.5))
    ci_upper = float(np.percentile(diff_scores, 97.5))

    return {
        "r2_a": float(r2_a),
        "r2_b": float(r2_b),
        "difference": float(diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": ci_lower > 0 or ci_upper < 0,
        "a_significantly_better": ci_lower > 0,
    }


def paired_ttest_per_sample(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """Paired t-test on per-sample MSE between two models.

    Args:
        y_true: Ground truth labels (n_samples, n_dims)
        pred_a: Predictions from model A
        pred_b: Predictions from model B

    Returns:
        Dict with t-statistic, p-value, and which model is better
    """
    # Compute per-sample MSE (average across dimensions)
    mse_a = ((y_true - pred_a) ** 2).mean(axis=1)
    mse_b = ((y_true - pred_b) ** 2).mean(axis=1)

    # Paired t-test (lower MSE is better)
    t_stat, p_value = stats.ttest_rel(mse_a, mse_b)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_mse_a": float(mse_a.mean()),
        "mean_mse_b": float(mse_b.mean()),
        "a_better": float(mse_a.mean()) < float(mse_b.mean()),
        "significant": p_value < 0.05,
    }


def wilcoxon_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        y_true: Ground truth labels
        pred_a: Predictions from model A
        pred_b: Predictions from model B

    Returns:
        Dict with statistic and p-value
    """
    mse_a = ((y_true - pred_a) ** 2).mean(axis=1)
    mse_b = ((y_true - pred_b) ** 2).mean(axis=1)

    # Wilcoxon requires non-zero differences
    diff = mse_a - mse_b
    if np.all(diff == 0):
        return {"stat": 0.0, "p_value": 1.0, "significant": False}

    stat, p_value = stats.wilcoxon(mse_a, mse_b)

    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "a_better": float(mse_a.mean()) < float(mse_b.mean()),
        "significant": p_value < 0.05,
    }


def cohens_d(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size for difference in MSE.

    Args:
        y_true: Ground truth labels
        pred_a: Predictions from model A
        pred_b: Predictions from model B

    Returns:
        Cohen's d (positive means A has lower MSE = better)
    """
    mse_a = ((y_true - pred_a) ** 2).mean(axis=1)
    mse_b = ((y_true - pred_b) ** 2).mean(axis=1)

    diff = mse_b - mse_a  # Positive if A is better (lower MSE)
    pooled_std = np.sqrt((mse_a.std() ** 2 + mse_b.std() ** 2) / 2)

    if pooled_std == 0:
        return 0.0

    return float(diff.mean() / pooled_std)


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        Tuple of (corrected_p_values, significant_mask)
    """
    n = len(p_values)
    corrected = np.minimum(np.array(p_values) * n, 1.0)
    significant = corrected < alpha
    return corrected, significant


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        Tuple of (corrected_p_values, significant_mask)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH procedure
    threshold = (np.arange(1, n + 1) / n) * alpha
    below_threshold = sorted_p <= threshold

    if not below_threshold.any():
        return p_values, np.zeros(n, dtype=bool)

    max_idx = np.max(np.where(below_threshold)[0])
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[: max_idx + 1]] = True

    # Corrected p-values
    corrected = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        corrected[idx] = sorted_p[i] * n / (i + 1)
    corrected = np.minimum(corrected, 1.0)

    return corrected, significant


# ============================================================================
# Fusion Strategy Functions
# ============================================================================


def simple_average_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
) -> np.ndarray:
    """Simple average of audio and symbolic predictions.

    Args:
        audio_preds: Audio model predictions (n_samples, n_dims)
        symbolic_preds: Symbolic model predictions (n_samples, n_dims)

    Returns:
        Fused predictions
    """
    return (audio_preds + symbolic_preds) / 2


def weighted_fusion_grid_search(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    n_weights: int = 21,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, List[float]]]:
    """Grid search for optimal per-dimension fusion weights with cross-validation.

    Uses leave-one-fold-out to find weights, then applies to held-out fold.
    This prevents overfitting of fusion weights to the test set.

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        labels: Ground truth (n_samples, 19)
        fold_assignments: Dict mapping sample_key -> fold_id (0-3)
        sample_keys: List of sample keys in same order as predictions
        n_weights: Number of weights to try in grid search

    Returns:
        Tuple of:
        - fused_predictions: CV-fused predictions
        - optimal_weights: Dict of dimension -> optimal audio weight
        - fold_weights: Dict of dimension -> list of weights per fold (for stability analysis)
    """
    n_dims = audio_preds.shape[1]
    weights = np.linspace(0, 1, n_weights)
    fused = np.zeros_like(audio_preds)

    # Build fold masks
    fold_ids = np.array([fold_assignments[k] for k in sample_keys])
    n_folds = len(set(fold_ids))

    # Track weights per fold for stability analysis
    fold_weights: Dict[str, List[float]] = {dim: [] for dim in PERCEPIANO_DIMENSIONS}

    for dim_idx, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
        for held_out_fold in range(n_folds):
            # Train on all folds except held_out
            train_mask = fold_ids != held_out_fold
            val_mask = fold_ids == held_out_fold

            # Grid search on training folds
            best_w, best_score = 0.5, -np.inf
            for w in weights:
                fused_train = (
                    w * audio_preds[train_mask, dim_idx]
                    + (1 - w) * symbolic_preds[train_mask, dim_idx]
                )
                score = r2_score(labels[train_mask, dim_idx], fused_train)
                if score > best_score:
                    best_w, best_score = w, score

            fold_weights[dim_name].append(best_w)

            # Apply best weight to held-out fold
            fused[val_mask, dim_idx] = (
                best_w * audio_preds[val_mask, dim_idx]
                + (1 - best_w) * symbolic_preds[val_mask, dim_idx]
            )

    # Compute final optimal weights (mean across folds)
    optimal_weights = {dim: np.mean(fold_weights[dim]) for dim in PERCEPIANO_DIMENSIONS}

    return fused, optimal_weights, fold_weights


def ridge_stacking_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    alphas: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Ridge regression stacking (per-dimension meta-learner).

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        labels: Ground truth (n_samples, 19)
        fold_assignments: Dict mapping sample_key -> fold_id
        sample_keys: List of sample keys
        alphas: Ridge regularization values to try

    Returns:
        Tuple of:
        - fused_predictions: Stacked predictions
        - coefficients: Dict of dimension -> {audio_coef, symbolic_coef, intercept, alpha}
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    n_dims = labels.shape[1]
    fused = np.zeros_like(audio_preds)
    coefficients: Dict[str, Dict[str, float]] = {}

    # Build fold masks
    fold_ids = np.array([fold_assignments[k] for k in sample_keys])
    n_folds = len(set(fold_ids))

    for dim_idx, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
        # Stack features: [audio_pred, symbolic_pred]
        X = np.column_stack([audio_preds[:, dim_idx], symbolic_preds[:, dim_idx]])
        y = labels[:, dim_idx]

        # CV predictions
        for held_out_fold in range(n_folds):
            train_mask = fold_ids != held_out_fold
            val_mask = fold_ids == held_out_fold

            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X[train_mask], y[train_mask])
            fused[val_mask, dim_idx] = ridge.predict(X[val_mask])

        # Final coefficients (trained on all data for interpretation)
        ridge_final = RidgeCV(alphas=alphas)
        ridge_final.fit(X, y)
        coefficients[dim_name] = {
            "audio_coef": float(ridge_final.coef_[0]),
            "symbolic_coef": float(ridge_final.coef_[1]),
            "intercept": float(ridge_final.intercept_),
            "alpha": float(ridge_final.alpha_),
        }

    return fused, coefficients


def confidence_weighted_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    audio_fold_r2: np.ndarray,
    symbolic_fold_r2: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Confidence-weighted fusion using fold-level R2 as confidence.

    Weights each modality by its R2 performance (higher R2 = more weight).

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        audio_fold_r2: Audio R2 per dimension (19,) - averaged across folds
        symbolic_fold_r2: Symbolic R2 per dimension (19,)

    Returns:
        Tuple of:
        - fused_predictions
        - weights: Dict of dimension -> audio_weight
    """
    # Use R2 as confidence (higher = better)
    # Normalize to get weights that sum to 1
    audio_conf = np.maximum(audio_fold_r2, 0)  # Clip negative R2
    symbolic_conf = np.maximum(symbolic_fold_r2, 0)

    total_conf = audio_conf + symbolic_conf + 1e-8
    w_audio = audio_conf / total_conf
    w_symbolic = symbolic_conf / total_conf

    # Apply per-dimension weighting
    fused = w_audio * audio_preds + w_symbolic * symbolic_preds

    weights = {
        dim: float(w_audio[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
    }

    return fused, weights


def category_weighted_fusion(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    n_weights: int = 21,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Optimize fusion weights by dimension category.

    Groups dimensions by category and finds one weight per category.

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        labels: Ground truth (n_samples, 19)
        fold_assignments: Dict mapping sample_key -> fold_id
        sample_keys: List of sample keys
        n_weights: Number of weights to try

    Returns:
        Tuple of:
        - fused_predictions
        - category_weights: Dict of category -> audio_weight
    """
    weights_grid = np.linspace(0, 1, n_weights)
    fused = np.zeros_like(audio_preds)
    category_weights: Dict[str, float] = {}

    # Build fold masks
    fold_ids = np.array([fold_assignments[k] for k in sample_keys])
    n_folds = len(set(fold_ids))

    # Create dimension -> index mapping
    dim_to_idx = {dim: i for i, dim in enumerate(PERCEPIANO_DIMENSIONS)}

    for category, dims in DIMENSION_CATEGORIES.items():
        dim_indices = [dim_to_idx[d] for d in dims]

        # Grid search for this category
        best_w, best_score = 0.5, -np.inf

        for w in weights_grid:
            scores = []
            for held_out_fold in range(n_folds):
                train_mask = fold_ids != held_out_fold

                # Fuse for all dims in this category
                for idx in dim_indices:
                    fused_train = (
                        w * audio_preds[train_mask, idx]
                        + (1 - w) * symbolic_preds[train_mask, idx]
                    )
                    scores.append(r2_score(labels[train_mask, idx], fused_train))

            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_w, best_score = w, avg_score

        category_weights[category] = best_w

        # Apply to all dims in category
        for idx in dim_indices:
            fused[:, idx] = (
                best_w * audio_preds[:, idx]
                + (1 - best_w) * symbolic_preds[:, idx]
            )

    return fused, category_weights


# ============================================================================
# Analysis Functions
# ============================================================================


def compute_error_correlation(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute correlation between audio and symbolic prediction errors.

    Low correlation indicates models make different errors -> fusion helps more.

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        labels: Ground truth (n_samples, 19)

    Returns:
        Dict with overall and per-dimension error correlations
    """
    error_audio = labels - audio_preds
    error_symbolic = labels - symbolic_preds

    # Overall correlation (flatten all errors)
    overall_corr, overall_p = stats.pearsonr(
        error_audio.flatten(), error_symbolic.flatten()
    )

    # Per-dimension
    per_dim = {}
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        corr, p = stats.pearsonr(error_audio[:, i], error_symbolic[:, i])
        per_dim[dim] = {"correlation": float(corr), "p_value": float(p)}

    return {
        "overall": {"correlation": float(overall_corr), "p_value": float(overall_p)},
        "per_dimension": per_dim,
        "interpretation": (
            "low" if abs(overall_corr) < 0.3 else
            "moderate" if abs(overall_corr) < 0.7 else
            "high"
        ),
    }


def compute_weight_stability(
    fold_weights: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute stability metrics for fusion weights across CV folds.

    Args:
        fold_weights: Dict of dimension -> list of weights per fold

    Returns:
        Dict with mean, std, and CV (coefficient of variation) per dimension
    """
    stability = {}
    for dim, weights in fold_weights.items():
        weights_arr = np.array(weights)
        mean_w = float(weights_arr.mean())
        std_w = float(weights_arr.std())
        cv = std_w / (mean_w + 1e-8) if mean_w > 0 else 0.0

        stability[dim] = {
            "mean": mean_w,
            "std": std_w,
            "cv": cv,
            "min": float(weights_arr.min()),
            "max": float(weights_arr.max()),
        }

    # Overall stability
    all_cvs = [v["cv"] for v in stability.values()]
    stability["_overall"] = {
        "mean_cv": float(np.mean(all_cvs)),
        "max_cv": float(np.max(all_cvs)),
        "stable": np.mean(all_cvs) < 0.2,
    }

    return stability


def compute_per_dimension_comparison(
    audio_preds: np.ndarray,
    symbolic_preds: np.ndarray,
    fusion_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-dimension comparison of all models.

    Args:
        audio_preds: Audio predictions
        symbolic_preds: Symbolic predictions
        fusion_preds: Fusion predictions
        labels: Ground truth

    Returns:
        Dict of dimension -> comparison metrics
    """
    comparison = {}

    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        audio_r2 = r2_score(labels[:, i], audio_preds[:, i])
        symbolic_r2 = r2_score(labels[:, i], symbolic_preds[:, i])
        fusion_r2 = r2_score(labels[:, i], fusion_preds[:, i])

        # Determine winner
        if fusion_r2 >= audio_r2 and fusion_r2 >= symbolic_r2:
            winner = "fusion"
        elif audio_r2 >= symbolic_r2:
            winner = "audio"
        else:
            winner = "symbolic"

        # Get category
        category = None
        for cat, dims in DIMENSION_CATEGORIES.items():
            if dim in dims:
                category = cat
                break

        comparison[dim] = {
            "audio_r2": float(audio_r2),
            "symbolic_r2": float(symbolic_r2),
            "fusion_r2": float(fusion_r2),
            "winner": winner,
            "fusion_improvement": float(fusion_r2 - max(audio_r2, symbolic_r2)),
            "category": category,
        }

    return comparison
