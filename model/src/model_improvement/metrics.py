"""Shared metrics suite for cross-experiment comparison."""

from typing import Dict

import numpy as np
import torch
from scipy import stats


class MetricsSuite:
    """Shared metrics suite for all model improvement experiments.

    Provides standardized metrics for comparing audio, symbolic, and fusion
    model variants: pairwise ranking accuracy, regression R-squared,
    difficulty correlation, and robustness evaluation.
    """

    def __init__(self, ambiguous_threshold: float = 0.05) -> None:
        self.ambiguous_threshold = ambiguous_threshold

    def pairwise_accuracy(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> Dict:
        """Compute pairwise ranking accuracy from logits and label pairs.

        Args:
            logits: Ranking logits with shape (n_pairs, n_dims). Positive
                values predict A > B.
            labels_a: Ground-truth scores for sample A, shape (n_pairs, n_dims).
            labels_b: Ground-truth scores for sample B, shape (n_pairs, n_dims).

        Returns:
            Dict with "overall" accuracy (float) and "per_dimension" dict
            mapping dimension index to accuracy. Ambiguous pairs where
            |label_diff| < threshold are excluded.
        """
        logits_np = logits.detach().cpu().numpy()
        labels_a_np = labels_a.detach().cpu().numpy()
        labels_b_np = labels_b.detach().cpu().numpy()

        label_diff = labels_a_np - labels_b_np
        non_ambiguous = np.abs(label_diff) >= self.ambiguous_threshold

        pred_ranking = logits_np > 0
        true_ranking = label_diff > 0

        # Overall accuracy across all non-ambiguous comparisons
        if non_ambiguous.any():
            correct = (pred_ranking == true_ranking) & non_ambiguous
            overall_acc = float(correct.sum() / non_ambiguous.sum())
        else:
            overall_acc = 0.5

        # Per-dimension accuracy
        n_dims = logits_np.shape[1]
        per_dim: Dict[int, float] = {}
        for d in range(n_dims):
            dim_mask = non_ambiguous[:, d]
            if dim_mask.any():
                dim_correct = (pred_ranking[:, d] == true_ranking[:, d]) & dim_mask
                per_dim[d] = float(dim_correct.sum() / dim_mask.sum())
            else:
                per_dim[d] = 0.5

        return {
            "overall": overall_acc,
            "per_dimension": per_dim,
            "n_comparisons": int(non_ambiguous.sum()),
        }

    def regression_r2(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute R-squared between predictions and targets.

        Args:
            predictions: Model predictions, shape (n_samples, n_dims).
            targets: Ground-truth targets, shape (n_samples, n_dims).

        Returns:
            R-squared value (float). Perfect predictions yield 1.0.

        Raises:
            ValueError: If predictions and targets have different shapes.
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} "
                f"vs targets {targets.shape}"
            )

        pred_np = predictions.detach().cpu().numpy().flatten()
        targ_np = targets.detach().cpu().numpy().flatten()

        ss_res = np.sum((targ_np - pred_np) ** 2)
        ss_tot = np.sum((targ_np - np.mean(targ_np)) ** 2)

        if ss_tot == 0.0:
            # All targets are identical; perfect predictions give R2 = 1.0
            return 1.0 if ss_res == 0.0 else 0.0

        return float(1.0 - ss_res / ss_tot)

    def difficulty_correlation(
        self,
        predictions: torch.Tensor,
        difficulties: torch.Tensor,
    ) -> Dict:
        """Compute Spearman rho between predictions and difficulty scores.

        Args:
            predictions: Model predictions, shape (n_samples, n_dims).
            difficulties: Scalar difficulty per sample, shape (n_samples,).

        Returns:
            Dict with "overall_rho" (Spearman rho of mean prediction vs
            difficulty) and "per_dimension" dict mapping dimension index
            to Spearman rho.

        Raises:
            ValueError: If first dimension sizes do not match.
        """
        if predictions.shape[0] != difficulties.shape[0]:
            raise ValueError(
                f"Sample count mismatch: predictions {predictions.shape[0]} "
                f"vs difficulties {difficulties.shape[0]}"
            )

        pred_np = predictions.detach().cpu().numpy()
        diff_np = difficulties.detach().cpu().numpy()

        # Overall: mean prediction across dimensions vs difficulty
        mean_pred = pred_np.mean(axis=1)
        overall_rho, _ = stats.spearmanr(mean_pred, diff_np)

        # Per-dimension
        n_dims = pred_np.shape[1]
        per_dim: Dict[int, float] = {}
        for d in range(n_dims):
            rho, _ = stats.spearmanr(pred_np[:, d], diff_np)
            per_dim[d] = float(rho)

        return {
            "overall_rho": float(overall_rho),
            "per_dimension": per_dim,
        }

    def full_report(self, model: object, test_data: dict) -> dict:
        """Run all applicable metrics and return unified results dict.

        Args:
            model: A model object that produces predictions.
            test_data: Dict with keys like "logits", "labels_a", "labels_b",
                "predictions", "targets", "difficulties", "clean_scores",
                "augmented_scores".

        Returns:
            Dict with metric group names as keys and result dicts as values.
        """
        report: dict = {}

        if all(k in test_data for k in ("logits", "labels_a", "labels_b")):
            report["pairwise"] = self.pairwise_accuracy(
                test_data["logits"],
                test_data["labels_a"],
                test_data["labels_b"],
            )

        if all(k in test_data for k in ("predictions", "targets")):
            report["r2"] = self.regression_r2(
                test_data["predictions"],
                test_data["targets"],
            )

        if all(k in test_data for k in ("predictions", "difficulties")):
            report["difficulty"] = self.difficulty_correlation(
                test_data["predictions"],
                test_data["difficulties"],
            )

        if all(k in test_data for k in ("clean_scores", "augmented_scores")):
            report["robustness"] = compute_robustness_metrics(
                test_data["clean_scores"],
                test_data["augmented_scores"],
            )

        return report


def compute_robustness_metrics(
    clean_scores: torch.Tensor,
    augmented_scores: torch.Tensor,
) -> Dict:
    """Compare clean vs augmented scores to measure model robustness.

    Args:
        clean_scores: Predictions on clean inputs, shape (n_samples, n_dims).
        augmented_scores: Predictions on augmented inputs, same shape.

    Returns:
        Dict with:
        - "pearson_r": Mean Pearson correlation across dimensions.
        - "score_drop_pct": Mean absolute difference as a percentage of the
          mean absolute clean score.

    Raises:
        ValueError: If input shapes do not match.
    """
    if clean_scores.shape != augmented_scores.shape:
        raise ValueError(
            f"Shape mismatch: clean {clean_scores.shape} "
            f"vs augmented {augmented_scores.shape}"
        )

    clean_np = clean_scores.detach().cpu().numpy()
    aug_np = augmented_scores.detach().cpu().numpy()

    n_dims = clean_np.shape[1]

    # Pearson correlation per dimension, then average
    correlations = []
    for d in range(n_dims):
        r, _ = stats.pearsonr(clean_np[:, d], aug_np[:, d])
        correlations.append(r)
    mean_pearson = float(np.mean(correlations))

    # Score drop as percentage of mean absolute clean score
    abs_diff = np.abs(clean_np - aug_np).mean()
    mean_abs_clean = np.abs(clean_np).mean()
    if mean_abs_clean > 0:
        score_drop_pct = float(abs_diff / mean_abs_clean * 100.0)
    else:
        score_drop_pct = 0.0

    return {
        "pearson_r": mean_pearson,
        "score_drop_pct": score_drop_pct,
        "per_dimension_r": {d: float(correlations[d]) for d in range(n_dims)},
    }


def format_comparison_table(results: Dict[str, Dict]) -> str:
    """Format experiment results as a readable ASCII table.

    Args:
        results: Dict mapping model name to a dict of metric values.
            Example: {"A1": {"r2": 0.55, "pairwise": 0.85, ...}}

    Returns:
        Formatted ASCII table string with models as rows, metrics as columns.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Results dict must not be empty")

    # Collect all metric names across all models
    all_metrics: list[str] = []
    for model_results in results.values():
        for metric_name in model_results:
            if metric_name not in all_metrics:
                all_metrics.append(metric_name)

    # Determine column widths
    model_col_width = max(len(name) for name in results) + 2
    model_col_width = max(model_col_width, len("Model") + 2)

    metric_col_widths: Dict[str, int] = {}
    for metric in all_metrics:
        header_len = len(metric) + 2
        value_len = 0
        for model_results in results.values():
            if metric in model_results:
                formatted = _format_metric_value(model_results[metric])
                value_len = max(value_len, len(formatted) + 2)
        metric_col_widths[metric] = max(header_len, value_len)

    # Build header
    header = "Model".ljust(model_col_width)
    for metric in all_metrics:
        header += metric.rjust(metric_col_widths[metric])
    separator = "-" * len(header)

    # Build rows
    rows = []
    for model_name, model_results in results.items():
        row = model_name.ljust(model_col_width)
        for metric in all_metrics:
            col_width = metric_col_widths[metric]
            if metric in model_results:
                formatted = _format_metric_value(model_results[metric])
                row += formatted.rjust(col_width)
            else:
                row += "-".rjust(col_width)
        rows.append(row)

    lines = [header, separator] + rows
    return "\n".join(lines)


def _format_metric_value(value: object) -> str:
    """Format a single metric value for table display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
