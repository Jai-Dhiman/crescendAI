"""
Analysis tools for piano performance evaluation.

Provides:
- Per-dimension analysis with statistical tests
- Model comparison utilities
- Cross-validation aggregation
- Significance testing
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings

from .metrics import (
    compute_r2,
    compute_mse,
    compute_mae,
    compute_pearson_r,
    compute_spearman_rho,
    compute_std_score,
    compute_all_metrics,
    MetricResult,
    DIMENSION_CATEGORIES,
)


@dataclass
class DimensionResult:
    """Results for a single dimension."""
    name: str
    r2: float
    mse: float
    mae: float
    pearson_r: float
    spearman_rho: float
    n_samples: int
    std_score_1: Optional[float] = None


@dataclass
class PerDimensionAnalysis:
    """
    Comprehensive per-dimension analysis of model predictions.

    Attributes:
        dimension_results: Dict mapping dimension name to DimensionResult
        overall_metrics: Dict of overall aggregated metrics
        category_metrics: Dict of metrics grouped by dimension category
        weak_dimensions: List of dimensions with R^2 < threshold
        strong_dimensions: List of dimensions with R^2 > threshold
    """
    dimension_results: Dict[str, DimensionResult] = field(default_factory=dict)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weak_dimensions: List[str] = field(default_factory=list)
    strong_dimensions: List[str] = field(default_factory=list)

    @classmethod
    def from_predictions(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        dimension_names: List[str],
        target_stds: Optional[np.ndarray] = None,
        weak_threshold: float = 0.15,
        strong_threshold: float = 0.30,
    ) -> "PerDimensionAnalysis":
        """
        Create analysis from predictions and targets.

        Args:
            predictions: [n_samples, n_dims] predictions
            targets: [n_samples, n_dims] ground truth
            dimension_names: Names for each dimension
            target_stds: Optional std deviations for Std-Score
            weak_threshold: R^2 below this is considered weak
            strong_threshold: R^2 above this is considered strong

        Returns:
            PerDimensionAnalysis instance
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n_samples, n_dims = predictions.shape

        analysis = cls()

        # Per-dimension analysis
        for i, dim_name in enumerate(dimension_names):
            pred = predictions[:, i]
            targ = targets[:, i]

            r2 = compute_r2(pred, targ, per_dimension=False)
            mse = compute_mse(pred, targ, per_dimension=False)
            mae = compute_mae(pred, targ, per_dimension=False)
            pearson = compute_pearson_r(pred, targ, per_dimension=False)
            spearman = compute_spearman_rho(pred, targ, per_dimension=False)

            std_score = None
            if target_stds is not None and i < len(target_stds):
                std_result = compute_std_score(
                    pred, targ,
                    np.array([target_stds[i]]),
                    alpha=1.0,
                    per_dimension=False,
                )
                std_score = std_result.value

            analysis.dimension_results[dim_name] = DimensionResult(
                name=dim_name,
                r2=r2.value,
                mse=mse.value,
                mae=mae.value,
                pearson_r=pearson.value,
                spearman_rho=spearman.value,
                n_samples=n_samples,
                std_score_1=std_score,
            )

            # Categorize
            if r2.value < weak_threshold:
                analysis.weak_dimensions.append(dim_name)
            elif r2.value > strong_threshold:
                analysis.strong_dimensions.append(dim_name)

        # Overall metrics
        all_metrics = compute_all_metrics(predictions, targets, dimension_names, target_stds)
        analysis.overall_metrics = {
            name: result.value for name, result in all_metrics.items()
        }

        # Category metrics
        dim_to_idx = {name: i for i, name in enumerate(dimension_names)}
        for category, dims in DIMENSION_CATEGORIES.items():
            indices = [dim_to_idx[d] for d in dims if d in dim_to_idx]
            if not indices:
                continue

            cat_preds = predictions[:, indices]
            cat_targets = targets[:, indices]

            r2 = compute_r2(cat_preds, cat_targets, per_dimension=False)
            mae = compute_mae(cat_preds, cat_targets, per_dimension=False)

            analysis.category_metrics[category] = {
                "r2": r2.value,
                "mae": mae.value,
                "n_dims": len(indices),
            }

        return analysis

    def get_ranked_dimensions(self, metric: str = "r2", ascending: bool = False) -> List[Tuple[str, float]]:
        """Get dimensions ranked by a metric."""
        results = []
        for dim_name, dim_result in self.dimension_results.items():
            value = getattr(dim_result, metric, None)
            if value is not None:
                results.append((dim_name, value))

        return sorted(results, key=lambda x: x[1], reverse=not ascending)

    def format_report(self) -> str:
        """Generate a formatted text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("Per-Dimension Analysis Report")
        lines.append("=" * 70)

        # Overall metrics
        lines.append("\nOverall Metrics:")
        lines.append("-" * 40)
        for name, value in sorted(self.overall_metrics.items()):
            lines.append(f"  {name}: {value:.4f}")

        # Category breakdown
        lines.append("\nCategory Breakdown:")
        lines.append("-" * 40)
        for category, metrics in sorted(self.category_metrics.items()):
            lines.append(f"  {category}: R^2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f} ({metrics['n_dims']} dims)")

        # Dimension ranking
        lines.append("\nDimensions Ranked by R^2:")
        lines.append("-" * 40)
        ranked = self.get_ranked_dimensions("r2")
        for dim, r2 in ranked:
            status = ""
            if dim in self.strong_dimensions:
                status = " [STRONG]"
            elif dim in self.weak_dimensions:
                status = " [WEAK]"
            lines.append(f"  {dim}: {r2:.4f}{status}")

        # Summary
        lines.append("\nSummary:")
        lines.append("-" * 40)
        lines.append(f"  Strong dimensions (R^2 > 0.30): {len(self.strong_dimensions)}")
        lines.append(f"  Weak dimensions (R^2 < 0.15): {len(self.weak_dimensions)}")
        lines.append(f"  Total dimensions: {len(self.dimension_results)}")

        if self.weak_dimensions:
            lines.append(f"\n  Weak dimensions to improve: {', '.join(self.weak_dimensions)}")

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ModelComparisonResult:
    """Result of comparing two models."""
    model_a_name: str
    model_b_name: str
    metric: str
    model_a_value: float
    model_b_value: float
    difference: float
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    effect_size: Optional[float] = None


class ModelComparison:
    """
    Compare multiple models with statistical testing.

    Uses paired t-tests across samples or cross-validation folds.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize model comparison.

        Args:
            significance_level: Alpha for significance testing
        """
        self.significance_level = significance_level
        self.results: Dict[str, Dict[str, Any]] = {}

    def add_model_results(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        dimension_names: Optional[List[str]] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add results for a model.

        Args:
            model_name: Name identifier for the model
            predictions: [n_samples, n_dims] predictions
            targets: [n_samples, n_dims] ground truth
            dimension_names: Optional dimension names
            fold_ids: Optional fold assignments for CV analysis
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        # Compute per-sample squared errors for paired testing
        squared_errors = (predictions - targets) ** 2
        abs_errors = np.abs(predictions - targets)

        # Overall metrics
        metrics = compute_all_metrics(predictions, targets, dimension_names)

        self.results[model_name] = {
            "predictions": predictions,
            "targets": targets,
            "squared_errors": squared_errors,
            "abs_errors": abs_errors,
            "metrics": metrics,
            "dimension_names": dimension_names,
            "fold_ids": fold_ids,
        }

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        metric: str = "r2",
    ) -> ModelComparisonResult:
        """
        Compare two models with statistical testing.

        Args:
            model_a: Name of first model
            model_b: Name of second model
            metric: Metric to compare

        Returns:
            ModelComparisonResult with comparison details
        """
        if model_a not in self.results or model_b not in self.results:
            raise ValueError(f"Model not found. Available: {list(self.results.keys())}")

        a_metrics = self.results[model_a]["metrics"]
        b_metrics = self.results[model_b]["metrics"]

        a_value = a_metrics[metric].value
        b_value = b_metrics[metric].value
        difference = a_value - b_value

        # Paired t-test on squared errors (for MSE-based comparison)
        a_errors = self.results[model_a]["squared_errors"].mean(axis=1)
        b_errors = self.results[model_b]["squared_errors"].mean(axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_rel(a_errors, b_errors)

        is_significant = p_value < self.significance_level

        # Cohen's d effect size
        diff = a_errors - b_errors
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        return ModelComparisonResult(
            model_a_name=model_a,
            model_b_name=model_b,
            metric=metric,
            model_a_value=a_value,
            model_b_value=b_value,
            difference=difference,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
        )

    def compare_all_pairs(self, metric: str = "r2") -> List[ModelComparisonResult]:
        """Compare all pairs of models."""
        model_names = list(self.results.keys())
        comparisons = []

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                comparisons.append(self.compare_models(model_a, model_b, metric))

        return comparisons

    def get_ranking(self, metric: str = "r2") -> List[Tuple[str, float]]:
        """Get models ranked by metric value."""
        rankings = []
        for model_name, data in self.results.items():
            value = data["metrics"][metric].value
            rankings.append((model_name, value))

        # Higher is better for most metrics
        reverse = metric in ["r2", "pearson_r", "spearman_rho", "std_score_1_0"]
        return sorted(rankings, key=lambda x: x[1], reverse=reverse)

    def format_comparison_table(self, metric: str = "r2") -> str:
        """Generate formatted comparison table."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Model Comparison (Metric: {metric})")
        lines.append("=" * 70)

        # Ranking
        ranking = self.get_ranking(metric)
        lines.append("\nRanking:")
        lines.append("-" * 40)
        for i, (model, value) in enumerate(ranking, 1):
            lines.append(f"  {i}. {model}: {value:.4f}")

        # Pairwise comparisons
        comparisons = self.compare_all_pairs(metric)
        lines.append("\nPairwise Comparisons:")
        lines.append("-" * 40)
        for comp in comparisons:
            sig_marker = "*" if comp.is_significant else ""
            lines.append(
                f"  {comp.model_a_name} vs {comp.model_b_name}: "
                f"diff={comp.difference:+.4f}, p={comp.p_value:.4f}{sig_marker}"
            )

        lines.append("\n* = statistically significant (p < 0.05)")
        lines.append("=" * 70)
        return "\n".join(lines)


class StatisticalTests:
    """
    Statistical significance tests for model evaluation.

    Implements tests commonly used in MIR research:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Bootstrap confidence intervals
    """

    @staticmethod
    def paired_ttest(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: str = "two-sided",
    ) -> Tuple[float, float]:
        """
        Paired t-test for comparing two models.

        Args:
            scores_a: Per-sample or per-fold scores for model A
            scores_b: Per-sample or per-fold scores for model B
            alternative: "two-sided", "less", or "greater"

        Returns:
            (t_statistic, p_value)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
        return t_stat, p_value

    @staticmethod
    def wilcoxon_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: str = "two-sided",
    ) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test (non-parametric alternative to t-test).

        More robust when normality assumption is violated.

        Args:
            scores_a: Per-sample scores for model A
            scores_b: Per-sample scores for model B
            alternative: "two-sided", "less", or "greater"

        Returns:
            (statistic, p_value)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value = stats.wilcoxon(
                scores_a, scores_b,
                alternative=alternative,
                zero_method="wilcox",
            )
        return stat, p_value

    @staticmethod
    def bootstrap_confidence_interval(
        scores: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for a metric.

        Args:
            scores: Array of scores (per-sample or per-fold)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility

        Returns:
            (mean, lower_bound, upper_bound)
        """
        rng = np.random.RandomState(random_state)
        n = len(scores)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean = np.mean(scores)

        return mean, lower, upper

    @staticmethod
    def compute_effect_size(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            scores_a: Scores for model A
            scores_b: Scores for model B

        Returns:
            Cohen's d (effect size)
        """
        diff = scores_a - scores_b
        return np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    @staticmethod
    def interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


def aggregate_cv_results(
    fold_results: List[Dict[str, MetricResult]],
) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
    """
    Aggregate results across cross-validation folds.

    Args:
        fold_results: List of metric dictionaries, one per fold

    Returns:
        Dict mapping metric name to (mean, std, 95% CI)
    """
    aggregated = {}

    # Get all metric names
    metric_names = fold_results[0].keys()

    for metric_name in metric_names:
        values = [fold[metric_name].value for fold in fold_results]
        values = np.array(values)

        mean = np.mean(values)
        std = np.std(values)

        # 95% confidence interval (assuming t-distribution for small n)
        n = len(values)
        if n > 1:
            t_value = stats.t.ppf(0.975, n - 1)
            margin = t_value * std / np.sqrt(n)
            ci = (mean - margin, mean + margin)
        else:
            ci = (mean, mean)

        aggregated[metric_name] = (mean, std, ci)

    return aggregated


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    n_samples, n_dims = 100, 19

    # Simulate predictions
    targets = np.random.rand(n_samples, n_dims)
    predictions_a = targets + np.random.randn(n_samples, n_dims) * 0.15
    predictions_b = targets + np.random.randn(n_samples, n_dims) * 0.20

    dims = [
        "timing", "tempo", "articulation_length", "articulation_touch",
        "pedal_amount", "pedal_clarity", "timbre_variety", "timbre_depth",
        "timbre_brightness", "timbre_loudness", "dynamic_range", "sophistication",
        "space", "balance", "drama", "mood_valence", "mood_energy",
        "mood_imagination", "interpretation",
    ]

    # Per-dimension analysis
    analysis = PerDimensionAnalysis.from_predictions(
        predictions_a, targets, dims
    )
    print(analysis.format_report())

    # Model comparison
    comparison = ModelComparison()
    comparison.add_model_results("Model A", predictions_a, targets, dims)
    comparison.add_model_results("Model B", predictions_b, targets, dims)
    print(comparison.format_comparison_table())
