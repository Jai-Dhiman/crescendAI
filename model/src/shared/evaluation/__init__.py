"""Evaluation module for piano performance assessment."""

from .metrics import (
    MetricResult,
    compute_r2,
    compute_mse,
    compute_mae,
    compute_pearson_r,
    compute_spearman_rho,
    compute_std_score,
    compute_all_metrics,
    compute_category_metrics,
    DIMENSION_CATEGORIES,
)
from .analysis import (
    DimensionResult,
    PerDimensionAnalysis,
    ModelComparison,
    ModelComparisonResult,
    StatisticalTests,
    aggregate_cv_results,
)
from .visualization import (
    plot_per_dimension_results,
    plot_confusion_matrix,
    plot_prediction_scatter,
    plot_dimension_comparison,
    plot_training_curves,
    plot_error_distribution,
    create_results_table,
)
from .sota_baselines import (
    BaselineResult,
    PERCEPIANO_BASELINES,
    DIMENSION_BASELINES,
    compare_to_sota,
    format_comparison_table,
    get_target_metrics,
)

__all__ = [
    # Metrics
    "MetricResult",
    "compute_r2",
    "compute_mse",
    "compute_mae",
    "compute_pearson_r",
    "compute_spearman_rho",
    "compute_std_score",
    "compute_all_metrics",
    "compute_category_metrics",
    "DIMENSION_CATEGORIES",
    # Analysis
    "DimensionResult",
    "PerDimensionAnalysis",
    "ModelComparison",
    "ModelComparisonResult",
    "StatisticalTests",
    "aggregate_cv_results",
    # Visualization
    "plot_per_dimension_results",
    "plot_confusion_matrix",
    "plot_prediction_scatter",
    "plot_dimension_comparison",
    "plot_training_curves",
    "plot_error_distribution",
    "create_results_table",
    # SOTA Baselines
    "BaselineResult",
    "PERCEPIANO_BASELINES",
    "DIMENSION_BASELINES",
    "compare_to_sota",
    "format_comparison_table",
    "get_target_metrics",
]
