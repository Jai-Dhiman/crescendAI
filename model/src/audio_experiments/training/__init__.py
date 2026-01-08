"""Training utilities for audio experiments."""

from .metrics import bootstrap_r2, compute_comprehensive_metrics
from .runner import (
    experiment_completed,
    load_existing_results,
    run_4fold_mert_experiment,
    run_4fold_mel_experiment,
    run_4fold_stats_experiment,
)
from .sync import (
    get_completed_experiments,
    print_experiment_status,
    restore_all_from_gdrive,
    should_run_experiment,
    sync_experiment_to_gdrive,
)
from .fusion import (
    # Experiment runners
    run_bootstrap_experiment,
    run_paired_tests_experiment,
    run_multiple_correction_experiment,
    run_simple_fusion_experiment,
    run_weighted_fusion_experiment,
    run_ridge_fusion_experiment,
    run_confidence_fusion_experiment,
    run_weight_stability_experiment,
    run_category_fusion_experiment,
    run_error_correlation_experiment,
    save_fusion_experiment,
    # Statistical testing
    bootstrap_r2_extended,
    bootstrap_r2_comparison,
    paired_ttest_per_sample,
    wilcoxon_test,
    cohens_d,
    bonferroni_correction,
    fdr_correction,
    # Fusion strategies
    simple_average_fusion,
    weighted_fusion_grid_search,
    ridge_stacking_fusion,
    confidence_weighted_fusion,
    category_weighted_fusion,
    # Analysis
    compute_error_correlation,
    compute_weight_stability,
    compute_per_dimension_comparison,
)

__all__ = [
    # Metrics
    "bootstrap_r2",
    "compute_comprehensive_metrics",
    # Runner
    "experiment_completed",
    "load_existing_results",
    "run_4fold_mert_experiment",
    "run_4fold_mel_experiment",
    "run_4fold_stats_experiment",
    # Sync
    "get_completed_experiments",
    "print_experiment_status",
    "restore_all_from_gdrive",
    "should_run_experiment",
    "sync_experiment_to_gdrive",
    # Fusion - Experiment runners
    "run_bootstrap_experiment",
    "run_paired_tests_experiment",
    "run_multiple_correction_experiment",
    "run_simple_fusion_experiment",
    "run_weighted_fusion_experiment",
    "run_ridge_fusion_experiment",
    "run_confidence_fusion_experiment",
    "run_weight_stability_experiment",
    "run_category_fusion_experiment",
    "run_error_correlation_experiment",
    "save_fusion_experiment",
    # Fusion - Statistical testing
    "bootstrap_r2_extended",
    "bootstrap_r2_comparison",
    "paired_ttest_per_sample",
    "wilcoxon_test",
    "cohens_d",
    "bonferroni_correction",
    "fdr_correction",
    # Fusion - Strategies
    "simple_average_fusion",
    "weighted_fusion_grid_search",
    "ridge_stacking_fusion",
    "confidence_weighted_fusion",
    "category_weighted_fusion",
    # Fusion - Analysis
    "compute_error_correlation",
    "compute_weight_stability",
    "compute_per_dimension_comparison",
]
