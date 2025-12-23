"""
Visualization tools for piano performance evaluation.

Provides publication-quality plots for:
- Per-dimension results (bar charts, radar plots)
- Prediction scatter plots
- Confusion matrices
- Model comparison charts
- Results tables
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .metrics import DIMENSION_CATEGORIES, MetricResult
from .sota_baselines import DIMENSION_BASELINES


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        )


def _check_seaborn():
    if not HAS_SEABORN:
        raise ImportError(
            "seaborn is required for this visualization. Install with: pip install seaborn"
        )


def plot_per_dimension_results(
    dimension_results: Dict[str, float],
    metric_name: str = "R^2",
    baseline_values: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show_categories: bool = True,
) -> "plt.Figure":
    """
    Create bar chart of per-dimension results.

    Args:
        dimension_results: Dict mapping dimension name to metric value
        metric_name: Name of the metric for y-axis label
        baseline_values: Optional baseline values to show as reference line
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
        show_categories: Color bars by dimension category

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    # Sort dimensions by value
    sorted_dims = sorted(dimension_results.items(), key=lambda x: x[1], reverse=True)
    dims = [d[0] for d in sorted_dims]
    values = [d[1] for d in sorted_dims]

    fig, ax = plt.subplots(figsize=figsize)

    # Color by category if requested
    if show_categories:
        dim_to_category = {}
        for cat, cat_dims in DIMENSION_CATEGORIES.items():
            for d in cat_dims:
                dim_to_category[d] = cat

        category_colors = {
            "timing": "#1f77b4",
            "articulation": "#ff7f0e",
            "pedal": "#2ca02c",
            "timbre": "#d62728",
            "dynamics": "#9467bd",
            "musical": "#8c564b",
            "emotion": "#e377c2",
            "interpretation": "#7f7f7f",
        }

        colors = [
            category_colors.get(dim_to_category.get(d, ""), "#333333") for d in dims
        ]
    else:
        colors = "#1f77b4"

    bars = ax.bar(range(len(dims)), values, color=colors)

    # Add baseline reference if provided
    if baseline_values:
        baseline_vals = [baseline_values.get(d, None) for d in dims]
        for i, (bv, v) in enumerate(zip(baseline_vals, values)):
            if bv is not None:
                ax.scatter(i, bv, color="red", s=50, marker="_", linewidths=2, zorder=5)

    # Formatting
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Dimension", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Per-Dimension {metric_name}", fontsize=14)

    # Add horizontal line at 0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Add legend for categories
    if show_categories:
        handles = [
            mpatches.Patch(color=color, label=cat)
            for cat, color in category_colors.items()
            if any(dim_to_category.get(d) == cat for d in dims)
        ]
        if baseline_values:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="_",
                    color="red",
                    label="SOTA Baseline",
                    markersize=10,
                    linewidth=0,
                )
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_names: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Create scatter plots of predictions vs targets for each dimension.

    Args:
        predictions: [n_samples, n_dims] predictions
        targets: [n_samples, n_dims] ground truth
        dimension_names: Names for each dimension
        n_cols: Number of columns in subplot grid
        figsize: Figure size (auto-calculated if None)
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    n_dims = predictions.shape[1]

    if dimension_names is None:
        dimension_names = [f"Dim {i}" for i in range(n_dims)]

    n_rows = (n_dims + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_dims > 1 else [axes]

    for i in range(n_dims):
        ax = axes[i]
        pred = predictions[:, i]
        targ = targets[:, i]

        ax.scatter(targ, pred, alpha=0.5, s=10)

        # Perfect prediction line
        min_val = min(targ.min(), pred.min())
        max_val = max(targ.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

        # Compute R^2
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

        ax.set_xlabel("Ground Truth", fontsize=8)
        ax.set_ylabel("Prediction", fontsize=8)
        ax.set_title(f"{dimension_names[i]}\nR^2={r2:.3f}", fontsize=10)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_name: str,
    n_bins: int = 5,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Create confusion matrix by binning continuous predictions.

    Useful for understanding where the model makes errors.

    Args:
        predictions: [n_samples] predictions for one dimension
        targets: [n_samples] ground truth for one dimension
        dimension_name: Name of the dimension
        n_bins: Number of bins
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    _check_seaborn()

    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    pred_bins = np.digitize(predictions, bin_edges[1:-1])
    target_bins = np.digitize(targets, bin_edges[1:-1])

    # Create confusion matrix
    confusion = np.zeros((n_bins, n_bins), dtype=int)
    for p, t in zip(pred_bins, target_bins):
        confusion[p, t] += 1

    # Normalize by row (predicted class)
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.nan_to_num(confusion_norm)

    fig, ax = plt.subplots(figsize=figsize)

    # Create labels
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(n_bins)]

    sns.heatmap(
        confusion_norm,
        annot=confusion,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Ground Truth", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title(f"Confusion Matrix: {dimension_name}", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dimension_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metric_name: str = "R^2",
    dimension_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Compare multiple models across dimensions.

    Args:
        results_dict: Dict mapping model name to dimension results
        metric_name: Name of the metric
        dimension_names: Order of dimensions (auto-detected if None)
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    model_names = list(results_dict.keys())

    if dimension_names is None:
        dimension_names = list(results_dict[model_names[0]].keys())

    n_dims = len(dimension_names)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_dims)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, results) in enumerate(results_dict.items()):
        values = [results.get(d, 0) for d in dimension_names]
        offset = width * (i - n_models / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(dimension_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_title(f"Model Comparison: {metric_name} by Dimension", fontsize=14)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_results_table(
    metrics: Dict[str, MetricResult],
    dimension_names: Optional[List[str]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Create formatted results table as string.

    Args:
        metrics: Dict of metric name -> MetricResult
        dimension_names: Names of dimensions
        baseline_metrics: Optional baseline for comparison

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 80)

    # Overall metrics
    lines.append("\nOverall Metrics:")
    lines.append("-" * 60)
    lines.append(
        f"{'Metric':<20} {'Value':>10} {'Std':>10} {'Baseline':>10} {'Diff':>10}"
    )
    lines.append("-" * 60)

    for name, result in sorted(metrics.items()):
        std_str = f"{result.std:.4f}" if result.std is not None else "N/A"
        if baseline_metrics and name in baseline_metrics:
            baseline = baseline_metrics[name]
            diff = result.value - baseline
            baseline_str = f"{baseline:.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            baseline_str = "N/A"
            diff_str = "N/A"

        lines.append(
            f"{name:<20} {result.value:>10.4f} {std_str:>10} {baseline_str:>10} {diff_str:>10}"
        )

    # Per-dimension breakdown for R^2
    if "r2" in metrics and metrics["r2"].per_dimension:
        lines.append("\nPer-Dimension R^2:")
        lines.append("-" * 60)
        lines.append(f"{'Dimension':<25} {'R^2':>10} {'Baseline':>10} {'Diff':>10}")
        lines.append("-" * 60)

        per_dim = metrics["r2"].per_dimension
        sorted_dims = sorted(per_dim.items(), key=lambda x: x[1], reverse=True)

        for dim, r2 in sorted_dims:
            if DIMENSION_BASELINES.get(dim) is not None:
                baseline = DIMENSION_BASELINES[dim]
                diff = r2 - baseline
                baseline_str = f"{baseline:.4f}"
                diff_str = f"{diff:+.4f}"
            else:
                baseline_str = "N/A"
                diff_str = "N/A"

            lines.append(f"{dim:<25} {r2:>10.4f} {baseline_str:>10} {diff_str:>10}")

    lines.append("=" * 80)

    return "\n".join(lines)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_r2: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot training curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        val_r2: Optional validation R^2 per epoch
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    n_plots = 2 if val_r2 is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss difference
    loss_diff = np.array(train_losses) - np.array(val_losses)
    axes[1].plot(epochs, loss_diff)
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train - Val Loss")
    axes[1].set_title("Overfitting Indicator")
    axes[1].grid(True, alpha=0.3)

    # R^2 if provided
    if val_r2 is not None:
        axes[2].plot(epochs, val_r2, color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("R^2")
        axes[2].set_title("Validation R^2")
        axes[2].grid(True, alpha=0.3)

        # Mark best epoch
        best_epoch = np.argmax(val_r2) + 1
        best_r2 = max(val_r2)
        axes[2].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5)
        axes[2].scatter([best_epoch], [best_r2], color="red", s=50, zorder=5)
        axes[2].annotate(
            f"Best: {best_r2:.4f}",
            (best_epoch, best_r2),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot error distribution across dimensions.

    Args:
        predictions: [n_samples, n_dims] predictions
        targets: [n_samples, n_dims] ground truth
        dimension_names: Names for each dimension
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    _check_seaborn()

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    n_dims = predictions.shape[1]

    if dimension_names is None:
        dimension_names = [f"Dim {i}" for i in range(n_dims)]

    # Compute errors
    errors = predictions - targets

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot of errors
    bp = ax.boxplot(
        [errors[:, i] for i in range(n_dims)],
        labels=dimension_names,
        patch_artist=True,
    )

    # Color by category
    dim_to_category = {}
    for cat, cat_dims in DIMENSION_CATEGORIES.items():
        for d in cat_dims:
            dim_to_category[d] = cat

    category_colors = {
        "timing": "#1f77b4",
        "articulation": "#ff7f0e",
        "pedal": "#2ca02c",
        "timbre": "#d62728",
        "dynamics": "#9467bd",
        "musical": "#8c564b",
        "emotion": "#e377c2",
        "interpretation": "#7f7f7f",
    }

    for i, (patch, dim) in enumerate(zip(bp["boxes"], dimension_names)):
        cat = dim_to_category.get(dim, "")
        color = category_colors.get(cat, "#333333")
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xticklabels(dimension_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Prediction Error (Pred - Target)", fontsize=12)
    ax.set_title("Error Distribution by Dimension", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Demo visualizations
    np.random.seed(42)
    n_samples, n_dims = 100, 19

    dims = [
        "timing",
        "tempo",
        "articulation_length",
        "articulation_touch",
        "pedal_amount",
        "pedal_clarity",
        "timbre_variety",
        "timbre_depth",
        "timbre_brightness",
        "timbre_loudness",
        "dynamic_range",
        "sophistication",
        "space",
        "balance",
        "drama",
        "mood_valence",
        "mood_energy",
        "mood_imagination",
        "interpretation",
    ]

    # Generate demo data
    targets = np.random.rand(n_samples, n_dims)
    predictions = targets + np.random.randn(n_samples, n_dims) * 0.15

    # Per-dimension R^2
    dim_r2 = {}
    for i, dim in enumerate(dims):
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        dim_r2[dim] = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

    print("Creating demo visualizations...")

    # Per-dimension bar chart
    fig1 = plot_per_dimension_results(
        dim_r2,
        baseline_values=DIMENSION_BASELINES,
        title="Demo: Per-Dimension R^2",
    )

    # Scatter plots
    fig2 = plot_prediction_scatter(
        predictions[:, :8],
        targets[:, :8],
        dims[:8],
    )

    print("Demo visualizations created successfully!")
    plt.show()
