"""Analyze skill eval results.

Computes Spearman correlations, bucket separation, confusion rates,
and config comparisons. Generates summary plots.

Usage:
    cd apps/evals/
    uv run python -m model.skill_eval.analyze --piece fur_elise
    uv run python -m model.skill_eval.analyze --piece all
    uv run python -m model.skill_eval.analyze --piece fur_elise --config ensemble_4fold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from paths import MODEL_DATA

DATA_DIR = MODEL_DATA / "evals" / "skill_eval"
FIGURES_DIR = DATA_DIR / "figures"
DIMENSIONS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]


def load_results(config_name: str, piece_id: str) -> list[dict] | None:
    """Load results for a config/piece. Returns None if not found."""
    path = DATA_DIR / config_name / piece_id / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("recordings", [])


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrapped confidence interval for a statistic.

    Returns (point_estimate, ci_low, ci_high).
    """
    point = stat_fn(x, y)
    boot_stats = []
    rng = np.random.default_rng(42)
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats.append(stat_fn(x[idx], y[idx]))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, lo, hi


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation, handling edge cases."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def confusion_rate(scores: np.ndarray, buckets: np.ndarray) -> float:
    """Fraction of cross-bucket pairs where lower-skill has higher score."""
    inversions = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if buckets[i] == buckets[j]:
                continue
            total += 1
            # Higher bucket should have higher score
            if buckets[i] < buckets[j] and scores[i] > scores[j]:
                inversions += 1
            elif buckets[j] < buckets[i] and scores[j] > scores[i]:
                inversions += 1
    return inversions / total if total > 0 else 0.0


def analyze_piece(results: list[dict], piece_id: str, config_name: str):
    """Print analysis for a single piece/config."""
    if not results:
        print(f"  No results for {config_name}/{piece_id}")
        return

    buckets = np.array([r["skill_bucket"] for r in results])
    mean_all = np.array([np.mean(list(r["mean_scores"].values())) for r in results])

    print(f"\n=== {piece_id} ({len(results)} recordings, config: {config_name}) ===")

    # Overall Spearman rho
    rho, ci_lo, ci_hi = bootstrap_ci(mean_all, buckets, spearman_rho)
    p_value = stats.spearmanr(mean_all, buckets).pvalue if len(mean_all) >= 3 else 1.0
    print(f"Overall Spearman rho: {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p_value:.4f})")

    # Per-dimension
    print("\nPer-dimension Spearman rho:")
    best_dim, best_rho = "", -1.0
    for dim in DIMENSIONS:
        dim_scores = np.array([r["mean_scores"][dim] for r in results])
        dim_rho, dim_lo, dim_hi = bootstrap_ci(dim_scores, buckets, spearman_rho)
        print(f"  {dim:15s}  {dim_rho:+.3f} [{dim_lo:+.3f}, {dim_hi:+.3f}]")
        if dim_rho > best_rho:
            best_rho = dim_rho
            best_dim = dim
    print(f"  Best dimension: {best_dim} ({best_rho:+.3f})")

    # Bucket separation (merged: low=1+2, mid=3, high=4+5)
    low = mean_all[buckets <= 2]
    mid = mean_all[buckets == 3]
    high = mean_all[buckets >= 4]

    print(f"\nBucket separation (Cohen's d, merged):")
    if len(low) >= 2 and len(mid) >= 2:
        d_low_mid = cohens_d(mid, low)
        print(f"  low(1+2) vs mid(3):   d={d_low_mid:+.2f}  (n={len(low)} vs {len(mid)})")
    if len(mid) >= 2 and len(high) >= 2:
        d_mid_high = cohens_d(high, mid)
        print(f"  mid(3) vs high(4+5):  d={d_mid_high:+.2f}  (n={len(mid)} vs {len(high)})")
    if len(low) >= 2 and len(high) >= 2:
        d_low_high = cohens_d(high, low)
        print(f"  low(1+2) vs high(4+5): d={d_low_high:+.2f}  (n={len(low)} vs {len(high)})")

    # Confusion rate
    conf = confusion_rate(mean_all, buckets)
    # Random baseline via permutation
    rng = np.random.default_rng(42)
    random_confs = []
    for _ in range(1000):
        perm = rng.permutation(buckets)
        random_confs.append(confusion_rate(mean_all, perm))
    random_baseline = float(np.mean(random_confs))
    print(f"\nConfusion rate: {conf:.3f} (random baseline: {random_baseline:.3f})")

    # Per-bucket means
    print(f"\nPer-bucket mean scores:")
    for b in sorted(set(buckets)):
        mask = buckets == b
        n = mask.sum()
        mean = float(mean_all[mask].mean())
        std = float(mean_all[mask].std()) if n > 1 else 0.0
        print(f"  Bucket {b}: {mean:.3f} +/- {std:.3f} (n={n})")

    # Latency summary
    times = [r.get("mean_processing_time_ms", 0) for r in results if r.get("mean_processing_time_ms")]
    if times:
        print(f"\nLatency: {np.mean(times):.0f}ms/chunk mean, {np.median(times):.0f}ms median")


def compare_configs(
    baseline_results: list[dict],
    variant_results: list[dict],
    baseline_name: str,
    variant_name: str,
    piece_id: str,
):
    """Compare a variant config against the baseline."""
    # Match by video_id
    baseline_by_id = {r["video_id"]: r for r in baseline_results}
    variant_by_id = {r["video_id"]: r for r in variant_results}
    common_ids = set(baseline_by_id) & set(variant_by_id)

    if len(common_ids) < 5:
        print(f"  Too few common recordings ({len(common_ids)}) to compare {variant_name}")
        return

    b_scores = np.array([np.mean(list(baseline_by_id[vid]["mean_scores"].values())) for vid in common_ids])
    v_scores = np.array([np.mean(list(variant_by_id[vid]["mean_scores"].values())) for vid in common_ids])
    b_buckets = np.array([baseline_by_id[vid]["skill_bucket"] for vid in common_ids])

    rank_corr = spearman_rho(b_scores, v_scores)
    score_mae = float(np.mean(np.abs(b_scores - v_scores)))

    b_rho = spearman_rho(b_scores, b_buckets)
    v_rho = spearman_rho(v_scores, b_buckets)
    rho_delta = v_rho - b_rho

    print(f"\n  {variant_name} vs {baseline_name} ({len(common_ids)} common recordings):")
    print(f"    Rank correlation:  {rank_corr:.3f}")
    print(f"    Score MAE:         {score_mae:.4f}")
    print(f"    Skill rho delta:   {rho_delta:+.3f} ({baseline_name}: {b_rho:.3f}, {variant_name}: {v_rho:.3f})")


def generate_plot(piece_id: str, all_configs: dict[str, list[dict]]):
    """Generate bucket vs score plot for a piece across configs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.linspace(-0.15, 0.15, len(all_configs))

    for idx, (config_name, results) in enumerate(all_configs.items()):
        if not results:
            continue
        buckets = sorted(set(r["skill_bucket"] for r in results))
        means = []
        stds = []
        for b in buckets:
            scores = [np.mean(list(r["mean_scores"].values())) for r in results if r["skill_bucket"] == b]
            means.append(np.mean(scores))
            stds.append(np.std(scores) if len(scores) > 1 else 0)

        ax.errorbar(
            np.array(buckets) + offsets[idx],
            means, yerr=stds,
            label=config_name, marker="o", capsize=3,
        )

    ax.set_xlabel("Skill Bucket")
    ax.set_ylabel("Mean Score (6 dims)")
    ax.set_title(f"Skill Eval: {piece_id}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Beginner", "Early Int", "Intermediate", "Advanced", "Professional"], rotation=15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / f"{piece_id}_skill_eval.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze skill eval results")
    parser.add_argument("--piece", required=True, choices=["fur_elise", "nocturne_op9no2", "all"])
    parser.add_argument("--config", default=None, help="Analyze single config (default: all available)")
    args = parser.parse_args()

    pieces = ["fur_elise", "nocturne_op9no2"] if args.piece == "all" else [args.piece]

    quality_configs = ["ensemble_4fold", "single_fold_0", "single_fold_best"]
    configs_to_analyze = [args.config] if args.config else quality_configs

    for piece_id in pieces:
        all_configs = {}

        for config_name in configs_to_analyze:
            results = load_results(config_name, piece_id)
            if results:
                all_configs[config_name] = results
                analyze_piece(results, piece_id, config_name)

        # Config comparisons (vs ensemble baseline)
        baseline = all_configs.get("ensemble_4fold")
        if baseline:
            for config_name, results in all_configs.items():
                if config_name == "ensemble_4fold":
                    continue
                compare_configs(baseline, results, "ensemble_4fold", config_name, piece_id)

        # Plot
        if all_configs:
            generate_plot(piece_id, all_configs)

        # Latency-only configs
        for latency_config in ["no_amt", "cpu_only"]:
            results = load_results(latency_config, piece_id)
            if results:
                times = [r.get("mean_processing_time_ms", 0) for r in results if r.get("mean_processing_time_ms")]
                if times:
                    print(f"\n  Latency ({latency_config}): {np.mean(times):.0f}ms/chunk mean")


if __name__ == "__main__":
    main()
