"""Validation gates for the teacher-grounded taxonomy.

All 5 gates must pass before Part 2 (training plan rewrite) begins.
"""

from __future__ import annotations

import numpy as np


def check_data_sufficiency(
    labels: np.ndarray,
    min_total: int = 1000,
    min_per_category: int = 30,
) -> dict:
    """Gate 1: Data sufficiency.

    Requires min_total non-noise moments and min_per_category per cluster.
    """
    valid = labels[labels >= 0]
    total = len(valid)
    cluster_ids = sorted(set(valid))
    counts = {int(cid): int((valid == cid).sum()) for cid in cluster_ids}
    small = {cid: cnt for cid, cnt in counts.items() if cnt < min_per_category}

    passed = total >= min_total and len(small) == 0
    return {
        "gate": "data_sufficiency",
        "passed": passed,
        "total_moments": total,
        "noise_count": int((labels == -1).sum()),
        "cluster_counts": counts,
        "small_categories": small,
    }


def check_stop_preservation(
    observed_auc: float,
    threshold: float = 0.80,
) -> dict:
    """Gate 2: STOP preservation.

    Composite dim STOP AUC must be >= threshold.
    """
    return {
        "gate": "stop_preservation",
        "passed": observed_auc >= threshold,
        "observed_auc": observed_auc,
        "threshold": threshold,
    }


def check_independence(
    composite_matrix: np.ndarray,
    threshold: float = 0.80,
    dim_names: list[str] | None = None,
) -> dict:
    """Gate 3: Independence.

    No pair of final dimensions should have |r| > threshold.
    """
    n_dims = composite_matrix.shape[1]
    corr = np.corrcoef(composite_matrix.T)
    violations = []

    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            r = abs(corr[i, j])
            if r > threshold:
                name_i = dim_names[i] if dim_names else f"dim_{i}"
                name_j = dim_names[j] if dim_names else f"dim_{j}"
                violations.append({"pair": (name_i, name_j), "abs_r": float(r)})

    return {
        "gate": "independence",
        "passed": len(violations) == 0,
        "violations": violations,
        "correlation_matrix": corr.tolist(),
    }


def check_actionability(
    quote_bank: dict[str, list],
    min_quotes: int = 5,
) -> dict:
    """Gate 4: Actionability.

    Every dimension must map to >= min_quotes real teacher quotes.
    """
    insufficient = {
        dim: len(quotes)
        for dim, quotes in quote_bank.items()
        if len(quotes) < min_quotes
    }
    return {
        "gate": "actionability",
        "passed": len(insufficient) == 0,
        "per_dimension": {dim: len(q) for dim, q in quote_bank.items()},
        "insufficient": insufficient,
    }


def check_coverage(
    labels: np.ndarray,
    threshold: float = 0.80,
) -> dict:
    """Gate 5: Coverage.

    Final dimensions must cover >= threshold of all teacher moments.
    """
    total = len(labels)
    assigned = int((labels >= 0).sum())
    coverage = assigned / total if total > 0 else 0.0

    return {
        "gate": "coverage",
        "passed": coverage >= threshold,
        "coverage": coverage,
        "assigned": assigned,
        "total": total,
        "noise": int((labels == -1).sum()),
    }


def run_all_gates(
    labels: np.ndarray,
    stop_auc: float,
    composite_matrix: np.ndarray,
    quote_bank: dict[str, list],
    dim_names: list[str] | None = None,
) -> dict:
    """Run all 5 validation gates and return a combined report."""
    gates = [
        check_data_sufficiency(labels),
        check_stop_preservation(stop_auc),
        check_independence(composite_matrix, dim_names=dim_names),
        check_actionability(quote_bank),
        check_coverage(labels),
    ]
    all_passed = all(g["passed"] for g in gates)
    return {"all_passed": all_passed, "gates": gates}
