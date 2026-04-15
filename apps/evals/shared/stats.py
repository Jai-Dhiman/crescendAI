"""Statistical helpers shared across eval modules."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Compute a bootstrap confidence interval for the sample mean.

    Returns None when N < 5 (not enough samples to produce a meaningful CI).
    Deterministic for a given seed.
    """
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boot_means = np.empty(n_bootstrap, dtype=float)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = float(sample.mean())
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return (low, high)


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    s1, s2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
    n1, n2 = len(group1), len(group2)
    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / denom)
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)
