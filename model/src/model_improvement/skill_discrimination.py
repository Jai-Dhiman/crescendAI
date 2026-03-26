"""T5 skill discrimination metric: can the model rank skill levels correctly?

Given model scores and ordinal skill buckets (1-5), compute pairwise
accuracy across all cross-bucket pairs. Higher bucket = higher expected score.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def skill_discrimination_pairwise(
    scores: np.ndarray,
    buckets: np.ndarray,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """Pairwise accuracy: does the model score higher-bucket recordings higher?

    Only cross-bucket pairs are counted (same-bucket pairs are ambiguous).
    For multi-dimensional scores, computes per-dimension and overall (mean score).

    Args:
        scores: Model predictions, shape (n_recordings,) or (n_recordings, n_dims).
        buckets: Ordinal skill bucket per recording, shape (n_recordings,). Values 1-5.
        n_bootstrap: If >0, compute bootstrap 95% CI with this many resamples.
        seed: Random seed for bootstrap.

    Returns:
        Dict with keys:
        - pairwise_accuracy: float (overall, using mean across dims if multi-dim)
        - n_pairs: int
        - per_dimension: dict[int, float] (only if scores is 2D)
        - ci_lower, ci_upper: float (only if n_bootstrap > 0)
    """
    scores = np.asarray(scores)
    buckets = np.asarray(buckets)
    n = len(scores)

    is_multidim = scores.ndim == 2
    if not is_multidim:
        scores_1d = scores
    else:
        scores_1d = scores.mean(axis=1)

    # Generate all cross-bucket pairs
    correct = 0
    total = 0
    per_dim_correct: dict[int, int] = {}
    per_dim_total: dict[int, int] = {}

    if is_multidim:
        n_dims = scores.shape[1]
        for d in range(n_dims):
            per_dim_correct[d] = 0
            per_dim_total[d] = 0

    for i, j in combinations(range(n), 2):
        if buckets[i] == buckets[j]:
            continue

        total += 1
        if buckets[i] < buckets[j]:
            low_idx, high_idx = i, j
        else:
            low_idx, high_idx = j, i

        if scores_1d[high_idx] > scores_1d[low_idx]:
            correct += 1

        if is_multidim:
            for d in range(n_dims):
                per_dim_total[d] += 1
                if scores[high_idx, d] > scores[low_idx, d]:
                    per_dim_correct[d] += 1

    result: dict = {
        "pairwise_accuracy": correct / total if total > 0 else 0.5,
        "n_pairs": total,
    }

    if is_multidim:
        result["per_dimension"] = {
            d: per_dim_correct[d] / per_dim_total[d] if per_dim_total[d] > 0 else 0.5
            for d in range(n_dims)
        }

    if n_bootstrap > 0 and total > 0:
        rng = np.random.RandomState(seed)
        boot_accs = []
        indices = np.arange(n)
        for _ in range(n_bootstrap):
            sample = rng.choice(indices, size=n, replace=True)
            s_scores = scores_1d[sample]
            s_buckets = buckets[sample]
            bc, bt = 0, 0
            for ii, jj in combinations(range(n), 2):
                if s_buckets[ii] == s_buckets[jj]:
                    continue
                bt += 1
                if s_buckets[ii] < s_buckets[jj]:
                    lo, hi = ii, jj
                else:
                    lo, hi = jj, ii
                if s_scores[hi] > s_scores[lo]:
                    bc += 1
            boot_accs.append(bc / bt if bt > 0 else 0.5)
        result["ci_lower"] = float(np.percentile(boot_accs, 2.5))
        result["ci_upper"] = float(np.percentile(boot_accs, 97.5))

    return result
