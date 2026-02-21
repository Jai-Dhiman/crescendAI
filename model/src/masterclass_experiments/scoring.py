"""Multi-signal scoring, selection, and hierarchy for taxonomy dimensions."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


# Per-dimension MuQ probing R2 from the PercePiano audit.
# Source: model/data/results/02_muq_fusion_experiments/definitive_all_results.json
PERCEPIANO_MUQ_R2 = {
    "timing": 0.393,
    "articulation_length": 0.671,
    "articulation_touch": 0.542,
    "pedal_amount": 0.557,
    "pedal_clarity": 0.469,
    "timbre_variety": 0.443,
    "timbre_depth": 0.582,
    "timbre_brightness": 0.500,
    "timbre_loudness": 0.578,
    "dynamic_range": 0.596,
    "tempo": 0.270,
    "space": 0.653,
    "balance": 0.554,
    "drama": 0.448,
    "mood_valence": 0.424,
    "mood_energy": 0.529,
    "mood_imagination": 0.485,
    "sophistication": 0.663,
    "interpretation": 0.477,
}


def compute_teacher_frequency(labels: np.ndarray) -> dict[int, float]:
    """Fraction of non-noise moments in each cluster.

    Args:
        labels: HDBSCAN cluster labels, -1 for noise.

    Returns:
        {cluster_id: frequency} where frequencies sum to 1.0.
    """
    valid = labels[labels >= 0]
    total = len(valid)
    if total == 0:
        return {}
    cluster_ids = sorted(set(valid))
    return {int(cid): float((valid == cid).sum() / total) for cid in cluster_ids}


def compute_muq_predictability(
    pp_mapping: dict[int, list[str]],
    r2_scores: dict[str, float] | None = None,
) -> dict[int, float]:
    """Average MuQ probing R2 for each cluster's mapped PercePiano dimensions.

    Args:
        pp_mapping: {cluster_id: [percepiano_dim_name, ...]}.
        r2_scores: Per-dimension R2 dict. Defaults to audit values.

    Returns:
        {cluster_id: mean_r2}.
    """
    if r2_scores is None:
        r2_scores = PERCEPIANO_MUQ_R2

    result = {}
    for cid, dims in pp_mapping.items():
        if not dims:
            result[cid] = 0.0
            continue
        values = [max(0.0, r2_scores.get(d, 0.0)) for d in dims]
        result[cid] = float(np.mean(values))
    return result


def select_dimensions(
    candidates: dict[int, dict],
    freq_threshold: float = 0.05,
    freq_drop_threshold: float = 0.03,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Apply selection criteria from the spec.

    Keep if: frequency > freq_threshold AND at least one soft signal > 0.
    Drop if: frequency < freq_drop_threshold AND no soft signal.

    Args:
        candidates: {cluster_id: {frequency, muq_r2, stop_delta_auc}}.

    Returns:
        (kept, dropped) dicts.
    """
    kept = {}
    dropped = {}
    for cid, scores in candidates.items():
        freq = scores["frequency"]
        has_soft = scores.get("muq_r2", 0) > 0 or scores.get("stop_delta_auc", 0) > 0

        if freq >= freq_threshold and has_soft:
            kept[cid] = scores
        elif freq < freq_drop_threshold and not has_soft:
            dropped[cid] = scores
        else:
            # Marginal: freq >= drop_threshold but < threshold, or no soft signal
            dropped[cid] = scores
    return kept, dropped


def build_hierarchy(
    dimensions: list[dict],
    n_groups: int = 4,
) -> list[dict]:
    """Group dimensions into top-level categories using agglomerative clustering.

    Args:
        dimensions: List of dicts with 'name' and 'centroid' (np.ndarray).
        n_groups: Target number of top-level groups.

    Returns:
        List of {group_name, dimensions} dicts.
    """
    if len(dimensions) <= n_groups:
        return [
            {"group_name": d["name"], "dimensions": [d["name"]]}
            for d in dimensions
        ]

    centroids = np.vstack([d["centroid"] for d in dimensions])
    Z = linkage(centroids, method="ward")
    group_labels = fcluster(Z, t=n_groups, criterion="maxclust")

    groups: dict[int, list[str]] = {}
    for dim, gid in zip(dimensions, group_labels):
        groups.setdefault(int(gid), []).append(dim["name"])

    return [
        {"group_name": f"group_{gid}", "dimensions": dims}
        for gid, dims in sorted(groups.items())
    ]
