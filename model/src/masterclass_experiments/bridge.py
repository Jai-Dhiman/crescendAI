"""PercePiano bridge: composite labels from 19 dims to teacher taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_weights(
    dim_mapping: dict[str, list[str]],
    r2_scores: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute normalized weights for each teacher dimension's PercePiano proxies.

    Weights are proportional to max(0, R2) and normalized to sum to 1.

    Args:
        dim_mapping: {teacher_dim: [percepiano_dim, ...]}.
        r2_scores: {percepiano_dim: R2_value}.

    Returns:
        {teacher_dim: {percepiano_dim: weight}}.
    """
    result = {}
    for teacher_dim, pp_dims in dim_mapping.items():
        if not pp_dims:
            result[teacher_dim] = {}
            continue

        raw = {d: max(0.0, r2_scores.get(d, 0.0)) for d in pp_dims}
        total = sum(raw.values())

        if total == 0:
            # Uniform fallback when all R2 values are zero
            uniform = 1.0 / len(pp_dims)
            result[teacher_dim] = {d: uniform for d in pp_dims}
        else:
            result[teacher_dim] = {d: v / total for d, v in raw.items()}

    return result


def compute_composite_labels(
    percepiano_labels: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]],
    dim_index: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Compute composite labels for all segments.

    Args:
        percepiano_labels: {segment_key: np.ndarray of shape [19]}.
        weights: {teacher_dim: {percepiano_dim: weight}} from compute_weights.
        dim_index: {percepiano_dim_name: index_in_label_vector}.

    Returns:
        {segment_key: {teacher_dim: composite_score}}.
    """
    composites: dict[str, dict[str, float]] = {}

    for seg_key, label_vec in percepiano_labels.items():
        seg_composites = {}
        for teacher_dim, dim_weights in weights.items():
            score = 0.0
            for pp_dim, w in dim_weights.items():
                idx = dim_index.get(pp_dim)
                if idx is not None and idx < len(label_vec):
                    score += w * float(label_vec[idx])
            seg_composites[teacher_dim] = score
        composites[seg_key] = seg_composites

    return composites


def save_composite_labels(
    composites: dict[str, dict[str, float]], path: Path
) -> None:
    """Save composite labels to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(composites, f, indent=2)


def load_composite_labels(path: Path) -> dict[str, dict[str, float]]:
    """Load composite labels from JSON."""
    with open(path) as f:
        return json.load(f)
