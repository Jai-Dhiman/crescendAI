"""Layer 1 validation experiment helpers.

Experiment 1: Competition correlation (does A1 quality signal predict placement?)
Experiment 2: AMT degradation (does S2 survive transcribed MIDI?)
Experiment 3: Dynamic range (does A1 differentiate intermediate players?)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from model_improvement.taxonomy import DIMENSIONS

logger = logging.getLogger(__name__)


def score_competition_segments(
    model: torch.nn.Module,
    embeddings: dict[str, torch.Tensor],
    max_frames: int = 1000,
) -> dict[str, np.ndarray]:
    """Score competition segments using A1 model's predict_scores.

    Args:
        model: A1 model with predict_scores(x, mask) -> [1, 6].
        embeddings: {segment_id: tensor [T, 1024]}.
        max_frames: Truncate embeddings longer than this.

    Returns:
        {segment_id: np.ndarray [6]} with per-dimension scores.
    """
    model.eval()
    device = next(model.parameters()).device
    results = {}

    with torch.no_grad():
        for seg_id, emb in embeddings.items():
            if emb.shape[0] > max_frames:
                emb = emb[:max_frames]
            x = emb.unsqueeze(0).to(device)  # [1, T, 1024]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            scores = model.predict_scores(x, mask)  # [1, 6]
            results[seg_id] = scores.squeeze(0).cpu().numpy()

    return results


def competition_correlation(
    segment_scores: dict[str, np.ndarray],
    metadata: list[dict],
    aggregations: tuple[str, ...] = ("mean", "median", "min"),
) -> dict[str, dict]:
    """Compute Spearman rho of aggregated scores vs competition placement.

    Groups segments by performer, aggregates per-dimension scores using each
    method, then correlates the mean-across-dimensions aggregate with placement.
    Also computes per-dimension correlations.

    Args:
        segment_scores: {segment_id: np.ndarray [6]}.
        metadata: List of dicts with segment_id, performer, placement.
        aggregations: Aggregation methods to try.

    Returns:
        {agg_name: {rho, p_value, per_dimension: {dim_name: {rho, p_value}}}}.
    """
    # Group scores by performer
    performer_segments: dict[str, list[np.ndarray]] = defaultdict(list)
    performer_placement: dict[str, int] = {}

    for meta in metadata:
        seg_id = meta["segment_id"]
        if seg_id not in segment_scores:
            continue
        performer = meta["performer"]
        performer_segments[performer].append(segment_scores[seg_id])
        performer_placement[performer] = meta["placement"]

    agg_fns = {
        "mean": lambda arrs: np.mean(arrs, axis=0),
        "median": lambda arrs: np.median(arrs, axis=0),
        "min": lambda arrs: np.min(arrs, axis=0),
    }

    results = {}
    performers = sorted(performer_segments.keys())
    placements = np.array([performer_placement[p] for p in performers])

    for agg_name in aggregations:
        fn = agg_fns[agg_name]
        agg_scores = np.array([fn(performer_segments[p]) for p in performers])  # [P, 6]

        # Overall: mean across dimensions vs placement
        overall = agg_scores.mean(axis=1)
        # Negate placement because lower placement = better, higher score = better
        rho, p_value = stats.spearmanr(overall, -placements)

        # Per-dimension
        per_dim = {}
        for d, dim_name in enumerate(DIMENSIONS):
            dim_rho, dim_p = stats.spearmanr(agg_scores[:, d], -placements)
            per_dim[dim_name] = {"rho": float(dim_rho), "p_value": float(dim_p)}

        results[agg_name] = {
            "rho": float(rho),
            "p_value": float(p_value),
            "per_dimension": per_dim,
            "n_performers": len(performers),
        }

    return results
