"""Recall@k, MRR, and open-set threshold curve for the piece-ID feasibility harness.

Rankings are lists of (query_true_piece_id, ranked_results) where
ranked_results is a list of (piece_id, score) tuples sorted descending by score.
"""
from __future__ import annotations

import numpy as np


# Type alias: list of (true_piece_id, ranked [(piece_id, score), ...])
Rankings = list[tuple[str, list[tuple[str, float]]]]


def recall_at_k(rankings: Rankings, k: int) -> float:
    """Fraction of queries where the true piece appears in the top-k results.

    Raises:
        ValueError: if k < 1 or rankings is empty.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not rankings:
        raise ValueError("rankings is empty")
    hits = 0
    for true_id, ranked in rankings:
        top_k_ids = [pid for pid, _ in ranked[:k]]
        if true_id in top_k_ids:
            hits += 1
    return hits / len(rankings)


def mrr(rankings: Rankings) -> float:
    """Mean reciprocal rank. Queries where true piece is not found contribute 0.

    Raises:
        ValueError: if rankings is empty.
    """
    if not rankings:
        raise ValueError("rankings is empty")
    total = 0.0
    for true_id, ranked in rankings:
        for rank, (pid, _) in enumerate(ranked, start=1):
            if pid == true_id:
                total += 1.0 / rank
                break
    return total / len(rankings)


def open_set_curve(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and return (false_accept_rates, true_accept_rates).

    At each threshold t, a query is "accepted" if its best-match score >= t.
    - true_accept_rate: fraction of in-catalog queries accepted (recall).
    - false_accept_rate: fraction of out-of-catalog queries accepted (FA).

    Returns:
        (fa_rates, ta_rates) — each shape == thresholds.shape, float64.

    Raises:
        ValueError: if in_scores or out_scores is empty.
    """
    if len(in_scores) == 0:
        raise ValueError("in_scores is empty")
    if len(out_scores) == 0:
        raise ValueError("out_scores is empty")
    fa_rates = np.array(
        [(out_scores >= t).mean() for t in thresholds], dtype=np.float64
    )
    ta_rates = np.array(
        [(in_scores >= t).mean() for t in thresholds], dtype=np.float64
    )
    return fa_rates, ta_rates


def open_set_ok(
    fa_rates: np.ndarray,
    ta_rates: np.ndarray,
    max_fa: float,
    min_ta: float,
) -> bool:
    """Return True iff some threshold achieves fa <= max_fa AND ta >= min_ta simultaneously."""
    return bool(np.any((fa_rates <= max_fa) & (ta_rates >= min_ta)))
