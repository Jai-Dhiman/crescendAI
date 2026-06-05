# model/src/piece_id_eval/open_set.py
"""Leave-one-out false-accept / true-accept curve for open-set evaluation.

In-catalog queries: the true piece IS in the catalog (standard recall).
LOO (leave-one-out) queries: the true piece has been removed from the catalog;
  any accept is a false accept (the only correct answer is "unknown").

A query is "accepted" if its top-1 score >= threshold.
"""
from __future__ import annotations

from typing import NamedTuple


class OperatingPoint(NamedTuple):
    """One point on the FA/TA curve at a given threshold."""
    threshold: float
    fa: float  # false-accept rate: fraction of LOO queries accepted
    ta: float  # true-accept rate: fraction of in-catalog queries accepted


def operating_points(
    in_catalog_scores: list[float],
    loo_scores: list[float],
    thresholds: list[float],
) -> list[OperatingPoint]:
    """Sweep thresholds and return one OperatingPoint per threshold.

    Args:
        in_catalog_scores: top-1 match score for each in-catalog query.
        loo_scores: top-1 match score for each LOO (out-of-catalog) query.
        thresholds: list of score thresholds to sweep (ascending order not required).

    Returns:
        List of OperatingPoint in the same order as thresholds.

    Raises:
        ValueError: if either score list is empty.
    """
    if not in_catalog_scores:
        raise ValueError("in_catalog_scores is empty")
    if not loo_scores:
        raise ValueError("loo_scores is empty")

    points: list[OperatingPoint] = []
    for t in thresholds:
        ta = sum(1 for s in in_catalog_scores if s >= t) / len(in_catalog_scores)
        fa = sum(1 for s in loo_scores if s >= t) / len(loo_scores)
        points.append(OperatingPoint(threshold=float(t), fa=fa, ta=ta))
    return points


def best_point(
    points: list[OperatingPoint],
    max_fa: float,
    min_ta: float,
) -> OperatingPoint | None:
    """Return the OperatingPoint with lowest FA that also satisfies TA >= min_ta,
    among all points with FA <= max_fa. Returns None if no such point exists.

    Args:
        points: list of OperatingPoint from operating_points().
        max_fa: maximum allowable false-accept rate.
        min_ta: minimum required true-accept rate.

    Returns:
        Best OperatingPoint, or None if no point qualifies.
    """
    qualifying = [p for p in points if p.fa <= max_fa and p.ta >= min_ta]
    if not qualifying:
        return None
    return min(qualifying, key=lambda p: p.fa)
