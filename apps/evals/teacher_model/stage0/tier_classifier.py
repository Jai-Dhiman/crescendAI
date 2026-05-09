"""Sonnet-anchored + absolute tier classification.

Used by the Stage 0 dossier aggregator to label each capability as
at-ceiling / mid-tier / absent, with compound labels when a 95% CI
straddles a tier boundary.
"""
from __future__ import annotations

from typing import Literal

Tier = str  # "at_ceiling" | "mid_tier" | "absent" | compound variants
Mode = Literal["relative", "absolute"]

_RELATIVE_CEILING_DELTA = 0.25  # within this much of baseline => at_ceiling
_RELATIVE_ABSENT_DELTA = 0.75   # more than this below baseline => absent

_ABSOLUTE_CEILING = 2.5
_ABSOLUTE_ABSENT = 1.5


def _point_tier_relative(value: float, baseline: float) -> Tier:
    delta = baseline - value
    if delta <= _RELATIVE_CEILING_DELTA:
        return "at_ceiling"
    if delta <= _RELATIVE_ABSENT_DELTA:
        return "mid_tier"
    return "absent"


def _point_tier_absolute(value: float) -> Tier:
    if value >= _ABSOLUTE_CEILING:
        return "at_ceiling"
    if value >= _ABSOLUTE_ABSENT:
        return "mid_tier"
    return "absent"


def classify_tier(
    value: float,
    baseline: float | None,
    mode: Mode,
    ci: tuple[float, float] | None = None,
) -> Tier:
    """Classify a measurement into a tier label.

    Args:
        value: the point estimate (e.g. mean dim score, discipline %)
        baseline: Sonnet baseline value (required when mode='relative')
        mode: 'relative' (Sonnet-anchored) or 'absolute' (fixed thresholds)
        ci: optional (low, high) 95% CI; if it straddles a tier boundary,
            the returned label is compound (e.g. 'mid_tier_with_ceiling_overlap')

    Returns the tier label string.
    """
    if mode not in ("relative", "absolute"):
        raise ValueError(f"mode must be 'relative' or 'absolute', got {mode!r}")
    if mode == "relative" and baseline is None:
        raise ValueError("baseline is required when mode='relative'")

    if mode == "relative":
        assert baseline is not None  # narrowed by the check above
        point = _point_tier_relative(value, baseline)
        if ci is None:
            return point
        ceiling_boundary = baseline - _RELATIVE_CEILING_DELTA
        absent_boundary = baseline - _RELATIVE_ABSENT_DELTA
    else:
        point = _point_tier_absolute(value)
        if ci is None:
            return point
        ceiling_boundary = _ABSOLUTE_CEILING
        absent_boundary = _ABSOLUTE_ABSENT

    low, high = ci
    crosses_ceiling = low < ceiling_boundary < high
    crosses_absent = low < absent_boundary < high

    if point == "mid_tier" and crosses_ceiling:
        return "mid_tier_with_ceiling_overlap"
    if point == "mid_tier" and crosses_absent:
        return "mid_tier_with_absent_overlap"
    if point == "at_ceiling" and crosses_ceiling:
        return "at_ceiling_with_mid_overlap"
    if point == "absent" and crosses_absent:
        return "absent_with_mid_overlap"
    return point
