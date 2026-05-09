"""Boundary tests for tier classification."""
from __future__ import annotations

import pytest

from teacher_model.stage0.tier_classifier import classify_tier


def test_relative_at_ceiling_within_quarter_of_baseline() -> None:
    assert classify_tier(value=2.80, baseline=2.84, mode="relative") == "at_ceiling"


def test_relative_mid_tier_quarter_to_three_quarters_below() -> None:
    assert classify_tier(value=2.30, baseline=2.84, mode="relative") == "mid_tier"


def test_relative_absent_more_than_three_quarters_below() -> None:
    assert classify_tier(value=1.50, baseline=2.84, mode="relative") == "absent"


def test_absolute_thresholds() -> None:
    assert classify_tier(value=2.50, baseline=None, mode="absolute") == "at_ceiling"
    assert classify_tier(value=2.00, baseline=None, mode="absolute") == "mid_tier"
    assert classify_tier(value=1.00, baseline=None, mode="absolute") == "absent"


def test_ci_straddling_ceiling_boundary_returns_compound_label() -> None:
    # value = 2.55, baseline = 2.84, point estimate is mid_tier (delta -0.29).
    # CI (2.45, 2.65) straddles the at_ceiling boundary at baseline-0.25 = 2.59.
    label = classify_tier(value=2.55, baseline=2.84, mode="relative", ci=(2.45, 2.65))
    assert label == "mid_tier_with_ceiling_overlap"


def test_ci_straddling_absent_boundary_returns_compound_label() -> None:
    # baseline = 2.84, absent boundary = 2.84 - 0.75 = 2.09. value = 2.15, CI overlaps 2.09.
    label = classify_tier(value=2.15, baseline=2.84, mode="relative", ci=(2.00, 2.30))
    assert label == "mid_tier_with_absent_overlap"


def test_relative_mode_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline is required"):
        classify_tier(value=2.5, baseline=None, mode="relative")


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        classify_tier(value=2.5, baseline=2.0, mode="weird")  # type: ignore[arg-type]
