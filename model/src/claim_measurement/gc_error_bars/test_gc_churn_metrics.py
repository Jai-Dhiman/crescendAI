"""Unit tests for the pure G-C churn math. No model / torch / network."""
from __future__ import annotations

import math

from gc_churn_metrics import (
    clip_per_note_churn,
    clip_statistic_churn,
    mean_velocity,
    pool,
    recommend_substrate_terms,
)


def test_mean_velocity():
    assert mean_velocity([60.0, 70.0, 80.0]) == 70.0


def test_statistic_churn_zero_when_no_variance():
    # A deterministic substrate (identical re-captures) has exactly zero churn.
    assert clip_statistic_churn([64.0, 64.0, 64.0]) == 0.0


def test_statistic_churn_is_sample_std():
    # ddof=1: std of [1,2,3] about mean 2 is sqrt((1+0+1)/2) = 1.0
    assert clip_statistic_churn([1.0, 2.0, 3.0]) == 1.0


def test_statistic_churn_nan_for_single_point():
    assert math.isnan(clip_statistic_churn([64.0]))


def test_per_note_churn_is_population_std():
    # deltas [-2, 0, 2] -> population std sqrt((4+0+4)/3) = sqrt(8/3)
    assert clip_per_note_churn([-2.0, 0.0, 2.0]) == math.sqrt(8.0 / 3.0)


def test_per_note_churn_nan_for_singleton():
    assert math.isnan(clip_per_note_churn([1.0]))


def test_pool_drops_nan_and_summarizes():
    out = pool([1.0, 2.0, 3.0, float("nan")])
    assert out["n"] == 3
    assert out["median"] == 2.0
    assert out["mean"] == 2.0
    assert out["max"] == 3.0


def test_pool_all_nan():
    out = pool([float("nan"), float("nan")])
    assert out["n"] == 0
    assert math.isnan(out["median"])


def test_recommend_returns_both_terms():
    # per-note p90 -> shrinking sigma_note; statistic p90 -> flat floor; max reported too
    stat = pool([1.0, 1.0, 2.5])     # p90 == 2.5, max == 2.5
    note = pool([3.0, 3.0])          # p90 == 3.0
    rec = recommend_substrate_terms(stat, note)
    assert rec["sigma_note"] == 3.0
    assert rec["statistic_floor"] == 2.5
    assert rec["statistic_floor_max"] == 2.5


def test_recommend_floor_is_flat_not_scaled_by_n():
    # The floor is the measured statistic churn AS-IS -- it must NOT be inflated by
    # sqrt(N) (the old bug): correlated churn does not shrink or grow with note count.
    stat = pool([1.4, 1.4])          # p90 == 1.4
    note = pool([2.7, 2.7])          # p90 == 2.7
    rec = recommend_substrate_terms(stat, note)
    assert rec["statistic_floor"] == 1.4
    assert rec["sigma_note"] == 2.7
