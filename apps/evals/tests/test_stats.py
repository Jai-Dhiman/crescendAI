"""Tests for shared.stats module."""
from __future__ import annotations

import pytest

from shared.stats import bootstrap_ci, cohens_d


def test_bootstrap_ci_contains_sample_mean_for_normal_data() -> None:
    values = [0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.55, 0.45, 0.52, 0.48]
    mean = sum(values) / len(values)
    ci = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert ci is not None
    low, high = ci
    assert low < mean < high
    assert high - low < 0.2


def test_bootstrap_ci_is_deterministic_for_same_seed() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ci_a = bootstrap_ci(values, n_bootstrap=500, seed=123)
    ci_b = bootstrap_ci(values, n_bootstrap=500, seed=123)
    assert ci_a == ci_b


def test_bootstrap_ci_returns_none_for_small_sample() -> None:
    assert bootstrap_ci([0.5, 0.5], n_bootstrap=100, seed=42) is None
    assert bootstrap_ci([], n_bootstrap=100, seed=42) is None


def test_cohens_d_zero_for_identical_groups() -> None:
    g = [0.5, 0.6, 0.4, 0.55, 0.45]
    assert cohens_d(g, g) == 0.0


def test_cohens_d_positive_when_group1_has_higher_mean() -> None:
    high = [0.8, 0.9, 0.85, 0.95, 0.88]
    low = [0.2, 0.3, 0.25, 0.35, 0.22]
    d = cohens_d(high, low)
    assert d > 1.5


def test_cohens_d_returns_zero_for_single_element_groups() -> None:
    assert cohens_d([0.5], [0.6]) == 0.0
