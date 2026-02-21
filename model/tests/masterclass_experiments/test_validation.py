"""Tests for validation gates."""

import numpy as np
import pytest


def test_check_data_sufficiency_pass():
    from masterclass_experiments.validation import check_data_sufficiency

    # 3 clusters, each with > 30 moments, total > 1000
    labels = np.array([0] * 400 + [1] * 350 + [2] * 300)
    result = check_data_sufficiency(labels, min_total=1000, min_per_category=30)
    assert result["passed"] is True
    assert result["total_moments"] == 1050


def test_check_data_sufficiency_fail_total():
    from masterclass_experiments.validation import check_data_sufficiency

    labels = np.array([0] * 50 + [1] * 50)
    result = check_data_sufficiency(labels, min_total=1000, min_per_category=30)
    assert result["passed"] is False


def test_check_data_sufficiency_fail_small_category():
    from masterclass_experiments.validation import check_data_sufficiency

    labels = np.array([0] * 500 + [1] * 500 + [2] * 10)
    result = check_data_sufficiency(labels, min_total=1000, min_per_category=30)
    assert result["passed"] is False
    assert 2 in result["small_categories"]


def test_check_stop_preservation_pass():
    from masterclass_experiments.validation import check_stop_preservation

    result = check_stop_preservation(observed_auc=0.83, threshold=0.80)
    assert result["passed"] is True


def test_check_stop_preservation_fail():
    from masterclass_experiments.validation import check_stop_preservation

    result = check_stop_preservation(observed_auc=0.75, threshold=0.80)
    assert result["passed"] is False


def test_check_independence_pass():
    from masterclass_experiments.validation import check_independence

    # 3 dims, no pair with r > 0.80
    rng = np.random.RandomState(42)
    composite_matrix = rng.randn(100, 3)
    result = check_independence(composite_matrix, threshold=0.80)
    assert result["passed"] is True


def test_check_independence_fail():
    from masterclass_experiments.validation import check_independence

    # Dim 0 and dim 1 are nearly identical
    rng = np.random.RandomState(42)
    base = rng.randn(100)
    composite_matrix = np.column_stack([base, base + 0.01 * rng.randn(100), rng.randn(100)])
    result = check_independence(composite_matrix, threshold=0.80)
    assert result["passed"] is False
    assert len(result["violations"]) >= 1


def test_check_actionability_pass():
    from masterclass_experiments.validation import check_actionability

    quote_bank = {
        "dynamics": [{"text": f"q{i}"} for i in range(5)],
        "phrasing": [{"text": f"q{i}"} for i in range(7)],
    }
    result = check_actionability(quote_bank, min_quotes=5)
    assert result["passed"] is True


def test_check_actionability_fail():
    from masterclass_experiments.validation import check_actionability

    quote_bank = {
        "dynamics": [{"text": "q1"}, {"text": "q2"}],
        "phrasing": [{"text": f"q{i}"} for i in range(5)],
    }
    result = check_actionability(quote_bank, min_quotes=5)
    assert result["passed"] is False
    assert "dynamics" in result["insufficient"]


def test_check_coverage_pass():
    from masterclass_experiments.validation import check_coverage

    # 100 moments, 15 noise, 85 assigned
    labels = np.array([0] * 40 + [1] * 45 + [-1] * 15)
    result = check_coverage(labels, threshold=0.80)
    assert result["passed"] is True
    assert result["coverage"] == pytest.approx(85 / 100)


def test_check_coverage_fail():
    from masterclass_experiments.validation import check_coverage

    # 100 moments, 50 noise
    labels = np.array([0] * 25 + [1] * 25 + [-1] * 50)
    result = check_coverage(labels, threshold=0.80)
    assert result["passed"] is False
