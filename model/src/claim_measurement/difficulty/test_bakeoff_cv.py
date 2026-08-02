"""Tests for bakeoff_cv (ported from the unmerged issue-104-mirex-difficulty
branch's phase5b_aria_probe.py, commit 7976b5e6 -- see the design spec for why
this is a port, not a cross-branch import).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import math

from claim_measurement.difficulty.bakeoff_cv import tau_c


def test_tau_c_perfect_agreement_is_one():
    assert tau_c([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_tau_c_perfect_disagreement_is_minus_one():
    assert tau_c([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_tau_c_none_for_constant_y():
    assert tau_c([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_tau_c_none_for_fewer_than_three_points():
    assert tau_c([1, 2], [1, 2]) is None


def test_tau_c_handles_ties_without_raising():
    result = tau_c([1, 1, 2, 3], [1, 2, 2, 3])
    assert result is not None
    assert not math.isnan(result)
