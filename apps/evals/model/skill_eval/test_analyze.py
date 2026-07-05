"""Tests for skill_eval.analyze -- currently just Stuart's tau-c (MIREX 2026 #104).

tau-c is kept a small isolated function so the exact variant can be swapped if the
MIREX organizers publish a different official scorer (the tau-c *formula* is not yet
public -- see docs/competitions/mirex-difficulty.md still-open item 1).
"""
from __future__ import annotations

import pytest
from scipy import stats

from model.skill_eval.analyze import kendall_tau_c


def test_perfect_order_is_plus_one():
    x = [1, 2, 3, 4, 5]
    assert kendall_tau_c(x, x) == pytest.approx(1.0, abs=1e-12)


def test_reverse_order_is_minus_one():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert kendall_tau_c(x, y) == pytest.approx(-1.0, abs=1e-12)


def test_square_fractional_hand_value():
    # x=[1,2,3,4], y=[1,2,4,3]: one discordant pair (3,4) of 6 -> C=5,D=1, m=4,n=4
    # tau_c = 2*4*(5-1)/(4^2*3) = 32/48 = 2/3
    assert kendall_tau_c([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(2 / 3, abs=1e-12)


def test_rectangular_hand_value():
    # x=[1,2,3,4,4] (4 distinct), y=[1,2,3,5,4] (5 distinct): C=9, D=0, one tie
    # m=min(4,5)=4, n=5 -> tau_c = 2*4*9 / (25*3) = 72/75 = 0.96
    assert kendall_tau_c([1, 2, 3, 4, 4], [1, 2, 3, 5, 4]) == pytest.approx(0.96, abs=1e-12)


def test_rectangular_ties_reach_one_unlike_tau_b():
    # The whole reason to use tau-c: a perfect ordering on a rectangular table with
    # ties still reaches 1.0, whereas scipy's tau-b is capped below 1.0.
    x = [1, 1, 2, 2]
    y = [1, 2, 3, 4]
    assert kendall_tau_c(x, y) == pytest.approx(1.0, abs=1e-12)
    assert stats.kendalltau(x, y).statistic < 1.0  # tau-b is penalized by the ties


def test_matches_tau_b_when_no_ties_and_square():
    # With no ties and equal category counts, tau-c reduces to tau-a == scipy tau-b.
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 4, 3, 5]
    assert kendall_tau_c(x, y) == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-12)


def test_constant_input_raises():
    with pytest.raises(ValueError):
        kendall_tau_c([1, 1, 1, 1], [1, 2, 3, 4])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        kendall_tau_c([1, 2, 3], [1, 2])
