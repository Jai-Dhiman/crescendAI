# model/tests/piece_id_eval/test_open_set.py
"""Verify operating_points and best_point through their public interfaces."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.open_set import OperatingPoint, best_point, operating_points


def _perfect_results() -> tuple[list[float], list[float]]:
    """In-catalog queries score 1.0; out-of-catalog score 0.0."""
    in_scores = [1.0] * 10
    loo_scores = [0.0] * 10
    return in_scores, loo_scores


def _random_results(seed: int = 0) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    in_scores = rng.uniform(0.5, 1.0, 10).tolist()
    loo_scores = rng.uniform(0.0, 0.5, 10).tolist()
    return in_scores, loo_scores


def test_operating_points_returns_one_per_threshold() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.0, 0.5, 1.0]
    pts = operating_points(in_s, loo_s, thresholds)
    assert len(pts) == 3


def test_operating_points_perfect_separation() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.5]
    pts = operating_points(in_s, loo_s, thresholds)
    pt = pts[0]
    assert pt.fa == pytest.approx(0.0)
    assert pt.ta == pytest.approx(1.0)


def test_operating_points_all_accepted_at_zero_threshold() -> None:
    in_s, loo_s = _random_results()
    pts = operating_points(in_s, loo_s, [0.0])
    pt = pts[0]
    assert pt.fa == pytest.approx(1.0)
    assert pt.ta == pytest.approx(1.0)


def test_best_point_returns_none_when_no_point_qualifies() -> None:
    in_s = [0.3] * 5  # all low scores
    loo_s = [0.3] * 5
    pts = operating_points(in_s, loo_s, [0.5])
    # At threshold 0.5, no in-catalog accepted: TA=0 < 0.6
    result = best_point(pts, max_fa=0.05, min_ta=0.60)
    assert result is None


def test_best_point_finds_qualifying_point() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.0, 0.5, 0.9, 1.1]
    pts = operating_points(in_s, loo_s, thresholds)
    result = best_point(pts, max_fa=0.05, min_ta=0.60)
    assert result is not None
    assert result.fa <= 0.05
    assert result.ta >= 0.60


def test_operating_point_is_namedtuple() -> None:
    in_s, loo_s = _perfect_results()
    pts = operating_points(in_s, loo_s, [0.5])
    pt = pts[0]
    assert isinstance(pt, OperatingPoint)
    assert hasattr(pt, "threshold")
    assert hasattr(pt, "fa")
    assert hasattr(pt, "ta")
