# model/tests/follower_bench/test_trajectory.py
"""Verify TrueTrajectory / from_alignment / build_trajectory_from_segments
through their public interface only."""
from __future__ import annotations

import pytest

from follower_bench.trajectory import TrueTrajectory


def test_score_position_at_interpolates_and_clamps() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)))
    assert traj.score_position_at(0.5) == pytest.approx(0.25)
    assert traj.score_position_at(1.5) == pytest.approx(0.75)
    assert traj.score_position_at(-1.0) == pytest.approx(0.0)   # clamp below range
    assert traj.score_position_at(5.0) == pytest.approx(1.0)    # clamp above range
    assert traj.score_position_at(1.0) == pytest.approx(0.5)    # exactly on an anchor
