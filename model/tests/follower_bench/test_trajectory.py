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


def test_is_monotonic_non_decreasing_true_for_ascending() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 1.0)))
    assert traj.is_monotonic_non_decreasing() is True


def test_is_monotonic_non_decreasing_false_for_a_regression() -> None:
    traj = TrueTrajectory(anchors=((0.0, 0.0), (1.0, 1.5), (2.0, 0.5), (3.0, 1.0)))
    assert traj.is_monotonic_non_decreasing() is False


from follower_bench.asap_alignment import load_alignment
from follower_bench.trajectory import from_alignment

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_from_alignment_matches_real_asap_beat_arrays_exactly() -> None:
    alignment = load_alignment(ALIGNED_PIECE)
    traj = from_alignment(alignment)
    assert len(traj.anchors) == len(alignment.performance_beats) == 92
    assert traj.anchors[0] == (alignment.performance_beats[0], alignment.midi_score_beats[0])
    assert traj.anchors[-1] == (alignment.performance_beats[-1], alignment.midi_score_beats[-1])
    assert traj.is_monotonic_non_decreasing() is True
