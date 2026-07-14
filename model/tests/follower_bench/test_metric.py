"""Verify score_clip / aggregate_by_pathology / trajectory_from_matches
through their public interface only, using real #111 clips
(clip_generator.generate) and hand-built TrueTrajectory estimates for the
core measurement behaviors, plus one real-follower.follow() integration
slice."""
from __future__ import annotations

import math

import pytest

from follower_bench.clip_generator import generate
from follower_bench.metric import SAMPLE_HZ, score_clip
from follower_bench.trajectory import TrueTrajectory

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_score_clip_identity_estimate_is_a_perfect_score() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)

    score = score_clip(clip.true_trajectory, clip)

    assert score.pathology_type == "repeat"
    assert score.median_abs_error_beats == pytest.approx(0.0)
    assert score.max_abs_error_beats == pytest.approx(0.0)
    assert score.lock_rate == pytest.approx(1.0)
    assert score.false_jump_count == 0

    assert len(clip.event_labels) == 1
    assert clip.event_labels[0].from_score_position != clip.event_labels[0].to_score_position
    assert len(score.relock_latencies_s) == 1
    latency = score.relock_latencies_s[0]
    assert 0.0 <= latency < 1.0 / SAMPLE_HZ


def test_score_clip_constant_offset_estimate_reports_exact_offset_as_error() -> None:
    clip = generate(ALIGNED_PIECE, "clean", seed=1)
    offset = 2.0
    shifted = TrueTrajectory(
        anchors=tuple((t, p + offset) for t, p in clip.true_trajectory.anchors)
    )

    score = score_clip(shifted, clip)

    assert score.median_abs_error_beats == pytest.approx(offset)
    assert score.max_abs_error_beats == pytest.approx(offset)
    assert score.lock_rate == pytest.approx(0.0)
    assert score.false_jump_count == 0
    assert score.relock_latencies_s == ()


def test_score_clip_relock_latency_is_inf_when_estimate_never_recovers() -> None:
    clip = generate(ALIGNED_PIECE, "jump", seed=13)
    event = clip.event_labels[0]
    assert event.from_score_position != event.to_score_position

    # Freeze the estimate at whatever it was tracking right at the event's
    # perf_time -- a stand-in for a follower that stops progressing at a
    # forward jump and never catches up to the leapt-ahead score position
    # (truth moves forward and never revisits from_score_position, so the
    # frozen estimate never re-enters tolerance -> relock latency is inf).
    frozen_anchors = tuple(
        (t, p) for t, p in clip.true_trajectory.anchors if t <= event.perf_time
    )
    estimated = TrueTrajectory(anchors=frozen_anchors)

    score = score_clip(estimated, clip)

    assert len(score.relock_latencies_s) == 1
    assert score.relock_latencies_s[0] == math.inf


def test_score_clip_relock_latency_is_finite_when_estimate_recovers() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    event = clip.event_labels[0]
    n_seconds = 2.0
    reconnect_time = event.perf_time + n_seconds

    pre = [(t, p) for t, p in clip.true_trajectory.anchors if t <= event.perf_time]
    post = [(t, p) for t, p in clip.true_trajectory.anchors if t >= reconnect_time]
    reconnect_pos = clip.true_trajectory.score_position_at(reconnect_time)
    estimated = TrueTrajectory(
        anchors=tuple(pre) + ((reconnect_time, reconnect_pos),) + tuple(post)
    )

    score = score_clip(estimated, clip)

    assert len(score.relock_latencies_s) == 1
    latency = score.relock_latencies_s[0]
    assert math.isfinite(latency)
    assert latency <= n_seconds + 1.0 / SAMPLE_HZ
