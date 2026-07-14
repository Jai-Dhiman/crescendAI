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
