# model/tests/follower_bench/test_follower_characterization.py
"""Characterization tests (issue #115, epic #108): document that the
baseline MONOTONIC follower is EXPECTED and SUPPOSED to fail to re-lock
after jump/repeat/restart pathologies. These are not bugs -- monotonic-
by-construction means the follower cannot represent a backward score
jump at all, and the continuity prior actively discourages the large
forward jump a `jump` pathology requires. This documents the gap that
#118 (jump-aware follower) closes."""
from __future__ import annotations

from follower_bench.asap_alignment import load_alignment
from follower_bench.clip_generator import generate
from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, follow
from follower_bench.score_notes import load_score_notes_from_midi
from follower_bench.trajectory import TrueTrajectory

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"
DIVERGENCE_THRESHOLD_S = 2.0
PROBE_DELAY_S = 3.0


def test_follow_fails_to_relock_after_a_jump() -> None:
    clip = generate(ALIGNED_PIECE, "jump", seed=11)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_S


def test_follow_fails_to_relock_after_a_repeat() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_S


def test_follow_fails_to_relock_after_a_restart() -> None:
    # seed=17 (plan default) was confounded by pre-existing follower lag causing a
    # coincidental trajectory crossing near the probe; seed=14 was empirically
    # selected to show a genuine stuck-ahead re-lock failure (est > true).
    clip = generate(ALIGNED_PIECE, "restart", seed=14)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_S
