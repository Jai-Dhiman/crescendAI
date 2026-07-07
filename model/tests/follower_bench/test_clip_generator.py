# model/tests/follower_bench/test_clip_generator.py
"""End-to-end verification of generate() through its public interface
only, on real ASAP fixtures."""
from __future__ import annotations

import pytest

from follower_bench.asap_alignment import AsapAlignmentMissingError, load_alignment
from follower_bench.clip_generator import generate
from follower_bench.trajectory import DISCONTINUITY_EPS_S

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"
UNALIGNED_PIECE = "Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid"


def test_generate_clean_is_monotonic_and_matches_asap_exactly() -> None:
    clip = generate(ALIGNED_PIECE, "clean", seed=1)
    alignment = load_alignment(ALIGNED_PIECE)

    assert clip.asap_piece == ALIGNED_PIECE
    assert clip.pathology_type == "clean"
    assert clip.seed == 1
    assert len(clip.notes) > 0
    assert clip.event_labels == ()
    assert clip.true_trajectory.is_monotonic_non_decreasing() is True

    for perf_beat, score_beat in zip(alignment.performance_beats, alignment.midi_score_beats):
        assert clip.true_trajectory.score_position_at(perf_beat) == pytest.approx(score_beat)


def test_generate_repeat_is_deterministic_and_has_the_back_jump() -> None:
    clip_a = generate(ALIGNED_PIECE, "repeat", seed=7)
    clip_b = generate(ALIGNED_PIECE, "repeat", seed=7)

    assert clip_a.notes == clip_b.notes
    assert clip_a.true_trajectory.anchors == clip_b.true_trajectory.anchors
    assert clip_a.event_labels == clip_b.event_labels

    assert len(clip_a.event_labels) == 1
    event = clip_a.event_labels[0]
    assert event.type == "repeat"

    before = clip_a.true_trajectory.score_position_at(event.perf_time)
    after = clip_a.true_trajectory.score_position_at(event.perf_time + DISCONTINUITY_EPS_S)
    assert before == pytest.approx(event.from_score_position)
    assert after == pytest.approx(event.to_score_position)
    assert after < before


def test_generate_propagates_missing_alignment_for_real_unaligned_piece() -> None:
    with pytest.raises(AsapAlignmentMissingError):
        generate(UNALIGNED_PIECE, "clean", seed=1)
