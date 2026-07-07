# model/tests/follower_bench/test_pathologies.py
"""Verify build_plan's per-pathology-type Segment/PathologyEvent
construction through its public interface only, on a real ClipAlignment."""
from __future__ import annotations

import random

import pytest

from follower_bench.asap_alignment import load_alignment
from follower_bench.pathologies import PATHOLOGY_TYPES, build_plan

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def _alignment():
    return load_alignment(ALIGNED_PIECE)


def test_build_plan_clean_is_one_identity_segment_no_events() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "clean", random.Random(0))
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.src_start == pytest.approx(t_min)
    assert seg.src_end == pytest.approx(t_max)
    assert seg.dst_start == pytest.approx(t_min)
    assert seg.time_scale == pytest.approx(1.0)
    assert plan.events == ()
    assert plan.note_mutations == ()


from follower_bench.trajectory import from_alignment


def test_build_plan_repeat_describes_the_back_jump() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "repeat", random.Random(0))
    clean_traj = from_alignment(alignment)

    assert len(plan.segments) == 3
    seg1, seg2, seg3 = plan.segments
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start < seg1.src_end
    assert seg2.src_end == pytest.approx(seg1.src_end)
    assert seg2.dst_start == pytest.approx(seg1.dst_end)
    assert seg3.src_start == pytest.approx(seg1.src_end)
    assert seg3.src_end == pytest.approx(t_max)

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "repeat"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position < event.from_score_position


def test_build_plan_jump_describes_the_forward_skip() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "jump", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start > seg1.src_end
    assert seg2.dst_start == pytest.approx(seg1.dst_end)
    assert seg2.src_end == pytest.approx(t_max)

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "jump"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position > event.from_score_position


def test_build_plan_restart_jumps_back_to_an_earlier_point() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "restart", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg1.dst_start == pytest.approx(t_min)
    assert seg2.src_start < seg1.src_end
    assert seg2.dst_start == pytest.approx(seg1.dst_end)
    assert seg2.src_end == pytest.approx(t_max)

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "restart"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))
    assert event.to_score_position == pytest.approx(clean_traj.score_position_at(seg2.src_start))
    assert event.to_score_position < event.from_score_position


from follower_bench.trajectory import build_trajectory_from_segments


def test_build_plan_hesitation_inserts_a_same_position_pause() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "hesitation", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 2
    seg1, seg2 = plan.segments
    assert seg1.src_start == pytest.approx(t_min)
    assert seg2.src_start == pytest.approx(seg1.src_end)
    assert seg2.src_end == pytest.approx(t_max)
    pause = seg2.dst_start - seg1.dst_end
    assert 1.0 <= pause <= 3.0

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "hesitation"
    assert event.perf_time == pytest.approx(seg1.dst_end)
    assert event.from_score_position == pytest.approx(event.to_score_position)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(seg1.src_end))

    # WATCH-ITEM (challenge review): the ground-truth trajectory must stay
    # FLAT at the paused score position for the whole pause, not merely at
    # its boundaries. Sample the MIDDLE of the destination-time pause gap.
    spliced = build_trajectory_from_segments(clean_traj, list(plan.segments))
    paused_pos = clean_traj.score_position_at(seg1.src_end)
    mid_pause_t = (seg1.dst_end + seg2.dst_start) / 2.0
    assert spliced.score_position_at(mid_pause_t) == pytest.approx(paused_pos)
    assert spliced.score_position_at(seg1.dst_end) == pytest.approx(paused_pos)
    assert spliced.score_position_at(seg2.dst_start) == pytest.approx(paused_pos)


def test_build_plan_wrong_note_is_a_pitch_mutation_with_no_timeline_change() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "wrong_note", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.src_start == pytest.approx(t_min)
    assert seg.src_end == pytest.approx(t_max)
    assert seg.time_scale == pytest.approx(1.0)

    assert len(plan.note_mutations) == 1
    mutation = plan.note_mutations[0]
    assert t_min <= mutation.target_onset <= t_max
    assert mutation.pitch_delta != 0

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "wrong_note"
    assert event.from_score_position == pytest.approx(event.to_score_position)
    assert event.from_score_position == pytest.approx(clean_traj.score_position_at(mutation.target_onset))


def test_build_plan_tempo_swing_is_a_contiguous_piecewise_time_ramp() -> None:
    alignment = _alignment()
    plan = build_plan(alignment, "tempo_swing", random.Random(0))
    clean_traj = from_alignment(alignment)
    t_min, t_max = alignment.performance_beats[0], alignment.performance_beats[-1]

    assert len(plan.segments) >= 4
    segs = plan.segments
    assert segs[0].src_start == pytest.approx(t_min)
    assert segs[0].time_scale == pytest.approx(1.0)
    assert segs[-1].src_end == pytest.approx(t_max)
    assert segs[-1].time_scale == pytest.approx(1.0)

    for prev, curr in zip(segs, segs[1:]):
        assert curr.src_start == pytest.approx(prev.src_end)
        assert curr.dst_start == pytest.approx(prev.dst_end)

    assert any(s.time_scale != pytest.approx(1.0) for s in segs[1:-1])

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.type == "tempo_swing"
    assert event.from_score_position == pytest.approx(event.to_score_position)


def test_build_plan_rejects_unknown_pathology_type() -> None:
    alignment = _alignment()
    with pytest.raises(ValueError, match="Unknown pathology_type"):
        build_plan(alignment, "does_not_exist", random.Random(0))


from follower_bench.asap_alignment import ClipAlignment


def test_build_plan_rejects_zero_duration_alignment() -> None:
    degenerate = ClipAlignment(
        asap_piece="fake/degenerate.mid",
        performance_midi_path=_alignment().performance_midi_path,
        score_midi_path=_alignment().score_midi_path,
        performance_beats=(1.0, 1.0, 1.0, 1.0),
        midi_score_beats=(0.0, 0.5, 1.0, 1.5),
    )
    with pytest.raises(ValueError, match="zero-duration"):
        build_plan(degenerate, "repeat", random.Random(0))
