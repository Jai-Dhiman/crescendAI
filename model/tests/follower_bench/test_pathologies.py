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
