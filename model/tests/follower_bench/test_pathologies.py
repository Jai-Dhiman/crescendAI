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
