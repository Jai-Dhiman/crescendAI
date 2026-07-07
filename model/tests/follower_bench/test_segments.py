# model/tests/follower_bench/test_segments.py
"""Verify the hard-splice engine (apply_segments / apply_note_mutations)
through its public interface only, on synthetic PerfNote lists."""
from __future__ import annotations

import pytest

from follower_bench.segments import PerfNote, Segment, apply_segments


def test_apply_segments_identity_reproduces_input_notes() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.4, pitch=62, velocity=70),
        PerfNote(onset=2.0, offset=2.6, pitch=64, velocity=90),
    ]
    identity = [Segment(src_start=0.0, src_end=3.0, dst_start=0.0, time_scale=1.0)]
    result = apply_segments(notes, identity)
    assert result == notes
