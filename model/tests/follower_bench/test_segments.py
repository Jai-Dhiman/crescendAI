# model/tests/follower_bench/test_segments.py
"""Verify the hard-splice engine (apply_segments / apply_note_mutations)
through its public interface only, on synthetic PerfNote lists."""
from __future__ import annotations

import pytest

from follower_bench.segments import (
    NoteMutation,
    PerfNote,
    Segment,
    apply_note_mutations,
    apply_segments,
)


def test_apply_segments_identity_reproduces_input_notes() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.4, pitch=62, velocity=70),
        PerfNote(onset=2.0, offset=2.6, pitch=64, velocity=90),
    ]
    identity = [Segment(src_start=0.0, src_end=3.0, dst_start=0.0, time_scale=1.0)]
    result = apply_segments(notes, identity)
    assert result == notes


def test_apply_segments_repeat_splice_duplicates_and_shifts_notes() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.4, pitch=60, velocity=80),   # before X
        PerfNote(onset=1.0, offset=1.4, pitch=62, velocity=80),   # inside [X=1, Y=2)
        PerfNote(onset=1.6, offset=1.9, pitch=64, velocity=80),   # inside [X=1, Y=2)
        PerfNote(onset=2.5, offset=2.9, pitch=65, velocity=80),   # after Y, the tail
    ]
    x, y, t_min, t_max = 1.0, 2.0, 0.0, 3.0
    seg1 = Segment(t_min, y, t_min, 1.0)
    seg2 = Segment(x, y, seg1.dst_end, 1.0)
    seg3 = Segment(y, t_max, seg2.dst_end, 1.0)
    result = apply_segments(notes, [seg1, seg2, seg3])

    onset_pitch_pairs = [(round(n.onset, 6), n.pitch) for n in result]
    assert len(result) == 6  # 4 original notes + 2 duplicated from the repeated span [1.0, 2.0)
    assert (0.0, 60) in onset_pitch_pairs   # seg1: before X, unchanged
    assert (1.0, 62) in onset_pitch_pairs   # seg1: first pass through X..Y
    assert (1.6, 64) in onset_pitch_pairs   # seg1: first pass through X..Y
    assert (2.0, 62) in onset_pitch_pairs   # seg2: repeat pass, X replayed at seg2.dst_start=2.0
    assert (2.6, 64) in onset_pitch_pairs   # seg2: repeat pass
    assert (3.5, 65) in onset_pitch_pairs   # seg3: tail note shifted forward by the repeat's duration


def test_apply_note_mutations_shifts_nearest_note_pitch_clamped() -> None:
    notes = [
        PerfNote(onset=0.0, offset=0.4, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.4, pitch=126, velocity=80),
        PerfNote(onset=2.0, offset=2.4, pitch=64, velocity=80),
    ]
    mutations = [NoteMutation(target_onset=1.05, pitch_delta=5)]
    result = apply_note_mutations(notes, mutations)

    assert result[0].pitch == 60
    assert result[1].pitch == 127  # 126 + 5 clamped to 127
    assert result[2].pitch == 64
    assert result[1].onset == 1.0 and result[1].offset == 1.4 and result[1].velocity == 80
