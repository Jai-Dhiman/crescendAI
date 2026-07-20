"""Behavioral tests for the opt-in Viterbi-HMM follower (issue #119)."""
from __future__ import annotations

import math

from follower_bench.follower import MatchedNote


def test_matched_note_confidence_defaults_none_and_accepts_a_value() -> None:
    # Existing positional construction (no confidence) is unchanged.
    m = MatchedNote(perf_index=0, score_index=0, perf_time=0.0, score_position=0.0)
    assert m.confidence is None
    # New: confidence can be supplied.
    m2 = MatchedNote(perf_index=1, score_index=2, perf_time=1.0, score_position=2.0, confidence=0.75)
    assert m2.confidence == 0.75


from follower_bench.hmm import HmmParams, follow_hmm
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote

MONO = HmmParams()  # p_jump_back = p_jump_fwd = 0.0 -> monotonic


def test_follow_hmm_monotonic_matches_in_order_and_skips_unmatchable() -> None:
    # Score C4 D4 E4 F4 G4 at positions 0..4.
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    # Perf: C4, D4, an unmatchable note (99), then F4.
    perf = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.5, pitch=62, velocity=80),
        PerfNote(onset=2.0, offset=2.5, pitch=99, velocity=80),
        PerfNote(onset=3.0, offset=3.5, pitch=65, velocity=80),
    ]
    result = follow_hmm(perf, score, MONO, transpose_candidates=(0,))
    assert result.transpose_semitones == 0
    assert [m.score_index for m in result.matches] == [0, 1, 3]
    assert result.unmatched_perf_indices == (2,)
    # monotonic: score indices non-decreasing
    idx = [m.score_index for m in result.matches]
    assert idx == sorted(idx)
