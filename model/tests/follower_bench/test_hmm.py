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


from follower_bench.follower import bar_boundary_columns

JUMPS = HmmParams(p_jump_back=0.02, p_jump_fwd=0.01)


def test_follow_hmm_backward_jump_relocks_after_a_repeat() -> None:
    # bar1 = C4,D4 (60@0,62@1); bar2 = E4,F4 (64@2,65@3). Bars start at 0.0, 2.0.
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65])]
    bars = bar_boundary_columns([s.position for s in score], [0.0, 2.0])  # (0, 2)
    # Perf: bar1, bar2, then REPEAT bar1.
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64, 65, 60, 62])]

    hit = follow_hmm(perf, score, JUMPS, bar_boundaries=bars, transpose_candidates=(0,))
    assert [m.score_index for m in hit.matches] == [0, 1, 2, 3, 0, 1]
    # a backward step in score position exists (the relock jump)
    assert any(b.score_position < a.score_position
               for a, b in zip(hit.matches, hit.matches[1:]))

    # Jumps off -> the replayed notes cannot relock (stay monotonic).
    mono = follow_hmm(perf, score, MONO, bar_boundaries=bars, transpose_candidates=(0,))
    idx = [m.score_index for m in mono.matches]
    assert idx == sorted(idx)
