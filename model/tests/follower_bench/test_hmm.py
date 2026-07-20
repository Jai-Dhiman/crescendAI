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


from follower_bench.hmm import HmmParams, column_posteriors, follow_hmm
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


def test_follow_hmm_forward_jump_relocks_after_a_skipped_passage() -> None:
    # bar1 C4,D4 (60,62); a 20-note filler bar the perf never plays (70..); bar3 G4,A4 (67,69).
    pitches = [60, 62] + list(range(70, 90)) + [67, 69]  # 2 + 20 + 2 = 24 notes
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate(pitches)]
    # downbeats at note-start seconds 0.0 (bar1), 2.0 (filler), 22.0 (bar3) -> columns (0, 2, 22)
    bars = bar_boundary_columns([s.position for s in score], [0.0, 2.0, 22.0])
    assert bars == (0, 2, 22)
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 67, 69])]

    hit = follow_hmm(perf, score, JUMPS, bar_boundaries=bars, transpose_candidates=(0,))
    assert [m.score_index for m in hit.matches] == [0, 1, 22, 23]

    mono = follow_hmm(perf, score, MONO, bar_boundaries=bars, transpose_candidates=(0,))
    assert [m.score_index for m in mono.matches] == [0, 1]
    assert mono.unmatched_perf_indices == (2, 3)


def _rows_sum_to_one(gamma) -> None:
    for row in gamma:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6), sum(row)


def test_column_posteriors_sum_to_one_per_step_monotonic_and_with_jumps() -> None:
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65])]
    bars = bar_boundary_columns([s.position for s in score], [0.0, 2.0])
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64, 65, 60, 62])]
    _rows_sum_to_one(column_posteriors(perf, score, MONO, transpose=0))
    _rows_sum_to_one(column_posteriors(perf, score, JUMPS, transpose=0, bar_boundaries=bars))


from follower_bench.hmm import alignment_logprob


def test_inserting_a_spurious_note_costs_about_log_p_ins() -> None:
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64, 65, 67])]
    base = alignment_logprob(perf, score, MONO, transpose=0)
    # Insert one clearly-spurious note (pitch 7, matches nothing) between notes.
    perf_ins = perf[:3] + [PerfNote(onset=2.5, offset=2.9, pitch=7, velocity=80)] + perf[3:]
    with_ins = alignment_logprob(perf_ins, score, MONO, transpose=0)
    drop = base - with_ins
    assert math.isclose(drop, -math.log(MONO.p_ins), rel_tol=0.05), drop


def test_follow_hmm_attaches_calibrated_confidence_to_matches() -> None:
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64, 65, 67])]
    result = follow_hmm(perf, score, MONO, transpose_candidates=(0,))
    assert len(result.matches) == 5
    for m in result.matches:
        assert m.confidence is not None
        assert 0.0 <= m.confidence <= 1.0
    # a clean, unambiguous run should be decoded confidently
    assert min(m.confidence for m in result.matches) > 0.5


from follower_bench.follower import ContinuityPrior, DEFAULT_SKIP_PENALTY, follow


def test_hmm_crosses_the_repeat_cliff_where_the_shipped_additive_pair_cannot() -> None:
    # 7 two-note bars; boundaries at columns (0,2,4,6,8,10,12). The skipped region
    # is widened to 6 score notes (bars 3-5) so the additive path provably cannot
    # relock bar 6 by margin (skip cost 6*0.5=3.0 > +2.0 match gain, and << jump_fwd=8.0).
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate(pitches)]
    bars = bar_boundary_columns([s.position for s in score], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    assert bars == (0, 2, 4, 6, 8, 10, 12)
    # Perf: play bars 0-2, REPEAT bars 0-2, then SKIP bars 3-5, resume bar 6.
    seq = [60, 62, 64, 65, 67, 69] + [60, 62, 64, 65, 67, 69] + [81, 83]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate(seq)]

    hmm = follow_hmm(perf, score, JUMPS, bar_boundaries=bars, transpose_candidates=(0,))
    hmm_idx = [m.score_index for m in hmm.matches]
    # HMM relocks BOTH (property assertions, not a brittle exact vector):
    # (1) backward relock to bar 0: index 0 reappears after the first pass, i.e. a
    #     backward score-position step exists.
    assert hmm_idx.count(0) >= 2
    assert any(b.score_position < a.score_position
               for a, b in zip(hmm.matches, hmm.matches[1:]))
    # (2) forward relock to bar 6: score indices {12, 13} are decoded.
    assert {12, 13} <= set(hmm_idx)
    # (Exact decoded vector, kept as documentation: [0,1,2,3,4,5,0,1,2,3,4,5,12,13].)

    # Additive follow() at the shipped #118 default (jump_back=5.0, jump_fwd=8.0):
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY, jump_back_penalty=5.0, jump_fwd_penalty=8.0)
    add = follow(perf, score, prior, bar_boundaries=bars, transpose_candidates=(0,))
    add_idx = [m.score_index for m in add.matches]
    # The additive pair misses the forward skip: bar 6 (indices 12, 13) never relocked.
    assert 12 not in add_idx and 13 not in add_idx
    assert set(add.unmatched_perf_indices) >= {12, 13}
