"""Verify follow() (the continuity-penalized monotonic fitting-DP with
inner transpose search) through its public interface only, on small
hand-built synthetic examples -- see test_follower_golden_fixture.py for
the real-fixture reproduction test and test_follower_characterization.py
for the required-to-fail pathology tests."""
from __future__ import annotations

from follower_bench.follower import ContinuityPrior, EstimatedTrajectory, MatchedNote, NO_PRIOR, bar_boundary_columns, follow, teleport_gaps
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote


def test_follow_matches_notes_in_score_order_and_skips_unmatchable_notes() -> None:
    # Score: C4 D4 E4 F4 G4 (pitches 60,62,64,65,67), positions 0..4
    score_notes = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    # Perf: C4, D4, then a wrong/unmatchable note (pitch 99), then F4
    perf_notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.5, pitch=62, velocity=80),
        PerfNote(onset=2.0, offset=2.5, pitch=99, velocity=80),
        PerfNote(onset=3.0, offset=3.5, pitch=65, velocity=80),
    ]

    result = follow(perf_notes, score_notes, NO_PRIOR, transpose_candidates=(0,))

    assert result.transpose_semitones == 0
    assert result.unmatched_perf_indices == (2,)
    assert [m.perf_index for m in result.matches] == [0, 1, 3]
    assert [m.score_index for m in result.matches] == [0, 1, 3]
    assert [m.score_position for m in result.matches] == [0.0, 1.0, 3.0]
    # monotonic non-decreasing in score_index by construction
    score_indices = [m.score_index for m in result.matches]
    assert score_indices == sorted(score_indices)


def test_teleport_gaps_returns_consecutive_match_position_deltas() -> None:
    trajectory = EstimatedTrajectory(
        transpose_semitones=0,
        matches=(
            MatchedNote(perf_index=0, score_index=0, perf_time=0.0, score_position=0.0),
            MatchedNote(perf_index=1, score_index=2, perf_time=1.0, score_position=2.0),
            MatchedNote(perf_index=2, score_index=9, perf_time=2.0, score_position=20.0),
        ),
        unmatched_perf_indices=(),
    )

    gaps = teleport_gaps(trajectory)

    assert gaps == [2.0, 18.0]


def test_continuity_prior_refuses_a_teleport_that_would_unlock_more_matches() -> None:
    # Score: idx0 pitch60@0 (true match for perf0), idx1 pitch62@1 (distractor),
    # idx2 pitch60@2 (the correct, NEARBY match for perf1 -- but nothing useful
    # follows it), idx3-9 filler pitches perf2/perf3 can't match, idx10
    # pitch60@10 (a coincidental FAR match for perf1), idx11 pitch61@11 and
    # idx12 pitch63@12 (the true continuation, immediately after idx10 -- only
    # reachable by taking the far match first).
    score_pitches = [60, 62, 60, 70, 71, 72, 73, 74, 75, 76, 60, 61, 63]
    score_notes = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate(score_pitches)]
    perf_notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.5, pitch=60, velocity=80),
        PerfNote(onset=2.0, offset=2.5, pitch=61, velocity=80),
        PerfNote(onset=3.0, offset=3.5, pitch=63, velocity=80),
    ]

    no_prior_result = follow(perf_notes, score_notes, NO_PRIOR, transpose_candidates=(0,))
    no_prior_gaps = teleport_gaps(no_prior_result)

    prior = ContinuityPrior(skip_penalty=0.5)
    with_prior_result = follow(perf_notes, score_notes, prior, transpose_candidates=(0,))
    with_prior_gaps = teleport_gaps(with_prior_result)

    # Without the prior: the DP takes the far trade (4 matches total,
    # jumping from score_index 2 to score_index 11 -- a gap of 9.0).
    assert len(no_prior_result.matches) == 4
    assert max(no_prior_gaps) == 9.0

    # With the prior: the far trade costs more (8 skipped notes * 0.5 =
    # 4.0) than the 2 extra matches it would unlock (+2.0), so the DP
    # stops after the correct local match instead.
    assert len(with_prior_result.matches) == 2
    assert max(with_prior_gaps) == 2.0


def test_follow_auto_detects_a_semitone_transpose() -> None:
    # Score: C4 D4 E4 F4 G4 (60,62,64,65,67) at positions 0..4.
    score_notes = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    # Perf: every note is score pitch + 1 semitone (transpose = +1).
    perf_notes = [
        PerfNote(onset=float(i), offset=float(i) + 0.5, pitch=p + 1, velocity=80)
        for i, p in enumerate([60, 62, 64, 65, 67])
    ]

    result = follow(perf_notes, score_notes, ContinuityPrior(skip_penalty=0.5))

    assert result.transpose_semitones == 1
    assert len(result.matches) == 5


def test_bar_boundary_columns_maps_downbeats_to_note_start_columns() -> None:
    # Notes at score-seconds 0,1,2,3; downbeats at 0.0 (bar1) and 2.0 (bar2).
    # Column = count of notes strictly before the downbeat.
    positions = [0.0, 1.0, 2.0, 3.0]
    assert bar_boundary_columns(positions, [0.0, 2.0]) == (0, 2)


def test_bar_boundary_columns_downbeat_between_notes_and_dedups() -> None:
    # Downbeat at 2.0 falls between notes at 1.5 and 3.0 -> column 2.
    # Duplicate/again-zero downbeats collapse to a sorted unique tuple.
    positions = [0.0, 1.5, 3.0]
    assert bar_boundary_columns(positions, [0.0, 2.0, 2.0]) == (0, 2)
