"""Verify follow() (the continuity-penalized monotonic fitting-DP with
inner transpose search) through its public interface only, on small
hand-built synthetic examples -- see test_follower_golden_fixture.py for
the real-fixture reproduction test and test_follower_characterization.py
for the required-to-fail pathology tests."""
from __future__ import annotations

from follower_bench.follower import EstimatedTrajectory, MatchedNote, NO_PRIOR, follow, teleport_gaps
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
