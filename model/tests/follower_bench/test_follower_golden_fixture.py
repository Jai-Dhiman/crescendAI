# model/tests/follower_bench/test_follower_golden_fixture.py
"""Reproduces the day-0 spike's recovered acceptance numbers (issue #115,
epic #108) on the real bach_inv1_chunk0 fixture. The exact historical
counts (62/82 matches, 3 teleports without the prior, max 6.9s) come from
a lost implementation and are not independently re-derivable -- see
docs/specs/2026-07-12-baseline-monotonic-follower-design.md's Open
Questions for the tolerance rationale. The structural claims (transpose
auto-detection, zero teleports WITH the prior, at least one teleport
WITHOUT it) are asserted exactly; the match count is asserted within a
tolerance band."""
from __future__ import annotations

from pathlib import Path

from follower_bench.follower import (
    DEFAULT_SKIP_PENALTY,
    NO_PRIOR,
    ContinuityPrior,
    follow,
    teleport_gaps,
)
from follower_bench.score_notes import load_golden_fixture_notes

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_FIXTURE_PATH = (
    REPO_ROOT / "apps/api/src/wasm/score-analysis/tests/fixtures/bach_inv1_chunk0.json"
)

# Grading constants (test-local, not part of follow()'s public API --
# see spec Open Questions). Verified empirically against this exact real
# fixture during planning (issue #115): at DEFAULT_SKIP_PENALTY (0.5),
# follow() returns transpose_semitones=-1, exactly 62 matches, 0 gaps
# over TELEPORT_THRESHOLD_S -- an exact reproduction of the day-0 spike's
# "62/82, zero teleports" result, not merely a tolerance-band match.
TELEPORT_THRESHOLD_S = 2.0
EXPECTED_MATCH_COUNT = 62


def test_follow_reproduces_day0_spike_on_golden_fixture() -> None:
    perf_notes, score_notes = load_golden_fixture_notes(GOLDEN_FIXTURE_PATH)

    result = follow(perf_notes, score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))

    assert abs(result.transpose_semitones) == 1
    assert len(result.matches) == EXPECTED_MATCH_COUNT

    gaps = teleport_gaps(result)
    teleport_count = sum(1 for g in gaps if g > TELEPORT_THRESHOLD_S)
    assert teleport_count == 0


def test_no_prior_regresses_to_multiple_teleports_on_golden_fixture() -> None:
    perf_notes, score_notes = load_golden_fixture_notes(GOLDEN_FIXTURE_PATH)

    result = follow(perf_notes, score_notes, NO_PRIOR)

    gaps = teleport_gaps(result)
    teleport_count = sum(1 for g in gaps if g > TELEPORT_THRESHOLD_S)
    assert teleport_count >= 1
    assert max(gaps, default=0.0) > 5.0  # same order of magnitude as the day-0 spike's 6.9s max, not exact
