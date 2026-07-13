"""Baseline MONOTONIC follower (issue #115, epic #108): a
continuity-penalized monotonic fitting-DP with an inner transposition
search over symbolic note sequences. Monotonic by construction (not
jump-aware -- see #118) and weight-free/deterministic (no learned
parameters -- see docs/specs/2026-07-12-baseline-monotonic-follower-
design.md for the full DP recurrence and rationale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote

DEFAULT_SKIP_PENALTY = 0.5


@dataclass(frozen=True)
class ContinuityPrior:
    """Cost charged per skipped score note (leading and internal skips are
    both charged; only the trailing, unconsumed tail of the score is
    free). skip_penalty=0.0 disables the prior entirely, letting the DP
    wander to any equally-good-looking coincidental match regardless of
    distance."""
    skip_penalty: float


NO_PRIOR = ContinuityPrior(skip_penalty=0.0)


@dataclass(frozen=True)
class MatchedNote:
    """One perf-note <-> score-note correspondence chosen by follow()."""
    perf_index: int
    score_index: int
    perf_time: float
    score_position: float


@dataclass(frozen=True)
class EstimatedTrajectory:
    """follow()'s output: the winning transpose, the chosen
    correspondence (monotonic non-decreasing in score_index by
    construction), and which perf notes were left unmatched."""
    transpose_semitones: int
    matches: tuple[MatchedNote, ...]
    unmatched_perf_indices: tuple[int, ...]


def _align_at_transpose(
    amt_notes: list[PerfNote], score_notes: list[ScoreNote], prior: ContinuityPrior, transpose: int
) -> EstimatedTrajectory:
    n = len(amt_notes)
    m = len(score_notes)
    neg_inf = -math.inf

    # B[i][j]: best cumulative match score aligning amt_notes[:i] against
    # score_notes[:j]. B[0][0] = 0 (nothing consumed yet); B[0][j] pays
    # prior.skip_penalty per leading score note skipped -- this is what
    # makes an unnecessarily-late first match strictly worse than an
    # early one, which is what prevents the DP from "resetting" its
    # anchor position for free (see Task 3's commit message for the bug
    # this fixes). Trailing skips ARE free: the final answer takes
    # max over j of B[n][j], so unconsumed score notes after the last
    # perf note cost nothing.
    B = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    B[0][0] = 0.0
    for j in range(1, m + 1):
        B[0][j] = B[0][j - 1] - prior.skip_penalty

    for i in range(1, n + 1):
        B[i][0] = B[i - 1][0]  # perf note i unmatched, 0 score notes available
        back[i][0] = ("skip_perf", i - 1, 0)

        for j in range(1, m + 1):
            best_val = neg_inf
            best_move: tuple[str, int, int] | None = None

            # skip score note j (charged prior.skip_penalty everywhere,
            # not just "between matches" -- see the design note above)
            cand = B[i][j - 1] - prior.skip_penalty
            if cand > best_val:
                best_val, best_move = cand, ("skip_score", i, j - 1)

            # leave perf note i unmatched, cost 0
            cand = B[i - 1][j]
            if cand >= best_val:
                best_val, best_move = cand, ("skip_perf", i - 1, j)

            # match perf note i to score note j
            if (score_notes[j - 1].pitch + transpose) == amt_notes[i - 1].pitch:
                cand = B[i - 1][j - 1] + 1.0
                if cand > best_val:
                    best_val, best_move = cand, ("match", i - 1, j - 1)

            B[i][j] = best_val
            back[i][j] = best_move

    # pick the best ending column (trailing skip is free, so we don't
    # require consuming every score note)
    best_j = max(range(m + 1), key=lambda j: B[n][j])

    matches: list[MatchedNote] = []
    unmatched: list[int] = []
    i, j = n, best_j
    while i > 0:
        move = back[i][j]
        if move is None:
            break
        kind, pi, pj = move
        if kind == "match":
            matches.append(
                MatchedNote(
                    perf_index=i - 1,
                    score_index=j - 1,
                    perf_time=amt_notes[i - 1].onset,
                    score_position=score_notes[j - 1].position,
                )
            )
            i, j = pi, pj
        elif kind == "skip_perf":
            unmatched.append(i - 1)
            i, j = pi, pj
        else:  # skip_score
            i, j = pi, pj

    matches.reverse()
    unmatched.reverse()

    return EstimatedTrajectory(
        transpose_semitones=transpose,
        matches=tuple(matches),
        unmatched_perf_indices=tuple(unmatched),
    )


def follow(
    amt_notes: list[PerfNote],
    score_notes: list[ScoreNote],
    prior: ContinuityPrior,
    transpose_candidates: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> EstimatedTrajectory:
    """Align amt_notes against score_notes via a continuity-penalized
    monotonic fitting-DP, searching transpose_candidates for the best
    semitone shift (most matches, ties broken toward transpose 0)."""
    best_result: EstimatedTrajectory | None = None
    best_key: tuple[int, int] | None = None
    for t in transpose_candidates:
        result = _align_at_transpose(amt_notes, score_notes, prior, t)
        key = (len(result.matches), -abs(t))
        if best_key is None or key > best_key:
            best_result, best_key = result, key
    assert best_result is not None
    return best_result


def teleport_gaps(trajectory: EstimatedTrajectory) -> list[float]:
    """Return the score_position delta between each pair of consecutively
    matched notes (always >= 0, since matches are monotonic non-decreasing
    in score_index by construction). A large gap indicates the alignment
    wandered forward to a coincidental match rather than following the
    performance continuously -- see docs/specs/2026-07-12-baseline-
    monotonic-follower-design.md for how this is used for grading."""
    return [
        b.score_position - a.score_position
        for a, b in zip(trajectory.matches, trajectory.matches[1:])
    ]
