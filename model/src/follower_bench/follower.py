"""Baseline MONOTONIC follower (issue #115, epic #108): a
continuity-penalized monotonic fitting-DP with an inner transposition
search over symbolic note sequences. Monotonic by construction (not
jump-aware -- see #118) and weight-free/deterministic (no learned
parameters -- see docs/specs/2026-07-12-baseline-monotonic-follower-
design.md for the full DP recurrence and rationale).
"""
from __future__ import annotations

import bisect
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
    distance. jump_back_penalty / jump_fwd_penalty are the fixed cost of a
    bar-boundary score-pointer relocation (backward = repeat/restart,
    forward = skip); both default to math.inf, which disables jumps and
    reproduces the monotonic baseline (#115) exactly. See #118."""
    skip_penalty: float
    jump_back_penalty: float = math.inf
    jump_fwd_penalty: float = math.inf


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
    correspondence (monotonic non-decreasing in score_index unless a
    bar-boundary jump was taken (#118)), and which perf notes were left
    unmatched."""
    transpose_semitones: int
    matches: tuple[MatchedNote, ...]
    unmatched_perf_indices: tuple[int, ...]


def _align_at_transpose(
    amt_notes: list[PerfNote],
    score_notes: list[ScoreNote],
    prior: ContinuityPrior,
    transpose: int,
    bar_boundaries: tuple[int, ...] | None = None,
) -> EstimatedTrajectory:
    n = len(amt_notes)
    m = len(score_notes)
    neg_inf = -math.inf

    B = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    B[0][0] = 0.0
    for j in range(1, m + 1):
        B[0][j] = B[0][j - 1] - prior.skip_penalty

    jumps_enabled = bool(bar_boundaries) and (
        prior.jump_back_penalty < math.inf or prior.jump_fwd_penalty < math.inf
    )

    for i in range(1, n + 1):
        B[i][0] = B[i - 1][0]  # perf note i unmatched, 0 score notes available
        back[i][0] = ("skip_perf", i - 1, 0)

        for j in range(1, m + 1):
            best_val = neg_inf
            best_move: tuple[str, int, int] | None = None

            cand = B[i][j - 1] - prior.skip_penalty
            if cand > best_val:
                best_val, best_move = cand, ("skip_score", i, j - 1)

            cand = B[i - 1][j]
            # >= (not >) intentionally biases ties toward skip_perf over skip_score.
            if cand >= best_val:
                best_val, best_move = cand, ("skip_perf", i - 1, j)

            if (score_notes[j - 1].pitch + transpose) == amt_notes[i - 1].pitch:
                cand = B[i - 1][j - 1] + 1.0
                if cand > best_val:
                    best_val, best_move = cand, ("match", i - 1, j - 1)

            B[i][j] = best_val
            back[i][j] = best_move

        if jumps_enabled:
            _relax_row_jumps(B[i], back[i], i, m, bar_boundaries, prior)

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
        elif kind == "jump":
            # same-row pointer relocation: no perf note consumed, no match.
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


def _relax_row_jumps(row, row_back, i, m, bar_boundaries, prior) -> None:
    """Apply at most the single best bar-boundary score-pointer jump to
    row i of the DP (a same-row relocation that consumes no perf note),
    then re-propagate skip_score forward from the jumped cell. Backward
    (source after target = repeat/restart) and forward (source before
    target = skip) branches carry independent penalties. At most one jump
    per row keeps traceback acyclic."""
    neg_inf = -math.inf

    # prefix max + argmax over row[0..j]
    pref_val = [neg_inf] * (m + 1)
    pref_arg = [0] * (m + 1)
    best, best_j = neg_inf, 0
    for j in range(m + 1):
        if row[j] > best:
            best, best_j = row[j], j
        pref_val[j], pref_arg[j] = best, best_j

    # suffix max + argmax over row[j..m]
    suf_val = [neg_inf] * (m + 2)
    suf_arg = [0] * (m + 2)
    best, best_j = neg_inf, m
    for j in range(m, -1, -1):
        if row[j] > best:
            best, best_j = row[j], j
        suf_val[j], suf_arg[j] = best, best_j

    best_cand, best_jb, best_src = neg_inf, None, None
    for jb in bar_boundaries:
        if not (0 <= jb <= m):
            continue
        cand, src = neg_inf, None
        if jb - 1 >= 0 and prior.jump_fwd_penalty < math.inf:
            c = pref_val[jb - 1] - prior.jump_fwd_penalty
            if c > cand:
                cand, src = c, pref_arg[jb - 1]
        if jb + 1 <= m and prior.jump_back_penalty < math.inf:
            c = suf_val[jb + 1] - prior.jump_back_penalty
            if c > cand:
                cand, src = c, suf_arg[jb + 1]
        if cand > row[jb] and cand > best_cand:
            best_cand, best_jb, best_src = cand, jb, src

    if best_jb is None:
        return
    row[best_jb] = best_cand
    row_back[best_jb] = ("jump", i, best_src)
    for j in range(best_jb + 1, m + 1):
        c = row[j - 1] - prior.skip_penalty
        if c > row[j]:
            row[j] = c
            row_back[j] = ("skip_score", i, j - 1)


def bar_boundary_columns(positions: list[float], downbeats) -> tuple[int, ...]:
    """Map bar-downbeat times (score-MIDI seconds) to score-note DP
    columns: each column is the count of notes with position strictly
    before the downbeat (i.e. the pointer sits at the start of that bar).
    `positions` must be sorted ascending (as load_score_notes_from_midi
    returns). Returns a sorted, de-duplicated tuple -- these are the only
    columns follow() may jump to."""
    cols = {bisect.bisect_left(positions, float(d)) for d in downbeats}
    return tuple(sorted(cols))


def follow(
    amt_notes: list[PerfNote],
    score_notes: list[ScoreNote],
    prior: ContinuityPrior,
    bar_boundaries: tuple[int, ...] | None = None,
    transpose_candidates: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> EstimatedTrajectory:
    """Align amt_notes against score_notes via a continuity-penalized
    fitting-DP, searching transpose_candidates for the best semitone
    shift (most matches, ties broken toward transpose 0). When
    bar_boundaries is given and prior.jump_back_penalty / jump_fwd_penalty
    are finite, the DP may relocate its score pointer to a bar boundary
    (backward = repeat/restart, forward = skip) for that fixed cost; with
    the defaults (no boundaries / inf penalties) it is the monotonic
    baseline (#115)."""
    best_result: EstimatedTrajectory | None = None
    best_key: tuple[int, int] | None = None
    for t in transpose_candidates:
        result = _align_at_transpose(amt_notes, score_notes, prior, t, bar_boundaries)
        key = (len(result.matches), -abs(t))
        if best_key is None or key > best_key:
            best_result, best_key = result, key
    assert best_result is not None
    return best_result


def teleport_gaps(trajectory: EstimatedTrajectory) -> list[float]:
    """Return the score_position delta between each pair of consecutively
    matched notes (usually >= 0; a bar-boundary jump (#118) yields a
    negative delta at the jump). A large gap indicates the alignment
    wandered forward to a coincidental match rather than following the
    performance continuously -- see docs/specs/2026-07-12-baseline-
    monotonic-follower-design.md for how this is used for grading."""
    return [
        b.score_position - a.score_position
        for a, b in zip(trajectory.matches, trajectory.matches[1:])
    ]
