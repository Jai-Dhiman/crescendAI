"""Opt-in Viterbi-HMM score follower (issue #119, epic #108): the "B" of
epic #108's A->B. Graduates #118's hand-tuned additive jump penalties to
log-probability emission/transition costs on the same perf-note x score-note
grid, and emits a calibrated position confidence from a forward-backward
posterior. Pitch-only (no tempo model). The additive follower
(follower.follow / _align_at_transpose) is untouched; this is a parallel
decoder. Jumps are row-advancing "jump-into-match" edges restricted to bar
boundaries (deliberately unlike #118's same-row _relax_row_jumps), so the
(i, j) grid stays a DAG and the sum-product forward-backward posterior is
cycle-free and correct. See docs/specs/2026-07-20-hmm-follower-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from follower_bench.follower import EstimatedTrajectory, MatchedNote
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote

NEG_INF = -math.inf


@dataclass(frozen=True)
class HmmParams:
    """Log-prob weights (given as probabilities in (0, 1]) for the HMM
    follower. Weights are unnormalized: the per-perf-note posterior normalizes
    by construction, so no simplex constraint is imposed. p_jump_back /
    p_jump_fwd default to 0.0 (log -inf) -> jumps disabled -> monotonic HMM.
    Defaults are sane hand-set values that pass the #119 unit tests; a post-ship
    /autoresearch pass tunes them for gap_report parity."""
    p_match: float = 0.9
    p_confuse: float = 0.001
    p_ins: float = 0.05
    p_del: float = 0.5
    p_adv: float = 0.9
    p_jump_back: float = 0.0
    p_jump_fwd: float = 0.0


def _lg(p: float) -> float:
    return math.log(p) if p > 0.0 else NEG_INF


def _logs(params: HmmParams) -> tuple[float, float, float, float, float, float, float]:
    return (_lg(params.p_match), _lg(params.p_confuse), _lg(params.p_ins),
            _lg(params.p_del), _lg(params.p_adv), _lg(params.p_jump_back),
            _lg(params.p_jump_fwd))


def _viterbi_at_transpose(amt_notes, score_notes, params, transpose, bar_boundaries=None):
    """Max-product Viterbi over the perf-note x score-note grid. Returns
    (V, back). Adds row-advancing bar-boundary jump-into-match edges when
    bar_boundaries is given and a jump prob is finite."""
    n, m = len(amt_notes), len(score_notes)
    lm, lc, li, ld, la, ljb, ljf = _logs(params)
    boundaries = tuple(b for b in (bar_boundaries or ()) if 0 <= b <= m)
    jumps_enabled = bool(boundaries) and (ljb > NEG_INF or ljf > NEG_INF)

    def emit(i, j):  # perf i-1 vs score j-1
        return lm if (score_notes[j - 1].pitch + transpose) == amt_notes[i - 1].pitch else lc

    V = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    V[0][0] = 0.0
    for j in range(1, m + 1):
        V[0][j] = V[0][j - 1] + ld
        back[0][j] = ("del", 0, j - 1)
    for i in range(1, n + 1):
        V[i][0] = V[i - 1][0] + li
        back[i][0] = ("ins", i - 1, 0)
        for j in range(1, m + 1):
            best, mv = V[i - 1][j] + li, ("ins", i - 1, j)          # insertion
            cand = V[i][j - 1] + ld                                  # deletion
            if cand > best:
                best, mv = cand, ("del", i, j - 1)
            cand = V[i - 1][j - 1] + la + emit(i, j)                 # match
            if cand > best:
                best, mv = cand, ("match", i - 1, j - 1)
            V[i][j] = best
            back[i][j] = mv
        if jumps_enabled:
            _relax_row_jumps_viterbi(V, back, i, m, boundaries, la, ld, ljb, ljf, emit)
    return V, back


def _relax_row_jumps_viterbi(V, back, i, m, boundaries, la, ld, ljb, ljf, emit):
    """Add the single best jump-into-match edge for each bar boundary in row i
    (a jump consuming perf note i-1 as a match on the bar's first note), then
    re-propagate deletions left-to-right so a jump can be followed by skips.
    Sources come from row i-1: backward = source column > target (repeat/
    restart, cost ljb), forward = source column < target (skip, cost ljf)."""
    prev = V[i - 1]
    # prefix max+arg over prev[0..s], suffix max+arg over prev[s..m]
    pref_val = [NEG_INF] * (m + 1)
    pref_arg = [0] * (m + 1)
    bv, ba = NEG_INF, 0
    for s in range(m + 1):
        if prev[s] > bv:
            bv, ba = prev[s], s
        pref_val[s], pref_arg[s] = bv, ba
    suf_val = [NEG_INF] * (m + 2)
    suf_arg = [0] * (m + 2)
    bv, ba = NEG_INF, m
    for s in range(m, -1, -1):
        if prev[s] > bv:
            bv, ba = prev[s], s
        suf_val[s], suf_arg[s] = bv, ba
    for b in boundaries:
        target = b + 1  # match the bar's first note (index b) -> pointer at b+1
        if not (1 <= target <= m):
            continue
        cand, src = NEG_INF, None
        if ljb > NEG_INF and b + 1 <= m:      # backward: source s > b
            c = suf_val[b + 1] + ljb
            if c > cand:
                cand, src = c, suf_arg[b + 1]
        if ljf > NEG_INF and b - 1 >= 0:      # forward: source s < b
            c = pref_val[b - 1] + ljf
            if c > cand:
                cand, src = c, pref_arg[b - 1]
        if src is None:
            continue
        cand = cand + la + emit(i, target)
        if cand > V[i][target]:
            V[i][target] = cand
            back[i][target] = ("jump", i - 1, src)
    # re-propagate deletions so a jump landing can be followed by skips
    for j in range(1, m + 1):
        c = V[i][j - 1] + ld
        if c > V[i][j]:
            V[i][j] = c
            back[i][j] = ("del", i, j - 1)


def _traceback(back, n, best_j, amt_notes, score_notes, conf=None):
    """Walk backpointers from (n, best_j). 'match' and 'jump' moves emit a
    MatchedNote; 'ins' records an unmatched perf note; 'del' just advances.
    `conf` is an optional (i, j) -> float supplying per-match confidence."""
    matches: list[MatchedNote] = []
    unmatched: list[int] = []
    i, j = n, best_j
    while i > 0:
        mv = back[i][j]
        if mv is None:
            break
        kind, pi, pj = mv
        if kind in ("match", "jump"):
            matches.append(MatchedNote(
                perf_index=i - 1,
                score_index=j - 1,
                perf_time=amt_notes[i - 1].onset,
                score_position=score_notes[j - 1].position,
                confidence=(conf(i, j) if conf is not None else None),
            ))
            i, j = pi, pj
        elif kind == "ins":
            unmatched.append(i - 1)
            i, j = pi, pj
        else:  # del
            i, j = pi, pj
    matches.reverse()
    unmatched.reverse()
    return matches, unmatched


def follow_hmm(amt_notes, score_notes, params, bar_boundaries=None,
               transpose_candidates=(-2, -1, 0, 1, 2)):
    """Align amt_notes to score_notes via a log-prob Viterbi-HMM, searching
    transpose_candidates for the best shift (most matches, ties toward 0).
    Confidence is attached in Task A7; here every match carries confidence=None."""
    n, m = len(amt_notes), len(score_notes)
    best = None
    best_key = None
    for t in transpose_candidates:
        V, back = _viterbi_at_transpose(amt_notes, score_notes, params, t, bar_boundaries)
        bj = max(range(m + 1), key=lambda j: V[n][j])
        matches, unmatched = _traceback(back, n, bj, amt_notes, score_notes)
        key = (len(matches), -abs(t))
        if best_key is None or key > best_key:
            best_key = key
            best = EstimatedTrajectory(
                transpose_semitones=t,
                matches=tuple(matches),
                unmatched_perf_indices=tuple(unmatched),
            )
    assert best is not None
    return best
