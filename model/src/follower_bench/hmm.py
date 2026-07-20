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


def _lse2(a: float, b: float) -> float:
    """logaddexp of two log-values, NaN/inf-safe."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _lse(values) -> float:
    acc = NEG_INF
    for v in values:
        acc = _lse2(acc, v)
    return acc


def _forward_backward(amt_notes, score_notes, params, transpose, bar_boundaries=None):
    """Sum-product alpha/beta over the same edges as _viterbi_at_transpose.
    Returns (gamma, logZ): gamma[i][j] is log P(pointer at column j right after
    consuming perf note i | all obs); logZ is the log marginal likelihood.
    Deletions are folded into the transition into each emitting note (del-
    closure), so sum_j exp(gamma[i][j]) == 1 for every i."""
    n, m = len(amt_notes), len(score_notes)
    lm, lc, li, ld, la, ljb, ljf = _logs(params)
    boundaries = tuple(b for b in (bar_boundaries or ()) if 0 <= b <= m)
    jumps_enabled = bool(boundaries) and (ljb > NEG_INF or ljf > NEG_INF)
    bset = set(boundaries)

    def emit(i, j):
        return lm if (score_notes[j - 1].pitch + transpose) == amt_notes[i - 1].pitch else lc

    def del_closure(row):
        # D[j] = logsumexp_{j'<=j} row[j'] + (j-j')*ld
        D = [NEG_INF] * (m + 1)
        D[0] = row[0]
        for j in range(1, m + 1):
            D[j] = _lse2(row[j], D[j - 1] + ld)
        return D

    # ---- forward ----
    alpha = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    alpha[0][0] = 0.0
    for j in range(1, m + 1):
        alpha[0][j] = alpha[0][j - 1] + ld
    for i in range(1, n + 1):
        prev = alpha[i - 1]
        D = del_closure(prev)
        # raw prefix/suffix logsumexp over prev for jump sources
        pref = [NEG_INF] * (m + 1)
        acc = NEG_INF
        for s in range(m + 1):
            acc = _lse2(acc, prev[s])
            pref[s] = acc
        suf = [NEG_INF] * (m + 2)
        acc = NEG_INF
        for s in range(m, -1, -1):
            acc = _lse2(acc, prev[s])
            suf[s] = acc
        row = alpha[i]
        for j in range(0, m + 1):
            terms = [D[j] + li]  # insertion at column j (dels then a noise note)
            if j >= 1:
                terms.append(D[j - 1] + la + emit(i, j))  # match into j
            if jumps_enabled and j >= 1 and (j - 1) in bset:
                # Intentional asymmetry vs _relax_row_jumps_viterbi: jump sources
                # here are raw landing columns (suf/pref over prev), not the
                # del-extended sources the Viterbi jump reads from V[i-1].
                b = j - 1
                jt = []
                if ljb > NEG_INF and b + 1 <= m:
                    jt.append(ljb + suf[b + 1])   # backward sources s > b
                if ljf > NEG_INF and b - 1 >= 0:
                    jt.append(ljf + pref[b - 1])  # forward sources s < b
                if jt:
                    terms.append(_lse(jt) + la + emit(i, j))
            row[j] = _lse(terms)
    logZ = _lse(alpha[n])
    if logZ == NEG_INF:
        raise ValueError("HMM forward pass found no decodable path (logZ = -inf); "
                         "check p_ins > 0 or the score/perf inputs")

    # ---- backward ----
    beta = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        beta[n][j] = 0.0
    for i in range(n - 1, -1, -1):
        nb = beta[i + 1]
        # E_match[j2] = la + emit(i+1,j2) + nb[j2]; E_ins[j2] = li + nb[j2]
        E_match = [NEG_INF] * (m + 1)
        E_ins = [NEG_INF] * (m + 1)
        for j2 in range(m + 1):
            E_ins[j2] = li + nb[j2]
            if j2 >= 1:
                E_match[j2] = la + emit(i + 1, j2) + nb[j2]
        # Cmatch[j] = logsumexp_{k>=0} k*ld + E_match[j+1+k]
        Cmatch = [NEG_INF] * (m + 1)
        for j in range(m - 1, -1, -1):
            Cmatch[j] = _lse2(E_match[j + 1], Cmatch[j + 1] + ld)
        # Cins[j] = logsumexp_{j2>=j} (j2-j)*ld + E_ins[j2]
        Cins = [NEG_INF] * (m + 1)
        Cins[m] = E_ins[m]
        for j in range(m - 1, -1, -1):
            Cins[j] = _lse2(E_ins[j], Cins[j + 1] + ld)
        # jump-out contributions per source column j
        jump_out = [NEG_INF] * (m + 1)
        if jumps_enabled:
            # E_jump_b = la + emit(i+1, b+1) + nb[b+1] for each boundary b with b+1<=m
            ejb = {b: la + emit(i + 1, b + 1) + nb[b + 1] for b in boundaries if b + 1 <= m}
            sb = sorted(ejb)
            # backward: source j > b (cost ljb) -> boundaries below j
            if ljb > NEG_INF:
                run = NEG_INF
                bi = 0
                for j in range(m + 1):
                    while bi < len(sb) and sb[bi] < j:
                        run = _lse2(run, ejb[sb[bi]])
                        bi += 1
                    jump_out[j] = _lse2(jump_out[j], (ljb + run) if run != NEG_INF else NEG_INF)
            # forward: source j < b (cost ljf) -> boundaries above j
            if ljf > NEG_INF:
                run = NEG_INF
                bi = len(sb) - 1
                for j in range(m, -1, -1):
                    while bi >= 0 and sb[bi] > j:
                        run = _lse2(run, ejb[sb[bi]])
                        bi -= 1
                    jump_out[j] = _lse2(jump_out[j], (ljf + run) if run != NEG_INF else NEG_INF)
        for j in range(m + 1):
            beta[i][j] = _lse([Cmatch[j], Cins[j], jump_out[j]])

    gamma = [[alpha[i][j] + beta[i][j] - logZ for j in range(m + 1)] for i in range(n + 1)]
    return gamma, logZ


def column_posteriors(amt_notes, score_notes, params, transpose, bar_boundaries=None):
    """Return the per-perf-note posterior over score columns as probabilities:
    out[i][j] = P(pointer at column j right after consuming perf note i | obs).
    Each row sums to ~1. Rows 1..n correspond to perf notes; row 0 is the start."""
    gamma, _ = _forward_backward(amt_notes, score_notes, params, transpose, bar_boundaries)
    return [[math.exp(g) for g in row] for row in gamma]


def follow_hmm(amt_notes, score_notes, params, bar_boundaries=None,
               transpose_candidates=(-2, -1, 0, 1, 2)):
    """Align amt_notes to score_notes via a log-prob Viterbi-HMM, searching
    transpose_candidates for the best shift (most matches, ties toward 0). Each
    match carries confidence = the forward-backward posterior mass on its
    decoded column (in [0, 1])."""
    n, m = len(amt_notes), len(score_notes)
    best_t = None
    best_key = None
    best_back = None
    best_bj = None
    for t in transpose_candidates:
        V, back = _viterbi_at_transpose(amt_notes, score_notes, params, t, bar_boundaries)
        bj = max(range(m + 1), key=lambda j: V[n][j])
        matches, _ = _traceback(back, n, bj, amt_notes, score_notes)
        key = (len(matches), -abs(t))
        if best_key is None or key > best_key:
            best_key, best_t, best_back, best_bj = key, t, back, bj
    assert best_t is not None
    gamma, _ = _forward_backward(amt_notes, score_notes, params, best_t, bar_boundaries)

    def conf(i, j):
        return math.exp(gamma[i][j])

    matches, unmatched = _traceback(best_back, n, best_bj, amt_notes, score_notes, conf=conf)
    return EstimatedTrajectory(
        transpose_semitones=best_t,
        matches=tuple(matches),
        unmatched_perf_indices=tuple(unmatched),
    )


def alignment_logprob(amt_notes, score_notes, params, transpose, bar_boundaries=None):
    """The log marginal likelihood (logsumexp over all alignment paths) of the
    perf notes under the score at the given transpose. Exposed so the no-free-
    skip property is directly testable: a spurious note costs ~log(p_ins)."""
    _, logZ = _forward_backward(amt_notes, score_notes, params, transpose, bar_boundaries)
    return logZ
