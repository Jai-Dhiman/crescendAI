# HMM Follower Implementation Plan (#119)

> **For the build agent:** Dispatch each task group in order. Group A is
> sequential (all tasks touch `hmm.py` / `follower.py`). Groups B and C depend on
> A. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Add an opt-in Viterbi-HMM decoder beside the untouched additive DP,
with log-prob costs and a forward-backward position confidence.
**Spec:** docs/specs/2026-07-20-hmm-follower-design.md
**Style:** Follow CLAUDE.md + model/CLAUDE.md. Python via `uv`. Explicit
exceptions over silent fallbacks. Tests through public interfaces only.
**Tests:** `cd model && uv run pytest tests/follower_bench/` (61 existing must stay green).

## Task Groups

- **Group A (sequential):** A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> A8. All
  touch `hmm.py` or `follower.py`; no parallelism.
- **Group B (depends on A):** B1. New file `calibration.py`.
- **Group C (depends on A, B):** C1. Modifies `gap_report.py`.

`[SHIPS INDEPENDENTLY]` note: Group A alone is a usable, tested HMM follower with
confidence; B adds the calibration measurement; C wires it into the batch driver.

---

### Task A1: MatchedNote gains a confidence field
**Group:** A (first)

**Behavior being verified:** a MatchedNote can carry an optional confidence, and
every existing construction (no confidence) still works unchanged.
**Interface under test:** `MatchedNote` dataclass.

**Files:**
- Modify: `model/src/follower_bench/follower.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_hmm.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'confidence'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/src/follower_bench/follower.py`, add a field to the `MatchedNote`
dataclass (append AFTER `score_position` so positional construction is
unaffected):

```python
@dataclass(frozen=True)
class MatchedNote:
    """One perf-note <-> score-note correspondence chosen by follow()."""
    perf_index: int
    score_index: int
    perf_time: float
    score_position: float
    confidence: float | None = None  # HMM follower (#119) fills this; additive path leaves None
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q && uv run pytest tests/follower_bench/ -q
```
Expected: PASS (new test green; all 61 existing still green).

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(follower): MatchedNote.confidence field for HMM follower (#119)"
```

---

### Task A2: HmmParams + monotonic Viterbi follow_hmm
**Group:** A (after A1)

**Behavior being verified:** with jumps disabled, follow_hmm decodes the obvious
monotonic correspondence and drops an unmatchable note as an insertion.
**Interface under test:** `hmm.follow_hmm`, `hmm.HmmParams`.

**Files:**
- Create: `model/src/follower_bench/hmm.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_hmm.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_follow_hmm_monotonic_matches_in_order_and_skips_unmatchable -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.hmm'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/hmm.py
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
    (V, back). Monotonic (match/insertion/deletion) unless bar_boundaries is
    given with a finite jump prob (jump edges added in the jump block below)."""
    n, m = len(amt_notes), len(score_notes)
    lm, lc, li, ld, la, ljb, ljf = _logs(params)

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
    return V, back


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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/hmm.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(hmm): monotonic Viterbi follow_hmm + HmmParams (#119)"
```

---

### Task A3: bar-boundary jump-into-match (Viterbi)
**Group:** A (after A2)

**Behavior being verified:** with a finite jump prob, follow_hmm relocks after a
repeat via a backward bar-boundary jump; with jumps off it does not.
**Interface under test:** `hmm.follow_hmm` with `bar_boundaries` + jump params.

**Files:**
- Modify: `model/src/follower_bench/hmm.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_hmm.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_follow_hmm_backward_jump_relocks_after_a_repeat -q
```
Expected: FAIL — the assertion `[... 0, 1]` fails (no jump edges yet; the replay
is decoded as insertions, so `[0, 1, 2, 3]`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add a jump block to `_viterbi_at_transpose` in `hmm.py`. Replace the entire
`_viterbi_at_transpose` function with this version (adds the boundary set,
prefix/suffix argmax over the previous row, the jump-into-match candidates for
both directions, and a left-to-right deletion re-propagation so a jump can be
followed by deletions):

```python
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: PASS (backward-jump test green; A1/A2 still green).

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/hmm.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(hmm): bar-boundary jump-into-match Viterbi edges (#119)"
```

---

### Task A4: forward-jump relock (skipped passage)
**Group:** A (after A3)

**Behavior being verified:** follow_hmm relocks after a long skipped passage via
a forward bar-boundary jump; jumps off leaves the resumed notes unmatched. (No
new implementation — exercises the forward branch added in A3.)
**Interface under test:** `hmm.follow_hmm` forward jump.

**Files:**
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test** (this test must pass immediately given
  A3; if it fails, A3's forward branch is wrong — fix A3, do not weaken this test)

```python
# append to model/tests/follower_bench/test_hmm.py
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
```

- [ ] **Step 2: Run test — verify it FAILS then PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_follow_hmm_forward_jump_relocks_after_a_skipped_passage -q
```
Expected: PASS given A3 is correct. (If FAIL: the forward `pref_val[b-1] + ljf`
branch in `_relax_row_jumps_viterbi` is wrong — fix it in A3's function.)

- [ ] **Step 3: Implement** — none; this task pins the forward-jump behavior.

- [ ] **Step 4: Run full suite**

```bash
cd model && uv run pytest tests/follower_bench/ -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_hmm.py
git commit -m "test(hmm): forward-jump relock after a skipped passage (#119)"
```

---

### Task A5: forward-backward posterior + column_posteriors
**Group:** A (after A4)

**Behavior being verified:** the sum-product forward-backward posterior over
score columns sums to ~1 at every perf-note step (forward-backward correctness),
both monotonic and with jumps.
**Interface under test:** `hmm.column_posteriors`.

**Files:**
- Modify: `model/src/follower_bench/hmm.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_hmm.py
from follower_bench.hmm import column_posteriors


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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_column_posteriors_sum_to_one_per_step_monotonic_and_with_jumps -q
```
Expected: FAIL — `ImportError: cannot import name 'column_posteriors'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the log-sum-exp forward-backward to `hmm.py`. Deletions are silent
(non-emitting) and are attributed to the transition INTO the next emitting note
(a "del-closure"), so each perf note has exactly one well-defined column-after
marginal — this is what makes the per-step posterior sum to 1. Add:

```python
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: PASS. If the sums-to-1 assertion fails, the forward and backward edge
sets disagree — reconcile them (they must mirror `_viterbi_at_transpose`
exactly: match from del-closure of j-1, insertion from del-closure of j, jump
from raw prefix/suffix of the previous row). Do NOT relax the tolerance.

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/hmm.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(hmm): log-sum-exp forward-backward + column_posteriors (#119)"
```

---

### Task A6: alignment_logprob + the no-free-skip property
**Group:** A (after A5)

**Behavior being verified:** inserting one spurious noise note drops the total
log marginal by ~log(p_ins) — proving insertions are charged, not free (the
mechanism that defeats the repeat-cliff).
**Interface under test:** `hmm.alignment_logprob`.

**Files:**
- Modify: `model/src/follower_bench/hmm.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_hmm.py
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
    assert math.isclose(drop, -math.log(MONO.p_ins), rel_tol=0.02), drop
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_inserting_a_spurious_note_costs_about_log_p_ins -q
```
Expected: FAIL — `ImportError: cannot import name 'alignment_logprob'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# append to model/src/follower_bench/hmm.py
def alignment_logprob(amt_notes, score_notes, params, transpose, bar_boundaries=None):
    """The log marginal likelihood (logsumexp over all alignment paths) of the
    perf notes under the score at the given transpose. Exposed so the no-free-
    skip property is directly testable: a spurious note costs ~log(p_ins)."""
    _, logZ = _forward_backward(amt_notes, score_notes, params, transpose, bar_boundaries)
    return logZ
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: PASS. (The dominant path for the spurious note is a single insertion,
so `base - with_ins ≈ -log(p_ins)`; the `rel_tol=0.02` absorbs the small
del/emit path-sum corrections.)

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/hmm.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(hmm): alignment_logprob + no-free-skip test (#119)"
```

---

### Task A7: follow_hmm attaches confidence from the posterior
**Group:** A (after A6)

**Behavior being verified:** every decoded match carries a confidence in [0, 1]
(the posterior mass on its column), and a cleanly-locked run yields high
confidence.
**Interface under test:** `hmm.follow_hmm` (confidence-bearing).

**Files:**
- Modify: `model/src/follower_bench/hmm.py`
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_hmm.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_follow_hmm_attaches_calibrated_confidence_to_matches -q
```
Expected: FAIL — `assert m.confidence is not None` (follow_hmm still passes
confidence=None).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `follow_hmm` in `hmm.py` with this version (runs the forward-backward on
the winning transpose and looks up gamma at each decoded cell):

```python
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py -q
```
Expected: PASS (A1-A6 still green).

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/hmm.py model/tests/follower_bench/test_hmm.py
git commit -m "feat(hmm): attach forward-backward posterior confidence to matches (#119)"
```

---

### Task A8: cliff-crossing capstone
**Group:** A (after A7)

**Behavior being verified:** on one clip with BOTH a repeat and a forward skip,
the HMM (one param set) relocks both, while the additive follow() at the shipped
5.0/8.0 pair misses the forward skip — demonstrating the mechanism #118 could
not. (Verification-only; no new implementation.)
**Interface under test:** `hmm.follow_hmm` vs `follower.follow`.

**Files:**
- Test: `model/tests/follower_bench/test_hmm.py`

- [ ] **Step 1: Write the failing/passing test**

```python
# append to model/tests/follower_bench/test_hmm.py
from follower_bench.follower import ContinuityPrior, DEFAULT_SKIP_PENALTY, follow


def test_hmm_crosses_the_repeat_cliff_where_the_shipped_additive_pair_cannot() -> None:
    # 6 two-note bars; boundaries at columns (0,2,4,6,8,10).
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate(pitches)]
    bars = bar_boundary_columns([s.position for s in score], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert bars == (0, 2, 4, 6, 8, 10)
    # Perf: play bars 0-2, REPEAT bars 0-2, then SKIP bars 3-4, resume bar 5.
    seq = [60, 62, 64, 65, 67, 69] + [60, 62, 64, 65, 67, 69] + [77, 79]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate(seq)]

    hmm = follow_hmm(perf, score, JUMPS, bar_boundaries=bars, transpose_candidates=(0,))
    hmm_idx = [m.score_index for m in hmm.matches]
    # HMM relocks BOTH: backward to bar 0 (repeat) AND forward to bar 5 (skip).
    assert hmm_idx == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 10, 11]

    # Additive follow() at the shipped #118 default (jump_back=5.0, jump_fwd=8.0):
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY, jump_back_penalty=5.0, jump_fwd_penalty=8.0)
    add = follow(perf, score, prior, bar_boundaries=bars, transpose_candidates=(0,))
    add_idx = [m.score_index for m in add.matches]
    # The additive pair misses the forward skip: bar 5 (indices 10, 11) is never relocked.
    assert 10 not in add_idx and 11 not in add_idx
    assert set(add.unmatched_perf_indices) >= {12, 13}
```

- [ ] **Step 2: Run test**

```bash
cd model && uv run pytest tests/follower_bench/test_hmm.py::test_hmm_crosses_the_repeat_cliff_where_the_shipped_additive_pair_cannot -q
```
Expected: PASS. If the HMM index sequence differs, the default `HmmParams` need a
small hand-adjustment (keep p_match >> p_ins and jumps small); adjust defaults
until BOTH events relock, then re-run all of Group A. Do NOT change the additive
assertions (they document #118's shipped behavior).

- [ ] **Step 3: Implement** — none.

- [ ] **Step 4: Run full suite**

```bash
cd model && uv run pytest tests/follower_bench/ -q
```
Expected: PASS (all Group A + 61 existing).

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_hmm.py
git commit -m "test(hmm): cliff-crossing capstone — HMM relocks where additive 5/8 cannot (#119)"
```

---

### Task B1: calibration_stats (Spearman rho + risk-coverage)
**Group:** B (depends on A)

**Behavior being verified:** given a confidence-bearing trajectory and a clip's
truth, calibration_stats reports a positive rank correlation between confidence
and negative position error, and a risk-coverage curve whose high-confidence
head has lower error than the overall pool.
**Interface under test:** `calibration.calibration_stats`.

**Files:**
- Create: `model/src/follower_bench/calibration.py`
- Test: `model/tests/follower_bench/test_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_calibration.py
"""Behavioral tests for the follower confidence-calibration measurement (#119)."""
from __future__ import annotations

from follower_bench.calibration import CalibrationStats, calibration_stats
from follower_bench.clip_generator import SynthClip
from follower_bench.follower import MatchedNote
from follower_bench.trajectory import TrueTrajectory


def _clip(true_anchors):
    return SynthClip(
        asap_piece="synthetic", pathology_type="clean", seed=0, notes=(),
        true_trajectory=TrueTrajectory(anchors=tuple(true_anchors)), event_labels=(),
    )


def test_calibration_stats_rewards_confident_low_error_and_reports_risk_coverage() -> None:
    # Truth: position == time over [0, 10].
    clip = _clip([(0.0, 0.0), (10.0, 10.0)])
    # Estimate: perfectly tracks truth in [0,5) with HIGH confidence; in [5,10)
    # it is off by 4.0 with LOW confidence. Matches are (perf_time, score_position).
    matches = tuple(
        MatchedNote(perf_index=k, score_index=k, perf_time=float(k),
                    score_position=(float(k) if k < 5 else float(k) + 4.0),
                    confidence=(0.95 if k < 5 else 0.1))
        for k in range(11)
    )
    stats = calibration_stats(matches, clip)
    assert isinstance(stats, CalibrationStats)
    # higher confidence => lower error: rho(confidence, -|error|) is strongly positive
    assert stats.spearman_rho > 0.5
    # risk-coverage: the top-20%-confident head has lower median error than the whole pool
    head_cov, head_risk = stats.risk_coverage[0]
    assert head_cov <= 0.2
    assert head_risk < stats.overall_median_error
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_calibration.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.calibration'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/calibration.py
"""Confidence-calibration measurement for the HMM follower (#119). Kept out of
metric.py so the position scorer stays follower-agnostic. Samples per-note
confidence onto the same uniform time grid metric.score_clip uses, then reports
Spearman rho(confidence, -|position error|) [the gate] and a risk-coverage curve
[the 'can I trust the cursor' diagnostic]. Confidence between matched notes is a
zero-order hold (last decoded confidence); position uses the follower's own
TrueTrajectory interpolation. Fails loud if any match lacks a confidence."""
from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass

from scipy.stats import spearmanr

from follower_bench.clip_generator import SynthClip
from follower_bench.follower import MatchedNote
from follower_bench.metric import SAMPLE_HZ, _sample_grid, trajectory_from_matches

DEFAULT_COVERAGE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class CalibrationStats:
    """spearman_rho: rank corr between confidence and -|error| (higher = better
    calibrated; the gate). overall_median_error: median |error| over all grid
    samples. risk_coverage: (coverage_fraction, median |error| among the top
    coverage_fraction most-confident samples) points, most-confident first."""
    n_samples: int
    spearman_rho: float
    overall_median_error: float
    risk_coverage: tuple[tuple[float, float], ...]


def _confidence_at(matches: tuple[MatchedNote, ...], t: float) -> float:
    """Zero-order hold: the confidence of the most recent match at or before t
    (or the first match's confidence before it)."""
    times = [m.perf_time for m in matches]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        i = 0
    return float(matches[i].confidence)


def calibration_stats(matches, clip, *, sample_hz: float = SAMPLE_HZ,
                      coverage_fractions=DEFAULT_COVERAGE_FRACTIONS) -> CalibrationStats:
    if not matches:
        raise ValueError("cannot compute calibration on empty matches")
    if any(m.confidence is None for m in matches):
        raise ValueError("every match must carry a confidence to compute calibration")
    est = trajectory_from_matches(matches)
    true = clip.true_trajectory
    t_min, t_max = true.anchors[0][0], true.anchors[-1][0]
    times = _sample_grid(t_min, t_max, sample_hz)
    errors = [abs(est.score_position_at(t) - true.score_position_at(t)) for t in times]
    confs = [_confidence_at(matches, t) for t in times]

    neg_err = [-e for e in errors]
    rho, _ = spearmanr(confs, neg_err)
    if math.isnan(rho):  # zero variance (e.g. all-equal confidence) -> uninformative
        rho = 0.0
    overall_median = statistics.median(errors)

    order = sorted(range(len(times)), key=lambda k: confs[k], reverse=True)
    rc: list[tuple[float, float]] = []
    for frac in sorted(coverage_fractions):
        k = max(1, round(frac * len(order)))
        head = [errors[order[t]] for t in range(k)]
        rc.append((k / len(order), statistics.median(head)))
    return CalibrationStats(
        n_samples=len(times),
        spearman_rho=float(rho),
        overall_median_error=float(overall_median),
        risk_coverage=tuple(rc),
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_calibration.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/calibration.py model/tests/follower_bench/test_calibration.py
git commit -m "feat(calibration): Spearman + risk-coverage calibration measurement (#119)"
```

---

### Task C1: gap_report --hmm flag + routing seam
**Group:** C (depends on A, B)

**Behavior being verified:** a testable routing seam selects the additive DP by
default and the HMM decoder under `--hmm`, verified on a tiny synthetic (no ASAP
data); the `--hmm` CLI flag exists.
**Interface under test:** `gap_report._follow_for_cell`, the argparse config.

**Files:**
- Modify: `model/src/follower_bench/gap_report.py`
- Test: `model/tests/follower_bench/test_gap_report_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_gap_report_routing.py
"""The gap_report follower-routing seam (#119): additive by default, HMM under --hmm."""
from __future__ import annotations

from follower_bench.gap_report import _follow_for_cell
from follower_bench.hmm import HmmParams
from follower_bench.follower import ContinuityPrior, DEFAULT_SKIP_PENALTY
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote


def _tiny():
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64])]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64])]
    return perf, score


def test_follow_for_cell_routes_additive_by_default_and_hmm_when_requested() -> None:
    perf, score = _tiny()
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY)
    add = _follow_for_cell(False, perf, score, prior, HmmParams(), None, (0,))
    assert len(add.matches) == 3
    # additive path leaves confidence None
    assert all(m.confidence is None for m in add.matches)

    hmm = _follow_for_cell(True, perf, score, prior, HmmParams(), None, (0,))
    assert len(hmm.matches) == 3
    # HMM path attaches confidence
    assert all(m.confidence is not None for m in hmm.matches)


def test_gap_report_has_hmm_flag() -> None:
    import argparse
    from follower_bench import gap_report
    # The CLI parser must expose --hmm (introspect via a fresh parser build).
    ap = argparse.ArgumentParser()
    gap_report._add_cli_args(ap)
    ns = ap.parse_args(["--hmm"])
    assert ns.hmm is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_gap_report_routing.py -q
```
Expected: FAIL — `ImportError: cannot import name '_follow_for_cell'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/src/follower_bench/gap_report.py`:

(a) Add the import near the top:

```python
from follower_bench.hmm import HmmParams, follow_hmm
```

(b) Add the routing seam (place it above `_run_cell`):

```python
def _follow_for_cell(use_hmm, amt_notes, score_notes, prior, hmm_params, bar_boundaries, transpose_candidates):
    """Route one cell to the additive DP (default) or the #119 HMM decoder.
    transpose_candidates defaults to follow()/follow_hmm's own default when None."""
    if use_hmm:
        if transpose_candidates is None:
            return follow_hmm(amt_notes, score_notes, hmm_params, bar_boundaries=bar_boundaries)
        return follow_hmm(amt_notes, score_notes, hmm_params, bar_boundaries=bar_boundaries,
                          transpose_candidates=transpose_candidates)
    if transpose_candidates is None:
        return follow(amt_notes, score_notes, prior, bar_boundaries=bar_boundaries)
    return follow(amt_notes, score_notes, prior, bar_boundaries=bar_boundaries,
                  transpose_candidates=transpose_candidates)
```

(c) Refactor `_run_cell` to call the seam. Change its signature to accept
`use_hmm` and `hmm_params` and replace its `est = follow(...)` line:

```python
def _run_cell(performance, pathology, seed, score_notes, bar_boundaries,
              jump_back_penalty, jump_fwd_penalty, use_hmm=False, hmm_params=None):
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY,
                            jump_back_penalty=jump_back_penalty,
                            jump_fwd_penalty=jump_fwd_penalty)
    if hmm_params is None:
        hmm_params = HmmParams(p_jump_back=0.02, p_jump_fwd=0.01)
    t0 = time.perf_counter()
    try:
        clip = generate(performance, pathology, seed)
        est = _follow_for_cell(use_hmm, list(clip.notes), score_notes, prior, hmm_params, bar_boundaries, None)
        est_traj = trajectory_from_matches(est.matches)
        score = score_clip(est_traj, clip)
        return RunOutcome(performance, pathology, seed, score, None, time.perf_counter() - t0)
    except Exception as exc:
        return RunOutcome(performance, pathology, seed, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)
```

(d) Thread `use_hmm` through `_run_performance` (unpack it from the task tuple)
and `run_gap_report` (add a `use_hmm: bool = False` param, include it in each
task tuple, and pass it to `_run_cell`). Match the existing task-tuple pattern:
add `use_hmm` as the final element of the per-performance task tuple and unpack
it alongside the existing fields.

(e) Extract the argparse config into a reusable `_add_cli_args(ap)` function
that adds ALL existing arguments plus the new flag, and have `main()` call it:

```python
def _add_cli_args(ap):
    ap.add_argument("--per-composer", type=int, default=5, help="max performances sampled per composer")
    ap.add_argument("--seeds", type=int, default=5, help="pathology seeds per performance (clean runs once)")
    ap.add_argument("--workers", type=int, default=8, help="parallel worker processes")
    ap.add_argument("--max-score-notes", type=int, default=None,
                    help="exclude performances whose score MIDI exceeds this many notes")
    ap.add_argument("--jump-back-penalty", type=float, default=None,
                    help="enable #118 backward bar-boundary jumps at this penalty (additive path)")
    ap.add_argument("--jump-fwd-penalty", type=float, default=None,
                    help="enable #118 forward bar-boundary jumps at this penalty (additive path)")
    ap.add_argument("--hmm", action="store_true",
                    help="use the #119 Viterbi-HMM decoder instead of the additive DP")
    ap.add_argument("--trackio", action="store_true", help="log the baseline to Trackio")
    ap.add_argument("--out", type=Path, default=None, help="write the text report to this path")
    return ap
```

Then in `main()`, replace the inline `ap.add_argument(...)` block with
`_add_cli_args(ap)`, and pass `use_hmm=args.hmm` into `run_gap_report(...)`. When
`args.hmm` is set, after building `evaluation`, also compute and print a
calibration line by aggregating `calibration_stats` over the OK cells: for each
OK outcome re-run is not needed — instead compute calibration inside `_run_cell`
when `use_hmm` and attach it to `RunOutcome`. To keep this task minimal and
testable without ASAP data, ONLY the routing seam + flag are required to pass
the tests; wire the calibration line as follows (add to `_run_cell`'s `use_hmm`
branch, storing on RunOutcome via a new optional field `calibration=None`, and
print the pooled median rho in `_format_report` when present). Keep this
additive-path-neutral: when `use_hmm` is False, `RunOutcome.calibration` stays
None and the report is byte-identical to today.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_gap_report_routing.py -q && uv run pytest tests/follower_bench/ -q
```
Expected: PASS (all follower_bench tests green, including the 61 pre-existing).

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/gap_report.py model/tests/follower_bench/test_gap_report_routing.py
git commit -m "feat(gap-report): --hmm decoder routing seam + calibration line (#119)"
```

---

## Plan Self-Review

- **Spec coverage:** every spec behavior (1-8) maps to a task (A2,A3,A4,A8,A5,B1,A6,C1); confidence attach = A7; HmmParams = A2; MatchedNote.confidence = A1. Complete.
- **Placeholder scan:** no TBD/TODO; every step has exact code or exact commands. C1 step 3(d)/(e) describe threading an existing tuple pattern the build agent can see in the file — acceptable (it is a mechanical refactor of visible code), and the two C1 tests pin the observable contract.
- **Type consistency:** `HmmParams`, `follow_hmm`, `column_posteriors`, `alignment_logprob`, `_forward_backward`, `_viterbi_at_transpose`, `_relax_row_jumps_viterbi`, `_traceback`, `calibration_stats`, `CalibrationStats`, `_follow_for_cell`, `_add_cli_args` names are consistent across tasks. `MatchedNote.confidence` used identically everywhere.
- **Group correctness:** Group A tasks all touch hmm.py/follower.py -> sequential (correct). B is a new file. C modifies gap_report.py only.
- **Vertical slice check:** A1,A2,A3,A5,A6,A7,B1,C1 are one-test-one-impl. A4 and A8 are verification-only capstones (explicitly flagged; they pin behavior of code landed in A3/A7 and carry no new impl) — a deliberate, declared exception.
- **Behavior test check:** all tests exercise public functions (follow_hmm, column_posteriors, alignment_logprob, calibration_stats, _follow_for_cell) and assert on returned values, never on private state.

**Highest-risk component:** the `_forward_backward` log-sum-exp pass (A5). Its
`column_posteriors` sums-to-1 test is the correctness guard; if it fails, the
forward and backward edge sets disagree and must be reconciled to mirror
`_viterbi_at_transpose` exactly. Do not relax the tolerance.

**Post-ship (NOT this plan):** `/autoresearch` tunes the seven `HmmParams`
against `gap_report --hmm` on the capped subset for #118 parity + calibration
rho >= ~0.3.

---

## Challenge Review

Reviewer read: this plan in full, the linked spec, and all listed source files
(`follower.py`, `metric.py`, `score_notes.py`, `trajectory.py`,
`clip_generator.py`, `asap_alignment.py`, `segments.py`, `gap_report.py`,
`test_follower.py`). Verified externally: `scipy.stats.spearmanr` imports (OK),
61 existing `follower_bench` tests collect.

### CEO Pass

- **Premise — right problem, real pain?** Confirmed against code. The additive
  DP's `skip_perf` really is free (`follower.py:96-97`: the `>=` tie-branch adds
  `B[i-1][j]` with no penalty). So the spec's root-cause claim ("no free skip is
  the load-bearing fix") is grounded in the actual code, not hand-waving. The
  pain (repeat-cliff, no confidence signal) is real and measured (#118).
- **Existing coverage / not reinventing.** `_relax_row_jumps` (the #118
  same-row jump) is correctly identified and deliberately *replaced* (not
  reused) by the row-advancing jump-into-match, with the reason stated
  (posterior must be cycle-free). Sound.
- **Scope.** Tight. 4 files touched (2 new, 2 modified), 2 new deep modules —
  under the complexity-smell thresholds. Param tuning correctly deferred to a
  separate post-ship autoresearch. No drift from spec.
- **[OBS]** MVP-if-halved: Group A alone (HMM + confidence) is the core bet; B
  (calibration measurement) and C (gap_report wiring) are the earning-their-keep
  proof. All three are needed to satisfy the issue's "calibrated confidence"
  success criterion, so nothing is cuttable without dropping a success bar.

### Engineering Pass

**Architecture — the forward-backward (A5), the highest-risk component.** I
traced the sum-product recursions against the Viterbi edge set by hand:
- Forward `match`/`insertion` correctly use the del-closure `D` of the previous
  row (deletions folded into the transition *into* each emitting note); `jump`
  correctly uses the *raw* prefix/suffix of the previous row (mirroring
  `_relax_row_jumps_viterbi`, which sources jumps from raw `V[i-1]`). Consistent.
- Backward `Cmatch[j] = lse(E_match[j+1], Cmatch[j+1]+ld)` expands to
  `logsumexp_{k>=0} k*ld + E_match[j+1+k]` (verified) and `Cins` likewise —
  these correctly mirror the forward's deletions-before-emission. The jump-out
  running accumulators partition sources by `s>b` (backward) / `s<b` (forward),
  exactly mirroring the forward's `suf[b+1]` / `pref[b-1]`. Edge sets match.
- The unnormalized start distribution (`alpha[0][j]=j*ld`, geometric, not
  summing to 1) does **not** break the sums-to-1 identity — that identity is
  algebraic in the recursions given `logZ = lse(alpha[n])`, independent of start
  normalization. Verified.
Conclusion: the forward-backward is very likely correct, and `column_posteriors`
sums-to-1 (A5) is a genuine guard. No blocker here, but it is the component most
likely to need a reconcile pass — see the plan's own note (line 1296).

**Module Depth.** `hmm` (interface: `HmmParams`, `follow_hmm`,
`column_posteriors`, `alignment_logprob`; hides ~230 LOC of log-space Viterbi +
DAG jumps + forward-backward) = **DEEP**. `calibration` (interface:
`calibration_stats`; hides grid sampling + Spearman + risk-coverage) = **DEEP**.
Both pass the depth audit.

**Findings:**

- `[RISK]` (confidence: 6/10) — **A8 additive-baseline assertion may be
  tie-fragile.** The capstone asserts `10 not in add_idx and 11 not in add_idx`
  for the shipped 5.0/8.0 additive `follow()`. The skipped region is 4 score
  notes (cols 6->10): reaching bar 5 via monotonic `skip_score` costs
  `4*0.5=2.0` and gains `+2.0` (two matches) — a genuine **reward tie** with
  leaving them unmatched. The assertion currently holds only because tie-breaking
  favors the smaller `best_j` and the load-bearing `>=` skip_perf bias
  (`follower.py:96`) — i.e. it's correct-by-accident, not by margin. Fallback
  (do this in A8, it does not weaken the assertion): **widen the skipped region**
  to >=6 score notes so `skip_score` cost (>=3.0) strictly exceeds the `+2.0`
  match gain AND stays above `jump_fwd=8.0` — then the additive path *provably*
  cannot relock bar 5, and the capstone proves the mechanism by margin.

- `[RISK]` (confidence: 6/10) — **A8's exact-vector HMM assertion is brittle and
  tuning-coupled.** `hmm_idx == [0,1,2,3,4,5,0,1,2,3,4,5,10,11]` couples a unit
  test to the hand-set default `HmmParams`. The mechanism it proves is: a
  backward relock to bar 0 AND a forward relock to bar 5 both occur under one
  param set. Prefer asserting those two properties (`0` reappears after the
  first pass = backward relock; `{10,11} <= set(hmm_idx)` = forward relock)
  rather than the full 14-element vector, so a later autoresearch re-tune of the
  defaults doesn't spuriously break a unit test. Keep the exact-vector as a
  comment if desired.

- `[RISK]` (confidence: 5/10) — **A6 `rel_tol=0.02` is tight.** `drop ≈ -log(p_ins)
  = 3.0`; the forward sum's confuse-path correction is `~exp(lc-li)=exp(-3.9)≈0.02`
  per alternative placement, which can push the relative error toward the 0.02
  bound. If it fails, loosen to `rel_tol=0.05` (still proves the ~`log(p_ins)`
  drop) — do NOT switch to an insertion-only assertion, which would stop testing
  the marginal.

- `[RISK]` (confidence: 7/10) — **C1 threading is prose-only and untested
  end-to-end.** Steps 3(d)/(e) describe threading `use_hmm` through the
  multiprocessing task tuple + `run_gap_report` in prose. The two C1 tests cover
  `_follow_for_cell` and the `--hmm` flag existence, but **nothing verifies
  `use_hmm` actually propagates** through `run_gap_report -> _run_performance ->
  _run_cell`. A mis-threaded tuple would silently run the additive path under
  `--hmm`. Add one small test: `run_gap_report(..., use_hmm=True)` on a tiny
  in-process case (workers=1, no ASAP) and assert the outcomes carry HMM
  confidence — or at minimum assert the task tuple round-trips `use_hmm`.

- `[RISK]` (confidence: 6/10) — **C1 calibration-line wiring is prose-only and
  untested.** The `RunOutcome.calibration` field, its computation in `_run_cell`,
  and its print in `_format_report` are described but have no test. Post-ship
  autoresearch depends on `gap_report --hmm` emitting rho. Not a build-blocker
  (tests pass without it), but flag it so the build agent implements it and adds
  a smoke assertion rather than treating it as optional prose.

- `[OBS]` — **scipy hard-imported without the fallback the plan's Open Question
  promises.** `calibration.py` does `from scipy.stats import spearmanr` with no
  numpy fallback. Verified scipy imports in this env, so this is fine as-is;
  just noting the code and the Open Question disagree. Leave as hard import
  (scipy is a guaranteed ML-stack dep); delete the "numpy fallback" from the
  Open Question or keep as documented non-goal.

- `[OBS]` — **`calibration.py` imports private `metric._sample_grid` /
  `metric.SAMPLE_HZ`.** Acceptable intra-package coupling and keeps the grid
  definition single-sourced (correct call — do NOT duplicate the grid). Noted
  only for awareness.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Forward & backward edge sets mirror exactly -> sums-to-1 | VALIDATE | Hand-traced as consistent; the A5 test is the guard if not |
| Additive 5.0/8.0 cannot relock the A8 forward skip | RISKY | Reward-tie; holds only by tie-break bias (see RISK) |
| Default HmmParams produce A8's exact index vector | RISKY | Tuning-coupled brittle assertion (see RISK) |
| `use_hmm` threads correctly through multiprocessing tuple | RISKY | Prose-only, no end-to-end test (see RISK) |
| Default (no --hmm) gap_report is byte-identical | SAFE | `_follow_for_cell(False,...)` calls `follow(...)` with the same args as today |
| `MatchedNote.confidence` default None keeps 61 tests green | SAFE | Field appended last with a default; positional constructions unaffected |
| Jump-into-match preserves #118 relock behavior | SAFE | Traced A3/A4: backward lands bar-first-note, forward leaps filler; both correct |
| scipy.stats.spearmanr available | SAFE | Verified import in this env |

### Summary

[BLOCKER] count: 0
[RISK]    count: 5
[QUESTION] count: 0

The plan is unusually rigorous: exact code for every emitting task, a genuine
forward-backward correctness guard, deep modules, correct vertical slicing
(A4/A8 are declared verification-only capstones over already-landed code, not
horizontal slicing). All five risks have concrete, in-plan fallbacks that do
not weaken any assertion, and none block starting the build.

VERDICT: PROCEED_WITH_CAUTION — monitor during build: (1) widen A8's skipped
region so the additive-can't-relock assertion holds by margin, not by tie-break;
(2) prefer A8's property assertions over the exact 14-element vector; (3) loosen
A6 rel_tol to 0.05 if the marginal correction trips 0.02; (4) add an end-to-end
`use_hmm` propagation test in C1; (5) actually implement + smoke-test C1's
calibration line rather than leaving it as prose.
