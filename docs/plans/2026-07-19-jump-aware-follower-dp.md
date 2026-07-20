# Jump-Aware Follower DP Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.
> All work happens in the worktree `.worktrees/issue-118-jump-aware-dp` (branch `issue-118-jump-aware-dp`).
> Run all commands from `model/` (e.g. `cd model && uv run pytest ...`). The ASAP dataset is
> symlinked into the worktree at `model/data/raw/asap-dataset`.

**Goal:** The score follower can relock after a performer repeats, restarts, or skips a passage — following backward and forward score-pointer jumps at bar boundaries — instead of staying lost until the performance replays forward past the abandoned position.
**Spec:** docs/specs/2026-07-19-jump-aware-follower-dp-design.md
**Style:** Follow CLAUDE.md / model/CLAUDE.md. Explicit exceptions over silent fallbacks; no emojis; partitura not music21; `uv` for Python. Tests verify behavior through `follow()`'s public return value, never internal DP state.

## Task Groups

- **Group 0 (first):** Task 0 — commit the already-implemented iteration cap, re-confirm the capped baseline.
- **Group 1 (parallel, different files):** Task 1 (`follower.py`: `bar_boundary_columns`), Task 5 (`asap_alignment.py`: downbeats).
- **Group 2 (depends on Group 1 / Task 1):** Task 2 (`follower.py`: backward jump).
- **Group 3 (depends on Group 2):** Task 3 (`follower.py`: forward jump).
- **Group 4 (depends on Group 3 + Task 5):** Task 6 (`gap_report.py`: wire penalties + boundaries; integration verification).

Task 1 and Task 5 touch different files → parallel. Tasks 2 and 3 both modify `_align_at_transpose` → strictly sequential. Task 6 depends on the follower API (Tasks 1–3) and downbeats (Task 5).

---

### Task 0: Commit the iteration-speed cap and re-confirm the capped baseline
**Group:** 0

**Behavior being verified:** the `--max-score-notes` cap (already implemented in `gap_report.py`, currently uncommitted) excludes over-length performances as reported skips and, with jumps off, reproduces the #117 baseline shape on a capped subset.
**Interface under test:** `python -m follower_bench.gap_report` CLI.

**Files:**
- Modify (commit only — already edited): `model/src/follower_bench/gap_report.py`

- [ ] **Step 1: Confirm the cap is present and the suite is green**

```bash
cd model && grep -n "max_score_notes" src/follower_bench/gap_report.py && uv run pytest tests/follower_bench/ -q
```
Expected: `max_score_notes` appears (arg + threading); 56 passed.

- [ ] **Step 2: Run the capped baseline (jumps off — no penalty args)**

```bash
cd model && uv run python -m follower_bench.gap_report --per-composer 1 --seeds 3 --max-score-notes 1800 --workers 6
```
Expected: clean lock ~0.96 with `false_jmp 0`; repeat/restart lock ~0.65, median relock ~50s; jump lock ~0.48. (Matches the locked #117 shape; this is the monotonic starting point.)

- [ ] **Step 3: Commit**

```bash
cd model && git add src/follower_bench/gap_report.py && git commit -m "feat(follower-bench): --max-score-notes iteration cap for #118 gap runs"
```

---

### Task 1: `bar_boundary_columns` — map downbeat times to DP columns
**Group:** 1 (parallel with Task 5)

**Behavior being verified:** downbeat times in score-MIDI seconds map to the DP column where each bar starts (count of score notes strictly before the downbeat).
**Interface under test:** `bar_boundary_columns(positions, downbeats)`.

**Files:**
- Modify: `model/src/follower_bench/follower.py`
- Test: `model/tests/follower_bench/test_follower.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_follower.py
from follower_bench.follower import bar_boundary_columns


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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k bar_boundary_columns
```
Expected: FAIL — `ImportError: cannot import name 'bar_boundary_columns' from 'follower_bench.follower'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `import bisect` alongside the existing `import math` at the top of `follower.py`, and add this public function (place it just above `def follow(`):

```python
def bar_boundary_columns(positions: list[float], downbeats) -> tuple[int, ...]:
    """Map bar-downbeat times (score-MIDI seconds) to score-note DP
    columns: each column is the count of notes with position strictly
    before the downbeat (i.e. the pointer sits at the start of that bar).
    `positions` must be sorted ascending (as load_score_notes_from_midi
    returns). Returns a sorted, de-duplicated tuple -- these are the only
    columns follow() may jump to."""
    cols = {bisect.bisect_left(positions, float(d)) for d in downbeats}
    return tuple(sorted(cols))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k bar_boundary_columns
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/follower.py tests/follower_bench/test_follower.py && git commit -m "feat(follower): bar_boundary_columns maps downbeats to DP columns (#118)"
```

---

### Task 5: Expose `midi_score_downbeats` on `ClipAlignment`
**Group:** 1 (parallel with Task 1)

**Behavior being verified:** `load_alignment` exposes the score-MIDI downbeat times (already in the annotations) so callers can compute bar boundaries.
**Interface under test:** `load_alignment(piece).midi_score_downbeats`.

**Files:**
- Modify: `model/src/follower_bench/asap_alignment.py`
- Test: `model/tests/follower_bench/test_asap_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_asap_alignment.py
def test_load_alignment_exposes_score_downbeats() -> None:
    from follower_bench.asap_alignment import load_alignment
    alignment = load_alignment("Bach/Fugue/bwv_846/Shi05M.mid")
    # downbeats are score-MIDI seconds, sorted ascending, a subset of the beat grid
    assert len(alignment.midi_score_downbeats) >= 4
    dbs = list(alignment.midi_score_downbeats)
    assert dbs == sorted(dbs)
    assert all(db in alignment.midi_score_beats for db in dbs)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q -k downbeats
```
Expected: FAIL — `AttributeError: 'ClipAlignment' object has no attribute 'midi_score_downbeats'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `model/src/follower_bench/asap_alignment.py`, add the field to `ClipAlignment` (after `midi_score_beats`):

```python
    midi_score_downbeats: tuple[float, ...]
```

Then in `load_alignment`, after the `perf_beats`/`score_beats` validation and before constructing `performance_midi_path`, read the downbeats, and add `midi_score_downbeats=...` to the `ClipAlignment(...)` return. The full return becomes:

```python
    score_downbeats = entry.get("midi_score_downbeats") or []
    performance_midi_path = asap_root / asap_piece
    if not performance_midi_path.exists():
        raise FileNotFoundError(f"ASAP performance MIDI not found: {performance_midi_path}")
    score_midi_path = performance_midi_path.parent / "midi_score.mid"
    if not score_midi_path.exists():
        raise FileNotFoundError(f"ASAP score MIDI not found: {score_midi_path}")
    return ClipAlignment(
        asap_piece=asap_piece,
        performance_midi_path=performance_midi_path,
        score_midi_path=score_midi_path,
        performance_beats=tuple(float(b) for b in perf_beats),
        midi_score_beats=tuple(float(b) for b in score_beats),
        midi_score_downbeats=tuple(float(b) for b in score_downbeats),
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_asap_alignment.py -q
```
Expected: PASS (the new test and all existing ones in the file).

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/asap_alignment.py tests/follower_bench/test_asap_alignment.py && git commit -m "feat(follower-bench): expose midi_score_downbeats on ClipAlignment (#118)"
```

---

### Task 2: Backward jump — relock after a repeat/restart
**Group:** 2 (depends on Task 1)

**Behavior being verified:** with bar boundaries and a cheap `jump_back_penalty`, `follow()` relocates its score pointer *backward* to a bar start when the performer replays a passage, producing matches whose `score_index` steps backward; with a high penalty it stays monotonic (false-jump suppression).
**Interface under test:** `follow(perf, score, prior, bar_boundaries=...)`.

**Files:**
- Modify: `model/src/follower_bench/follower.py`
- Test: `model/tests/follower_bench/test_follower.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_follower.py
def test_backward_jump_relocks_after_a_repeat() -> None:
    # Score: bar1 = C4,D4 (60@0,62@1); bar2 = E4,F4 (64@2,65@3). Bars start
    # at score-seconds 0.0 and 2.0 -> columns (0, 2).
    score_notes = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65])]
    bars = bar_boundary_columns([n.position for n in score_notes], [0.0, 2.0])
    # Perf: bar1, bar2, then REPEAT bar1.
    perf_notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.5, pitch=62, velocity=80),
        PerfNote(onset=2.0, offset=2.5, pitch=64, velocity=80),
        PerfNote(onset=3.0, offset=3.5, pitch=65, velocity=80),
        PerfNote(onset=4.0, offset=4.5, pitch=60, velocity=80),
        PerfNote(onset=5.0, offset=5.5, pitch=62, velocity=80),
    ]

    # Cheap backward jump -> the replay is followed back to bar 1.
    prior = ContinuityPrior(skip_penalty=0.5, jump_back_penalty=0.5)
    result = follow(perf_notes, score_notes, prior, bar_boundaries=bars, transpose_candidates=(0,))
    assert len(result.matches) == 6
    assert [m.score_index for m in result.matches] == [0, 1, 2, 3, 0, 1]
    # a backward step in score position exists (the jump)
    assert any(b.score_position < a.score_position
               for a, b in zip(result.matches, result.matches[1:]))
    assert result.unmatched_perf_indices == ()

    # Expensive backward jump -> no jump is profitable, stays monotonic.
    strict = ContinuityPrior(skip_penalty=0.5, jump_back_penalty=10.0)
    mono = follow(perf_notes, score_notes, strict, bar_boundaries=bars, transpose_candidates=(0,))
    assert [m.score_index for m in mono.matches] == [0, 1, 2, 3]
    idx = [m.score_index for m in mono.matches]
    assert idx == sorted(idx)  # monotonic, no jump
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k backward_jump
```
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'jump_back_penalty'` (and/or `follow() got an unexpected keyword argument 'bar_boundaries'`).

- [ ] **Step 3: Implement the minimum to make the test pass**

3a. Extend `ContinuityPrior` (keep the existing docstring, append the two fields):

```python
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
```

3b. Replace `follow()` to accept and forward `bar_boundaries`:

```python
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
```

3c. Replace `_align_at_transpose` with the version below — it adds a `bar_boundaries` parameter, a **backward-only** jump relaxation after each row, and the `"jump"` traceback case. (The forward branch is added in Task 3.)

```python
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
```

3d. Add the row-jump relaxation helper (place it just below `_align_at_transpose`). Backward branch only for now:

```python
def _relax_row_jumps(row, row_back, i, m, bar_boundaries, prior) -> None:
    """Apply at most the single best bar-boundary score-pointer jump to
    row i of the DP (a same-row relocation that consumes no perf note),
    then re-propagate skip_score forward from the jumped cell. At most one
    jump per row keeps traceback acyclic (a jump's source is always a cell
    reached by normal transitions). Backward branch only in this task."""
    neg_inf = -math.inf

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
```

3e. Loosen the two docstrings whose invariant this weakens:
- In `EstimatedTrajectory`, change "monotonic non-decreasing in score_index by construction" to "monotonic non-decreasing in score_index unless a bar-boundary jump was taken (#118)".
- In `teleport_gaps`, change "always >= 0, since matches are monotonic non-decreasing" to "usually >= 0; a bar-boundary jump (#118) yields a negative delta at the jump".

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k backward_jump && uv run pytest tests/follower_bench/ -q
```
Expected: the backward-jump test passes; the full suite (existing 56 + the new Task 1/Task 2 tests) stays green (default `inf` penalties keep every existing test monotonic).

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/follower.py tests/follower_bench/test_follower.py && git commit -m "feat(follower): backward bar-boundary jump for repeat/restart relock (#118)"
```

---

### Task 3: Forward jump — relock after a skipped passage
**Group:** 3 (depends on Task 2)

**Behavior being verified:** with a cheap `jump_fwd_penalty`, `follow()` leaps its pointer *forward* to a later bar the performer jumped to — matching post-skip notes the continuity prior would otherwise refuse (too many skipped notes); with `inf` (default) those notes stay unmatched.
**Interface under test:** `follow(perf, score, prior, bar_boundaries=...)`.

**Files:**
- Modify: `model/src/follower_bench/follower.py`
- Test: `model/tests/follower_bench/test_follower.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_follower.py
def test_forward_jump_relocks_after_a_skipped_passage() -> None:
    # Score: bar1 C4,D4 (60@0,62@1); a long filler bar2 of 8 notes the perf
    # never plays (pitches 70..77 @ 2..9); bar3 G4,A4 (67@10,69@11).
    pitches = [60, 62, 70, 71, 72, 73, 74, 75, 76, 77, 67, 69]
    score_notes = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate(pitches)]
    bars = bar_boundary_columns([n.position for n in score_notes], [0.0, 2.0, 10.0])  # (0, 2, 10)
    # Perf: bar1 then bar3 -- the 8-note bar2 is skipped.
    perf_notes = [
        PerfNote(onset=0.0, offset=0.5, pitch=60, velocity=80),
        PerfNote(onset=1.0, offset=1.5, pitch=62, velocity=80),
        PerfNote(onset=2.0, offset=2.5, pitch=67, velocity=80),
        PerfNote(onset=3.0, offset=3.5, pitch=69, velocity=80),
    ]

    # Cheap forward jump -> leaps over the 8-note gap to bar 3.
    prior = ContinuityPrior(skip_penalty=0.5, jump_fwd_penalty=0.5)
    result = follow(perf_notes, score_notes, prior, bar_boundaries=bars, transpose_candidates=(0,))
    assert len(result.matches) == 4
    assert [m.score_index for m in result.matches] == [0, 1, 10, 11]

    # No forward jump (inf, the default): skipping 8 notes (cost 4.0) to gain
    # 2 matches (+2.0) is refused, so perf notes 2,3 stay unmatched.
    strict = ContinuityPrior(skip_penalty=0.5)  # jump penalties default to inf
    mono = follow(perf_notes, score_notes, strict, bar_boundaries=bars, transpose_candidates=(0,))
    assert [m.score_index for m in mono.matches] == [0, 1]
    assert mono.unmatched_perf_indices == (2, 3)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k forward_jump
```
Expected: FAIL — the cheap-jump assertion fails (`[0, 1, 10, 11] != [0, 1]` or unmatched (2,3)), because `_relax_row_jumps` has no forward branch yet.

- [ ] **Step 3: Implement the minimum to make the test pass**

Extend `_relax_row_jumps` to also compute a prefix max/argmax and add the forward candidate. Replace the whole helper body with:

```python
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_follower.py -q -k "forward_jump or backward_jump" && uv run pytest tests/follower_bench/ -q
```
Expected: forward- and backward-jump tests pass; full suite green.

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/follower.py tests/follower_bench/test_follower.py && git commit -m "feat(follower): forward bar-boundary jump for skip relock (#118)"
```

---

### Task 6: Wire jump penalties + bar boundaries into the gap report
**Group:** 4 (depends on Tasks 1–3 and Task 5)

**Behavior being verified:** the gap report computes each performance's bar boundaries and, given finite `--jump-back-penalty`/`--jump-fwd-penalty`, runs the jump-aware follower — improving repeat/restart/jump relock while keeping clean `false_jumps` at 0. With no penalty args it reproduces the monotonic baseline.
**Interface under test:** `python -m follower_bench.gap_report` CLI (its printed per-pathology report).

**Files:**
- Modify: `model/src/follower_bench/gap_report.py`

- [ ] **Step 1: Write the failing check (run the CLI with penalties — it must not error and must activate jumps)**

```bash
cd model && uv run python -m follower_bench.gap_report --per-composer 1 --seeds 3 --max-score-notes 1800 --workers 6 --jump-back-penalty 1.0 --jump-fwd-penalty 1.0
```
Expected before implementation: FAIL — `error: unrecognized arguments: --jump-back-penalty 1.0 --jump-fwd-penalty 1.0`.

- [ ] **Step 2: Confirm the failure reason**

The `main()` argparse has no jump-penalty flags and `_run_cell` never builds bar boundaries or passes jump penalties — so jumps can't be exercised end-to-end. Proceed to implement.

- [ ] **Step 3: Implement the wiring**

3a. In `gap_report.py` imports, add `bar_boundary_columns` and `load_alignment` is already imported; ensure the follower import line reads:

```python
from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, bar_boundary_columns, follow
```

3b. Replace `_run_cell` to accept bar boundaries + jump penalties and build the prior/boundaries:

```python
def _run_cell(performance: str, pathology: str, seed: int, score_notes: list,
              bar_boundaries: tuple[int, ...], jump_back_penalty: float, jump_fwd_penalty: float) -> RunOutcome:
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY,
                            jump_back_penalty=jump_back_penalty,
                            jump_fwd_penalty=jump_fwd_penalty)
    t0 = time.perf_counter()
    try:
        clip = generate(performance, pathology, seed)
        est = follow(list(clip.notes), score_notes, prior, bar_boundaries=bar_boundaries)
        est_traj = trajectory_from_matches(est.matches)
        score = score_clip(est_traj, clip)
        return RunOutcome(performance, pathology, seed, score, None, time.perf_counter() - t0)
    except Exception as exc:  # loud: recorded, never silently dropped
        return RunOutcome(performance, pathology, seed, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)
```

3c. Replace `_run_performance` to compute bar boundaries from the alignment's downbeats and thread the penalties (task tuple grows):

```python
def _run_performance(task: tuple[str, list[int], int | None, float, float]) -> tuple[list[RunOutcome], dict | None]:
    """Load one performance's score MIDI once, compute its bar-boundary
    columns from the ASAP downbeats, then run all its (pathology, seed)
    cells. Returns (outcomes, skip_record-or-None). Pickle-safe top-level
    function so multiprocessing.Pool can dispatch it. `clean` is
    RNG-invariant so it runs once regardless of seed count. When
    max_score_notes is set, a performance whose score MIDI exceeds it is
    recorded as an explicit skip (the #118 iteration-speed cap). jump
    penalties default to inf upstream, giving the monotonic baseline."""
    perf, seeds, max_score_notes, jump_back_penalty, jump_fwd_penalty = task
    try:
        alignment = load_alignment(perf)
        score_notes = load_score_notes_from_midi(alignment.score_midi_path)
    except Exception as exc:  # loud: recorded as a skip, never silently dropped
        return [], {"performance": perf, "reason": f"{type(exc).__name__}: {exc}"}
    if max_score_notes is not None and len(score_notes) > max_score_notes:
        return [], {"performance": perf,
                    "reason": f"excluded by --max-score-notes cap ({len(score_notes)} > {max_score_notes})"}
    bar_boundaries = bar_boundary_columns([n.position for n in score_notes], alignment.midi_score_downbeats)
    outcomes: list[RunOutcome] = []
    for pathology in PATHOLOGY_TYPES:
        cell_seeds = [seeds[0]] if pathology == "clean" else seeds
        for seed in cell_seeds:
            outcomes.append(_run_cell(perf, pathology, seed, score_notes,
                                      bar_boundaries, jump_back_penalty, jump_fwd_penalty))
    return outcomes, None
```

3d. Replace `run_gap_report` signature + task construction to carry the penalties:

```python
def run_gap_report(
    performances: list[str], seeds: list[int], workers: int = 1, max_score_notes: int | None = None,
    jump_back_penalty: float = math.inf, jump_fwd_penalty: float = math.inf,
) -> dict:
    """Run every (performance, pathology, seed) cell, score it, and
    aggregate per pathology. Parallelizes over performances when
    workers > 1. `clean` is RNG-invariant so it runs once per performance
    regardless of seed count. `max_score_notes` excludes whole
    performances whose score MIDI exceeds the cap (recorded as skips).
    jump_back_penalty / jump_fwd_penalty (default inf = monotonic
    baseline) enable the #118 bar-boundary jumps."""
    outcomes: list[RunOutcome] = []
    skipped: list[dict] = []
    tasks = [(perf, seeds, max_score_notes, jump_back_penalty, jump_fwd_penalty) for perf in performances]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = pool.map(_run_performance, tasks)
    else:
        results = [_run_performance(t) for t in tasks]
    for perf_outcomes, skip in results:
        outcomes.extend(perf_outcomes)
        if skip is not None:
            skipped.append(skip)

    ok = [o for o in outcomes if o.score is not None]
    failed = [o for o in outcomes if o.error is not None]
    aggregates = aggregate_by_pathology([o.score for o in ok])
    return {
        "aggregates": aggregates,
        "outcomes": outcomes,
        "ok": ok,
        "failed": failed,
        "skipped_performances": skipped,
        "n_performances": len(performances),
    }
```

Add `import math` to the top-of-file imports (with the existing `import json`, `import time`).

3e. In `main()`, add the two CLI flags (after `--max-score-notes`) and pass them through:

```python
    ap.add_argument("--jump-back-penalty", type=float, default=None,
                    help="enable #118 backward (repeat/restart) bar-boundary jumps at this penalty (default: off/inf)")
    ap.add_argument("--jump-fwd-penalty", type=float, default=None,
                    help="enable #118 forward (skip) bar-boundary jumps at this penalty (default: off/inf)")
```

and change the `run_gap_report(...)` call to:

```python
    result = run_gap_report(
        performances, seeds, workers=args.workers, max_score_notes=args.max_score_notes,
        jump_back_penalty=args.jump_back_penalty if args.jump_back_penalty is not None else math.inf,
        jump_fwd_penalty=args.jump_fwd_penalty if args.jump_fwd_penalty is not None else math.inf,
    )
```

- [ ] **Step 4: Run the check — verify jumps activate and clean does not regress**

```bash
cd model && uv run pytest tests/follower_bench/ -q
cd model && uv run python -m follower_bench.gap_report --per-composer 1 --seeds 3 --max-score-notes 1800 --workers 6 --jump-back-penalty 1.0 --jump-fwd-penalty 1.0
```
Expected: 58+ tests green. The report runs without error; versus the Task 0 monotonic baseline: `repeat`/`restart` median relock drops well below 50s, `jump` `relock_ok` rises above 0.11, and `clean` `false_jmp` stays **0** (also tempo_swing/wrong_note/hesitation false_jmp stay 0). Exact values are not asserted here — they are the `/autoresearch` target — but a clean-control regression (any `false_jmp > 0`) is a FAIL.

- [ ] **Step 5: Commit**

```bash
cd model && git add src/follower_bench/gap_report.py && git commit -m "feat(follower-bench): wire bar-boundary jump penalties into the gap report (#118)"
```

---

## Post-build: `/autoresearch`

After all tasks are committed and green, `/autoresearch` sweeps `--jump-back-penalty` and `--jump-fwd-penalty` against the #113 metric on the capped subset, maximizing repeat/restart/jump relock subject to the hard constraint **clean/tempo_swing/wrong_note/hesitation false_jumps == 0**. The winning penalties become the follower's defaults for the production path (a follow-up wiring, tracked separately). This plan delivers the mechanism + the two tunable knobs; it does not hard-code final penalty values.
