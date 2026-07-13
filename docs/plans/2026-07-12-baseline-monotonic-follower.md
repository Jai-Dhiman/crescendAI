# Baseline MONOTONIC Follower Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task) where marked parallel.
> Do NOT start execution until `/challenge` returns VERDICT: PROCEED.

**Goal:** `follow(amt_notes, score, prior) -> estimated_trajectory` reproduces the day-0 spike's continuity-penalized monotonic fitting-DP result on the `bach_inv1_chunk0` fixture and documents its expected failure to re-lock on non-monotonic pathologies.
**Spec:** `docs/specs/2026-07-12-baseline-monotonic-follower-design.md`
**Style:** Follow `CLAUDE.md` / `model/CLAUDE.md`. Match `model/src/follower_bench/{segments,trajectory,asap_alignment,pathologies,clip_generator}.py` conventions exactly: `from __future__ import annotations`, frozen dataclasses, module-level docstring explaining the file's one responsibility, public-interface-only tests, one test file per `test_*.py` module under `model/tests/follower_bench/`.

All commands below assume `cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-115-baseline-monotonic-follower/model` unless stated otherwise. Test runner: `uv run python -m pytest tests/follower_bench/ -v` (whole suite) or `uv run python -m pytest tests/follower_bench/test_X.py -v` (single file, used per-task below).

## Task Groups

```
Group 0 (harness, sequential, must complete first): Task 1
Group A (sequential, all touch follower.py): Task 2, Task 3, Task 4, Task 5
Group B (sequential, depends on Group 0 + Group A; both touch new golden-fixture test file): Task 6, Task 7
Group C (parallel with Group B — different files/fixtures, depends only on Group 0 + Group A): Task 8
Group D (sequential, depends on Task 8; all touch the same characterization test file): Task 9, Task 10, Task 11
```

Task 6 depends on Task 7 only in file-ordering (same file, sequential). Group C (Task 8) can be dispatched in parallel with Group B since it touches `score_notes.py` + a new characterization test file, not the golden-fixture test file. Group D must wait for Task 8 (needs `load_score_notes_from_midi`).

**Prerequisite / environment setup (before Group C):** Tasks 8-11 and the
22 pre-existing #111 tests (`test_asap_alignment.py`, `test_clip_generator.py`,
`test_pathologies.py`, `test_trajectory.py`) require the ASAP dataset at
`data/raw/asap-dataset`. It is provisioned as a symlink to the canonical
`data/raw/asap` clone (content-identical) — already done in this worktree.
If it is ever absent (e.g. a fresh worktree), recreate it idempotently from
the repo root:

```bash
[ -e model/data/raw/asap-dataset ] || ln -s asap model/data/raw/asap-dataset
```

(Run from repo root; a worktree's `model/data/raw/asap-dataset` symlink
points at the main checkout's `model/data/raw/asap`, per `model/CLAUDE.md`'s
offload table.) With the symlink in place, `uv run python -m pytest
tests/follower_bench/` reports 33 passed (verified in this worktree).

---

### Task 1: Golden fixture loader (harness)

**Group:** 0 (harness, must run first)

**Behavior being verified:** Loading the real, already-on-disk WASM fixture (`apps/api/src/wasm/score-analysis/tests/fixtures/bach_inv1_chunk0.json`) through a new follower_bench-native adapter produces exactly 82 `PerfNote`s and 458 `ScoreNote`s, with the documented boundary values (first perf onset 0.70s, last perf onset 14.92s, first score note pitch 60 at position 0.1875 — verified directly against the fixture's `score_bars[0].notes[0].onset_seconds`).

**Interface under test:** `load_golden_fixture_notes(json_path: Path) -> tuple[list[PerfNote], list[ScoreNote]]`

**Files:**
- Create: `model/src/follower_bench/score_notes.py`
- Test: `model/tests/follower_bench/test_score_notes.py`

- [x] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_score_notes.py
"""Verify the score-note loaders (golden-fixture JSON, score MIDI) through
their public interface only, against real committed/on-disk fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from follower_bench.score_notes import load_golden_fixture_notes

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_FIXTURE_PATH = (
    REPO_ROOT / "apps/api/src/wasm/score-analysis/tests/fixtures/bach_inv1_chunk0.json"
)


def test_load_golden_fixture_notes_matches_day0_spike_counts() -> None:
    perf_notes, score_notes = load_golden_fixture_notes(GOLDEN_FIXTURE_PATH)

    assert len(perf_notes) == 82
    assert len(score_notes) == 458

    assert perf_notes[0].onset == pytest.approx(0.70)
    assert perf_notes[-1].onset == pytest.approx(14.92)

    assert score_notes[0].pitch == 60
    assert score_notes[0].position == pytest.approx(0.1875)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run python -m pytest tests/follower_bench/test_score_notes.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.score_notes'` (or `ImportError: cannot import name 'load_golden_fixture_notes'`).

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/score_notes.py
"""Score-note representation and loaders for the baseline monotonic
follower (issue #115). Two source formats collapse into one ScoreNote
shape: the WASM score-analysis crate's bar/tick-based fixture JSON (read
in place, not duplicated -- see docs/specs/2026-07-12-baseline-monotonic-
follower-design.md), and a raw score MIDI file via partitura.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from follower_bench.segments import PerfNote


@dataclass(frozen=True)
class ScoreNote:
    """One score note event: MIDI pitch and its position in the score's
    own timeline. `position` is an opaque monotonic label -- seconds for
    a fixed-tempo score render, beats for a partitura-loaded score MIDI --
    follow() never interprets its unit, only compares/reports it."""
    pitch: int
    position: float


def load_golden_fixture_notes(json_path: Path) -> tuple[list[PerfNote], list[ScoreNote]]:
    """Load the WASM score-analysis crate's bach_inv1_chunk0.json fixture,
    returning (perf_notes, score_notes) in follower_bench's own types.
    `perf_notes` entries already match PerfNote's fields exactly (pitch,
    onset, offset, velocity in seconds). `score_notes` are flattened from
    the fixture's per-bar `notes` lists, in bar order, position =
    onset_seconds.

    Raises:
        FileNotFoundError: json_path does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"golden fixture not found: {json_path}")
    data = json.loads(json_path.read_text())

    perf_notes = [
        PerfNote(
            onset=float(n["onset"]),
            offset=float(n["offset"]),
            pitch=int(n["pitch"]),
            velocity=int(n["velocity"]),
        )
        for n in data["perf_notes"]
    ]

    score_notes = [
        ScoreNote(pitch=int(n["pitch"]), position=float(n["onset_seconds"]))
        for bar in data["score_bars"]
        for n in bar["notes"]
    ]

    return perf_notes, score_notes
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_score_notes.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/score_notes.py model/tests/follower_bench/test_score_notes.py
git commit -m "feat(follower-bench): load_golden_fixture_notes reproduces day-0 spike note counts (#115)"
```

---

### Task 2: Basic fitting-DP match on a tiny synthetic example

**Group:** A (sequential — first `follower.py` slice)

**Behavior being verified:** With the continuity prior disabled (`NO_PRIOR`) and a fixed transpose of 0, `follow()` matches each performance note to the score note with the same pitch that appears in score order, and leaves a performance note unmatched if no matching pitch exists in the score at all.

**Interface under test:** `follow(amt_notes, score_notes, prior, transpose_candidates=(0,)) -> EstimatedTrajectory`

**Files:**
- Create: `model/src/follower_bench/follower.py`
- Test: `model/tests/follower_bench/test_follower.py`

- [x] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_follower.py
"""Verify follow() (the continuity-penalized monotonic fitting-DP with
inner transpose search) through its public interface only, on small
hand-built synthetic examples -- see test_follower_golden_fixture.py for
the real-fixture reproduction test and test_follower_characterization.py
for the required-to-fail pathology tests."""
from __future__ import annotations

from follower_bench.follower import NO_PRIOR, follow
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
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.follower'`

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/follower.py
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
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower.py
git commit -m "feat(follower-bench): follow() basic fitting-DP matches notes in score order (#115)"
```

---

### Task 3: `teleport_gaps()` reports consecutive-match position deltas

**Group:** A (sequential, depends on Task 2 — same file)

**Behavior being verified:** `teleport_gaps()` returns the `score_position` delta between each pair of consecutively matched notes, in match order, without needing to run `follow()` (pure function over an `EstimatedTrajectory`).

**Interface under test:** `teleport_gaps(trajectory: EstimatedTrajectory) -> list[float]`

**Files:**
- Modify: `model/src/follower_bench/follower.py`
- Modify: `model/tests/follower_bench/test_follower.py`

- [x] **Step 1: Write the failing test**

```python
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
```

Add to the test file's imports:

```python
from follower_bench.follower import EstimatedTrajectory, MatchedNote, teleport_gaps
```

(replaces the Task 2 import line, adding `EstimatedTrajectory, MatchedNote, teleport_gaps`)

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: FAIL — `ImportError: cannot import name 'teleport_gaps'`

- [x] **Step 3: Implement the minimum to make the test pass**

Append to `model/src/follower_bench/follower.py`:

```python
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
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower.py
git commit -m "feat(follower-bench): teleport_gaps reports consecutive-match position deltas (#115)"
```

---

### Task 4: Continuity prior refuses a teleport that would unlock more matches

**Group:** A (sequential, depends on Task 3 — same file)

**Behavior being verified:** When matching a distant coincidental pitch would unlock a longer run of subsequent matches (more total matches overall than staying local), `NO_PRIOR` lets `follow()` take that trade and "teleport" to the far note (a large `score_position` gap between consecutive matches, per `teleport_gaps()`). A nonzero `ContinuityPrior` makes the trip not worth it: it stops matching once the correct local continuation runs out, rather than pay to reach the distant run.

**Interface under test:** `follow(amt_notes, score_notes, prior, transpose_candidates=(0,))`, `teleport_gaps()`

**Files:**
- Modify: `model/src/follower_bench/follower.py` (no change expected — this task should pass against Task 2's implementation as written; if it does not, fix `_align_at_transpose` per Step 3 below)
- Modify: `model/tests/follower_bench/test_follower.py`

- [x] **Step 1: Write the failing test**

```python
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
```

Add `ContinuityPrior` to the test file's existing `follower_bench.follower` import line (already importing `NO_PRIOR, follow` from Task 2 and `EstimatedTrajectory, MatchedNote, teleport_gaps` from Task 3).

- [x] **Step 2: Run test — verify it FAILS or confirms the behavior is already correct**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: this should PASS against Task 2's `_align_at_transpose` exactly
as written, with no further code change — the corrected recurrence
(leading skips charged via `B[0][j] = B[0][j-1] - prior.skip_penalty`,
not free) already makes an unnecessarily-late match strictly worse than
an early one, which is what makes the `NO_PRIOR` vs. `ContinuityPrior`
cases resolve differently. If it does NOT pass, inspect the DP trace: the
likely cause is the `skip_perf` transition's `>=` comparison (`if cand >=
best_val`) being checked in the wrong order relative to `skip_score`'s
`>` — verify the checked order in `_align_at_transpose` matches Task 2's
code exactly (skip_score, then skip_perf, then match, each only
overwriting the running best on its own comparator).

- [x] **Step 3: Implement the minimum to make the test pass**

Only needed if Step 2 failed; the fix is localized to
`_align_at_transpose`'s transition ordering/comparators in `follower.py`.
No new public interface.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower.py
git commit -m "test(follower-bench): continuity prior refuses a teleport that would unlock more matches (#115)"
```

---

### Task 5: Transpose search auto-detects a semitone shift

**Group:** A (sequential, depends on Task 4 — same file)

**Behavior being verified:** When every performance note is a fixed semitone shift away from its true score pitch, `follow()`'s default `transpose_candidates` auto-detects that shift (more matches at the correct transpose than at 0), without the caller specifying a sign or magnitude.

**Interface under test:** `follow(amt_notes, score_notes, prior)` (default `transpose_candidates`)

**Files:**
- Modify: `model/tests/follower_bench/test_follower.py`

- [x] **Step 1: Write the failing test**

```python
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
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: at transpose=0, no perf note's pitch matches any score note's
pitch (all off by 1), so `follow()` with only `transpose_candidates=(0,)`
would find 0 matches; with the default candidate set already implemented
in Task 2, this test should actually PASS already if `_align_at_transpose`
and the outer search are correct. Run it to confirm — if it fails, the bug
is in the outer `follow()` selection loop (`key = (len(result.matches),
-abs(t))` comparison), not in `_align_at_transpose`.

- [x] **Step 3: Implement the minimum to make the test pass**

If Step 2 revealed a bug in the outer search's `key` comparison or
`transpose_candidates` default, fix it directly in `follow()` in
`follower.py`. No new public interface.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower.py
git commit -m "test(follower-bench): follow() auto-detects a semitone transpose via the inner search (#115)"
```

---

### Task 6: Golden-fixture reproduction

**Group:** B (sequential, depends on Group 0 Task 1 + Group A Tasks 2-5)

**Behavior being verified:** Running `follow()` on the real `bach_inv1_chunk0` fixture with the continuity prior enabled reproduces the day-0 spike's structural result: an auto-detected ±1 semitone transpose, zero teleports, and a match count within the documented tolerance band around the historical 62/82.

**Interface under test:** `follow(amt_notes, score_notes, prior)` end-to-end on real data; `teleport_gaps()`

**Files:**
- Create: `model/tests/follower_bench/test_follower_golden_fixture.py`

- [x] **Step 1: Write the failing test**

```python
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

from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, follow, teleport_gaps
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
```

- [x] **Step 2: Run test — verify it FAILS or confirms the already-verified result**

```bash
uv run python -m pytest tests/follower_bench/test_follower_golden_fixture.py -v
```
Expected: PASS, with `DEFAULT_SKIP_PENALTY = 0.5` exactly as set in Task 2
— this was verified directly against the real fixture during planning
(not merely predicted). If it does NOT pass, one of two things happened:
1. The build agent's `_align_at_transpose`/`follow()` deviates from
   Task 2/3/4/5's exact code (re-diff against the plan's code blocks).
2. A genuine environment difference (e.g. a different `json.loads`
   float-parsing edge case) shifted a tie-break — inspect
   `result.matches` and `result.transpose_semitones` directly (temporary
   `print`, not committed) and adjust `DEFAULT_SKIP_PENALTY` only if the
   investigation shows the DP itself is behaving correctly and the exact
   count of 62 is not reachable; do not silently loosen
   `EXPECTED_MATCH_COUNT` into a tolerance band without first
   understanding why the verified number changed.

- [x] **Step 3: Fix only if Step 2 failed**

If Step 2 failed for reason 1 above, fix `follower.py` to match Task
2-5's code exactly. If for reason 2, document the actual observed
values and the adjusted constant with a one-line comment above
`DEFAULT_SKIP_PENALTY` explaining the deviation from the planning-time
verification.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower_golden_fixture.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower_golden_fixture.py
git commit -m "test(follower-bench): follow() reproduces day-0 spike transpose + zero-teleport result on golden fixture (#115)"
```

---

### Task 7: Continuity-prior ablation regresses teleports

**Group:** B (sequential, depends on Task 6 — same file)

**Behavior being verified:** Running the same golden fixture with `NO_PRIOR` (the prior disabled) reintroduces multiple teleports — proving the continuity prior is load-bearing. Verified empirically during planning: `NO_PRIOR` on this fixture yields 65 matches with 5 gaps exceeding `TELEPORT_THRESHOLD_S`, max gap 16.69 -- same order of magnitude as the day-0 spike's "teleports_without_prior = 3, max 6.9s" (not bit-exact, since the lost implementation's specific tie-breaking cannot be recovered — see spec Open Questions), but the qualitative and structural claim (removing the prior reintroduces teleports) reproduces exactly.

**Interface under test:** `follow(amt_notes, score_notes, NO_PRIOR)`; `teleport_gaps()`

**Files:**
- Modify: `model/tests/follower_bench/test_follower_golden_fixture.py`

- [x] **Step 1: Write the failing test**

```python
def test_no_prior_regresses_to_multiple_teleports_on_golden_fixture() -> None:
    perf_notes, score_notes = load_golden_fixture_notes(GOLDEN_FIXTURE_PATH)

    result = follow(perf_notes, score_notes, NO_PRIOR)

    gaps = teleport_gaps(result)
    teleport_count = sum(1 for g in gaps if g > TELEPORT_THRESHOLD_S)
    assert teleport_count >= 1
    assert max(gaps, default=0.0) > 5.0  # same order of magnitude as the day-0 spike's 6.9s max, not exact
```

Add `NO_PRIOR` to the file's existing `follower_bench.follower` import line.

- [x] **Step 2: Run test — verify it PASSES as verified during planning**

```bash
uv run python -m pytest tests/follower_bench/test_follower_golden_fixture.py -v
```
Expected: PASS. This was directly verified during planning against the
real fixture: `NO_PRIOR` yields `teleport_count == 5` and `max(gaps) ==
16.69` (both comfortably clearing this test's `>= 1` and `> 5.0`
thresholds). If it does NOT pass (no-prior run also has zero teleports,
i.e. the DP behaves identically regardless of the prior), that means
`_align_at_transpose`'s skip transition is not actually being exercised
differently between `skip_penalty=0.0` and `skip_penalty=
DEFAULT_SKIP_PENALTY` on this real data — investigate and fix the DP's
skip-cost application in `follower.py` (this would indicate the prior
isn't wired into the recurrence correctly, a real bug, not a tuning
issue). Re-diff against Task 2's exact code first.

- [x] **Step 3: Implement the minimum to make the test pass**

Only needed if Step 2 failed; the fix is localized to
`_align_at_transpose`'s skip-transition cost application.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower_golden_fixture.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/follower.py model/tests/follower_bench/test_follower_golden_fixture.py
git commit -m "test(follower-bench): continuity prior is load-bearing -- NO_PRIOR regresses to a teleport on golden fixture (#115)"
```

---

### Task 8: Score-MIDI loader for ASAP characterization fixtures

**Group:** C (parallel with Group B — different files; depends only on Group 0 + Group A)

**Behavior being verified:** `load_score_notes_from_midi()` loads a real ASAP score MIDI file (via partitura) into `ScoreNote`s with `position` in the same beat units as `TrueTrajectory`'s anchors (needed so characterization tests in Task 9-11 can compare `follow()`'s output against `clip.true_trajectory` directly).

**Interface under test:** `load_score_notes_from_midi(path: Path) -> list[ScoreNote]`

**Files:**
- Modify: `model/src/follower_bench/score_notes.py`
- Modify: `model/tests/follower_bench/test_score_notes.py`

- [x] **Step 1: Write the failing test**

```python
def test_load_score_notes_from_midi_matches_real_asap_score() -> None:
    from follower_bench.asap_alignment import load_alignment
    from follower_bench.score_notes import load_score_notes_from_midi

    alignment = load_alignment("Liszt/Transcendental_Etudes/1/LuoJ05M.mid")
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    assert len(score_notes) > 0
    positions = [n.position for n in score_notes]
    assert positions == sorted(positions)
    assert positions[0] >= 0.0
    assert positions[-1] <= alignment.midi_score_beats[-1] + 1.0
```

Add the new import (`load_score_notes_from_midi`) to `test_score_notes.py`'s existing import line, alongside `load_golden_fixture_notes`.

- [x] **Step 2: Run test — verify it FAILS**

```bash
uv run python -m pytest tests/follower_bench/test_score_notes.py -v
```
Expected: FAIL — `ImportError: cannot import name 'load_score_notes_from_midi'`

- [x] **Step 3: Implement the minimum to make the test pass**

Append to `model/src/follower_bench/score_notes.py` (add `import partitura as pa` to the top-of-file imports alongside the existing `json`/`dataclasses`/`pathlib` imports):

```python
def load_score_notes_from_midi(path: Path) -> list[ScoreNote]:
    """Load a score MIDI file's notes via partitura, sorted by onset
    beat. `position` is in score beats -- the same unit as
    follower_bench.trajectory.TrueTrajectory's anchors -- so characterization
    tests can compare follow()'s output directly against a clip's
    true_trajectory."""
    spart = pa.load_score_midi(str(path))
    note_array = spart.note_array()
    notes = [ScoreNote(pitch=int(row["pitch"]), position=float(row["onset_beat"])) for row in note_array]
    notes.sort(key=lambda n: n.position)
    return notes
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_score_notes.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/src/follower_bench/score_notes.py model/tests/follower_bench/test_score_notes.py
git commit -m "feat(follower-bench): load_score_notes_from_midi loads real ASAP score notes in beat units (#115)"
```

---

### Task 9: Characterization — jump pathology fails to re-lock

**Group:** D (sequential, depends on Task 8)

**Behavior being verified:** On a `SynthClip` with an injected forward `jump` (score position abruptly skips ahead), `follow()`'s implied trajectory diverges measurably from `clip.true_trajectory` shortly after the jump — documenting that a monotonic-only follower does not re-lock across a jump (the gap #118 closes).

**Interface under test:** `follow()`, `follower_bench.clip_generator.generate()`, `follower_bench.trajectory.TrueTrajectory` (reused directly, not modified)

**Files:**
- Create: `model/tests/follower_bench/test_follower_characterization.py`

- [x] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_follower_characterization.py
"""Characterization tests (issue #115, epic #108): document that the
baseline MONOTONIC follower is EXPECTED and SUPPOSED to fail to re-lock
after jump/repeat/restart pathologies. These are not bugs -- monotonic-
by-construction means the follower cannot represent a backward score
jump at all, and the continuity prior actively discourages the large
forward jump a `jump` pathology requires. This documents the gap that
#118 (jump-aware follower) closes."""
from __future__ import annotations

from follower_bench.asap_alignment import load_alignment
from follower_bench.clip_generator import generate
from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, follow
from follower_bench.score_notes import load_score_notes_from_midi
from follower_bench.trajectory import TrueTrajectory

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"
DIVERGENCE_THRESHOLD_BEATS = 2.0
PROBE_DELAY_S = 3.0


def test_follow_fails_to_relock_after_a_jump() -> None:
    clip = generate(ALIGNED_PIECE, "jump", seed=11)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_BEATS
```

- [x] **Step 2: Run test — verify it FAILS or confirms the expected failure exists**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: this asserts a FAILURE MODE exists (divergence), so a PASS here
means the characterization is correctly documented. If the assertion
fails (i.e. `follow()` unexpectedly tracks the jump correctly, divergence
below threshold), inspect `result.matches` around the jump — this would
be a surprising result worth investigating (e.g. the jump lands on a
coincidental nearby match that happens to look locked), not a reason to
weaken the test. Adjust the seed or `DIVERGENCE_THRESHOLD_BEATS` only if
investigation shows the test construction itself is flawed (e.g. the
jump distance is too small to be a meaningful pathology), never to force
a pass without understanding why.

- [x] **Step 3: No implementation change expected**

This task only adds a test proving already-implemented (Groups 0/A/B)
behavior. If Step 2 required an adjustment, document it here; otherwise
this step is empty by design (the point of a characterization test is
that the code under test does not change).

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_follower_characterization.py
git commit -m "test(follower-bench): characterize follow()'s expected failure to re-lock after a jump (#115)"
```

---

### Task 10: Characterization — repeat pathology fails to re-lock

**Group:** D (sequential, depends on Task 9 — same file)

**Behavior being verified:** On a `SynthClip` with an injected `repeat` (score position jumps backward, then replays forward), `follow()` cannot represent the backward jump at all (monotonic by construction) and its implied trajectory diverges from `clip.true_trajectory` shortly after the repeat event.

**Interface under test:** same as Task 9, `pathology_type="repeat"`

**Files:**
- Modify: `model/tests/follower_bench/test_follower_characterization.py`

- [x] **Step 1: Write the failing test**

```python
def test_follow_fails_to_relock_after_a_repeat() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_BEATS
```

- [x] **Step 2: Run test — verify it FAILS or confirms the expected failure exists**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: same reasoning as Task 9 Step 2, applied to the `repeat`
pathology (a backward score jump, structurally impossible for a
monotonic-by-construction follower to represent — the divergence here
should be, if anything, easier to trigger than the `jump` case).

- [x] **Step 3: No implementation change expected**

Empty by design unless Step 2 investigation reveals a test construction
flaw (see Task 9 Step 3's guidance).

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_follower_characterization.py
git commit -m "test(follower-bench): characterize follow()'s expected failure to re-lock after a repeat (#115)"
```

---

### Task 11: Characterization — restart pathology fails to re-lock

**Group:** D (sequential, depends on Task 10 — same file)

**Behavior being verified:** On a `SynthClip` with an injected `restart` (performer stops and restarts from an earlier point), `follow()`'s implied trajectory diverges from `clip.true_trajectory` shortly after the restart event, for the same structural reason as `repeat`.

**Interface under test:** same as Task 9, `pathology_type="restart"`

**Files:**
- Modify: `model/tests/follower_bench/test_follower_characterization.py`

- [x] **Step 1: Write the failing test**

```python
def test_follow_fails_to_relock_after_a_restart() -> None:
    clip = generate(ALIGNED_PIECE, "restart", seed=17)
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)

    result = follow(list(clip.notes), score_notes, ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY))
    estimated = TrueTrajectory(
        anchors=tuple((m.perf_time, m.score_position) for m in result.matches)
    )

    assert len(clip.event_labels) == 1
    probe_time = clip.event_labels[0].perf_time + PROBE_DELAY_S

    true_position = clip.true_trajectory.score_position_at(probe_time)
    estimated_position = estimated.score_position_at(probe_time)

    assert abs(estimated_position - true_position) > DIVERGENCE_THRESHOLD_BEATS
```

- [x] **Step 2: Run test — verify it FAILS or confirms the expected failure exists**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: same reasoning as Tasks 9-10, applied to `restart`.

- [x] **Step 3: No implementation change expected**

Empty by design unless Step 2 investigation reveals a test construction
flaw.

- [x] **Step 4: Run test — verify it PASSES**

```bash
uv run python -m pytest tests/follower_bench/test_follower_characterization.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_follower_characterization.py
git commit -m "test(follower-bench): characterize follow()'s expected failure to re-lock after a restart (#115)"
```

---

## Final verification (after all tasks)

```bash
cd model && uv run python -m pytest tests/follower_bench/ -v
```
Expected: with the ASAP dataset symlink in place (see the Prerequisite /
environment setup note above Group C), the pre-existing #111 suite
(`test_asap_alignment.py`, `test_clip_generator.py`, `test_package.py`,
`test_pathologies.py`, `test_segments.py`, `test_trajectory.py`) is a
33-passing baseline in this worktree — confirmed, not merely predicted.
Tasks 8-11 build on top of that baseline: after this plan's new files
(`test_score_notes.py`, `test_follower.py`, `test_follower_golden_fixture.py`,
`test_follower_characterization.py`) are added, the full suite should pass
in its entirety.

## Challenge Review

### CEO Pass

**Premise Challenge.** Right problem, real pain: epic #108's later work
(#118 jump-aware, #119 HMM, #120 WASM port) genuinely has no committed
baseline to extend or regression-test against — the day-0 spike's source
is lost. This plan is a spec-driven reimplementation against recovered
acceptance numbers, not gold-plating. No simpler framing beats "reimplement
the DP against the recovered numbers" — the alternative (skip straight to
#118's jump-aware follower) would leave #118 without a monotonic baseline
to diff against, which is exactly the gap this issue exists to close.

**Scope Check.** Matches the spec goal exactly: two new modules
(`score_notes.py`, `follower.py`), four test files, no drift beyond what
`docs/specs/2026-07-12-baseline-monotonic-follower-design.md` describes.
6 files touched, 2 new modules — under the "8 files / 2 services" complexity
smell threshold. The hardest problem (the continuity-penalized DP
recurrence and its interaction with the transpose search) is being solved
directly, not avoided — Tasks 2-5's synthetic tests specifically target the
DP's trickiest edge case (the free-leading-skip vs. charged-internal-skip
distinction that prevents "free anchor reset").

**Twelve-Month Alignment.**
```
CURRENT STATE                  THIS PLAN                    12-MONTH IDEAL
follower_bench has #111's  →   adds follow() + a committed →  #118 jump-aware
clip/pathology harness but     monotonic baseline,             follower extends/
no follower under test         characterization tests          regression-tests
(day-0 spike lost)              documenting its limits          against this baseline
```
Moves toward the ideal; no tech debt created that conflicts with it — the
spec explicitly scopes jump-awareness, HMM, and WASM port out to sibling
issues rather than smuggling them in here.

**Alternatives Check.** The spec does not enumerate alternative DP
formulations (e.g., a banded/windowed DP, a greedy nearest-pitch matcher)
kept as `[QUESTION]` per the skill's default — but this is low-stakes since
the day-0 spike already empirically validated this exact algorithm class
(symbolic fitting-DP) against the audio-chroma alternative (12x lift), so
re-litigating algorithm choice here would be redundant with #108's already-
settled decision.

### Engineering Pass

**Architecture.** Data flow: `load_golden_fixture_notes` /
`load_score_notes_from_midi` → `follow()` (DP + transpose search) →
`teleport_gaps()` (post-hoc diagnostic). Traced against the actual fixture
and actual `partitura` API (see verification below) — the approach matches
how the code works, not an assumed shape. `follower_bench.segments.PerfNote`
and `follower_bench.trajectory.TrueTrajectory` are reused unmodified exactly
as the spec claims (read `segments.py`/`trajectory.py` directly — the field
names line up with no adapter needed). No security-relevant data flow (no
SQL/shell/LLM-prompt injection surface — pure numeric DP over local files).
No N+1/fan-out concerns — the DP is O(N·M) run at most 5 times (bounded,
82×458 on the golden fixture).

**Module Depth.** `score_notes.py`: 3 exports (`ScoreNote`,
`load_golden_fixture_notes`, `load_score_notes_from_midi`), hides real
bar/tick-iteration and partitura note-array indexing — DEEP. `follower.py`:
6 exports, hides an O(N·M) two-layer DP table + backtracking run per
transpose candidate — DEEP. Matches the spec's own verdicts; no shallow
modules introduced.

**Code Quality / Test Philosophy / Vertical Slice.** All tests exercise
public interfaces only (`follow()`, `teleport_gaps()`, the two loaders) —
no internal-collaborator mocking, no DP-table inspection. Each task is one
test → one (possibly no-op, explicitly justified) implementation → one
commit; Tasks 4/5/7/9/10/11 correctly flag "no implementation change
expected" as characterization/ablation tests of already-built behavior
rather than deferred implementation — this is a legitimate TDD pattern here
(the point of a characterization test is that the code doesn't change), not
horizontal slicing.

**Verification performed this review (not just re-reading the plan's
claims):**
1. Loaded the real fixture directly: confirmed 82 `perf_notes`
   (onset 0.70s→14.92s) and 458 total `score_bars[*].notes` — matches the
   plan's counts exactly.
2. Implemented Task 2-5's exact DP code from the plan's code blocks in a
   scratch script and ran it against the real fixture with
   `skip_penalty=0.5`: got `transpose=-1, matches=62, teleport_count(>2.0s)=0,
   max_gap=0.1875` — an exact match to Task 6's claimed numbers. With
   `NO_PRIOR`: got `matches=65, teleport_count=5, max_gap=16.6875` — matches
   Task 7's claimed `16.69` (rounding). **The DP algorithm and its claimed
   golden-fixture numbers are real, not fabricated or optimistic
   extrapolation.**
3. Checked `partitura`'s actual API: `pa.load_score_midi(path)` returns a
   `partitura.score.Score`, and `Score.note_array()` works directly
   (verified against a real MIDI file on disk) and does include an
   `onset_beat` field alongside `pitch` — Task 8's implementation is
   API-compatible with the installed `partitura` version.
4. **Checked Task 1's harness test against the actual fixture bytes** —
   see BLOCKER below.
5. **Checked whether the ASAP dataset Tasks 8-11 depend on is present
   locally** — see BLOCKER below.

**Test Coverage.**
```
[+] model/src/follower_bench/score_notes.py
    │
    ├── load_golden_fixture_notes()
    │   ├── [TESTED] ★★  counts + boundary values — Task 1 (BUT: one
    │   │                assertion is factually wrong against real data,
    │   │                see BLOCKER 1)
    │   └── [GAP]        FileNotFoundError path — docstring documents it,
    │                    no test exercises it (RISK, non-critical: this is
    │                    a research harness, not a user-facing path)
    │
    └── load_score_notes_from_midi()
        └── [TESTED] ★★  real ASAP score, sortedness, boundary — Task 8
                         (BUT: cannot execute in this environment, see
                         BLOCKER 2)

[+] model/src/follower_bench/follower.py
    │
    ├── follow()
    │   ├── [TESTED] ★★★ basic match + unmatched — Task 2
    │   ├── [TESTED] ★★★ continuity-prior teleport refusal — Task 4
    │   ├── [TESTED] ★★★ transpose auto-detection — Task 5
    │   ├── [TESTED] ★★★ golden-fixture reproduction — Task 6
    │   └── [TESTED] ★★★ NO_PRIOR ablation regression — Task 7
    │
    └── teleport_gaps()
        └── [TESTED] ★★  consecutive-match deltas — Task 3
```

**Failure Modes.** All failure modes here are research-harness-appropriate
(loud `FileNotFoundError`/`ValueError`/`AsapAlignmentMissingError`, no
silent fallbacks) and consistent with CLAUDE.md's "explicit exception
handling over silent fallbacks" rule. No async operations, no database
writes, no partial-state/rollback concerns — this module is pure,
deterministic, and stateless.

---

**[BLOCKER 1]** (confidence: 10/10) — Task 1's golden-fixture test asserts
`score_notes[0].position == pytest.approx(0.0)`, but the real fixture's
first score note (`score_bars[0].notes[0]`, pitch 60) has
`onset_seconds: 0.1875`, not `0.0` (verified directly: `python3 -c
"import json; d=json.load(open('apps/api/src/wasm/score-analysis/tests/
fixtures/bach_inv1_chunk0.json')); print(d['score_bars'][0]['notes'][0])"`
→ `{'pitch': 60, ..., 'onset_seconds': 0.1875, ...}`). The pitch assertion
(60) is correct; the position assertion is not. This test as written will
FAIL at Step 4 ("verify it PASSES") against real data, contradicting the
plan's claim that this was "verified this session" (Verification
Architecture section: "verified this session: 82 perf_notes with onsets
0.70s-14.92s, 458 total notes... an exact match"). The note-count and
perf-note-boundary numbers ARE verified correct — only the
`score_notes[0].position` boundary value is wrong. **Fix:** change the
assertion to `pytest.approx(0.1875)` in Task 1's test code block before
building.

**[BLOCKER 2]** (confidence: 9/10) — The ASAP dataset that Task 8's test
and Tasks 9-11's characterization tests depend on
(`data/raw/asap-dataset/`, referenced via `asap_alignment.DEFAULT_ASAP_ROOT`)
is **not present in this worktree** (`ls data/raw/asap-dataset` →
"No such file or directory"). This is not a new problem this plan
introduces — running the existing #111 suite right now already shows 22
pre-existing failures, all `FileNotFoundError: ASAP annotations file not
found`, across `test_asap_alignment.py`, `test_clip_generator.py`,
`test_pathologies.py`, `test_trajectory.py` (only 11 pass). But the plan's
Task 8 test (`load_score_notes_from_midi` against
`Liszt/Transcendental_Etudes/1/LuoJ05M.mid`) and Tasks 9-11's three
characterization tests all call `load_alignment()` / `clip_generator.generate()`
against this same missing data, and the plan's own "Final verification"
section asserts "all tests pass, including the pre-existing #111 tests" —
a claim that is currently false and unaddressed anywhere in the plan. Group
C and Group D (4 of 11 tasks, plus the load-bearing characterization tests
that are the plan's second acceptance criterion alongside the golden
fixture) cannot be verified to pass without first acquiring the ASAP
dataset (`git clone https://github.com/CPJKU/asap-dataset.git` into
`data/raw/asap-dataset`, per `model/CLAUDE.md`'s offload table's own regen
instructions for this exact path). This is a plain `git clone`, not the R2
`rclone` offload path the user explicitly said to leave alone — so
acquiring it does not conflict with that instruction — but the plan
neither mentions this prerequisite nor includes a step to acquire it.
**Fix:** add an explicit setup step (before Group C) that clones ASAP into
`data/raw/asap-dataset` if absent, or descope Tasks 8-11 to a separate
follow-up if ASAP acquisition is out of scope for this session, and correct
the Final Verification section's claim to reflect the current (currently
false) 22-pre-existing-failure baseline.

**[RISK]** (confidence: 5/10) — Task 8's assumption that `partitura`'s
`onset_beat` (from `Score.note_array()`) is on the exact same beat scale as
ASAP's `midi_score_beats` annotations (used by `TrueTrajectory` and
Tasks 9-11's probe comparisons) is plausible but unverified in this review
— it could not be checked without the ASAP dataset (BLOCKER 2). If the
scales differ (e.g., a pickup-measure offset or a different beat-zero
convention), Tasks 9-11's `DIVERGENCE_THRESHOLD_BEATS` comparisons could
silently compare mismatched units. Fallback: once ASAP is available, spot-
check `load_score_notes_from_midi(alignment.score_midi_path)` positions
against a handful of `alignment.midi_score_beats` anchors before trusting
the characterization tests' probe logic.

**[OBS]** — The DP recurrence, transpose search, and golden-fixture/
ablation numbers were independently re-derived from the plan's exact code
blocks in this review (not merely re-read) and matched the plan's claims
exactly. This is the strongest part of the plan — the algorithm work is
sound and the historical-number reproduction claims are genuine, not
optimistic rounding.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `score_notes[0].position == 0.0` on the golden fixture | RISKY (confirmed wrong) | First note's `onset_seconds` is 0.1875, verified directly against the fixture (BLOCKER 1) |
| Golden-fixture DP numbers (62/82, transpose ±1, 0 teleports; NO_PRIOR 65 matches, 5 teleports, max 16.69s) | SAFE | Independently re-implemented and re-run against the real fixture in this review; matched exactly |
| `partitura.load_score_midi(...).note_array()` returns `onset_beat`/`pitch` fields directly on the `Score` object | SAFE | Verified against a real MIDI file with the installed partitura version |
| ASAP dataset (`data/raw/asap-dataset`) is present/acquirable for Tasks 8-11 | RISKY (confirmed absent) | Directory does not exist; 22 pre-existing #111 tests already fail for this reason (BLOCKER 2) |
| `partitura` `onset_beat` units match ASAP's `midi_score_beats` scale | VALIDATE | Could not verify without the ASAP dataset; plausible but unconfirmed (RISK above) |
| No new user-facing security/scaling surface (pure local-file DP) | SAFE | Read all touched files; no SQL/shell/LLM/network I/O introduced |

### Summary

[BLOCKER] count: 2
[RISK]    count: 2
[QUESTION] count: 1

VERDICT: NEEDS_REWORK — (1) Task 1's `score_notes[0].position == pytest.approx(0.0)` assertion is factually wrong against the real fixture (actual value 0.1875) and will fail as written; (2) Tasks 8-11 depend on the ASAP dataset at `data/raw/asap-dataset`, which is absent from this worktree (confirmed: 22 pre-existing #111 tests already fail for the same reason) and the plan has no step to acquire it, making Group C/D unexecutable and the Final Verification section's "all tests pass" claim currently false.
