# Follower Trajectory Metric Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Score a symbolic score-follower's estimated trajectory against a `SynthClip`'s exact ground truth, producing per-clip and per-pathology-type numbers (position error, lock rate, relock latency, false-jump count).
**Spec:** docs/specs/2026-07-12-follower-trajectory-metric-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md). `from __future__ import annotations`, frozen dataclasses, explicit exceptions, no emojis, no mocking of internal collaborators — tests exercise `score_clip` / `aggregate_by_pathology` / `trajectory_from_matches` only.

## File Structure

| File | Responsibility | Interface | Depth | New / Modify |
|------|----------------|-----------|-------|--------------|
| `model/src/follower_bench/metric.py` | Score an estimated `TrueTrajectory` against a `SynthClip`'s ground truth on a common time grid; aggregate scores by pathology type; adapt follower `MatchedNote`s into a `TrueTrajectory`. | `TrajectoryScore`, `AggregateScore`, `score_clip()`, `aggregate_by_pathology()`, `trajectory_from_matches()` | DEEP | New |
| `model/tests/follower_bench/test_metric.py` | Behavior tests through the public interface only, incl. one real-`follower.follow()` integration slice. | test functions | — | New |

No other files change. `metric.py` imports `follower_bench.trajectory.TrueTrajectory`, `follower_bench.clip_generator.SynthClip`, `follower_bench.follower.MatchedNote` (type only, for the adapter signature) — it does **not** import `follower_bench.follower`'s `follow`/`ContinuityPrior`/`DEFAULT_SKIP_PENALTY`. Only `test_metric.py`'s final integration test imports those.

## Task Groups

Group A (sequential — every task edits `model/src/follower_bench/metric.py` and/or `model/tests/follower_bench/test_metric.py`, each building on the previous task's committed state): Tasks 1-9, in order. No parallelism — this is one cohesive deep module built as one continuously-growing file.

`[SHIPS INDEPENDENTLY]`: N/A — this whole plan is one deep module; it has no useful partial-ship point below "all 9 tasks done" (an incomplete `score_clip` that can't compute relock latency or false-jump counts is not usable as #115's acceptance evidence or #118's regression baseline).

---

### Task 1: `score_clip` computes position error, lock rate, false-jump count, and relock latency for the identity case

**Group:** A (first task)

**Behavior being verified:** Scoring a `SynthClip`'s own ground truth against itself (`estimated == clip.true_trajectory`) is a perfect score on every dimension: zero position error, full lock rate, zero false jumps, and a near-zero relock latency for the clip's one position-changing event (a `repeat` clip has exactly one).

**Interface under test:** `score_clip(estimated: TrueTrajectory, clip: SynthClip) -> TrajectoryScore`

**Files:**
- Create: `model/src/follower_bench/metric.py`
- Create: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/follower_bench/test_metric.py
"""Verify score_clip / aggregate_by_pathology / trajectory_from_matches
through their public interface only, using real #111 clips
(clip_generator.generate) and hand-built TrueTrajectory estimates for the
core measurement behaviors, plus one real-follower.follow() integration
slice."""
from __future__ import annotations

import math

import pytest

from follower_bench.clip_generator import generate
from follower_bench.metric import SAMPLE_HZ, score_clip

ALIGNED_PIECE = "Liszt/Transcendental_Etudes/1/LuoJ05M.mid"


def test_score_clip_identity_estimate_is_a_perfect_score() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)

    score = score_clip(clip.true_trajectory, clip)

    assert score.pathology_type == "repeat"
    assert score.median_abs_error_beats == pytest.approx(0.0)
    assert score.max_abs_error_beats == pytest.approx(0.0)
    assert score.lock_rate == pytest.approx(1.0)
    assert score.false_jump_count == 0

    assert len(clip.event_labels) == 1
    assert clip.event_labels[0].from_score_position != clip.event_labels[0].to_score_position
    assert len(score.relock_latencies_s) == 1
    latency = score.relock_latencies_s[0]
    assert 0.0 <= latency < 1.0 / SAMPLE_HZ
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'follower_bench.metric'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/metric.py
"""Score a symbolic score-follower's estimated trajectory against a
SynthClip's exact ground truth (issue #113): position error, lock rate,
relock latency, and false-jump count, sampled on a common uniform time
grid over the clip's true-trajectory time span. Hides grid construction,
interpolation via TrueTrajectory.score_position_at, event-relative
relock search, and backward-move detection guarded by truth-
monotonicity. Does NOT import follower_bench.follower -- the metric
stays follower-agnostic; trajectory_from_matches is the one adapter
point a caller uses to bridge follow()'s output in.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from follower_bench.clip_generator import SynthClip
from follower_bench.trajectory import TrueTrajectory

SAMPLE_HZ = 20.0
POSITION_TOL_BEATS = 0.5
FALSE_JUMP_BEATS = 1.0


@dataclass(frozen=True)
class TrajectoryScore:
    """One clip's measurement: position error (beats), lock rate,
    relock latency per position-changing event (seconds, math.inf if the
    estimate never re-locks before the clip ends), and false-jump
    count."""
    pathology_type: str
    median_abs_error_beats: float
    max_abs_error_beats: float
    lock_rate: float
    relock_latencies_s: tuple[float, ...]
    false_jump_count: int


def _sample_grid(t_min: float, t_max: float, sample_hz: float) -> list[float]:
    """Uniform time grid over [t_min, t_max] at ~sample_hz, always
    including both endpoints exactly."""
    duration = t_max - t_min
    n_steps = max(1, round(duration * sample_hz))
    return [t_min + i * duration / n_steps for i in range(n_steps + 1)]


def score_clip(
    estimated: TrueTrajectory,
    clip: SynthClip,
    *,
    sample_hz: float = SAMPLE_HZ,
    position_tol_beats: float = POSITION_TOL_BEATS,
    false_jump_beats: float = FALSE_JUMP_BEATS,
) -> TrajectoryScore:
    """Score estimated against clip.true_trajectory on a uniform grid
    over the true trajectory's own time span (the clip's real
    duration -- an estimate that starts/ends elsewhere is fairly
    penalized since score_position_at clamps outside its own anchors)."""
    true = clip.true_trajectory
    t_min = true.anchors[0][0]
    t_max = true.anchors[-1][0]
    times = _sample_grid(t_min, t_max, sample_hz)
    true_positions = [true.score_position_at(t) for t in times]
    est_positions = [estimated.score_position_at(t) for t in times]
    errors = [abs(e - t) for e, t in zip(est_positions, true_positions)]

    median_abs_error_beats = statistics.median(errors)
    max_abs_error_beats = max(errors)
    lock_rate = sum(1 for e in errors if e <= position_tol_beats) / len(errors)

    false_jump_count = 0
    for i in range(1, len(times)):
        true_non_decreasing = true_positions[i] >= true_positions[i - 1]
        backward_move = est_positions[i - 1] - est_positions[i]
        if true_non_decreasing and backward_move > false_jump_beats:
            false_jump_count += 1

    relock_latencies_s: list[float] = []
    for event in clip.event_labels:
        if event.from_score_position == event.to_score_position:
            continue
        latency = math.inf
        for t, e in zip(times, errors):
            if t >= event.perf_time and e <= position_tol_beats:
                latency = t - event.perf_time
                break
        relock_latencies_s.append(latency)

    return TrajectoryScore(
        pathology_type=clip.pathology_type,
        median_abs_error_beats=median_abs_error_beats,
        max_abs_error_beats=max_abs_error_beats,
        lock_rate=lock_rate,
        relock_latencies_s=tuple(relock_latencies_s),
        false_jump_count=false_jump_count,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/metric.py model/tests/follower_bench/test_metric.py && git commit -m "feat(follower-bench): score_clip identity case (issue #113)"
```

---

### Task 2: `score_clip` reports exact position error and a degraded lock rate for a constant-offset estimate

**Group:** A (depends on Task 1)

**Behavior being verified:** An estimate whose score position is uniformly offset from truth by a constant amount produces exactly that offset as both median and max error (linear interpolation commutes with a constant shift), and a lock rate of 0.0 when the offset exceeds `position_tol_beats`.

**Interface under test:** `score_clip`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
# append to model/tests/follower_bench/test_metric.py, alongside the existing imports:
# from follower_bench.trajectory import TrueTrajectory   <- add this import near the top

def test_score_clip_constant_offset_estimate_reports_exact_offset_as_error() -> None:
    clip = generate(ALIGNED_PIECE, "clean", seed=1)
    offset = 2.0
    shifted = TrueTrajectory(
        anchors=tuple((t, p + offset) for t, p in clip.true_trajectory.anchors)
    )

    score = score_clip(shifted, clip)

    assert score.median_abs_error_beats == pytest.approx(offset)
    assert score.max_abs_error_beats == pytest.approx(offset)
    assert score.lock_rate == pytest.approx(0.0)
    assert score.false_jump_count == 0
    assert score.relock_latencies_s == ()
```

Add `from follower_bench.trajectory import TrueTrajectory` to the top-level imports in `test_metric.py`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_score_clip_constant_offset_estimate_reports_exact_offset_as_error -q
```
Expected: FAIL if `_sample_grid` or the lock-rate/error computation has an off-by-one or boundary-clamping bug (the assertions require an *exact* match, not just "some" degradation) — `NameError: name 'TrueTrajectory' is not defined` until the import is added, then an `AssertionError` if the grid/interpolation math is wrong.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change is expected: Task 1's `score_clip` is already a general grid-sampling comparison (it does not special-case the identity input), so this test exercises the same code with a new, independently-computable scenario. If the assertions fail, fix `_sample_grid` / the error computation in `model/src/follower_bench/metric.py` so this test's exact values hold without breaking Task 1's test.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): score_clip constant-offset error/lock-rate case (issue #113)"
```

---

### Task 3: `score_clip` reports `math.inf` relock latency when the estimate never recovers after a backward event

**Group:** A (depends on Task 2)

**Behavior being verified:** An estimate that matches truth exactly up to a `repeat` event's `perf_time` and then freezes (never continues) never comes back within `position_tol_beats` of truth — the north-star #115 characterization ("a monotonic follower re-locks after a forward jump but never after a backward repeat/restart"), now expressed as `relock_latencies_s == (math.inf,)`.

**Interface under test:** `score_clip`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_clip_relock_latency_is_inf_when_estimate_never_recovers() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    event = clip.event_labels[0]
    assert event.from_score_position != event.to_score_position

    # Freeze the estimate at whatever it was tracking right at the event's
    # perf_time -- a stand-in for a monotonic follower that cannot
    # represent the backward jump and simply stops progressing.
    frozen_anchors = tuple(
        (t, p) for t, p in clip.true_trajectory.anchors if t <= event.perf_time
    )
    estimated = TrueTrajectory(anchors=frozen_anchors)

    score = score_clip(estimated, clip)

    assert len(score.relock_latencies_s) == 1
    assert score.relock_latencies_s[0] == math.inf
```

If seed=13 does not produce a frozen position far enough from the clip's final score position for this to hold (unlikely given `_pick_two_points` places the repeat's `from_score_position` at 55-75% of the piece, well short of the end), try nearby seeds — see `test_follower_characterization.py`'s precedent for empirical seed selection on this exact piece.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_score_clip_relock_latency_is_inf_when_estimate_never_recovers -q
```
Expected: FAIL if the relock search loop in `score_clip` has a bug that lets it return a finite (spuriously "successful") latency instead of exhausting to `math.inf` — this is the first test in the suite to exercise an estimate that genuinely never re-locks, since Task 1's identity case always relocks near-instantly.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change is expected — Task 1's relock search loop already returns `math.inf` when no grid sample at/after `event.perf_time` is within tolerance. If this fails, fix the loop in `score_clip` (`model/src/follower_bench/metric.py`) so it does not terminate early or misidentify a match.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): score_clip inf relock latency for a non-recovering estimate (issue #113)"
```

---

### Task 4: `score_clip` reports a finite relock latency when the estimate re-converges after a delay

**Group:** A (depends on Task 3)

**Behavior being verified:** An estimate that matches truth up to a `repeat` event, then reconnects exactly to truth's position `N` seconds later, produces a *finite* relock latency bounded by `N` plus one grid step — proving the relock search correctly finds and reports an early termination, not just correctly returning `inf` in the failure case (Task 3).

**Interface under test:** `score_clip`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_clip_relock_latency_is_finite_when_estimate_recovers() -> None:
    clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    event = clip.event_labels[0]
    n_seconds = 2.0
    reconnect_time = event.perf_time + n_seconds

    pre = [(t, p) for t, p in clip.true_trajectory.anchors if t <= event.perf_time]
    post = [(t, p) for t, p in clip.true_trajectory.anchors if t >= reconnect_time]
    reconnect_pos = clip.true_trajectory.score_position_at(reconnect_time)
    estimated = TrueTrajectory(
        anchors=tuple(pre) + ((reconnect_time, reconnect_pos),) + tuple(post)
    )

    score = score_clip(estimated, clip)

    assert len(score.relock_latencies_s) == 1
    latency = score.relock_latencies_s[0]
    assert math.isfinite(latency)
    assert latency <= n_seconds + 1.0 / SAMPLE_HZ
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_score_clip_relock_latency_is_finite_when_estimate_recovers -q
```
Expected: FAIL if the relock search does not terminate at the first in-tolerance sample at/after `event.perf_time` (e.g. it keeps searching past the reconnect point, or measures latency from the wrong origin).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change is expected — this exercises the same search loop from Task 1/3 with a scenario guaranteed by construction to hit an exact match at `reconnect_time`. If it fails, fix the loop's break condition or latency computation (`latency = t - event.perf_time`) in `model/src/follower_bench/metric.py`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): score_clip finite relock latency for a recovering estimate (issue #113)"
```

---

### Task 5: `score_clip` counts a backward teleport on a monotonic clip as a false jump

**Group:** A (depends on Task 4)

**Behavior being verified:** On a `clean` clip (truth is monotonic non-decreasing throughout, no injected discontinuities), an estimate that jumps backward by more than `FALSE_JUMP_BEATS` between consecutive grid samples is counted as a false jump — catching a follower that teleports on a coincidental pitch match (the day-0 "3 teleports without the continuity prior" failure mode) rather than a real discontinuity, which is excluded because the guard requires truth to be non-decreasing across that step.

**Interface under test:** `score_clip`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_clip_false_jump_count_detects_a_backward_teleport() -> None:
    clip = generate(ALIGNED_PIECE, "clean", seed=1)
    true = clip.true_trajectory
    assert true.is_monotonic_non_decreasing() is True

    mid_idx = len(true.anchors) // 2
    t_mid, pos_mid = true.anchors[mid_idx]
    teleport_time = t_mid + 0.3
    teleport_pos = pos_mid - (FALSE_JUMP_BEATS + 2.0)

    estimated = TrueTrajectory(
        anchors=true.anchors[: mid_idx + 1] + ((teleport_time, teleport_pos),)
    )

    score = score_clip(estimated, clip)

    assert score.false_jump_count >= 1
```

Add `FALSE_JUMP_BEATS` to the `from follower_bench.metric import (...)` line at the top of `test_metric.py`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_score_clip_false_jump_count_detects_a_backward_teleport -q
```
Expected: FAIL if the false-jump backward-move comparison or the truth-monotonicity guard in `score_clip` is wrong (e.g. comparing the wrong pair of consecutive samples, or not guarding on truth's own direction at all).

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change is expected — Task 1's false-jump loop already compares consecutive grid samples and guards on `true_positions[i] >= true_positions[i - 1]`. If it fails, fix that loop in `model/src/follower_bench/metric.py`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): score_clip false-jump detection on a monotonic clip (issue #113)"
```

---

### Task 6: `aggregate_by_pathology` groups scores by type and computes correct per-group statistics

**Group:** A (depends on Task 5)

**Behavior being verified:** Given a mix of hand-built `TrajectoryScore`s across two pathology types, `aggregate_by_pathology` groups them correctly and computes: `n_clips` per group; `median_abs_error_beats` as the median of the group's per-clip medians; `mean_lock_rate` as the mean of per-clip lock rates; `relock_success_rate` as (finite latencies) / (all latencies) in the group, defined as `1.0` when the group has zero position-changing events; `median_relock_latency_s` as the median of only the finite latencies (excluded when computing the median, but counted against `relock_success_rate`), `0.0` when the group has zero events, and `math.inf` when the group has events but none succeeded; `total_false_jumps` as the sum of `false_jump_count` across the group.

**Interface under test:** `aggregate_by_pathology(scores: Iterable[TrajectoryScore]) -> dict[str, AggregateScore]`

**Files:**
- Modify: `model/src/follower_bench/metric.py`
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_aggregate_by_pathology_groups_scores_and_computes_stats() -> None:
    repeat_scores = (
        TrajectoryScore(
            pathology_type="repeat",
            median_abs_error_beats=0.1,
            max_abs_error_beats=0.3,
            lock_rate=0.8,
            relock_latencies_s=(2.0,),
            false_jump_count=0,
        ),
        TrajectoryScore(
            pathology_type="repeat",
            median_abs_error_beats=0.3,
            max_abs_error_beats=0.5,
            lock_rate=0.6,
            relock_latencies_s=(math.inf,),
            false_jump_count=1,
        ),
    )
    clean_score = TrajectoryScore(
        pathology_type="clean",
        median_abs_error_beats=0.0,
        max_abs_error_beats=0.0,
        lock_rate=1.0,
        relock_latencies_s=(),
        false_jump_count=0,
    )

    result = aggregate_by_pathology(repeat_scores + (clean_score,))

    assert set(result.keys()) == {"repeat", "clean"}

    repeat_agg = result["repeat"]
    assert repeat_agg.n_clips == 2
    assert repeat_agg.median_abs_error_beats == pytest.approx(0.2)
    assert repeat_agg.mean_lock_rate == pytest.approx(0.7)
    assert repeat_agg.relock_success_rate == pytest.approx(0.5)
    assert repeat_agg.median_relock_latency_s == pytest.approx(2.0)
    assert repeat_agg.total_false_jumps == 1

    clean_agg = result["clean"]
    assert clean_agg.n_clips == 1
    assert clean_agg.median_abs_error_beats == pytest.approx(0.0)
    assert clean_agg.mean_lock_rate == pytest.approx(1.0)
    assert clean_agg.relock_success_rate == pytest.approx(1.0)
    assert clean_agg.median_relock_latency_s == pytest.approx(0.0)
    assert clean_agg.total_false_jumps == 0
```

Update the top-of-file import in `test_metric.py` to also pull in `AggregateScore`, `TrajectoryScore`, `aggregate_by_pathology` from `follower_bench.metric`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_aggregate_by_pathology_groups_scores_and_computes_stats -q
```
Expected: FAIL — `ImportError: cannot import name 'AggregateScore' from 'follower_bench.metric'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# add to model/src/follower_bench/metric.py

from typing import Iterable


@dataclass(frozen=True)
class AggregateScore:
    """Per-pathology-type rollup of TrajectoryScores."""
    n_clips: int
    median_abs_error_beats: float
    mean_lock_rate: float
    relock_success_rate: float
    median_relock_latency_s: float
    total_false_jumps: int


def aggregate_by_pathology(scores: Iterable[TrajectoryScore]) -> dict[str, AggregateScore]:
    """Group scores by pathology_type and compute per-group stats.
    relock_success_rate is 1.0 for a group with zero position-changing
    events (vacuously perfect); median_relock_latency_s excludes
    math.inf entries, is 0.0 for a group with zero events, and is
    math.inf for a group that had events but none succeeded."""
    by_type: dict[str, list[TrajectoryScore]] = {}
    for score in scores:
        by_type.setdefault(score.pathology_type, []).append(score)

    result: dict[str, AggregateScore] = {}
    for pathology_type, group in by_type.items():
        all_latencies = [lat for s in group for lat in s.relock_latencies_s]
        finite_latencies = [lat for lat in all_latencies if math.isfinite(lat)]

        if not all_latencies:
            relock_success_rate = 1.0
            median_relock_latency_s = 0.0
        else:
            relock_success_rate = len(finite_latencies) / len(all_latencies)
            median_relock_latency_s = (
                statistics.median(finite_latencies) if finite_latencies else math.inf
            )

        result[pathology_type] = AggregateScore(
            n_clips=len(group),
            median_abs_error_beats=statistics.median(s.median_abs_error_beats for s in group),
            mean_lock_rate=statistics.mean(s.lock_rate for s in group),
            relock_success_rate=relock_success_rate,
            median_relock_latency_s=median_relock_latency_s,
            total_false_jumps=sum(s.false_jump_count for s in group),
        )
    return result
```

Move the `from typing import Iterable` import to the top-of-file import block alongside the existing `math`/`statistics`/`dataclasses` imports rather than inline.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all six tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/metric.py model/tests/follower_bench/test_metric.py && git commit -m "feat(follower-bench): aggregate_by_pathology (issue #113)"
```

---

### Task 7: `trajectory_from_matches` adapts follower output into a `TrueTrajectory`, sorted by perf_time

**Group:** A (depends on Task 6)

**Behavior being verified:** `trajectory_from_matches` builds a `TrueTrajectory` from a tuple of `MatchedNote`s, sorting anchors by `perf_time` regardless of input order (a follower's `matches` are monotonic in `score_index`, not necessarily pre-sorted the way this adapter needs).

**Interface under test:** `trajectory_from_matches(matches: tuple[MatchedNote, ...]) -> TrueTrajectory`

**Files:**
- Modify: `model/src/follower_bench/metric.py`
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trajectory_from_matches_sorts_anchors_by_perf_time() -> None:
    matches = (
        MatchedNote(perf_index=2, score_index=1, perf_time=1.0, score_position=0.6),
        MatchedNote(perf_index=0, score_index=0, perf_time=0.2, score_position=0.1),
        MatchedNote(perf_index=1, score_index=2, perf_time=0.5, score_position=0.4),
    )

    traj = trajectory_from_matches(matches)

    assert traj.anchors == ((0.2, 0.1), (0.5, 0.4), (1.0, 0.6))
```

Add `from follower_bench.follower import MatchedNote` and `trajectory_from_matches` (to the `follower_bench.metric` import line) at the top of `test_metric.py`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_trajectory_from_matches_sorts_anchors_by_perf_time -q
```
Expected: FAIL — `ImportError: cannot import name 'trajectory_from_matches' from 'follower_bench.metric'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# add to model/src/follower_bench/metric.py, at the bottom.
# Note: this is a type-only reference to MatchedNote -- metric.py does
# NOT import follower_bench.follower's follow/ContinuityPrior/
# DEFAULT_SKIP_PENALTY, keeping the metric follower-agnostic.

from follower_bench.follower import MatchedNote


def trajectory_from_matches(matches: tuple[MatchedNote, ...]) -> TrueTrajectory:
    """Adapt a follower's EstimatedTrajectory.matches into a
    TrueTrajectory, so score_clip stays follower-agnostic."""
    anchors = tuple(sorted((m.perf_time, m.score_position) for m in matches))
    return TrueTrajectory(anchors=anchors)
```

Move the `from follower_bench.follower import MatchedNote` import to the top-of-file import block. This deliberately omits the empty-input guard — Task 8 adds it with its own failing test.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all seven tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/follower_bench/metric.py model/tests/follower_bench/test_metric.py && git commit -m "feat(follower-bench): trajectory_from_matches adapter (issue #113)"
```

---

### Task 8: `trajectory_from_matches` raises `ValueError` on empty matches

**Group:** A (depends on Task 7)

**Behavior being verified:** A follower that matched nothing cannot be scored — `trajectory_from_matches(())` fails loud with `ValueError` rather than silently producing a zero-length `TrueTrajectory` that would later crash or misbehave inside `score_clip`'s `anchors[0]` / `anchors[-1]` access.

**Interface under test:** `trajectory_from_matches`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trajectory_from_matches_raises_on_empty_matches() -> None:
    with pytest.raises(ValueError):
        trajectory_from_matches(())
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_trajectory_from_matches_raises_on_empty_matches -q
```
Expected: FAIL — `Failed: DID NOT RAISE <class 'ValueError'>`. Task 7's `trajectory_from_matches` has no empty-input guard, so `trajectory_from_matches(())` currently returns `TrueTrajectory(anchors=())` silently instead of raising.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/follower_bench/metric.py -- replace the trajectory_from_matches
# body added in Task 7 with:

def trajectory_from_matches(matches: tuple[MatchedNote, ...]) -> TrueTrajectory:
    """Adapt a follower's EstimatedTrajectory.matches into a
    TrueTrajectory, so score_clip stays follower-agnostic.

    Raises:
        ValueError: matches is empty (a follower that matched nothing
            cannot be scored -- fail loud, not a silent zero-length
            trajectory).
    """
    if not matches:
        raise ValueError("cannot build TrueTrajectory from empty matches")
    anchors = tuple(sorted((m.perf_time, m.score_position) for m in matches))
    return TrueTrajectory(anchors=anchors)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all eight tests)

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): trajectory_from_matches rejects empty matches (issue #113)"
```

---

### Task 9: Real-follower integration — `follower.follow()` scores well on a clean clip and never re-locks on a repeat clip

**Group:** A (depends on Task 8)

**Behavior being verified:** The shipped baseline follower (#115), scored through `trajectory_from_matches` + `score_clip`, reproduces its own characterization as a *number*: a high `lock_rate` on a `clean` clip, and `math.inf` relock latency on a `repeat` clip's backward event. This also empirically validates that `MatchedNote.score_position` and `TrueTrajectory`'s anchors share the same score-beats unit — if they didn't, the `clean`-clip `lock_rate` would collapse, failing this test loudly rather than silently mis-scoring.

**Interface under test:** `score_clip` + `trajectory_from_matches`, driven by the real `follower.follow()`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
LOCK_RATE_FLOOR = 0.5


def test_real_follower_locks_well_on_clean_and_never_relocks_on_repeat() -> None:
    alignment = load_alignment(ALIGNED_PIECE)
    score_notes = load_score_notes_from_midi(alignment.score_midi_path)
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY)

    clean_clip = generate(ALIGNED_PIECE, "clean", seed=1)
    clean_result = follow(list(clean_clip.notes), score_notes, prior)
    clean_estimated = trajectory_from_matches(clean_result.matches)
    clean_score = score_clip(clean_estimated, clean_clip)
    assert clean_score.lock_rate > LOCK_RATE_FLOOR

    repeat_clip = generate(ALIGNED_PIECE, "repeat", seed=13)
    repeat_result = follow(list(repeat_clip.notes), score_notes, prior)
    repeat_estimated = trajectory_from_matches(repeat_result.matches)
    repeat_score = score_clip(repeat_estimated, repeat_clip)
    assert len(repeat_score.relock_latencies_s) == 1
    assert repeat_score.relock_latencies_s[0] == math.inf
```

Add these imports to the top of `test_metric.py`:
```python
from follower_bench.asap_alignment import load_alignment
from follower_bench.follower import DEFAULT_SKIP_PENALTY, ContinuityPrior, follow
from follower_bench.score_notes import load_score_notes_from_midi
```

This is the only place `follower_bench.follower`'s `follow`/`ContinuityPrior`/`DEFAULT_SKIP_PENALTY` are imported in this test file (the metric module itself never imports them).

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py::test_real_follower_locks_well_on_clean_and_never_relocks_on_repeat -q
```
Expected: FAIL — `NameError: name 'ContinuityPrior' is not defined` until the imports are added; if it still fails after adding imports, it means either the unit-alignment assumption (`MatchedNote.score_position` vs. `TrueTrajectory` anchors, both score-beats) does not hold, or seed=13's repeat event does happen to re-lock before the clip ends (unexpected given `test_follower_characterization.py`'s existing divergence evidence for this exact seed) — in either case this is a real finding, not a plan bug: report it rather than loosening the assertion.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production code change is expected in `model/src/follower_bench/metric.py` — this test only composes the already-shipped `follower.follow()` (#115) with `trajectory_from_matches` and `score_clip` (Tasks 1-8). Add the imports listed in Step 1.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/follower_bench/test_metric.py -q
```
Expected: PASS (all nine tests in `test_metric.py`, one per task across Tasks 1-9)

```bash
cd model && uv run pytest tests/follower_bench/ -q
```
Expected: PASS — the full `tests/follower_bench/` suite stays green (54 passed: the 45 pre-existing tests observed before this plan, plus this plan's 9 new tests in `test_metric.py`).

- [ ] **Step 5: Commit**

```bash
git add model/tests/follower_bench/test_metric.py && git commit -m "test(follower-bench): real-follower integration slice for score_clip (issue #113)"
```

---

## Verification Architecture (from spec)

- **Canonical success state:** `score_clip(clip.true_trajectory, clip)` (estimate == truth) returns zero error, `lock_rate == 1.0`, near-zero relock latency, `false_jump_count == 0` — verified in Task 1.
- **Automated check:** `cd model && uv run pytest tests/follower_bench/test_metric.py` (and the full `tests/follower_bench/` suite must stay green — verified at the end of Task 8).
- **Real-follower integration:** verified in Task 8.
- **Harness:** none needed beyond the tests (per spec — no golden fixture files; #111's `generate()` supplies real clips).
