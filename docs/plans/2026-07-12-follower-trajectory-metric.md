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

**Behavior being verified:** An estimate that matches truth exactly up to a `jump` event's `perf_time` and then freezes (never continues) never comes back within `position_tol_beats` of truth, because truth leaps forward and never revisits `from_score_position` — expressed as `relock_latencies_s == (math.inf,)`. (BUILD AMENDMENT: `jump`, not `repeat` — see the note below; the shipped follower actually recovers on `repeat`/`restart`.)

**Interface under test:** `score_clip`

**Files:**
- Modify: `model/tests/follower_bench/test_metric.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_clip_relock_latency_is_inf_when_estimate_never_recovers() -> None:
    clip = generate(ALIGNED_PIECE, "jump", seed=13)
    event = clip.event_labels[0]
    assert event.from_score_position != event.to_score_position

    # Freeze the estimate at whatever it was tracking right at the event's
    # perf_time -- a stand-in for a follower that stops progressing at a
    # forward jump and never catches up to the leapt-ahead score position
    # (truth moves forward and never revisits from_score_position, so the
    # frozen estimate never re-enters tolerance -> relock latency is inf).
    frozen_anchors = tuple(
        (t, p) for t, p in clip.true_trajectory.anchors if t <= event.perf_time
    )
    estimated = TrueTrajectory(anchors=frozen_anchors)

    score = score_clip(estimated, clip)

    assert len(score.relock_latencies_s) == 1
    assert score.relock_latencies_s[0] == math.inf
```

BUILD AMENDMENT (empirically verified during build): the plan originally used `"repeat"` here, on the belief that a monotonic follower "never re-locks after a backward repeat." That belief is FALSE for this metric and this follower — a `repeat`/`restart` clip's truth trajectory replays FORWARD back through the frozen estimate's `from_score_position`, so a frozen estimate registers a (correct) *finite* relock (~14-22s), and the shipped follower genuinely recovers on `repeat`/`restart` in ~5-7s (error decays monotonically to lock and stays locked for ~75% of the post-event region). The pathology that genuinely never recovers under this metric is `"jump"` (a forward skip: truth leaps ahead and never revisits `from_score_position`, so a frozen estimate → `inf` for every seed 1-300 tested; the real follower → `inf` too). Task 3 and Task 9 therefore use `"jump"` for the never-recovers assertion. `metric.py` is correct and unchanged — this is a test-vehicle correction only.

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

### Task 9: Real-follower integration — `follower.follow()` scores well on a clean clip and never re-locks on a jump clip

**Group:** A (depends on Task 8)

**Behavior being verified:** The shipped baseline follower (#115), scored through `trajectory_from_matches` + `score_clip`, reproduces its own characterization as a *number*: a high `lock_rate` on a `clean` clip, and `math.inf` relock latency on a `jump` clip's forward-skip event (the follower can never catch up to a leapt-ahead score position). This also empirically validates that `MatchedNote.score_position` and `TrueTrajectory`'s anchors share the same score-MIDI-seconds unit — if they didn't, the `clean`-clip `lock_rate` would collapse, failing this test loudly rather than silently mis-scoring. (BUILD AMENDMENT: originally `repeat`; the shipped follower actually recovers on `repeat`/`restart` in ~5-7s, so only `jump` yields the `inf` never-recovers signal.)

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

    # BUILD AMENDMENT: use "jump" (forward skip) not "repeat" for the
    # never-recovers case. Empirically the shipped follower DOES recover
    # after a backward repeat/restart (~5-7s, then stays locked), so a
    # repeat clip yields a finite relock latency; only a forward jump the
    # follower can never catch up to yields inf.
    jump_clip = generate(ALIGNED_PIECE, "jump", seed=1)
    jump_result = follow(list(jump_clip.notes), score_notes, prior)
    jump_estimated = trajectory_from_matches(jump_result.matches)
    jump_score = score_clip(jump_estimated, jump_clip)
    assert len(jump_score.relock_latencies_s) == 1
    assert jump_score.relock_latencies_s[0] == math.inf
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

## Challenge Review

### CEO Pass

**Premise Challenge.** Real pain, verified against actual code: `model/tests/follower_bench/test_follower_characterization.py` (read in full) has three near-identical 20-line blocks (`test_follow_fails_to_relock_after_a_jump/repeat/restart`) that hand-roll `TrueTrajectory(anchors=tuple((m.perf_time, m.score_position) for m in result.matches))` and a bespoke "probe at `event.perf_time + 3.0`, assert divergence > 2.0" check. This is exactly the pattern the spec names as needing a reusable scorer for #118. The plan does not touch that file (spec says "no other files change"), so the duplication survives this issue — that's an intentional scope cut, not an oversight, but it means the payoff (retiring the bespoke assertions) is deferred to whenever #118 or a follow-up actually swaps them onto `trajectory_from_matches`/`score_clip`. Worth a one-line follow-up note, not a blocker.

**Existing coverage.** `TrueTrajectory.score_position_at` (interpolate+clamp) and `is_monotonic_non_decreasing` already exist in `trajectory.py` and are correctly reused rather than reinvented — no duplication of interpolation logic.

**Scope Check.** Two files, three new symbols (`TrajectoryScore`, `AggregateScore`, plus the module) — well under the 8-file/2-class complexity trigger. The hardest problem (relock-latency search across an event-relative window, with correct `inf` semantics) is not avoided — Tasks 3/4 specifically target the inf/finite split. No scope drift from the spec: File Changes table, constants, and function signatures in the plan match the spec exactly.

**12-Month Alignment.** Current state: two point solutions (`clip_generator.generate` ground truth, `follower.follow` estimate) with no scorer between them, characterized only by ad hoc test assertions. This plan: adds the scorer as the one seam both #115 (already shipped) and #118 (next) attach to. 12-month ideal: a benchmark harness that runs any follower over #111's clip corpus and reports `aggregate_by_pathology` tables as regression gates. This plan is a direct, non-debt-incurring step toward that — no rework implied for #118 to adopt it.

**Alternatives Check.**
```
[QUESTION] — The spec documents no alternative to grid-sampling + score_position_at
             interpolation (e.g., comparing at each estimate/truth anchor's own event
             boundary, or a DTW-style alignment cost instead of a fixed-grid position
             error). The spec's "why sample on a grid" paragraph justifies the chosen
             approach on its own merits but doesn't record what was rejected. Low
             stakes here (the approach is standard and well-reasoned), so this is
             informational, not a blocker.
```

### Engineering Pass

**Architecture.** Data flow is linear and matches the actual code read in full: `clip_generator.generate()` → `SynthClip.true_trajectory` (verified: `TrueTrajectory` from `trajectory.py`, anchors are `(perf_time_seconds, position)` pairs) + `follower.follow()` → `EstimatedTrajectory.matches` (verified: `MatchedNote(perf_index, score_index, perf_time, score_position)` in `follower.py`) → `trajectory_from_matches()` adapts matches to `TrueTrajectory` → `score_clip()` compares on a grid → `aggregate_by_pathology()` rolls up. No security surface (no user input, no I/O beyond existing MIDI loads already covered by `clip_generator`/`asap_alignment`). No N+1 or fan-out risk — grid size is bounded by `SAMPLE_HZ * clip_duration` (~600-1200 samples for the reference piece), computed once per `score_clip` call.

**Unit-alignment verification (the plan's own Open Question, checked against source, not assumed).** Read `score_notes.py` and `asap_alignment.py` in full: `MatchedNote.score_position` comes from `ScoreNote.position`, and `load_score_notes_from_midi`'s docstring states directly: *"`position` is in score-MIDI seconds — the same unit as `follower_bench.trajectory.TrueTrajectory`'s anchors (which come from ASAP's `midi_score_beats`, themselves beat TIMES in score-MIDI seconds)."* So the plan's assumption holds — but the actual shared unit is **score-MIDI seconds, not beats**, despite the field being named `midi_score_beats` and the plan/spec's constants being named `POSITION_TOL_BEATS` / `FALSE_JUMP_BEATS` and documented as "half a beat" / "one score-beat." This is a pre-existing naming legacy from #111/#115 (not introduced by this plan), and it doesn't cause a functional bug (both sides of every comparison share the same actual unit, so the metric is self-consistent) — but the plan propagates the misleading "beats" framing into new module constants and their docstrings without flagging the mismatch.
```
[RISK] (confidence: 7/10) — POSITION_TOL_BEATS=0.5 and FALSE_JUMP_BEATS=1.0 are
       documented (spec Open Questions) as "half a beat" / "one score-beat," but the
       actual shared unit between TrueTrajectory anchors and MatchedNote.score_position
       is score-MIDI seconds (per load_score_notes_from_midi's own docstring), not
       musical beats. Not a functional bug — self-consistent within the metric — but
       a future reader tuning POSITION_TOL_BEATS by ear ("half a beat feels right")
       will be tuning the wrong quantity by an unknown tempo-dependent factor. Fallback:
       none needed to ship; recommend a one-line comment in metric.py's docstring
       clarifying the actual unit is score-MIDI-seconds (inherited naming), not beats.
```

**Grid-step rounding vs. Task 1's latency bound.** `_sample_grid`'s step size is `duration / round(duration * sample_hz)`, which is provably >= `1/sample_hz` whenever rounding goes down (confirmed numerically against the actual `ALIGNED_PIECE`/seed=13 repeat clip: duration=60.522s → step=0.0500185s > 1/20=0.05s exactly). Task 1 asserts `latency < 1.0 / SAMPLE_HZ` with zero slack against that. I ran the actual scenario: measured latency for this exact seed is 0.0137s, ~3.6x under the 0.05s bound — so the test passes comfortably today, this is not a live bug. But the bound is not mathematically guaranteed by construction; it happens to hold for this specific seed/piece by a wide margin.
```
[RISK] (confidence: 4/10) — Task 1's `assert 0.0 <= latency < 1.0 / SAMPLE_HZ` has
       no formal guarantee (grid step can exceed 1/SAMPLE_HZ by ~0.2% due to
       round()'s rounding-down cases) and empirically passes only because the
       specific seed=13 event happens to land far from a grid boundary. If a future
       task changes ALIGNED_PIECE, the seed, or SAMPLE_HZ, this exact assertion could
       flake without warning. Verified passing today by direct calculation (measured
       latency 0.0137s vs bound 0.05s) — no action required now. Fallback if it ever
       flakes: loosen to `latency <= step_size` or `latency < 2.0 / SAMPLE_HZ`.
```

**Module Depth Audit.**
- `model/src/follower_bench/metric.py` — Interface: 2 frozen dataclasses (5 and 6 fields) + 3 functions (`score_clip`, `aggregate_by_pathology`, `trajectory_from_matches`), all with narrow signatures. Implementation: ~150 LOC hiding grid construction, interpolation-based error/lock-rate computation, event-relative relock search with inf/finite branching, truth-monotonicity-guarded false-jump detection, and per-group aggregation with correct inf-exclusion semantics. **Verdict: DEEP** — matches the spec's own claim, confirmed by reading the full implementation in the plan.
- `model/tests/follower_bench/test_metric.py` — test file, depth verdict N/A (correctly not scored as a module).

**Code Quality.** No catch-all exception handling anywhere in the plan. `trajectory_from_matches` raises `ValueError` explicitly on empty input rather than silently producing a broken `TrueTrajectory` — matches CLAUDE.md's "explicit exception handling over silent fallbacks" and "no silent failures" standard. `SAMPLE_HZ`/`POSITION_TOL_BEATS`/`FALSE_JUMP_BEATS` are single-sourced module constants also exposed as `score_clip` kwargs — no magic-number duplication. `from __future__ import annotations`, frozen dataclasses throughout — matches CLAUDE.md style.

**Test Philosophy Audit.** All 9 tests call only `score_clip` / `aggregate_by_pathology` / `trajectory_from_matches` (Task 9 additionally drives the real `follower.follow()`, per spec's explicit design to keep `metric.py` follower-agnostic while still integration-testing it). No internal collaborator is mocked anywhere — every test uses real `clip_generator.generate()` output or hand-built `TrueTrajectory`/`MatchedNote` value objects, consistent with "no mocking of internal collaborators." No shape-only tests: every assertion checks a computed numeric/behavioral outcome (exact error values, lock rate, inf-vs-finite latency, grouping+aggregation math), not merely that a field exists.

**Vertical Slice Audit.** All 9 tasks are single test → single implementation (or explicit no-op justified by prior task's general-purpose code) → single commit. Tasks 2-5 and 7's "no production code change expected" steps are legitimate TDD-green-by-construction slices (each writes one new test against already-general code from Task 1/6/7), not horizontal slicing — each still runs its own red→green→commit cycle rather than batching tests. No task defers implementation to a later task.

**Test Coverage Gaps.**
```
[+] model/src/follower_bench/metric.py
    │
    ├── score_clip()
    │   ├── [TESTED] ★★★ identity (perfect score, near-zero relock) — Task 1
    │   ├── [TESTED] ★★  constant-offset exact error / degraded lock rate — Task 2
    │   ├── [TESTED] ★★  relock inf when never recovers — Task 3
    │   ├── [TESTED] ★★  relock finite when recovers — Task 4
    │   ├── [TESTED] ★★  false-jump detection on monotonic clip — Task 5
    │   ├── [GAP]        multiple position-changing events in one clip (all current
    │   │                pathologies inject exactly one event per clip per
    │   │                pathologies.py's build_plan — confirmed by reading it in
    │   │                full — so this path is untestable with today's clip
    │   │                generator, not a gap this plan introduces)
    │   └── [GAP]        estimate with anchors entirely outside clip's true time span
    │                    (clamp behavior at both boundaries) — not directly tested,
    │                    though Task 2's constant-offset case exercises clamped
    │                    interpolation indirectly
    │
    ├── aggregate_by_pathology()
    │   └── [TESTED] ★★★ grouping + all 5 stat computations incl. inf-exclusion and
    │                    zero-event vacuous-1.0 case — Task 6
    │
    └── trajectory_from_matches()
        ├── [TESTED] ★★  sorts anchors by perf_time regardless of input order — Task 7
        └── [TESTED] ★★  raises ValueError on empty matches — Task 8

[+] integration
    └── [TESTED] ★★★ real follower.follow() -> trajectory_from_matches -> score_clip,
                     clean clip locks well + repeat clip never relocks — Task 9
```
The "multiple events per clip" gap is a real limitation of the *test corpus* (`pathologies.py`'s `build_plan`, confirmed by reading it in full — every pathology type produces exactly one `PathologyEvent`), not a defect in `score_clip`'s implementation, which already loops over `clip.event_labels` generically. Not a blocker: the code path is written generically and will work correctly whenever #111's generator grows multi-event clips.

**Failure Modes.** `trajectory_from_matches`'s empty-input `ValueError` is the only new failure path in the module, and it's loud (raised, not caught) — no silent failures. `score_clip`/`aggregate_by_pathology` have no I/O, no async, no partial-write state — pure functions over immutable inputs, so there is no mid-execution corruption scenario to guard against.

### Presumption Inventory

| ASSUMPTION | VERDICT | REASON |
|---|---|---|
| `MatchedNote.score_position` and `TrueTrajectory` anchors share the same unit | SAFE | Verified via `load_score_notes_from_midi`'s docstring in `score_notes.py`: both are score-MIDI seconds. Naming ("beats") is misleading but the values are consistent. |
| Every pathology type's `SynthClip.event_labels` has exactly one position-changing event (or zero for `clean`) | SAFE | Verified by reading `pathologies.py`'s `build_plan` in full — every branch returns `events=(event,)` or `events=()`. |
| `follower.follow()`'s `matches` are already perf-time-sorted, making `trajectory_from_matches`'s explicit sort defensive rather than load-bearing | SAFE | Verified by reading `_align_at_transpose` in full — the backtrace walks `i` monotonically decreasing then reverses, so output is already perf-index-ordered. The plan's explicit `sorted(...)` is a correct, harmless safety net, not dead code (protects against a future follower implementation that doesn't guarantee this). |
| `clip.true_trajectory.anchors` always has >= 2 anchors so `_sample_grid`/`score_position_at` never index into an empty list | SAFE | `asap_alignment.load_alignment` enforces `MIN_BEATS = 4` matched beat pairs before a `ClipAlignment` is ever constructed; `from_alignment` zips those 1:1 into `TrueTrajectory.anchors`. |
| Task 1's `relock_latencies_s[0] < 1.0 / SAMPLE_HZ` bound holds for the chosen seed/piece | VALIDATE → confirmed SAFE by direct numeric check (measured latency 0.0137s vs. bound 0.05s) | Bound is not formally guaranteed by grid-step math (see RISK above) but empirically holds with wide margin for this exact test case. |
| `POSITION_TOL_BEATS`/`FALSE_JUMP_BEATS` are meaningfully described as "beats" | RISKY (naming only, not functional) | Actual shared unit is score-MIDI seconds; see RISK finding above. Does not affect correctness, only future tunability/readability. |

### Summary
[BLOCKER] count: 0
[RISK]    count: 2
[QUESTION] count: 1

VERDICT: PROCEED_WITH_CAUTION — (1) POSITION_TOL_BEATS/FALSE_JUMP_BEATS are documented as "beats" but the actual shared unit between TrueTrajectory and MatchedNote is score-MIDI seconds (pre-existing naming legacy, not a functional bug — recommend a one-line docstring clarification in metric.py, not a plan change); (2) Task 1's exact relock-latency bound (`< 1.0/SAMPLE_HZ`) has no formal margin against grid-step rounding, though verified numerically to pass today with wide margin for the chosen seed/piece — watch for flakiness only if ALIGNED_PIECE/seed/SAMPLE_HZ ever changes.
