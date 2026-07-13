# Follower Trajectory Metric Design

**Goal:** Score a symbolic score-follower's estimated trajectory against a `SynthClip`'s exact ground truth, producing per-clip and per-pathology-type numbers (position error, lock rate, relock latency, false-jump count) so #115's baseline follower has a falsifiable acceptance test.
**Not in scope:** the follower itself (#115); audio/AMT (#112); any WASM/Rust port (#120); tuning the follower to hit a target — this issue only *measures*.

## Problem
Issue #108's Path-A retarget (see #115) makes #111's ASAP synthetic clips the follower benchmark, replacing the unrecoverable `bach_inv1_chunk0` fixture. But #111 only *generates* clips with ground-truth trajectories — there is no scorer. Without one, #115's spec ("high coverage on clean/tempo/wrong-note; *fails* to re-lock on jump/repeat/restart") is unmeasurable prose. This issue supplies the scorer: given a follower's `estimated_trajectory: TrueTrajectory` and a `SynthClip`, return numbers that make each of those claims true or false.

## Solution (from the user's perspective)
A developer (or a build agent) runs any follower over #111's clips and calls `score_clip(estimated, clip)` per clip, then `aggregate_by_pathology(scores)`. They get a dict keyed by pathology type — `clean`, `tempo_swing`, `wrong_note`, `jump`, `repeat`, `restart`, `hesitation` — each carrying median position error, lock rate, relock success rate + latency, and false-jump totals. That table is #115's acceptance evidence and every later phase's regression baseline.

## Design
The metric samples both trajectories on a common uniform time grid over the clip's true-trajectory time span and compares them in **score-beats** (the unit of `TrueTrajectory` anchors' second element). Four behaviors, one deep module:

1. **Position error** — at each grid time `t`, `abs(estimated.score_position_at(t) - true.score_position_at(t))`. Report median and max (beats). Median resists the post-discontinuity error spike that is *expected* and not a follower defect.
2. **Lock rate** — fraction of grid samples with error `<= POSITION_TOL_BEATS`. A single scalar for "how much of the clip was tracked."
3. **Relock latency** — for each *position-changing* `PathologyEvent` (jump/repeat/restart; identified by `from_score_position != to_score_position`), the seconds from `event.perf_time` until the estimate first re-locks (error `<= POSITION_TOL_BEATS`) at or after the event. `math.inf` if it never re-locks before the clip ends. This is the north-star signal: a monotonic follower re-locks (late) after a forward *jump* but **never** after a backward *repeat/restart* — exactly the #115 gap.
4. **False-jump count** — number of grid steps where the estimate moves *backward* by more than `FALSE_JUMP_BEATS` while the true trajectory is non-decreasing across that step (no real discontinuity there). Catches a follower that teleports on coincidental pitch matches (the day-0 "3 teleports without the continuity prior" failure mode).

**Why sample on a grid rather than compare anchors directly?** The two trajectories have unrelated anchor times (the estimate's anchors come from the follower's own decisions). A common time grid is the only apples-to-apples comparison, and `score_position_at` already interpolates+clamps, so sampling is exact for piecewise-linear inputs between anchors.

**Why take the true trajectory's time span as canonical?** `clip.true_trajectory` defines the clip's real duration; the follower's estimate may start/end anywhere, and `score_position_at` clamps outside its own range, so evaluating on truth's span fairly penalizes an estimate that stops early.

**Tunable constants (named, single source):** `SAMPLE_HZ = 20.0`, `POSITION_TOL_BEATS = 0.5`, `FALSE_JUMP_BEATS = 1.0`. These are module constants and also keyword args of `score_clip` so tests and future tuning can override without editing the module. Rationale in Open Questions.

## Modules
### `model/src/follower_bench/metric.py` (New) — DEEP
- **Interface:**
  - `TrajectoryScore` (frozen dataclass): `pathology_type: str`, `median_abs_error_beats: float`, `max_abs_error_beats: float`, `lock_rate: float`, `relock_latencies_s: tuple[float, ...]`, `false_jump_count: int`.
  - `score_clip(estimated: TrueTrajectory, clip: SynthClip, *, sample_hz: float = SAMPLE_HZ, position_tol_beats: float = POSITION_TOL_BEATS, false_jump_beats: float = FALSE_JUMP_BEATS) -> TrajectoryScore`
  - `AggregateScore` (frozen dataclass): `n_clips: int`, `median_abs_error_beats: float`, `mean_lock_rate: float`, `relock_success_rate: float`, `median_relock_latency_s: float`, `total_false_jumps: int`.
  - `aggregate_by_pathology(scores: Iterable[TrajectoryScore]) -> dict[str, AggregateScore]`
- **Hides:** uniform grid construction over truth's time span; interpolation via `score_position_at`; identification of position-changing events; event-relative relock search with a "first sample at/after `perf_time` within tol" rule; backward-move detection guarded by truth-monotonicity; median/rate aggregation with correct handling of `inf` latencies (excluded from `median_relock_latency_s`, counted in `relock_success_rate`) and pathology types that have zero position-changing events (relock_success_rate defined as 1.0 when there are no such events).
- **Tested through:** `score_clip` / `aggregate_by_pathology` only, using hand-built `TrueTrajectory` estimates and real #111 clips from `clip_generator.generate(...)`. No follower needed — the metric is independently testable by supplying estimated trajectories directly.
- **Depth verdict:** DEEP — two functions + two value types hide four distinct measurement algorithms and their aggregation; the interface never needs to change when the follower changes.

## Verification Architecture
- **Canonical success state:** `score_clip(clip.true_trajectory, clip)` (estimate == truth) returns `median_abs_error_beats == 0.0`, `max_abs_error_beats == 0.0`, `lock_rate == 1.0`, every entry of `relock_latencies_s == 0.0`, `false_jump_count == 0`. A deliberately-degraded estimate (clean monotonic trajectory scored against a `repeat` clip) returns `lock_rate < 1.0` and a `math.inf` relock latency for the backward event.
- **Automated check:** `uv run pytest tests/follower_bench/test_metric.py` (part of the existing green `tests/follower_bench/` suite).
- **Harness:** none needed beyond the tests — this issue *is* the harness for #115. #111's `generate()` supplies real clips; hand-built `TrueTrajectory` objects supply the estimated side. No golden fixture files.

## File Changes
| File | Change | Type |
|------|--------|------|
| `model/src/follower_bench/metric.py` | New deep module: `TrajectoryScore`, `AggregateScore`, `score_clip`, `aggregate_by_pathology` | New |
| `model/tests/follower_bench/test_metric.py` | Behavior tests through the public interface | New |
| `model/src/follower_bench/__init__.py` | (No change required — subpackage imports are by module path, matching #111 convention) | — |

## Open Questions
- Q: Is `POSITION_TOL_BEATS = 0.5` the right "locked" threshold? Default: 0.5 score-beats (half a beat). Tunable via `score_clip` kwarg; #115 can sweep it when reporting acceptance. It only sets the pass/fail line for lock/relock, not the raw error numbers, so downstream analysis is unaffected.
- Q: `SAMPLE_HZ = 20.0` (50 ms grid) — fine enough to time relock to ~0.05 s, coarse enough to stay fast on ~30 s clips. Default: 20 Hz. Tunable.
- Q: `FALSE_JUMP_BEATS = 1.0` — a backward estimate move larger than one score-beat between consecutive 50 ms samples is a teleport, not tracking drift. Default: 1.0 beat. Tunable.
