# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: score_clip identity case
- Created metric.py + test_metric.py. Commit 3c901c0f.
- Incorporated /challenge advisory: module docstring has a "NOTE ON UNITS" paragraph clarifying the _beats suffix actually carries score-MIDI-seconds units. No constants/fields renamed.
- Review MINOR (non-blocking, pre-noted in challenge review): identity-case latency bound `< 1.0/SAMPLE_HZ` has no formal grid-step margin; passes with wide margin today (~0.0137s vs 0.05s). Watch only if ALIGNED_PIECE/seed/SAMPLE_HZ change.
- Module does not import follower_bench.follower (follower-agnostic).

## Task 2: constant-offset error/lock-rate case
- Test-only. Commit 2bf7def5. No metric.py change needed (Task 1 score_clip already general).
- Reviewer mutation-tested (broke lock-rate comparison, confirmed test catches it, reverted). Working tree verified clean afterward.
- APPROVED. MINOR: the false_jump_count==0 assertion is a sanity check (analytically guaranteed 0 under a monotonic constant shift), not a targeted false-jump test.

## BUILD AMENDMENT (Tasks 3 & 9): plan's "never recovers on repeat" was empirically FALSE
- Task 3 implementer hit a genuine plan defect (BLOCKED), independently verified by the controller.
- Finding: the plan assumed a monotonic follower never re-locks after a backward `repeat`/`restart`. FALSE for this metric + shipped follower:
  - Frozen estimate on `repeat`/`restart` -> FINITE relock (~14-22s), because truth replays FORWARD back through the frozen `from_score_position`. (0/300 seeds gave inf.)
  - Real follower on `repeat` seed=13 -> lock_rate 0.647, relock 5.47s FINITE; on `restart` seed=1 -> 6.49s FINITE. Error decays monotonically to lock and STAYS locked ~75-78% of the post-event region -> genuine sustained recovery, not a spurious crossing.
  - `jump` (forward skip) -> inf for BOTH frozen and real follower (truth leaps ahead, never revisits from_score_position). This is the true never-recovers pathology.
- #115's `test_follow_fails_to_relock_after_a_repeat` only probes divergence at a fixed +3s (transient), consistent with a ~5-7s recovery; it never asserted permanent failure over the full clip. The "fails_to_relock" name is about the 3s window, not the whole clip.
- RESOLUTION: `metric.py` is CORRECT and unchanged. Amended Tasks 3 & 9 to use `"jump"` instead of `"repeat"` for the inf/never-recovers assertion. Plan file code blocks + prose updated with BUILD AMENDMENT notes. This is a test-vehicle correction only, verified end-to-end before dispatch.
- Latent metric note (not a bug): relock-latency is a clean never-recover signal for forward-divergence (jump). For backward repeat/restart the follower recovers, so relock is finite — which is the correct, truthful measurement.

## Task 3: inf relock latency (jump)
- Test-only, green-by-construction. Commit b0285e73. metric.py untouched. Uses "jump" per amendment. PASS + APPROVED.

## Task 4: finite relock latency (repeat, forced reconnect)
- Test-only, green-by-construction. Commit 4a4ca282. metric.py untouched. Complements Task 3's inf branch. PASS + APPROVED.

## BUILD AMENDMENT (Task 5): teleport gap must be sub-grid-step
- Task 5 implementer hit a real plan defect (BLOCKED), verified by controller.
- Plan placed the backward teleport anchor at t_mid + 0.3s. But score_position_at is piecewise-LINEAR: over 0.3s (~6 grid samples at 0.05s) the 3.0-unit drop smooths to ~0.5/step, all below the FALSE_JUMP_BEATS=1.0 per-consecutive-sample threshold -> false_jump_count=0.
- Verified: dt=0.3 -> 0; dt in {1e-6, 1e-3, 0.025} -> count>=1 (max(3f,3(1-f))>=1.5>1.0 for any grid alignment).
- RESOLUTION: metric.py CORRECT and unchanged. Amended Task 5 test to use teleport_time = t_mid + 1e-3 (< one grid step) so the drop is a single step. Test-input correction only.

## Task 5: false-jump detection (fixed teleport gap)
- Test-only, green-by-construction. Commit 9d37d5b3. metric.py untouched. Uses teleport_time = t_mid + 1e-3. PASS + APPROVED.

## Task 6: aggregate_by_pathology (+ coverage fix)
- Added AggregateScore + aggregate_by_pathology to metric.py. Commit 0969f23f (note: a transient commit-timing race produced a phantom SHA 4e34d093 in one implementer notification; real HEAD is 0969f23f, verified additive-only, Task 1-5 untouched, 6 passed).
- Code review NEEDS_FIXES (1 IMPORTANT): the "events exist but none succeeded" branch (relock_success_rate=0.0, median_relock_latency_s=inf) was correct but untested. Fixed with commit 101eac28 adding 2 tests (all-inf group + empty-input -> {}). Full file 8 passed. metric.py untouched by the fix.
- MINOR left as-is (non-blocking): AggregateScore.median_abs_error_beats is median-of-per-clip-medians (documented in function docstring); field name reused from TrajectoryScore is a defensible aggregation choice.

## Task 7: trajectory_from_matches adapter
- Added trajectory_from_matches + `from follower_bench.follower import MatchedNote` (type-only, module stays follower-agnostic — no follow/ContinuityPrior/DEFAULT_SKIP_PENALTY import). Commit e23a2f37. Load-bearing sort (unsorted input). No empty guard yet (Task 8). PASS + APPROVED. 9 passed.
