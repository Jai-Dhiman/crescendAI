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
