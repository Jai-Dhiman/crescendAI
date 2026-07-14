# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: score_clip identity case
- Created metric.py + test_metric.py. Commit 3c901c0f.
- Incorporated /challenge advisory: module docstring has a "NOTE ON UNITS" paragraph clarifying the _beats suffix actually carries score-MIDI-seconds units. No constants/fields renamed.
- Review MINOR (non-blocking, pre-noted in challenge review): identity-case latency bound `< 1.0/SAMPLE_HZ` has no formal grid-step margin; passes with wide margin today (~0.0137s vs 0.05s). Watch only if ALIGNED_PIECE/seed/SAMPLE_HZ change.
- Module does not import follower_bench.follower (follower-agnostic).
