# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline
- Rust crate `piece-identify`: 24 tests pass; 1 PRE-EXISTING failure `real_recording_test::identify_real_recording` (env-gated, requires NOTES_JSON; legacy test retained per additive-PR constraint). Not caused by this work.
- Python Task-1 verify (`test_parity_fixtures.py`): 1 passed against committed golden fixture.
