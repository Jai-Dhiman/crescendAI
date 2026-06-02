# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline
- 5 pre-existing failures from MISSING UNTRACKED score-data JSON (model/data/scores/*.json): test_piece_score_map.py (4) + test_run_eval_bar_analysis.py (1). Both test thin-framing code the plan deletes/supersedes. Clean baseline excluding them: 82 passed, 4 skipped.
- Tests require the `teacher-model-stage0` uv extra (jsonschema). Run with: uv run --extra teacher-model-stage0 pytest ...
