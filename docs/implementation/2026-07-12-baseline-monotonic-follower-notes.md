# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: Golden fixture loader (harness)
- Implemented `ScoreNote` frozen dataclass + `load_golden_fixture_notes` in `src/follower_bench/score_notes.py`; matches plan verbatim, no deviations.
- Verified PerfNote field order against `segments.py` before writing the mapping.
- Added the challenge-requested extra test `test_load_golden_fixture_notes_raises_on_missing_file` (FileNotFoundError path) in the same commit.
- REPO_ROOT `parents[3]` confirmed correct for the worktree's directory depth.
- Note: module docstring forward-references the partitura `load_score_notes_from_midi` loader (added in Task 8) — expected vertical-slicing, not creep.
- Commit 9cacef1e. Tests: 2 passed.
