# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: `parse_key_to_pc` and `transpose_interval` pure helpers
- Implemented exactly as specified. `load_passage_key` included in same file per plan.
- `_DEFAULT_SCORES_DIR` anchored to `__file__` (not CWD) per project preference.
- 25/25 tests pass.

## Task 2: `load_passage_key` fixture tests
- Added import + 3 new tests to test_keys.py.
- `parents[3]` path depth is noted as fragile in the challenge review; accepted per plan.
- 28/28 tests pass.

## Task 3: TagSet.key field + toml update + test updates
- Added `key: str` as required field to TagSet dataclass.
- All 22 technique_tags.toml entries annotated with `key = "C"` (all Hanon/Czerny/Burgmuller are C major).
- Updated `_tags()` helper in test_briefing.py to pass `key="C"`.
- Updated `_diagnosis()` default `piece_id` from `"fur_elise"` to `"bach.prelude.bwv_846"` per plan blocker resolution.
- DEVIATION: test_match.py also constructs TagSet directly and was not in Task 3's file list. The regression was caught in Task 6 and fixed with a separate commit. Not scope expansion -- completing the invariant the plan missed.

## Task 4: test_piece_eb.json fixture
- Created `model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json` with `key_signature: "Eb"`.
- 29/29 test_keys.py tests pass.

## Task 5: transpose_semitones/target_key + excerpt from bar_range
- Added `transpose_semitones: int | None` and `target_key: str | None` to ExerciseBriefing dataclass.
- Removed `_EXCERPT_BARS` constant (dead after change) per plan.
- `_transform_params` now takes `bar_range` and computes `end_bar = bar_range[1] - bar_range[0] + 1`.
- Key resolution calls `load_passage_key` unconditionally; FileNotFoundError propagates (explicit exceptions per project standard).
- Instruction appended with `" Transpose into the key of {target_key}."` when target_key is set.
- 13/13 test_briefing.py pass, 1 skipped (real catalog absent).

## Task 6: Full suite verification
- DEVIATION: test_match.py had 2 regressions from Task 3's TagSet.key addition (plan omitted test_match.py from Task 3's file list). Fixed by adding `key="C"` to 4 TagSet constructors in test_match.py.
- Final: 88 passed, 8 failed (pre-existing test_transforms.py MIDI-absent failures), 1 skipped.
