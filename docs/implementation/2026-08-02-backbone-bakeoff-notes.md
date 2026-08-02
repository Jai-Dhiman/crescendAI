# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 6 (npz multi-pooling round trip)
Task 5's `write_embedding_npz`/`read_embedding_npz` already generalized to an
arbitrary number of pooling vectors — the Task 6 test passed immediately with
no implementation change, exactly as the plan anticipated.

## Task 10 (extraction orchestrator) — test fix, not spec deviation
The plan's literal test code for `test_extract_embeddings_records_failures_and_continues`
used `if "b" in str(midi_path)` to simulate a failure on entry "b". This is
fragile: macOS pytest `tmp_path` directories (e.g.
`.../pytest-of-jdhiman/pytest-41/...`) frequently contain the literal letter
"b" somewhere in the path, so the substring check spuriously matched entry
"a" too, making the test fail deterministically in this environment. The
implementer changed the check to `midi_path.stem == "b"`, which tests the
intended behavior (failure isolated to the "b" entry) without depending on
OS temp-dir naming. No other test assertions or the orchestrator's
implementation were changed.

## Task 12b (MoonBeam construction without a loader)
`ValueError` was already raised in `__init__` by Task 12a's implementation —
the regression test passed immediately, no implementation change needed, per
the plan's explicit allowance.

## Task 15 (run_bakeoff eval stage)
`_stage_eval` and the `eval` CLI branch already existed from Task 14 and
already guarded against empty `.npz` directories (`if not npz_paths:
continue` before any `np.stack` call). The eval-stage test passed
immediately, no implementation change needed.

## Known follow-ups flagged by review (not blockers, noted per /challenge's PROCEED_WITH_CAUTION)
- `extract_embeddings` has no resume/skip-existing logic — a human running
  the real ~900-piece MoonBeam-839M GPU extraction should pre-filter
  `entries` to exclude already-written `seg_id`s before calling it, or a
  crash/interrupt mid-run means re-extracting everything, not just the
  remainder. Not fixed in this build per the challenge review's guidance
  (worth a one-line pre-filter or documented limitation before the real GPU
  run, not before this offline harness build).
- `write_embedding_npz` writes directly via `np.savez` (no write-to-temp-then-rename).
  A crash mid-write during the real GPU run could leave a truncated `.npz`.
- `run_bakeoff.py`'s `_stage_eval` has no dedicated regression test for the
  empty-`.npz`-directory guard path, though the code correctly handles it
  (verified by wave-2 review via code inspection).
- `bakeoff_cv.py`'s `n_folds > n_composers` and `<3-train-rows` fold-guard
  branches are exercised by the guard logic but have no dedicated unit test
  by name.
