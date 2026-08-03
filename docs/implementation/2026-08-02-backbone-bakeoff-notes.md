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

## Pre-GPU-run resume guard (DONE 2026-08-02)
`extract_embeddings` now takes `skip_existing=True` (default): an entry whose
`.npz` is already on disk is counted in the new `ExtractionReport.skipped` and
the backbone is never called, so an interrupted GPU extraction resumes on the
remainder. That is only safe because `write_embedding_npz` is now atomic —
`tempfile.mkstemp` in the destination directory, `np.savez` into the open file
handle, then `os.replace` — so a present `.npz` is a complete `.npz`, and a
crash mid-write leaves neither a truncated target nor a stray temp file. The
temp suffix is `.npz.tmp`, which deliberately does not match the eval stage's
`*.npz` glob. Caveat: any `.npz` written before this change would be trusted by
the resume path without a completeness check (none exist yet — no real
extraction has run).

## Known follow-ups flagged by review (not blockers, noted per /challenge's PROCEED_WITH_CAUTION)
- `run_bakeoff.py`'s `_stage_eval` has no dedicated regression test for the
  empty-`.npz`-directory guard path, though the code correctly handles it
  (verified by wave-2 review via code inspection).
- `bakeoff_cv.py`'s `n_folds > n_composers` and `<3-train-rows` fold-guard
  branches are exercised by the guard logic but have no dedicated unit test
  by name.
