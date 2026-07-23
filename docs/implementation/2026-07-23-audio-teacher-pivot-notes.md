# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline

Baseline suite (`cd model && uv run python -m pytest tests/`): 748 passed, 47 failed, 31 skipped, 2 xfailed.
All 47 failures are pre-existing data-availability failures in this fresh worktree (missing gitignored
`model/data/raw/asap-dataset/` and `model/data/weights/aria-medium-base/` -- R2-offloaded datasets,
the documented worktree gotcha). None relate to audio_teacher. Accepted as baseline; final suite must
show the identical failure set plus all audio_teacher tests green.

## Task 1: Bootstrap package + valid-WAV validation
Code verbatim from plan. uv.lock unchanged by the sync (hatch packages list is not a lock input), so
committed without it. MalformedClipError and manifest_factory are pre-staged per plan for Tasks 2/6+.
Spec review PASS; quality APPROVED (minor notes only).

## Task 17: Label + epic issue (Group I)
Label epic:audio-teacher pre-existed; epic created as #129. Verified via gh issue list.

## Task 18: Gate 0 + contingency issues (Group I)
Gate 0 = #130, contingency = #131 (epic:audio-teacher,blocked). Per accepted /challenge caution, the
Gate 0 body carries an explicit run-protocol step: re-verify tinker_client.py's Tinker API shape
against current cookbook docs before funding a live run (written without SDK access).
