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

## Task 19: Migration batch (Group I)
Executed after controller pre-verified all 12 targets' live state matched the plan. #71 #79 #80 #81
#82 #83 #84 #16 #55 closed not-planned with successor comment linking epic #129; #32 #33 #40
relabeled deferred,epic:audio-teacher with retarget comments. Verified epic:teacher-finetune and
epic:model-v2 both have zero open issues. STATE comment posted to #127. Before/after snapshots in
the session scratchpad (issue-migration-before/after.txt).

## Task 2: Malformed WAVs abort naming the file
Code verbatim from plan (commit 5a395a53). Spec PASS; quality APPROVED. Reviewer MINOR notes: the
"not readable" and "zero-length" branches have no dedicated test case (manually verified working);
acceptable -- plan scope covers stereo/wrong-rate/truncated only.

## Task 3: Budget guard
Code verbatim from plan (commit 0695628d). Spec PASS; quality APPROVED. Reviewer MINOR note: no
exact-boundary (projected == cap) test; strict > semantics verified by reading.

## Task 4: Per-axis elicitation questions
Code verbatim from plan (commit 33f24264). Spec PASS; quality APPROVED. MINOR style notes only.
