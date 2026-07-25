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

## Task 5: Forced-choice answer parsing
Code verbatim from plan (commit ab55c792, made before the session interruption; reviews ran on
resume). Spec PASS; quality APPROVED. Reviewer MINOR notes: `_ANSWER_RE` is unanchored (a substring
like "MEANSWER: A" would match) -- accepted, the prompt tightly controls output format; no
whitespace-variant parametrize cases.

## Task 6: Manifest loader
Code verbatim from plan (commit ccc52983, pre-interruption; reviewed on resume). Spec PASS
(test file byte-identical to plan); quality APPROVED. Error-path tests deliberately deferred to
Tasks 8/9 per plan. MODEL_ROOT duplication with probe.py already accepted by /challenge as MINOR.

## Task 7: Recorded-response client
Base code verbatim from plan (commit f436e70e, pre-interruption; reviewed on resume). Spec PASS.
Quality review found one IMPORTANT: duplicate pair_id lines in a recorded JSONL were silently
overwritten, contradicting the module's fail-loud contract. Fixed in c91f8959 (TDD: new test
test_duplicate_pair_id_in_recording_fails_loudly failed with DID NOT RAISE, then ValueError naming
the pair id + file added before the dict write). Re-review APPROVED. This is a deliberate,
reviewed deviation from the plan's verbatim client.py code.
