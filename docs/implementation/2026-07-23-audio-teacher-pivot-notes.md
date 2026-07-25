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

## Task 8: Rehydrate-hint lock tests (Group D)
Commit 32e81df3, test-only (41 insertions); manifest.py needed NO change -- Task 6's
_ensure_clip_local already satisfied both behaviors, both tests passed on first run.
HONESTY CHECK EXECUTED: assertion temporarily mutated to "rclone WRONG"; pytest failed with
AssertionError showing the real message ("...Rehydrate with:\n    rclone copy
r2:crescendai-bucket/mirex-probe/clips clips"); assertion restored exactly; re-run 3 passed.
No deviations.

## Task 10: Population-partitioned scorer (Group D)
Code verbatim from plan (commit 15c03597). TDD observed (ModuleNotFoundError first). Spec PASS;
quality APPROVED. MINOR notes: one 89-char line (E501, not CI-gated); report returns the internal
cells dict without defensive copy (fresh per call, accepted). `import json` retained per plan for
Task 13.

## Task 12: Tinker client behind the seam (Group D)
Base code verbatim from plan (commit bad92839); test exercised the real not-installed path
(1 passed, not skipped -- SDK genuinely absent). Spec PASS. Quality review found one IMPORTANT:
ask()'s getattr/hasattr response-text extraction was a silent fallback contradicting the fail-loud
rule. Fixed in b8621b88 (direct message.content.text in try/except AttributeError raising TypeError
naming the pair and actual type, from exc). Re-review APPROVED. This is the second deliberate,
reviewed deviation from the plan's verbatim code. SDK API surface remains knowingly unverified
(accepted /challenge caution; Gate 0 issue #130 carries the re-verify step). MINOR accepted:
single except ImportError collapses partial-install cases; no test for live paths (by design).

## Task 9: Manifest schema rejection lock tests (Group E)
Commit 8f2d3a6a, test-only; manifest.py needed NO change (all six violations already rejected;
each parametrized case traced to a distinct validation branch by the reviewer). HONESTY CHECK
EXECUTED: implementer first verified ManifestError subclasses Exception (not ValueError), then
weakened to pytest.raises(ValueError) -- all 6 cases failed with the uncaught ManifestError (e.g.
missing_key: "pair[0] missing keys ['description']"); restored, 9 passed. Spec PASS; quality
APPROVED. No deviations.

## Task 11: Ex-ante verdict rules lock tests (Group E)
Commit bf95f922, test-only (scorer.py needed NO change -- Task 10's _verdict_reasons already
satisfied all 5 rules). HONESTY CHECK EXECUTED: expected_verdict of pass_synthetic_never_gates
mutated PASS->FAIL; pytest failed "AssertionError: assert 'PASS' == 'FAIL'"; restored, 6 passed.
Spec PASS; quality APPROVED. No deviations.

## Task 13: Deterministic report rendering (Group F)
Code verbatim from plan (commit 4dd20295). TDD observed (ImportError first, then 7 passed).
Spec PASS; quality APPROVED -- reviewer confirmed the order-independence test is not a tautology
(dropping sort_keys would fail it). No deviations.

## Task 14: CSV->YAML curation script (Group F)
Code verbatim from plan (commit c3a275a6). TDD observed (ModuleNotFoundError first, then 1 passed).
Spec PASS; quality APPROVED -- reviewer confirmed validation strictly precedes the write (bad_out
never created), __file__-anchored MODEL_ROOT default, no silent fallbacks. No deviations.
