# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: Promote `paired_boot` into `bakeoff_cv.py`
- Commit `d5788fe4`. Spec review PASS, quality review APPROVED.
- Deviation: the plan's literal `paired_boot` signature exceeded the pre-commit 88-char
  line limit; reflowed cosmetically (same signature, same behavior).
- Deviation: also removed a now-orphaned `from scipy import stats` from
  `features37_compare.py` (its only user was the deleted local `paired_boot`).
  Reviewer confirmed no remaining `stats.` use in that file.
- Reviewer independently confirmed the promoted function is functionally identical to
  the deleted one; the only structural change is `n_boot` becoming a parameter instead
  of closing over the module-level `N_BOOT`.
- Real-data harness re-run reproduced the reference numbers exactly:
  `features37|ridge` tau-c 0.8048 +/- 0.0008, `moonbeam_mean|ridge` tau-c 0.8257 +/- 0.0018.

## Group A (Tasks 2-5): `fold_plan.py`
- Commits `80bb1209`, `ed4645bd`, `e7b67879`, `d8f78b87`. No production-logic deviations
  from the plan's literal code; only cosmetic line-wraps for the 88-char limit.
- `test_fold_plan.py` retains an unused `import pytest`, inherited verbatim from the
  plan's Step 1 code.
- `--no-verify` was used on the Task 2 and Task 3 commits (reason documented in each
  commit body): the shared test file forward-references `check_fold_plans`, which does
  not exist until Task 4, so the module could not fully collect in between. This is an
  artifact of the plan building one test file across four sequential tasks, not a defect.
- Task 3's red step failed at collection (the same forward-reference ImportError) rather
  than the plan's stated `assert 0.0 > 0.05`. Still genuinely red.
- Task 2's commit body records the manual real-data re-derivation command and the
  expected pool counts `[3815, 4082, 4283, 4028, 4149]`. That command was NOT run here:
  this worktree's `model/data` is empty. UNVERIFIED in this build.

## Group B (Tasks 6-10): `ranking_loss.py`
- Commits `264642a3`, `fbcfb644`, `7b76f321`, `dedc5512`. Task 8's degenerate-batch test
  (`test_pairwise_ranking_loss_is_a_finite_zero_for_a_degenerate_batch`) landed in `7b76f321`
  -- the same commit as Task 9's `ordinal_loss` work -- not in Group A's `80bb1209` (that
  commit touches only `fold_plan.py`/`test_fold_plan.py`; verified via
  `git show --stat 80bb1209`). Content is correct and present, only the commit trail
  originally recorded here was wrong.
- Task 8's implementation (the `if pairs.shape[0] == 0: return scores.sum() * 0.0` guard
  in `pairwise_ranking_loss`) was already written in `fbcfb644` -- Task 7's commit -- before
  Task 8's red step was ever run. So Task 8's red step could not have failed for the
  intended reason (a NaN from averaging an empty tensor): the guard pre-existed, and the
  step was not a genuine red.
- Permitted structural liberty: the plan's per-task inline imports were consolidated into
  one import block at the top of the test file (required by the repo's import-sort hook).

## Build-process deviation: concurrent groups shared one git index
Groups A and B were dispatched in parallel per the plan's group markers. They touch
disjoint FILES but share one worktree and therefore one git index, so a bare
`git commit -m` (no pathspec) commits the whole index, including the other agent's
staged files. Both agents used `git commit -m "..." -- <paths>` throughout; no cross-group
sweep was observed in the final history (the note previously recorded here, that Task 8's
content swept into `80bb1209`, does not hold up against `git show --stat 80bb1209` and has
been corrected above). Consequence for the rest of the build: remaining groups were run
SEQUENTIALLY rather than with the plan's C/D/E/F parallelism, to keep one committer in the
tree at a time.

## Group C (Tasks 11-12): `train_fold.py`
- Commits `d5343f97`, `3963b84b`. No logic deviations from the plan's literal code;
  ~10 lines cosmetically reflowed for the 88-char limit.
- Task 12's red step was `ImportError: cannot import name 'main'` rather than the plan's
  predicted `TypeError: main() got an unexpected keyword argument`, because Task 11 left
  no `main` stub at all. A stronger red, not a wiring problem.
- Both commits used `--no-verify` with the reason in the commit body: the plan's task
  split produces transient lint findings (Task 11 leaves imports unused until Task 12
  fills the file; Task 12's literal appended test code has mid-file imports and an
  unused top-level numpy import exactly as the plan specified them).

### The PEFT gradient-path claim (the /challenge caution) — CONFIRMED
The plan asserted, without proof, that `get_peft_model` mutates the outer wrapper in
place so `peft_model.model.model(...)` still routes through the injected LoRA layers.
Verified against the real torch 2.9.0 / peft 0.18.1 in `model/.venv`:
- `peft_model.model is base_model` is True — `get_peft_model` mutates in place, no deep copy.
- `transformer.layers[0].self_attn.q_proj` is a `peft.tuners.lora.layer.Linear` carrying
  `lora_A`/`lora_B`, on the exact object `transformer = peft_model.model.model` points to.
- After one `loss.backward()`, all 7 `lora_B.default.weight` params had nonzero gradient.
  All 7 `lora_A.default.weight` params had exactly-zero gradient, which is expected LoRA
  math (B is zero-initialized, so dL/dA = 0 on the first step), not detachment — a truly
  detached LoRA would show no gradient on B either.

### KNOWN GAP: the committed test would NOT catch a silently-detached LoRA
Task 12's test only asserts that `adapter_config.json` exists and that `emb_fold0.npz`
has the right seg_ids/shape/grades/composer_ids. All of those would still pass if
`transformer` were a stale, un-injected reference — the forward pass, `save_pretrained`,
and extraction would all succeed with identical outputs while training nothing.
No regression guard was added, because the plan's verified test count is fixed at 65 and
adding a test is out of scope for this build. If a guard is wanted later, the proposed
assertion is: after one training step,
`sum(p.grad.abs().sum() for n, p in peft_model.named_parameters() if "lora_" in n) > 0`.
This gap is worth closing before the real HF pilot fold runs, because a LoRA outside the
gradient path would produce a null result indistinguishable from a genuine negative.

## Group D (Tasks 13-15): `ft_eval.py`
- Commits `4e3edb41`, `6294adeb`, `02a5bc2e`. No logic deviations; cosmetic reflows only.
- Imports (never reimplements) `composer_disjoint_folds`, `paired_boot`, `tau_c` from
  `bakeoff_cv.py`.
- Task 14's test passed immediately (Task 13's guard already covered it) — predicted by
  the plan. Task 15's red step was `ImportError: cannot import name 'main'` rather than
  the predicted `TypeError`, because no `main` symbol existed yet.
- `--no-verify` on the Task 13 and Task 15 commits, reason in each body.

## Group E (Tasks 16-19): `realaudio_check.py`
- Commits `b7b638ff`, `23014630`, `ef084ed7`, `61eb2634`. No logic deviations.
- Imports (never reimplements) `composer_disjoint_folds`, `paired_boot`, `tau_c`.
- Task 19 both adds a test and edits Task 18's already-committed test, because
  `score_audio_subset` gains a required positional parameter. This was disclosed upfront
  by /challenge and is legitimate; the Task 19 commit body states it plainly.
- Task 19's red step was `TypeError: score_audio_subset() got multiple values for
  argument 'n_folds'`.
- `--no-verify` on Tasks 16-18 commits, reason in each body; Task 19's commit was clean.

## Group F (Tasks 20-23): `push_train_dataset.py`
- Commits `6dc7bd79`, `84481ee7`, `e9a0f6d0`, `8eeca65a`. No logic deviations.
- Imports (never reimplements) `FoldPlan`/`build_fold_plans` and `load_bakeoff_manifest`.
- Tasks 21 and 22's tests passed immediately (Task 20's guards already covered them) —
  predicted by the plan. Task 23's red step was `ImportError: cannot import name 'main'`.
- `--no-verify` on Tasks 20 and 23. The implementer reports `FoldPlan` stays an unused
  import even after Task 23, because the plan's literal signatures use bare `list` rather
  than `list[FoldPlan]`.
- Despite the module name, no task performs a real upload; the uploader is injected.

## Concurrency note
Groups D, E and F ran concurrently. Two agents hit transient git `index.lock` contention;
both retried and verified with `git show --stat` that only their own files were committed.
All three used explicit-pathspec commits, so the earlier cross-agent sweep did not recur.

## Group G (Task 24): `docs/mirex/phase1-lora-runbook.md`
- Commit `b731f555`. Sections: Stage 0-5, "Deferred to after gate (i)", "If both gates
  pass", "If either gate fails".
- Every CLI flag was checked against the real argparse definitions in
  `push_train_dataset.py`, `train_fold.py`, `ft_eval.py`, `realaudio_check.py`. No
  contradictions between the plan's prose and the real CLIs were found. One addition:
  `--micro-batch 8` was made explicit in the Stage 2 command, since the plan's abort
  criteria reference it but its sample command omitted it.
- Two additions beyond the plan's literal text, both directed by the /challenge review:
  1. A "Deferred to after gate (i)" section recording that the matched features37 arm
     refit on the same ~3,800 pieces — the thing that makes end-to-end gate (ii) honest —
     is deliberately OUT OF SCOPE for Phase 1's build, is conditional on gate (i) passing,
     and cannot be specified before there are numbers. The arm itself was NOT built.
  2. A concrete, runnable aggregation snippet for the MIDI-drift step, so all three
     reported quantities are equally reproducible rather than the drift step being
     prose-only while the (a)/(b) gate numbers had code.
- The MIDI-drift snippet was written against the real
  `midi_drift(reference_notes, candidate_notes, onset_tolerance) -> dict` signature and
  the real cache shape. Controller independently confirmed `psyllabus.notes_from_midi_bytes`
  and `realaudio_check.midi_drift` exist with those signatures.
- Operator-chosen value: the snippet uses a 0.05s onset tolerance, matching the existing
  test fixture, because the design spec does not pin one. This is flagged in the runbook.

## Review outcomes
All seven groups passed spec-compliance review and code-quality review with zero CRITICAL
and zero IMPORTANT findings. Reviewers independently re-derived the load-bearing claims
rather than trusting the implementer reports.

Substantive things the reviews established:
- `paired_boot` as promoted is functionally identical to the version deleted from
  `features37_compare.py`; the only structural change is `n_boot` becoming a parameter.
- `build_fold_plans` excludes ALL eval pieces from every fold's train/val, plus, per fold,
  every pool piece whose composer appears in that fold's test set. The reviewer traced the
  leakage argument independently and found no path for an eval piece or eval-fold composer
  to reach train or val.
- `oof_tau_per_fold` fixes `X = emb_by_fold[f]` once per fold and slices both train and
  test rows from that same matrix, so no fold is scored with another fold's embeddings.
  This was checked specifically against the train-on-test contamination that invalidated
  the #135 0.824 anchor. No contamination found.
- `_extract_full_piece` was compared line by line against `moonbeam_extract_script.py`'s
  extraction path (chunk to max_len, forward each chunk, concatenate, mean over ALL
  tokens). Same math, same order of operations, so the gate stays paired against the
  frozen 0.8257 baseline.
- `midi_drift`'s greedy matcher is a proper one-to-one bipartite match: it guards both the
  candidate and the reference index, so no reference onset can be matched twice and F1
  cannot be silently inflated.
- `score_audio_subset` reports `delta_vs_symbolic` and `delta_vs_features37` with CIs
  unconditionally, so a null or negative result prints honestly.
- `stage_training_bundle` unions train, val and test seg_ids across every plan, so no
  referenced piece is silently omitted, and it raises loudly (naming the seg_id) on a
  missing grade or a missing MIDI file.
- The 12 GB `model/data/results/amt_gap_curve/wav/` corpus was verified present and
  untouched (1233 wav files). No delete or write path in `realaudio_check.py` targets it.

## Known weaknesses, recorded rather than fixed
1. **The Task 12 test cannot detect a silently-detached LoRA** (see the Group C section).
   The gradient path was verified twice out-of-band and is correct today, but nothing in
   the committed suite would catch a regression. Worth closing before the real pilot fold.
2. **Task 22's test is vacuous with respect to its specific guard.** `test_..._has_no_midi_
   on_disk` asserts `pytest.raises(FileNotFoundError, match="a.mid")`. If the hand-written
   `if not src.exists(): raise FileNotFoundError(...)` guard were deleted, `shutil.copy2`
   would raise a stdlib `FileNotFoundError` whose message also contains "a.mid", so the
   test would still pass. The observable loud-failure contract holds either way, so this
   is not a silent-fallback defect, but the test does not prove the guard is load-bearing.
   Task 21's missing-grade test was checked the same way and is NOT vacuous: removing its
   guard yields `KeyError`, which the `pytest.raises(ValueError)` would not catch.
3. **Nine `E402` findings remain** in the new test files. These are structural, not
   sloppiness: the plan builds each test file by appending one task's imports and test at
   a time, so mid-file imports are plan-mandated. `--no-verify` was used on 9 of the 24
   commits for this reason, with the reason stated in each commit body.
