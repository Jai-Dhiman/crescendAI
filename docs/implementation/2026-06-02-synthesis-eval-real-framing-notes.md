# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline
- 5 pre-existing failures from MISSING UNTRACKED score-data JSON (model/data/scores/*.json): test_piece_score_map.py (4) + test_run_eval_bar_analysis.py (1). Both test thin-framing code the plan deletes/supersedes. Clean baseline excluding them: 82 passed, 4 skipped.
- Tests require the `teacher-model-stage0` uv extra (jsonschema). Run with: uv run --extra teacher-model-stage0 pytest ...

## Task 1 (remove build_synthesis_user_msg)
- Removed the function + stubbed legacy run() try-block with NotImplementedError per plan.
- DEVIATION (orphan cleanup): also removed 3 now-unused top-level imports the deletion orphaned: get_style_guidance, format_teacher_voice_blocks, select_clusters. They were used ONLY by build_synthesis_user_msg. Verified no surviving consumer (atomic_gate/judge_family/provenance/split_flag tests + scripts/split.py + scripts/tag_dataset.py) imports them; they only use preserved symbols.

## Task 2 (delete bar_analysis_local)
- Deleted bar_analysis_local.py + test_bar_analysis_local.py + test_run_eval_bar_analysis.py per plan.
- NOTE: piece_score_map.py is NOT deleted by this plan; its test (test_piece_score_map.py) remains a pre-existing failure due to missing untracked score-data JSON. Out of scope.

## Task 3 (delete stage0 run_synthesis)
- Deleted run_synthesis.py + test_run_synthesis.py + test_run_eval_blocks.py + test_run_eval_style_injection.py per plan.
- PLAN INACCURACY (harmless): plan said run_synthesis.py does NOT import build_synthesis_user_msg, but it DID (line 15). Already broken by Task 1, so deletion is the right call either way.
- EXPECTED INTERMEDIATE BREAKAGE: cli.py:11 module-top import of run_synthesis is now unsatisfiable until Task 6. Do not run the full teacher_model/stage0 suite between Task 3 and Task 6 (documented in plan).

## Task 4 (build_do_row)
- Implemented verbatim per plan. 3 tests pass (success, DO-failure, unresolved-piece).

## Task 5 (run_do_baseline + CLI)
- Implemented run_do_baseline + _default_driver per plan; errors counted once (post-row check only) per plan note.
- DEVIATION (per challenge RISK 2 directive): extracted _judge_provider_for(judge_model) shared helper to kill the triplicated autodetect rule. Used in main() --do-path branch. Tasks 6/9 will reuse it where they need provider autodetect. Legacy run() no longer holds a copy (removed in Task 1 stub).
- Confirmed run_recording signature (wrangler_url, recording_cache, student_id=, piece_query=) matches _default_driver call.

## Task 6 (repoint cli.py synthesis)
- Removed module-top run_synthesis import (was line 11); repointed synthesis branch to run_do_baseline with judge_fn=judge_extended (9-dim preserved per BLOCKER 2); added --wrangler-url to synthesis subparser.
- DEVIATION (orphan cleanup): removed now-unused _SYNTH_SYSTEM path constant (only consumer was the old synthesis branch). LLMClient kept (still used by tool/continuation branches). judge_extended import kept (used by repointed branch).
- Verified pre-existing test_cli.py collects + passes again after cli.py:11 removal (collectability restored, intermediate-break window closed).

## Task 9 (repoint regen_calibration_baseline)
- Removed build_synthesis_user_msg/extract_teacher_response + orphan chain (SESSION_SYNTHESIS_SYSTEM, aggregate_muq, _build_row, LLMClient + synthesis_client); added build_do_row import + module-level _default_driver; added --wrangler-url.
- Kept judge_synthesis_v2 (7-dim v2-rubric calibration corpus, separate from stage0 dossier) per plan.
- DEVIATION (orphan cleanup): removed `import time` (only consumers were the deleted time.monotonic() calls). sys kept (ImportError handler).
- build_do_row uses default judge_provider=workers-ai / judge_model=gemma per plan (matches regen's prior Gemma judge defaults).

## Tasks 7 + 8 (Group D, manual) — DEFERRED to maintainer with credentials
- Task 7 (live DO measurement) requires `wrangler dev` SessionBrain DO with WORKING synthesis (Anthropic teacher key) + judge (Workers AI) LLM calls over 98 holdout recordings. This environment has NO apps/api/.dev.vars (no local Anthropic/AI-Gateway secrets), so every eval_chunk->synthesis call would error -> 98 error rows. Plan Task 7 Step 1 says STOP if ERROR: appears. Cannot produce an honest baseline here. This is exactly the deferral the plan's Open Questions anticipated ("If the build environment lacks a live DO, Tasks 7-8 are deferred to a maintainer").
- Task 8 (re-lock _SONNET_BASELINE) depends on Task 7's results/baseline_v2_do_aggregate.json. Its verification (Step 2: dossier test passes with the NEW measured numbers) cannot be validated without Task 7's output. Editing _SONNET_BASELINE with fabricated numbers would corrupt the locked baseline, so it was NOT done. EVAL_CHECKLIST.md update was also held (it is part of the same task/commit and Step 4 gates on the dossier test).
- TO COMPLETE (maintainer, from apps/evals/, with apps/api/.dev.vars in place):
  1. Terminal 1: `just api`  (boot DO + local DB)
  2. Terminal 2 smoke: `curl -sf http://localhost:8787/health` then
     `uv run python -m teaching_knowledge.run_eval --do-path --limit 1 --teacher-model claude-sonnet-4-6 --judge-model '@cf/google/gemma-4-26b-a4b-it' --out results/smoke_do_$(date -u +%Y%m%d).jsonl`
     -> expect [1] <rid> | <slug> with NO ERROR:, synthesis_latency_ms>500.
  3. Full: `uv run python -m teaching_knowledge.run_eval --do-path --teacher-model claude-sonnet-4-6 --judge-model '@cf/google/gemma-4-26b-a4b-it' --out results/baseline_v2_do.jsonl` (98 rows, error rate <5%).
  4. Precondition guard then aggregate (run from apps/evals/): verify teaching_knowledge/data/dataset_index.jsonl exists and `missing: 0` for holdout ids, then
     `uv run python -m teaching_knowledge.scripts.aggregate results/baseline_v2_do.jsonl --dataset-index teaching_knowledge/data/dataset_index.jsonl --out results/baseline_v2_do_aggregate.json`
  5. Task 8: copy dimensions (name/mean_outcome/n) + composite_mean verbatim from baseline_v2_do_aggregate.json into _SONNET_BASELINE in teacher_model/stage0/tests/test_aggregator.py; add provenance comment; run `uv run pytest teacher_model/stage0/tests/test_aggregator.py -q`.
  6. EVAL_CHECKLIST.md: run_eval.py now REQUIRES wrangler dev (--do-path); document DO-path runbook; STATE PLAINLY (RISK 1) the re-locked baseline is HOLDOUT-ONLY (n<=98) and NOT numerically comparable to the old train-split ASCF 1.387 (different population AND framing). Remove build_synthesis_user_msg/bar_analysis_local/run_synthesis as synthesis sources.

## Final test state (Groups A-C)
- teaching_knowledge + teacher_model/stage0: 82 passed, 4 skipped, 4 failed.
  The 4 failures are PRE-EXISTING test_piece_score_map.py cases (missing untracked model/data/scores/*.json), unrelated to this work and untouched by it.
- tests/ + teacher_model/calibration: 143 passed, 2 failed. The 2 failures (test_analyze_e2e.py build_confusion_matrix) are PRE-EXISTING and in pipeline/ code this branch never touches.
- Net: zero new failures introduced; full-suite collectability restored (no broken imports from deletions).
