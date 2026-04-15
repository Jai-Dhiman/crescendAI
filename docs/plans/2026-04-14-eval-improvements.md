# Eval System Improvements — P0–P3 Roadmap

**Goal:** Turn `apps/evals/teaching_knowledge/` from infrastructure-that-can-run into a legible, calibrated quality gate that can serve as the "beat this" baseline for the teacher voice finetune.

**North star metric:** synthesis quality composite (mean of 7 dimension scores from locked judge v2), reported with bootstrap CI and stratified by composer era + skill level.

---

## Context

- **Track A (research) COMPLETE.** 73 transcripts, 379 teaching moments, 5-cluster playbook, 7-dim rubric, judge v2.
- **Track B (infra) MOSTLY COMPLETE.** Inference cache at 890 recordings in `model/data/eval/inference_cache/auto-t5_http/` (MIDI key bug fixed, `midi_notes` vs `notes`). `teaching_knowledge/run_eval.py` exists as a 379-line resume-safe orchestrator.
- **What's missing as of 2026-04-14:**
  - `apps/evals/teaching_knowledge/results/` is empty — baseline has never been scored end-to-end.
  - `run_eval.py` never loads `data/playbook.yaml` or injects composer-era style rules into the synthesis prompt (composer passes through as passive metadata only).
  - `apps/api/src/services/prompts.ts` teacher prompt is style-agnostic (single generic "style" mention on line 41) — prod and eval both lack style guidance.
  - No train/test split. No per-dimension aggregation with CIs. No dual-judge. No human calibration. No regression harness. No A/B scaffold for finetuned teacher.

## Eval Strategy (locked)

**Principle: same-family judging is forbidden.** The judge must be a different model family from the teacher under evaluation to avoid same-family phrasing-preference bias.

### Phase 1 — Sonnet 4.6 teacher, hill-climb the prompt
- **Teacher under eval:** Claude Sonnet 4.6 (current prod model)
- **Judges:** Gemma-4 (Workers AI, cheap) + GPT-5.4-mini (cross-family, cheap)
- **Forbidden:** Sonnet-as-judge (same-family bias)
- **Goal:** Hill-climb the teacher prompt until composite plateaus or hits diminishing returns against the holdout set.
- **Locked baseline artifact:** `results/baseline_sonnet46_judge-gemma4-gpt54mini_YYYYMMDD.jsonl` — frozen, never modified, every future run diffs against it.

### Phase 2 — Finetuned Qwen teacher (conditional)
- **Gate to enter Phase 2:** Phase 1 prompt iteration has plateaued AND baseline composite is known AND dual-judge calibration is complete AND the four strategic gates from `project_teacher_model_finetuning.md` have passed.
- **Teacher under eval:** Finetuned Qwen3-27B (teacher voice finetune)
- **Judges:** GPT-5.4-mini + Sonnet 4.6 (Sonnet now permitted because teacher is no longer Sonnet)
- **Success criterion:** Finetuned composite > best Phase 1 prompt composite on holdout, with non-regression on every single dimension (no dimension regresses by >1 bootstrap CI).
- **Efficiency tier:** Once correctness is locked, track cost/token/latency deltas to confirm the finetune earns its maintenance burden.

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `apps/evals/teaching_knowledge/scripts/tag_dataset.py` | Enrich cache index with composer_era, estimated_skill, duration_bucket tags; writes `data/dataset_index.jsonl` |
| `apps/evals/teaching_knowledge/scripts/split.py` | 80/20 stratified train/holdout split by composer_era x skill; writes `data/splits.json` |
| `apps/evals/teaching_knowledge/scripts/aggregate.py` | Per-dim mean + bootstrap CI (n=1000) + headroom ranking; reads JSONL output, writes `results/<run_id>_aggregate.json` |
| `apps/evals/teaching_knowledge/scripts/regression_check.py` | Diff two eval runs, flag dims regressed > 1 CI |
| `apps/evals/teaching_knowledge/scripts/dual_judge.py` | Run Gemma + GPT-5.4-mini judges side-by-side, compute Spearman + Cohen's kappa per dim |
| `apps/evals/teaching_knowledge/scripts/phantom_audit.py` | Manual audit harness: present each dim with examples, capture keep/merge/kill decision |
| `apps/evals/shared/llm_client_gpt.py` | GPT-5.4-mini client via OpenAI SDK (new provider, cheap iteration judge) |
| `apps/evals/teacher_model/eval_ab.py` | Run `run_eval.py` twice (baseline + finetuned), diff per-dim with CIs |

### Modified Files

| File | Change |
|---|---|
| `apps/evals/teaching_knowledge/run_eval.py` | Load `playbook.yaml`, lookup by composer, inject style rules into synthesis user message; accept `--teacher-model`, `--judge-model`, `--split train\|holdout\|all`; write `run_id` + `git_sha` into output metadata |
| `apps/evals/teaching_knowledge/data/playbook.yaml` | (Read-only) — confirm `piece_style_dimension_rules` schema matches injection code |
| `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt` | Freeze. Tag as `judge-v2.0-locked`. No future edits without re-scoring baseline. |
| `apps/evals/shared/prompts/rubric_definition.json` | Freeze after phantom audit (P1 step 1). Version bump to v2.1 if any dims merged/killed. |
| `apps/api/src/services/prompts.ts` | Add composer-era style guidance to teacher prompt. Must match the injection logic in `run_eval.py` — prod and eval paths stay aligned. |
| `apps/evals/teaching_knowledge/llm_client.py` | Add GPT provider option alongside Workers AI + Anthropic |
| `apps/evals/EVAL_CHECKLIST.md` | Update to reflect new P0-P3 ordering, remove stale "cache at 2/269" references |

---

## Phase P0 — Make the current eval legible (2–3 days)

**Exit criterion:** A locked Sonnet 4.6 baseline exists on disk with run_id, git_sha, per-dim scores, and a holdout set that has never been touched.

- [ ] **P0.1** Write `scripts/tag_dataset.py`. Enrich each of the 890 cache entries with `composer_era` (Baroque/Classical/Romantic/Modern), `estimated_skill` (beginner/intermediate/advanced, inferred from piece difficulty), `duration_bucket` (<30s / 30-60s / 60s+). Persist to `data/dataset_index.jsonl`.
- [ ] **P0.2** Write `scripts/split.py`. 80/20 stratified split keyed on (composer_era, estimated_skill). Persist to `data/splits.json`. Print stratification table. Document in a header comment: "holdout is sacred — never iterate prompts against it."
- [ ] **P0.3** Extend `run_eval.py` to accept `--split {train,holdout,all}` and filter recordings by split membership. Default to `train`.
- [ ] **P0.4** Extend `run_eval.py` to load `data/playbook.yaml`, resolve the synthesis recording's composer to an era, and inject era-specific dimension rules into `build_synthesis_user_msg()`. Style injection must be deterministic and logged.
- [ ] **P0.5** Edit `apps/api/src/services/prompts.ts` teacher system prompt to add the same composer-era guidance. Run `gitnexus_impact` on `TEACHER_SYSTEM` (or whatever symbol holds the prompt) before editing.
- [ ] **P0.6** Add `run_id` (git short SHA + UTC timestamp) and `git_sha` fields to `run_eval.py` output JSONL. Every judgment row carries its provenance.
- [ ] **P0.7** Run the locked baseline: `run_eval.py --split train --teacher-model sonnet-4-6 --judge-model gemma-4 --no-dry-run`. Write to `results/baseline_sonnet46_judge-gemma4_2026-04-14.jsonl`. Estimate: ~$5 (Workers AI judge) + ~$15 (Sonnet synthesis) = ~$20, ~1 hour.
- [ ] **P0.8** Freeze `synthesis_quality_judge_v2.txt` and `rubric_definition.json`. Git-tag as `judge-v2.0-locked`. Document in `EVAL_CHECKLIST.md` that any future judge edits require re-scoring all historical runs.

## Phase P1 — Rubric rigor and dual-judge (3–4 days)

**Exit criterion:** The 7-dim rubric is human-audited, process/outcome split, and validated by cross-family judge agreement on a 100-recording sample.

- [ ] **P1.1** Phantom-criteria audit. Write `scripts/phantom_audit.py` — present each of the 7 dims with its definition, its 0/3 level descriptions, and 3 random example judgments from the P0.7 baseline. Decide per-dim: keep / merge / kill. Suspect candidates: "Autonomy-Supporting Motivation" and "Scaffolded Guided Discovery" (both LLM-extracted from `derive_rubrics.py`). Output: `data/rubric_audit_2026-04-14.md` with decisions and rationale.
- [ ] **P1.2** If any dims are killed or merged, version-bump the rubric to v2.1, re-lock, re-score baseline. If all 7 dims survive, document the audit and move on.
- [ ] **P1.3** Split process vs outcome in the judge output schema. Every dim now returns `{process: 0-3, outcome: 0-3}`. Process = "did the teacher notice / attempt the behavior." Outcome = "was the observation/advice correct given the performance." Update `judge_synthesis_v2()` accordingly.
- [ ] **P1.4** Re-run baseline with the process/outcome schema on a 100-recording sample. Verify the two signals actually decorrelate (Pearson < 0.7). If they're tightly coupled, the split isn't buying anything and can be reverted.
- [ ] **P1.5** Add GPT-5.4-mini as a second judge. Write `apps/evals/shared/llm_client_gpt.py` using the OpenAI SDK, add `--judge-model gpt-5.4-mini` support to `run_eval.py`.
- [ ] **P1.6** Write `scripts/dual_judge.py`. Runs Gemma-4 and GPT-5.4-mini judges on the same 100-recording sample. Computes per-dim Spearman rank correlation and Cohen's kappa. Output: `results/dual_judge_calibration_YYYYMMDD.md`.
- [ ] **P1.7** Classify dimensions by judge agreement: **high-trust** (Spearman > 0.7) — trust the cheap judge going forward; **uncertain** (0.4–0.7) — require dual-judge mean; **low-trust** (< 0.4) — route to human calibration in P1.8 before trusting any automated number on these dims.
- [ ] **P1.8** Human calibration, minimum 20 syntheses, scoped to low-trust dims only. Sit down with the rubric, score blind, compute Spearman vs each judge per dim. This replaces the originally-planned "50 expert syntheses" step by scoping human effort to dims where judges disagree.

## Phase P2 — Hill-climbing loop (3–5 days)

**Exit criterion:** A repeatable harness iteration loop that takes a proposed prompt change, validates it against the optimization (train) set, checks non-regression on holdout, and runs a human review gate.

- [ ] **P2.1** Write `scripts/aggregate.py`. Reads a run's JSONL, outputs `results/<run_id>_aggregate.json` with per-dim mean, stddev, bootstrap CI (n=1000), composite, and stratified breakdowns by composer_era and skill. ~120 lines numpy/scipy.
- [ ] **P2.2** Write `scripts/regression_check.py`. Given two run ids, prints a per-dim delta with significance (CIs overlapping = null, non-overlapping = signal). Flag any dim regressed by >1 CI.
- [ ] **P2.3** Wire `/autoresearch` skill to run `run_eval.py --split train` as the metric function, with composite as the optimized quantity. Guard: holdout split must not be touched during autoresearch.
- [ ] **P2.4** Run first prompt hill-climb experiment: one targeted change to teacher prompt (candidate: Style-Consistent dim, since it scored 0/3 in the old smoke test). Measure train-set delta, verify non-regression on other dims, then unlock holdout and verify the gain generalizes.
- [ ] **P2.5** Production trace flywheel. Sample ~5% of web-beta session syntheses into `data/production_trace_queue.jsonl`. Weekly human review promotes interesting failures into the eval as new cases, tagged with composer_era + skill + failure mode. This is the wiki's "dogfood → eval" flywheel and keeps the eval from going stale as the product evolves.
- [ ] **P2.6** Decide Phase 1 plateau. If 2–3 rounds of hill-climbing produce <0.1 composite gain on holdout, Phase 1 is done; Phase 2 (finetune) is justified. If gains are still coming, keep iterating the prompt — finetuning what you can still prompt-engineer is wasted effort.

## Phase P3 — Teacher finetune as quality gate (deferred, gated on P2.6)

**Exit criterion:** A one-command A/B harness that compares finetuned Qwen vs best-prompt Sonnet on train + holdout, with per-dim deltas and bootstrap CIs.

- [ ] **P3.1** Write `teacher_model/eval_ab.py`. Thin wrapper: runs `run_eval.py` twice (baseline Sonnet + finetuned Qwen), diffs aggregates, prints a per-dim delta table with significance flags. ~80 lines on top of `aggregate.py`.
- [ ] **P3.2** Swap the judge lineup for Phase 2 — Sonnet 4.6 now permitted as judge (teacher is no longer Sonnet). Dual-judge: GPT-5.4-mini + Sonnet 4.6. Re-run dual-judge calibration (P1.6) under the new lineup to confirm agreement holds on the new teacher's outputs.
- [ ] **P3.3** Gate the finetune against correctness first, efficiency second. Correctness: composite delta > 0 on holdout, no dim regressed. Efficiency: track cost/token/latency per synthesis — a finetune that matches Sonnet on composite but costs 10x less is a win; a finetune that scores +0.05 but takes 3x longer is not.
- [ ] **P3.4** If the finetune beats Sonnet under the gate, promote it to prod. Otherwise: either iterate finetune training data (return to `project_teacher_model_finetuning.md` pipeline) or revert the decision and continue with prompt-engineered Sonnet.

---

## Costs (rough)

| Phase | Cost | Notes |
|---|---|---|
| P0.7 baseline (Sonnet + Gemma) | ~$20 | 890 × Sonnet synthesis + 890 × Gemma judge |
| P1.4 process/outcome re-run (100 sample) | ~$3 | Partial re-judge |
| P1.6 dual-judge (100 sample) | ~$5 | GPT-5.4-mini adds ~$2, Gemma ~free |
| P2.4 per hill-climb iteration | ~$5 | Train-split only, cheap judge |
| P3.1 A/B run | ~$25 | Two full runs against 890 recordings |

Total through P2 plateau: roughly $50–$100 depending on hill-climb iteration count.

## Self-check before finetuning

Do not begin teacher model finetuning training until every box below is checked:

- [ ] Locked baseline exists on disk with frozen judge and rubric
- [ ] Holdout has never been touched during prompt iteration
- [ ] Dual-judge calibration complete; low-trust dims human-validated
- [ ] Prompt hill-climbing has demonstrably plateaued (<0.1 composite gain per iteration)
- [ ] `eval_ab.py` scaffold exists and has been dry-run successfully against two Sonnet baselines (to prove the A/B machinery works before introducing a real second model)
- [ ] The four strategic gates from `project_teacher_model_finetuning.md` (PMF, A/B test, rubric validation, 7B probe) have passed
