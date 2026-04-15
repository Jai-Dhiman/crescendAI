# Teaching Knowledge Eval -- Baseline Runbook

## What's shipped

All infrastructure for producing a locked Sonnet 4.6 baseline is in place as of 2026-04-15 (`feat/eval-baseline-readiness`). The eval pipeline is:

1. Inference cache → MuQ + AMT scores per 15s chunk (`model/data/eval/inference_cache/auto-t5_http/`)
2. `tag_dataset.py` → enrich with composer_era, skill bucket, duration bucket (`data/dataset_index.jsonl`)
3. `split.py` → 80/20 stratified train/holdout (`data/splits.json`) — **holdout is sacred**
4. `run_eval.py` → synthesis (teacher) + judge (cross-family), style-injected from playbook, provenance-stamped, writes JSONL
5. `scripts/aggregate.py` → per-dim means + bootstrap CIs + stratified breakdowns
6. `scripts/regression_check.py` → run-vs-run delta with CI-overlap significance
7. `scripts/dual_judge.py` → Spearman-based cross-family judge calibration
8. `teacher_model/eval_ab.py` → Sonnet vs finetuned Qwen A/B harness

Key files:
- Playbook: `apps/evals/teaching_knowledge/data/playbook.yaml`
- Style rules (single source): `apps/evals/shared/data/style_rules.json` (mirrored from `apps/api/src/lib/style-rules.json`)
- Judge prompt: `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`
- Rubric: `apps/evals/shared/prompts/rubric_definition.json`
- Judge function: `apps/evals/shared/judge.py` → `judge_synthesis_v2()` (process/outcome schema)
- Cross-family guard: `apps/evals/shared/judge_compatibility.py`
- LLM client: `apps/evals/teaching_knowledge/llm_client.py` (Workers AI + Anthropic + OpenRouter providers)

## Producing the locked baseline (P0.7)

Blocked on: refreshed MuQ inference cache (Model v2 training output). Once the cache is current:

```bash
cd apps/evals

# Sanity: run on 3 recordings to confirm wiring
uv run python -m teaching_knowledge.run_eval \
    --limit 3 \
    --split train \
    --teacher-model claude-sonnet-4-6 \
    --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --no-dry-run \
    --out results/smoke_$(date -u +%Y%m%d).jsonl

# Locked baseline: 80% of 890 recordings (train split only)
uv run python -m teaching_knowledge.run_eval \
    --split train \
    --teacher-model claude-sonnet-4-6 \
    --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --no-dry-run \
    --out results/baseline_sonnet46_judge-gemma4_$(date -u +%Y%m%d).jsonl

# Aggregate
uv run python -m teaching_knowledge.scripts.aggregate \
    results/baseline_sonnet46_judge-gemma4_YYYYMMDD.jsonl
```

Estimate: ~$20 total (Sonnet synthesis + Workers AI Gemma judge), ~1 hour wall time.

## Cross-family constraint

Same-family teacher/judge pairings are blocked at `run_eval.py` startup by `assert_judge_compatible`. Allowed pairings:

| Teacher family | Judge family |
|---|---|
| Anthropic (Claude) | Workers AI (Gemma-4) |
| Anthropic (Claude) | OpenRouter (GPT-5.4-mini) |
| Qwen (finetuned, post-Phase 1) | Anthropic (Sonnet 4.6) |
| Qwen (finetuned, post-Phase 1) | OpenRouter (GPT-5.4-mini) |

## Regression check (run-vs-run)

```bash
uv run python -m teaching_knowledge.scripts.regression_check \
    --baseline results/baseline_YYYYMMDD.jsonl \
    --candidate results/candidate_YYYYMMDD.jsonl
```

Flags any dimension where the candidate's bootstrap CI does not overlap the baseline's. Exit code nonzero on regression for CI integration.

## Dual-judge calibration (P1.6)

```bash
# Run the same recordings through two judges
uv run python -m teaching_knowledge.run_eval --judge-model '@cf/google/gemma-4-26b-a4b-it' --out results/judge_gemma.jsonl
uv run python -m teaching_knowledge.run_eval --judge-model 'openai/gpt-5.4-mini'       --out results/judge_gpt.jsonl

# Compute per-dim Spearman rank agreement
uv run python -m teaching_knowledge.scripts.dual_judge \
    --judge-a results/judge_gemma.jsonl \
    --judge-b results/judge_gpt.jsonl \
    --out results/dual_judge_agreement.json
```

Dimensions classify into: high-trust (ρ ≥ 0.7), uncertain (0.4–0.7), low-trust (< 0.4). Low-trust dims should be human-validated before trusting any automated number.

## A/B harness (P3.1)

```bash
uv run python -m teacher_model.eval_ab \
    results/baseline_sonnet46_judge-gemma4_YYYYMMDD.jsonl \
    results/candidate_qwen_judge-sonnet46_YYYYMMDD.jsonl
```

Prints per-dim delta, efficiency delta (synthesis + judge latency), and verdict: `CANDIDATE_WINS` / `CANDIDATE_LOSES` / `EQUIVALENT`.

## Still open

- P0.8: git-tag the judge prompt + rubric as `judge-v2.0-locked` after the first baseline run
- P1.1: phantom-criteria audit on the 7 dims after the baseline exists
- P1.8: human calibration on low-trust dims from dual-judge output
- P2.3: wire `/autoresearch` to run `run_eval.py --split train` as the optimized metric
- See `docs/plans/2026-04-14-eval-improvements.md` for the full P0–P3 roadmap
