# Teaching Knowledge Eval -- Baseline Runbook

## Prereqs

Two eval entrypoints, two different prereq sets. Get these right before running, or you'll burn 30 minutes diagnosing auth errors that look like pipeline bugs.

### Synthesis-quality eval (`run_eval.py`) — no wrangler dependency
- `ANTHROPIC_API_KEY` set (teacher) and either `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` (Workers AI judge) or `OPENROUTER_API_KEY` (OpenRouter judge).
- Inference cache present at `model/data/eval/inference_cache/auto-t5_http/`.
- Does **not** require `wrangler dev` running. Runs entirely Python-side.

### DO-level integration eval (`pipeline/practice_eval/eval_practice.py`) — needs local wrangler
This path drives the real `SessionBrain` Durable Object over WebSocket on `localhost:8787`. Before invoking:

1. **Cloudflare auth must be fresh.** Either:
   - Run `wrangler login` (browser SSO; persists in `~/.config/.wrangler/config/default.toml`), OR
   - Export a current `CLOUDFLARE_API_TOKEN` with Workers + AI + R2 + D1/Hyperdrive scopes.

   Stale auth manifests as a boot failure at `/accounts/.../workers/subdomain/edge-preview` because `env.AI` is a `remote` binding. The wrangler error is unhelpful — "edge-preview" failures almost always mean reauth, not a code problem.

2. **Boot wrangler dev:** `just api` (or `cd apps/api && bun run dev`). Wait for `Ready on http://localhost:8787` before invoking the harness.

3. **Verify the WebSocket is reachable:**
   ```bash
   curl -sf http://localhost:8787/health | head
   ```

4. **Then run the harness:**
   ```bash
   cd apps/evals
   uv run python -m pipeline.practice_eval.eval_practice --scenarios t5 --max-recordings 1
   ```

   Expected on a healthy run: `synthesis (N chars)` with N > 0 in stdout; `synthesis_latency_ms` in `reports/practice_eval_details.json` > 500 (real Anthropic call, not a short-circuit).

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

## Producing the HONEST baseline through the real DO (#22, canonical)

The `run_eval.py` path above feeds the teacher a thin Python summary; it overstates
quality. The honest baseline replays the holdout through the **real SessionBrain DO**
(`buildSynthesisFraming`), which is the production framing. This is the canonical
baseline as of #22 — locked into `teacher_model/stage0/tests/test_aggregator.py`
(`_SONNET_BASELINE`).

Prereqs:
- Local `wrangler dev` on :8787 (`cd apps/api && bun run dev`). Needs `.dev.vars`.
- The eval studentId override is **fail-closed**: set BOTH in `apps/api/.dev.vars`:
  `ALLOW_EVAL_STUDENT_OVERRIDE=true` and `EVAL_SHARED_SECRET=<dev secret>`. The harness
  sends `x-eval-secret`; export the same value: `export EVAL_SHARED_SECRET=<dev secret>`.
  These vars are absent in production, so the override is unreachable there.
- DB connection cap: `createDb` uses `max:5, idle_timeout:10` so sequential sessions
  don't exhaust Postgres (`too many clients`). If you still hit it, lower `max` further.

```bash
# Smoke one recording (confirm a real ~10s Anthropic call, no ERROR:)
cd apps/evals && uv run python -m teaching_knowledge.run_eval --do-path --limit 1 \
    --teacher-model claude-sonnet-4-6 --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --out results/smoke_do_$(date -u +%Y%m%d).jsonl

# Full holdout (98 recordings) through the DO — resume-safe, must be <5% errors
cd apps/evals && uv run python -m teaching_knowledge.run_eval --do-path \
    --teacher-model claude-sonnet-4-6 --judge-model '@cf/google/gemma-4-26b-a4b-it' \
    --out results/baseline_v2_do.jsonl

# Aggregate (run from apps/evals so the default --dataset-index resolves)
cd apps/evals && uv run python -m teaching_knowledge.scripts.aggregate \
    results/baseline_v2_do.jsonl \
    --dataset-index teaching_knowledge/data/dataset_index.jsonl \
    --out results/baseline_v2_do_aggregate.json
```

Then re-lock `_SONNET_BASELINE` in `teacher_model/stage0/tests/test_aggregator.py` from
the aggregate, and confirm `pytest teacher_model/stage0/tests/test_aggregator.py` passes.
Current locked numbers: ASCF `mean_outcome` 0.959, composite 1.060 (n=98) — below the old
thin-path 1.387/2.483 because the honest cold-start framing is leaner, not a regression.

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
