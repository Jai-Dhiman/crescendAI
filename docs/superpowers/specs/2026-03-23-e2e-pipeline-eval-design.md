# End-to-End Pipeline Evaluation Harness

**Date:** 2026-03-23
**Status:** Design approved
**Priority:** Observation quality > STOP accuracy > Piece ID accuracy

## Problem

The STOP classifier (logistic regression, LOVO AUC 0.649) was trained on masterclass audio. The full pipeline (STOP -> teaching moment selection -> subagent -> teacher observation) has never been stress-tested on intermediate students playing upright pianos with consumer microphones. We need to answer: does the pipeline produce useful, skill-appropriate observations on real-world audio?

## Approach

Three-layer composable eval built on the existing pipeline eval framework. Each layer is independently runnable and cacheable. All evaluation routes through the real system (wrangler dev via WebSocket).

### Test Corpus

T5 YouTube Skill dataset, 4 pieces (312 recordings):

| Piece | Recordings | Inference Cache |
|-------|-----------|-----------------|
| bach_prelude_c_wtc1 | 136 | needs generation |
| bach_invention_1 | 121 | needs generation |
| fur_elise | 28 | exists |
| nocturne_op9no2 | 27 | exists |

Each recording has a human-labeled skill bucket (1=beginner through 5=professional). A 5th piece (chopin_ballade_1, 6 recordings) is excluded due to insufficient sample size for meaningful per-bucket analysis.

## Layer 1: Inference Cache Auto-Generation

Extend `apps/evals/inference/eval_runner.py` to support T5 recordings.

The pipeline eval (`eval_practice.py`) reads inference cache from `model/data/eval/inference_cache/{fingerprint}/{recording_id}.json`, written by `eval_runner.py`. This is a different system from `run_inference.py` in skill_eval (which writes to `ensemble_4fold/` in a different format). Layer 1 uses `eval_runner.py` so cache is directly consumable by Layer 2.

**Cache format per recording** (existing, unchanged):
```json
{
  "recording_id": "str",
  "model_fingerprint": "str",
  "chunks": [
    {
      "chunk_index": 0,
      "predictions": {"dynamics": 0.55, "timing": 0.50, ...},
      "midi_notes": [{"pitch": 60, ...}],
      "pedal_events": [{"start": 0.0, "end": 1.5, ...}]
    }
  ]
}
```

**Changes to `eval_runner.py`:**
- Add `--auto-t5` flag: scans all 4 target pieces' manifests, identifies recordings without cached results, runs inference only for uncached recordings
- Targets local servers by default (`localhost:8000` for MuQ, `localhost:8001` for AMT)
- Health-checks both servers before starting; fails fast with clear message if either is down (tells user to run `just muq` / `just amt`)
- Reads T5 manifests from `model/data/evals/skill_eval/{piece}/manifest.yaml` for recording metadata (video_id, audio paths)
- Progress bar per piece (tqdm), estimated time remaining

**Existing behavior preserved:** Manual mode and other eval_runner workflows unchanged.

## Layer 2: Pipeline Eval Extension

Extend `apps/evals/pipeline/practice_eval/eval_practice.py` and supporting infrastructure.

### 2a. T5 Scenario Files

New scenario YAML files in `apps/evals/pipeline/practice_eval/scenarios/`, using the format `load_scenarios()` expects:

```yaml
# t5_bach_prelude_c_wtc1.yaml
# NOTE: no piece_query -- omitting it forces automatic piece identification
#       via the N-gram pipeline, which is what we want to test
candidates:
  - video_id: "abc123"
    include: true
    skill_level: 3
    title: "Bach Prelude in C - Student Performance"
    general_notes: "T5 skill corpus, bucket 3 (intermediate)"
```

Key design choice: T5 scenarios deliberately omit `piece_query`. Existing scenarios (fur_elise, nocturne) include `piece_query` which sends a `set_piece` WebSocket message, bypassing automatic piece identification. T5 scenarios skip this so we can test piece ID accuracy. The field `skill_level` is already supported by the existing loader (extracted at eval_practice.py line 149).

Generated from existing `manifest.yaml` files via a small script that maps `skill_bucket` to `skill_level` and `video_id` to the candidate format.

### 2b. Judge Prompt v3: Skill-Appropriateness Criterion

Extend v2 judge prompt with a 6th criterion:

**Skill Appropriateness (threshold: 0.60):** Given that this student is at skill level N (1=beginner, 5=professional), does the observation match their developmental stage? A beginner needs fundamentals (hand position, rhythm, basic dynamics). An intermediate needs musical concepts (phrasing, pedaling technique). An advanced/professional needs nuance (voicing, interpretive choices, stylistic references). Score YES if the observation's language, complexity, and focus area are appropriate for the stated level.

The judge receives the skill bucket as additional context alongside existing teaching moment metadata.

### 2c. New Metrics in Report

**STOP metrics** (from `eval_context.teaching_moment` in observation WebSocket messages):

STOP probability is already surfaced in observation messages when `is_eval_session=true`. Each observation includes `eval_context.teaching_moment.stop_probability` and `eval_context.teaching_moment.is_positive`. No API changes needed -- we extract these from the existing trace data.

- `stop_trigger_rate_by_bucket`: trigger rate per skill level
- `stop_probability_skill_correlation`: Spearman rho between STOP probability and skill bucket (expect negative)
- `stop_bucket_separation`: Cohen's d between adjacent bucket pairs

**Piece ID metrics** (from WebSocket `piece_identified` messages):

T5 scenarios omit `piece_query`, so the session attempts automatic piece identification via N-gram + rerank + DTW. Ground truth is the known piece from the scenario file.

- `piece_id_top1_accuracy`: fraction correctly identified
- `piece_id_top3_accuracy`: correct piece in top-3 candidates
- `piece_id_mean_notes_to_identify`: average notes before lock-in
- `piece_id_false_positive_rate`: wrong piece identified confidently

**Observation quality by bucket:**
- All 6 judge criteria broken down by skill bucket (5x6 matrix)
- Per-bucket N (sample size) alongside every metric -- flag cells where N < 5 as unreliable
- Worst-performing criterion per bucket
- Failure examples with full context

### 2d. Pipeline Client Extension

Add `piece_identified` to the WebSocket message types captured by `pipeline_client.py`. Currently captures `chunk_processed` and `observation` only.

## Layer 3: Analysis Script

**New file:** `apps/evals/pipeline/practice_eval/analyze_e2e.py`

Reads eval report JSON, produces cross-cutting analysis. No LLM calls.

**Outputs:**

1. **STOP Generalization Report** -- trigger rate by skill bucket, Cohen's d between adjacent buckets, comparison against masterclass training distribution, per-dimension STOP driver analysis
2. **Observation Quality Dashboard** -- 6-criteria pass rates by skill bucket (ASCII heatmap), worst criterion per bucket, skill-appropriateness breakdown, failure examples
3. **Piece ID Accuracy Report** -- confusion matrix, notes-to-identify distribution, failure analysis

**CLI:**
```bash
python analyze_e2e.py --report practice_eval.json           # full analysis
python analyze_e2e.py --report practice_eval.json --stop-only     # STOP only (cheap)
python analyze_e2e.py --report practice_eval.json --piece-id-only # piece ID only
```

## Data Flow

```
Layer 1: Inference Cache
  manifest.yaml (312 recordings)
       |
       v
  eval_runner.py --auto-t5
  (local MuQ:8000 + AMT:8001)
       |
       v
  model/data/eval/inference_cache/{fingerprint}/{recording_id}.json

Layer 2: Pipeline Eval
  scenario YAML + inference cache
       |
       v
  eval_practice.py --scenarios t5_*
  (wrangler dev WebSocket pipeline)
       |
       v
  practice_eval.json + practice_eval_observations.json

Layer 3: Analysis
  practice_eval.json
       |
       v
  analyze_e2e.py
       |
       v
  Terminal output (tables, correlations, failure examples)
```

## Prerequisites

- **Layer 1:** Local MuQ + AMT servers (`just muq` + `just amt`) -- only if cache missing
- **Layer 2:** Wrangler dev (`just api`), Groq + Anthropic API keys
- **Layer 3:** None (reads JSON)

## Justfile Additions

```just
# Run full E2E pipeline eval
eval-e2e: eval-cache eval-pipeline eval-analyze

# Generate missing inference cache (requires just muq + just amt)
eval-cache:
    cd apps/evals && uv run inference/eval_runner.py --auto-t5

# Run pipeline eval on T5 corpus (requires just api)
eval-pipeline:
    cd apps/evals && uv run pipeline/practice_eval/eval_practice.py --scenarios t5

# Analyze eval results
eval-analyze:
    cd apps/evals && uv run pipeline/practice_eval/analyze_e2e.py --report reports/practice_eval.json
```

## What We Are NOT Building

- No new WebSocket protocol changes (`eval_chunk` already carries scores + MIDI)
- No new API endpoints (piece ID messages already flow through WebSocket)
- No new judge infrastructure (extending existing `judge.py` with one criterion)
- No dashboard UI (terminal output sufficient)

## Next Steps: Autoresearch Loops

After the eval harness produces a baseline, set up autoresearch loops to iteratively improve the pipeline:

### Loop 1: STOP Threshold Tuning (cheapest)
- **Goal:** Maximize STOP skill-bucket correlation
- **Scope:** `apps/api/src/services/stop.rs` (weights, bias, threshold)
- **Metric:** `stop_probability_skill_correlation` (Spearman rho, target < -0.5)
- **Verify:** `python analyze_e2e.py --report practice_eval.json --stop-only`
- **Guard:** STOP trigger rate for bucket 5 must stay < 0.30

### Loop 2: Observation Skill-Appropriateness
- **Goal:** Maximize skill_appropriateness pass rate
- **Scope:** Teacher system prompt
- **Metric:** skill_appropriateness judge criterion pass rate (target > 0.75)
- **Verify:** `python eval_practice.py --pieces fur_elise --max-recordings 10`
- **Guard:** Other 5 criteria must not regress below thresholds

### Loop 3: Subagent Dimension Selection
- **Goal:** Improve observation grounding and technical specificity
- **Scope:** Subagent prompt
- **Metric:** mean(grounding, technical_specificity) pass rate
- **Verify:** `python eval_practice.py --pieces fur_elise --max-recordings 10`
- **Guard:** actionability must not regress

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO-0A | Test current per-observation pipeline, not upcoming session synthesis | P6 (action) | Session synthesis spec approved same day but not implemented. Eval baseline on current pipeline informs synthesis transition. | Wait for synthesis to ship first |
| 2 | CEO-0A | Add per-bucket N reporting to all metrics | P1 (completeness) | Fur_elise/nocturne have ~5 recordings per bucket. Statistics meaningless without sample size. | Report only aggregates |
| 3 | CEO-0C | Choose three-layer composable over monolithic | P1+P4 (completeness+DRY) | Already validated in brainstorming. Composable layers enable cheap re-analysis. | Monolithic single script |
| 4 | CEO-0D | Defer CI integration to TODOS | P3 (pragmatic) | Useful but separate concern. Get baseline first, then automate. | Add CI now |
| 5 | CEO-0D | Defer historical tracking to TODOS | P3 (pragmatic) | Need at least 2 runs before tracking adds value. | Add tracking now |
| 6 | CEO-0D | TASTE: Add bootstrap confidence intervals | P1 (completeness) | S effort, high value for sparse buckets. But could be deferred. Close call. | Defer to follow-up |
| 7 | CEO-S2 | Skip unavailable recordings with warning | P5 (explicit) | YouTube videos get taken down. Eval should not crash on missing audio. Report skipped count. | Crash on missing audio |
| 8 | CEO-S10 | Validate harness on existing 2 pieces first | P5 (explicit) | Compare fur_elise/nocturne results against existing practice_eval baseline before running on new bach pieces. | Run all 4 pieces immediately |
| 9 | ENG-S0 | Use HTTP client to call local servers (not direct model loading) | P5 (explicit) | Tests same interface as production HF endpoints. eval_runner currently loads models in-process; --auto-t5 should call localhost:8000/8001 via HTTP instead. | Reuse in-process model loading |
| 10 | ENG-S1 | SessionResult needs piece_identification field | P1 (completeness) | pipeline_client.py must capture piece_identified messages and expose them in SessionResult for metrics. | Parse from raw messages later |
| 11 | ENG-S1 | Add model_fingerprint to report metadata | P5 (explicit) | If local MuQ loads different model version than HF endpoint, STOP analysis is invalid. Fingerprint provides auditability. | Omit fingerprint |
| 12 | ENG-S2 | TASTE: Add per-recording checkpointing | P1 (completeness) | 312 recordings = hours of pipeline time. Wrangler crash loses all progress. But existing eval doesn't checkpoint either. | No checkpointing (match existing pattern) |
| 13 | ENG-S3 | Unit tests for all new code + integration test on existing pieces | P1 (completeness) | Every new codepath (manifest scan, HTTP client, scenario gen, analysis metrics) gets unit tests. Integration test validates against existing fur_elise baseline. | Skip tests |
| 14 | TASTE | Add bootstrap confidence intervals in analyze_e2e.py | P1 (completeness) | User approved. ~30 lines scipy, high value for sparse buckets. | Defer |
| 15 | TASTE | Add per-recording checkpointing in eval_practice.py | P1 (completeness) | User approved. ~30 lines, prevents losing 50min of pipeline work. | No checkpoint |

## Key Files

| File | Role |
|------|------|
| `apps/evals/inference/eval_runner.py` | Layer 1: inference cache generation (extend with --auto-t5) |
| `model/data/evals/skill_eval/{piece}/manifest.yaml` | T5 recording manifests with skill labels |
| `apps/evals/pipeline/practice_eval/eval_practice.py` | Layer 2: pipeline eval runner |
| `apps/evals/pipeline/practice_eval/analyze_e2e.py` | Layer 3: analysis (new) |
| `apps/evals/shared/pipeline_client.py` | WebSocket client (extend for piece_identified) |
| `apps/evals/shared/judge.py` | LLM judge framework |
| `apps/evals/shared/prompts/observation_quality_judge_v3.txt` | Judge prompt with skill-appropriateness (new) |
| `apps/evals/shared/reporting.py` | Metric aggregation (extend for STOP + piece ID) |
| `apps/api/src/services/stop.rs` | STOP classifier (target for autoresearch Loop 1) |

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | clean | Session synthesis transition noted. 3 expansions deferred. Per-bucket N reporting added. |
| Eng Review | `/plan-eng-review` | Architecture & tests | 1 | clean | HTTP client gap found and resolved. Test plan with 7 unit + 1 integration. Bootstrap CIs + checkpointing approved. |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | skipped | No UI scope (terminal-only eval) |

**VERDICT:** APPROVED via /autoplan. 15 decisions total (13 auto, 2 taste approved by user). Ready for implementation planning.
