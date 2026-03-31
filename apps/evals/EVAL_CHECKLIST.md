# Teaching Knowledge Eval -- Remaining Checklist

## Context

We built a teaching quality eval pipeline that:
1. Extracted 379 real teaching moments from YouTube piano masterclasses
2. Synthesized a Teaching Playbook grounded in 8 pedagogy frameworks
3. Derived a 7-dimension rubric for judging AI teacher synthesis quality
4. Built a judge (`judge_synthesis_v2()`) that scores synthesis output 0-3 per dimension
5. All LLM calls use Workers AI (GPT-OSS-120B + Qwen3) via CF AI Gateway -- NOT Anthropic

Key files:
- Playbook: `apps/evals/teaching_knowledge/data/playbook.yaml`
- Judge prompt: `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`
- Rubric: `apps/evals/shared/prompts/rubric_definition.json`
- LLM client: `apps/evals/teaching_knowledge/llm_client.py` (reads CF_API_TOKEN from `apps/api/.dev.vars`)
- Judge v2 function: `apps/evals/shared/judge.py` -> `judge_synthesis_v2()`
- Eval runner: `apps/evals/inference/eval_runner.py` (fixed MIDI key bug on 2026-03-30)
- Inference cache: `model/data/eval/inference_cache/auto-t5_http/` (JSON per recording)

## Step 1: Wait for inference cache to complete

The cache is rebuilding via `bash /tmp/cache_t5_loop.sh` (run from `apps/evals/`).
It processes 991 T5 recordings across 11 pieces (~2 min each, ~33 hours total).
Requires MuQ (port 8000) + AMT (port 8001) servers running (`just dev`).

Check progress:
```bash
ls model/data/eval/inference_cache/auto-t5_http/*.json | wc -l
# Target: ~991 (some may fail due to audio length limits)
```

Each cache entry is a JSON file with MuQ scores (6 dimensions) + AMT MIDI notes + pedal events per 15s chunk.

## Step 2: Write eval orchestrator script

No orchestrator exists yet. Create `apps/evals/teaching_knowledge/run_eval.py` that:

1. Loads all cache entries from `model/data/eval/inference_cache/auto-t5_http/`
2. For each recording:
   a. Aggregate MuQ scores across chunks (mean per dimension)
   b. Look up piece name, composer, skill level from T5 manifests in `model/data/evals/skill_eval/*/manifest.yaml`
   c. Generate a teacher synthesis using `LLMClient` (Workers AI, GPT-OSS-120B) -- see the smoke test prompt pattern in the conversation history or adapt from the pipeline's actual teacher prompt
   d. Judge the synthesis using `judge_synthesis_v2()` from `shared/judge.py`
   e. Save results (recording_id, piece, skill_level, dimension scores, synthesis text)
3. Aggregate results into a baseline scorecard

Use `--provider workers-ai` (default) for both synthesis and judging. Cost: ~$5 for 991 recordings.

## Step 3: Analyze baseline scores

After the eval run, analyze:
- Mean/median score per dimension (which are weakest?)
- Score distribution per skill bucket (1-5)
- Score distribution per piece/composer
- Correlation between dimensions
- Worst 10 syntheses (lowest mean score) -- read them manually

## Step 4: Fix Style-Consistent Musical Language (known issue)

The smoke test scored 0/3 on "Style-Consistent Musical Language" because the teacher prompt doesn't inject piece-style rules. The playbook has `piece_style_dimension_rules` with per-style priorities (Baroque, Classical, Romantic, Impressionist, Jazz, Contemporary).

Fix: when generating the synthesis prompt, look up the piece's style/composer and inject the relevant dimension priorities from the playbook. Example: for Bach, inject "articulation: very high, pedaling: minimal, dynamics: terraced, timing: strict."

## Step 5: Re-run eval and compare

After prompt fixes, re-run the eval on the same cache entries and compare scores. The cache doesn't need to be regenerated -- only the synthesis + judge step reruns.

## Step 6: Human calibration (50 syntheses)

Manually review 50 judge-scored syntheses across different pieces and skill levels. Check:
- Do the 0-3 scores match your intuition?
- Is the rubric calibrated (is a "3" actually excellent, is a "0" actually harmful)?
- Are there failure modes the rubric misses?

## Notes

- Workers AI client: `LLMClient(provider="workers-ai", tier="quality")` uses GPT-OSS-120B
- Qwen3 requires `/no_think` suffix (handled automatically by LLMClient)
- GPT-OSS-120B wraps JSON in markdown fences (handled by `complete_json()`)
- The judge prompt is self-contained -- rubric is baked into the prompt text
- T2 source (`--source t2`) pulls from skill_eval manifests which are PERFORMANCE videos, not masterclasses. Use `--source search` for teaching transcripts.
