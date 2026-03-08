# Memory Eval Improvements (2026-03-07)

Location: `apps/api/evals/memory/`
Run: `cd apps/api/evals/memory && uv run python -m src.run_all --layer report`

## First Run Baseline

| Metric | Score | Target | Verdict |
|--------|-------|--------|---------|
| Retrieval F1 | 0.94 | 0.95 | Near |
| JSON parse rate | 1.00 | 0.95 | Pass |
| Invalidation R/P | 0.94 | 0.85 | Pass |
| Temporal accuracy | 1.00 | 0.85 | Pass |
| Abstention | 1.00 | 0.80 | Pass |
| Fact type accuracy | 1.00 | 0.90 | Pass |
| Trend accuracy | 0.85 | 0.80 | Pass |
| Downstream win rate | 60% | 70% | Below |
| Continuity delta | +0.73 | +1.0 | Below |
| New fact recall | 0.55 | 0.80 | Below |
| Hallucination rate | 0.87 | 0.10 | Below |

## Root Cause

Most failures are **matching failures, not system failures**. The LLM produces correct facts but the eval's regex-only matching strategy is too brittle. Example: LLM says "Student struggles with dynamic contrast" but regex expects `(flat|uniform|limited|lack).*(dynamic|contrast)`.

## Improvement Plan (priority order)

### 1. Add sentence-transformer cosine similarity fallback (highest leverage)

- In `eval_synthesis.py` `_match_fact()`, after regex fails, compute cosine similarity between produced fact_text and expected fact_text_pattern (stripped of regex syntax)
- Threshold: >0.85 counts as match
- `sentence-transformers` already available in the model env; add to `apps/api/evals/memory/pyproject.toml`
- Expected impact: recall 0.55 -> ~0.80, hallucination 0.87 -> ~0.30

### 2. Broaden regex patterns in `build_dataset.py`

- Many patterns enforce word order that LLM doesn't follow
- E.g. `pedal.*(improv|clear|better)` misses "improvement in pedaling"
- Switch to unordered keyword sets or use `(?=.*word1)(?=.*word2)` lookaheads
- Affects: all 30 scenarios in `build_dataset.py`

### 3. Account for approach facts in scoring

- LLM consistently creates engagement/approach facts (the synthesis prompt encourages this)
- Currently these count as "hallucinations" when not in expected_new_facts
- Fix: either add approach facts to expected lists, or exclude `fact_type=approach` from hallucination denominator in `eval_synthesis.py`

### 4. Improve gold facts for downstream eval

- `eval_downstream.py` builds memory context using placeholder text derived from regex patterns (e.g. "[Pattern] flat/uniform/limited/lack dynamic/contrast")
- Replace with realistic natural-language fact text in `build_dataset.py` (add a `gold_fact_text` field to ExpectedFact)
- Expected impact: downstream win rate 60% -> 70%+

### 5. Knowledge update matching

- te-01 and te-02 fail knowledge_update because old facts aren't fully invalidated (LLM sometimes rewrites rather than invalidating + creating new)
- Consider matching invalidation by dimension + fact_text similarity, not just fact_id

## After Fixes

1. Clear synthesis cache: `rm apps/api/evals/memory/data/synthesis_cache.jsonl`
2. Clear downstream caches: `rm apps/api/evals/memory/data/downstream_cache.jsonl apps/api/evals/memory/data/judge_results.jsonl`
3. Re-run: `cd apps/api/evals/memory && uv run python -m src.run_all --live`
