# Memory System Eval Improvement + Supermemory-Inspired Experiments

**Date:** 2026-03-23
**Status:** Design approved
**Scope:** Eval suite upgrade, autoresearch loop on memory system, supermemory pattern experiments

## Context

The memory eval suite has recall=1.0, precision=0.722 on 38 synthetic scenarios. All scenarios are hand-crafted with clean observation sequences. Matching uses hand-crafted regex (Pass 1) with cosine similarity fallback at threshold 0.55. Two problems: scenarios don't reflect real student behavior, and matching is brittle to LLM paraphrasing.

Research into [supermemory](https://github.com/supermemoryai/supermemory) patterns identified three improvements worth testing: supersession chains, proactive staleness decay, and semantic dedup before synthesis. CrescendAI's bi-temporal system already handles most of what supermemory offers (dual-layer profiles, contradiction via invalidation, hybrid retrieval), but these three gaps could improve fact quality.

## Approach

Eval-first, then memory system. Upgrade the eval to produce trustworthy metrics, then use autoresearch to iterate on the memory system one experiment at a time.

## Phase 1: Eval Suite Upgrade

### 1a. Simulated-Realistic Scenarios

Generate 15-20 new synthesis scenarios and 5-8 temporal scenarios using an LLM to simulate messy real student behavior. Each scenario gets a student persona + practice arc, then the LLM generates 6-12 observations across 2-4 sessions.

**Scenario categories (synthesis):**

| Category | Count | What it tests |
|----------|-------|---------------|
| Incomplete sessions | 3 | Student plays 2 chunks then stops. Sparse data. |
| Mid-session piece changes | 3 | Switches pieces mid-practice. Facts should be piece-scoped. |
| Multi-session arcs | 4 | Same piece over 3 sessions: bad -> improved -> plateau. Trend detection. |
| Vague engagement | 2 | Ambiguous student responses. System should not over-infer. |
| Contradictory observations | 3 | Dynamics flat -> improved -> regressed. Invalidation + supersession. |
| Sparse data | 2 | Only 2 observations total. System should abstain from high-confidence facts. |

**Scenario categories (temporal):**

| Category | Count | What it tests |
|----------|-------|---------------|
| Delayed creation | 2 | Fact should appear after session 3, not after session 2 |
| Cross-session invalidation | 3 | Contradictory evidence 2+ weeks later triggers invalidation |
| Abstention | 3 | Insufficient evidence, system should NOT create a fact |

**Expected facts are hand-annotated** after reviewing generated observations. Ground truth comes from human judgment, not the LLM.

**Format:** Same `MemoryEvalScenario` dataclass. Stored in `data/realistic_scenarios.jsonl` (synthesis) and integrated into temporal layer.

### 1b. Matching Strategy as Experiment

Keep regex+cosine as the baseline matching strategy. LLM-as-judge is experiment #1 in the autoresearch queue -- tested empirically, not assumed better.

If tested, the judge would be a small model (Groq/Llama 8B) scoring semantic equivalence of (produced_fact, expected_fact) pairs with a music-teaching-aware prompt. Cosine similarity at 0.3 threshold as pre-filter to reduce LLM calls.

### 1c. Machine-Readable Output

Add `--json-output` flag to `run_all.py` that emits machine-readable scores. Also fix multi-layer CLI parsing (`--layer` flag currently only reads the first occurrence; update argparse to accept multiple `--layer` values):

```json
{
  "synthesis": {"recall": 0.72, "precision": 0.68, "f1": 0.70},
  "temporal": {"assertion_accuracy": 0.65, "creation_accuracy": 0.80, "invalidation_accuracy": 0.50},
  "chat_extraction": {"recall": 0.90, "precision": 0.72},
  "composite": 0.69
}
```

Composite = `0.4 * synthesis_recall + 0.3 * temporal_assertion_accuracy + 0.3 * chat_extraction_precision`

**Note:** New scenarios are added for synthesis and temporal layers only. Chat extraction uses the existing 38 scenarios as-is -- its 0.3 weight in the composite provides a regression signal, not a new optimization target.

## Phase 2: Autoresearch Loop

### Setup

```
Goal: Maximize memory system quality (synthesis + temporal + chat extraction)
Scope: apps/api/src/services/prompts.rs, apps/api/src/services/memory.rs, apps/evals/memory/src/eval_synthesis.py
Metric: Composite score (0.4 * synthesis_recall + 0.3 * temporal_assertion_accuracy + 0.3 * chat_extraction_precision)
Verify: cd apps/evals/memory && uv run python -m src.run_all --layer synthesis --layer temporal --layer chat_extraction --live --json-output
Guard: cd apps/evals/memory && uv run python -m src.run_all --layer retrieval
Iterations: 15
Target: 0.85
```

### Experiment Queue

Ordered by expected impact. Autoresearch reviews failures after each iteration and can re-prioritize.

| # | Experiment | Scope | Hypothesis |
|---|-----------|-------|-----------|
| 0 | Baseline | -- | Establish starting metrics on new + existing scenarios |
| 1 | LLM-as-judge matching | `eval_synthesis.py` | Semantic matching catches facts that regex misses, improving measured recall |
| 2 | Synthesis prompt: multi-session awareness | `prompts.rs` | Adding cross-session guidance improves recall on multi-session arc scenarios |
| 3 | Synthesis prompt: abstention guidance | `prompts.rs` | Explicit "don't create facts from < 3 observations" reduces false positives |
| 4 | Supersession chains | `memory.rs` + migration | `superseded_by` field makes contradiction handling auditable and more accurate |
| 5 | Proactive staleness decay | `memory.rs` | Confidence downgrade for facts with no evidence in 30+ days keeps fact set fresh |
| 6 | Semantic dedup before synthesis | `memory.rs` | Cosine-dedup active facts before synthesis reduces prompt noise |

### Stop Criteria

- Composite exceeds 0.85
- 15 iterations exhausted
- 5 consecutive no-improvements

## Phase 3: Supermemory-Inspired Experiments (Detail)

### Experiment 4: Supersession Chains

- Add `superseded_by: Option<String>` to `SynthesizedFact` struct
- D1 migration: `ALTER TABLE synthesized_facts ADD COLUMN superseded_by TEXT`
- `run_synthesis()`: When invalidating a fact and creating its replacement, set `superseded_by = new_fact_id` on the old fact
- Synthesis prompt gets instruction: "When invalidating a fact, explain what replaces it."
- No retrieval or client changes (queries already filter `invalid_at IS NULL`)

### Experiment 5: Proactive Staleness Decay

- New function `decay_stale_facts()` called during synthesis, before prompt construction
- Facts with `valid_at` > 30 days AND no supporting observations in 30 days get confidence downgraded: high -> medium, medium -> low
- Pre-synthesis filter, updates confidence in D1
- No new columns, no retrieval changes

### Experiment 6: Semantic Dedup Before Synthesis

- New function `dedup_facts_for_synthesis()` called before `build_synthesis_prompt()`
- Loads active facts, computes pairwise cosine similarity via Workers AI embeddings (production) or sentence-transformers (eval/local)
- Merges facts above 0.85 similarity (keeps more recent, combines evidence arrays)
- In-memory only -- D1 facts untouched. Synthesis LLM sees cleaner fact list.
- No schema changes, no permanent dedup
- Eval testing: since `--live` runs against the local API (`wrangler dev`), Workers AI is available. For offline eval, fall back to sentence-transformers.

## Key Design Decisions

1. **Session greeting dropped** -- Memory already flows into every `/api/chat` and `/api/ask` request. No separate greeting needed.
2. **Eval-first** -- Upgrade eval before touching memory system. Can't optimize against a metric you don't trust.
3. **LLM-as-judge is an experiment, not an assumption** -- Tested via autoresearch, kept only if it improves metrics.
4. **One change at a time** -- Autoresearch discipline. Each experiment is atomic, attributed, and reversible.
5. **No vector search** -- Domain is narrow enough for structured SQL + hybrid scoring. Supermemory's embedding-based retrieval is overkill here.
6. **No graph structure** -- 6 dimensions + ~240 pieces don't need entity graphs. Flat facts with metadata suffice.

## Files Changed

| File | Change |
|------|--------|
| `apps/evals/memory/data/realistic_scenarios.jsonl` | New: 15-20 simulated-realistic synthesis scenarios |
| `apps/evals/memory/src/build_realistic_scenarios.py` | New: LLM-powered scenario generator |
| `apps/evals/memory/src/run_all.py` | Add `--json-output` flag |
| `apps/evals/memory/src/eval_synthesis.py` | Optional: LLM-as-judge matching (experiment 1) |
| `apps/api/src/services/prompts.rs` | Synthesis prompt improvements (experiments 2-3) |
| `apps/api/src/services/memory.rs` | Supersession chains, staleness decay, semantic dedup (experiments 4-6) |
| `apps/api/migrations/NNNN_supersession.sql` | New: `superseded_by` column (experiment 4) |
| `apps/evals/memory/results.tsv` | New: autoresearch tracking |
| `apps/evals/memory/changelog.md` | New: autoresearch experiment log |

## Out of Scope

- Session opening greeting / proactive first message
- Vector/embedding-based retrieval (replace hybrid scoring)
- Graph-based entity relationships
- Document chunking (supermemory pattern, not relevant)
- Production deployment of memory system changes (this is eval + experiment only)
