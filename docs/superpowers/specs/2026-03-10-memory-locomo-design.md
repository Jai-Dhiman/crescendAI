# Memory System: LoCoMo Benchmark + Knowledge Graph Layer

**Date:** 2026-03-10
**Goal:** Best-in-class domain memory with competitive LoCoMo scores (target 70-80 F1).

## Context

CrescendAI's memory system uses bi-temporal facts + LLM extraction. Chat extraction eval scores: recall=1.000, precision=0.864, operation accuracy=0.989. LoCoMo has never been run -- adapter has structural bugs.

### Competitive Landscape (LoCoMo F1)

| System | F1 | Approach |
|--------|-----|----------|
| MemMachine | 84.87 | Multi-graph consolidation |
| Mem0 | 58.44 | Hybrid vector + graph |
| GPT-4 baseline | 32.1 | Zero-shot |
| CrescendAI | TBD | Bi-temporal facts + LLM extraction |

## Layer 1: Infrastructure Fixes

### 1a. Fix LoCoMo adapter fact processing

`locomo_adapter.py` lines 257-282 read `extract_result.get("actions", [])` but the API returns `{"add": [...], "update": [...]}`. Replace with correct field names matching `eval_chat_extraction.py`.

### 1b. Rate limit retry

Both extraction and QA calls hit Groq (12k TPM on free tier). Add retry wrapper: catch 429/503, read `retry-after` header or default 2s backoff, max 3 retries.

### 1c. Fact IDs for UPDATE tracking

Pass stable `id` fields on accumulated facts so the extraction prompt can reference them for UPDATE operations.

### 1d. Download LoCoMo data

`locomo10.json` from snap-research/LoCoMo repo into `data/`.

## Layer 2: Structured Retrieval + QA (Approach B)

### 2a. Relevance filtering

LoCoMo conversations accumulate 50-100+ facts. Two-stage filter before QA:
1. Keyword pre-filter: extract entities/terms from question, filter facts to matches
2. Cosine similarity re-rank: if >15 facts survive, sentence-transformer ranks by question relevance, keep top 15

Applies to LoCoMo adapter only -- domain fact sets are narrow enough.

### 2b. Chain-of-thought QA prompt

Replace minimal "answer concisely" prompt with CoT: identify relevant facts, trace reasoning chains for multi-hop, use dates for temporal questions, answer concisely.

### 2c. Fact grouping by topic

Organize facts into sections (People, Events, Preferences, etc.) in QA context instead of flat list.

### 2d. Temporal awareness

Include `current_date` in QA prompt for relative time reasoning.

## Layer 3: Knowledge Graph (Approach C)

### 3a. Entity extraction

Extend `CHAT_EXTRACTION_SYSTEM` output format to optionally include entities and relations:

```json
{
  "add": [{
    "fact_text": "...",
    "category": "...",
    "entities": ["Sarah", "Mike", "Bali"],
    "relations": [{"s": "Sarah", "r": "met_at", "o": "Mike", "ctx": "yoga retreat"}]
  }]
}
```

Backward compatible -- Rust API parses if present, ignores if absent.

### 3b. Entity-hop retrieval

For multi-hop QA:
1. Extract entities from question
2. Find facts mentioning those entities
3. Collect related entities from matched facts
4. One-hop expansion: find facts mentioning related entities
5. Deduplicate, rank, send combined set to QA

Implemented in LoCoMo adapter as retrieval strategy over accumulated facts.

### 3c. Domain compatibility

Same entity format naturally captures pieces, composers, teachers, events in piano domain. No domain integration now -- just compatible format for later wiring.

### 3d. Scope boundaries

- No graph DB (Neo4j, etc.)
- No vector embeddings for entities
- No entity resolution/deduplication
- No D1 schema changes (entities in eval adapter only)

## Files Touched

| File | Changes |
|------|---------|
| `evals/memory/src/locomo_adapter.py` | Bulk: adapter fixes, retry, relevance filter, CoT QA, entity-hop retrieval |
| `apps/api/src/services/prompts.rs` | Extend extraction output format for optional entities/relations |
| `apps/api/src/services/memory.rs` | Parse entities/relations if present (backward compatible) |

## Success Criteria

- LoCoMo overall F1 > 65 (beats Mem0)
- LoCoMo multi-hop F1 > 55 (proves graph layer)
- Chat extraction recall stays 1.0, precision stays > 0.85
