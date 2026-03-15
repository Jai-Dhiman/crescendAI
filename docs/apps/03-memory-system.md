# Student Memory System

How CrescendAI remembers what it knows about a student across sessions. This is the data inventory for the apps layer -- parallel to how `model/01-data.md` catalogs training data, this doc catalogs the runtime data structures that accumulate as a student practices.

> **Status (2026-03-14):** Observations table COMPLETE (ships with the `/api/ask` pipeline). Synthesized facts DEFERRED until the pipeline has enough real observation data to validate synthesis quality.

See `02-pipeline.md` for the teacher pipeline that produces observations.
See `model/02-teacher-grounded-taxonomy.md` for the 6 dimensions that define the baseline schema.
See `02-pipeline.md` for the subagent that consumes memory at inference time.

---

## Core Concept: Two Clocks

From Koratana's "How to build a context graph" and Marple's "Building the Event Clock": every system has a **state clock** (what is true now) and an **event clock** (what happened, in what order, with what reasoning). Most systems only build the state clock.

### State Clock (Baselines)

What level is the student at, right now, per dimension?

- Exponential moving average per dimension (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Inferred overall level
- Explicit goals (student-reported)
- Already implemented in SwiftData (iOS) + D1 (cloud)

The state clock answers: "Where does this student stand?"

### Event Clock (Reasoning Traces)

What happened, when, and why did the system say what it said?

- History of observations: what was flagged, why it was flagged, how it was framed
- Condensed subagent reasoning traces attached to each observation
- Temporal facts synthesized from observation patterns

The event clock answers: "What has the teacher noticed, and how has the student responded?"

### Why Piano Teaching Needs Both

A teacher who has worked with a student for months knows:
- "We have talked about pedaling three times. She knows it is an issue. Frame as progress check, not discovery."
- "His dynamics improved dramatically after we worked on the Chopin. That approach worked."
- "She always struggles with the development section of new pieces. That is her pattern."

Without the event clock, the subagent rediscovers patterns from scratch each session.

---

## Data Model

### Observations Table

Episode capture -- built with the `/api/ask` pipeline. Each row is one teaching observation delivered to the student.

| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | Unique observation ID |
| `student_id` | TEXT FK | References `students(apple_user_id)` |
| `session_id` | TEXT FK | References `sessions(id)` |
| `chunk_index` | INTEGER | Audio chunk that triggered this observation |
| `dimension` | TEXT NOT NULL | One of the 6 teacher-grounded dimensions (see `model/02-teacher-grounded-taxonomy.md`) |
| `observation_text` | TEXT NOT NULL | The observation as delivered to the student |
| `elaboration_text` | TEXT | Extended explanation (if requested) |
| `reasoning_trace` | TEXT (JSON) | Condensed subagent reasoning for this observation |
| `framing` | TEXT | One of: correction, recognition, encouragement, question |
| `dimension_score` | REAL | MuQ score for this dimension on this chunk |
| `student_baseline` | REAL | Student's baseline for this dimension at time of observation |
| `piece_context` | TEXT (JSON) | `{composer, title, section, bar_range}` |
| `learning_arc` | TEXT | One of: new, mid-learning, polishing |
| `created_at` | TEXT NOT NULL | Timestamp |

```sql
CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    chunk_index INTEGER,
    dimension TEXT NOT NULL,
    observation_text TEXT NOT NULL,
    elaboration_text TEXT,
    reasoning_trace TEXT,
    framing TEXT,
    dimension_score REAL,
    student_baseline REAL,
    piece_context TEXT,
    learning_arc TEXT,
    created_at TEXT NOT NULL
);
```

**What is stored:** Every observation the teacher pipeline delivers. The `reasoning_trace` preserves the subagent's analysis so that later synthesis can understand *why* the system said what it said, not just what it said.

**How it is queried:** By `student_id`, ordered by `created_at DESC`. Filtered by `dimension` when building dimension-specific context. Filtered by `piece_context` for piece-specific history.

### Synthesized Facts Table

Event clock -- produced asynchronously by the memory synthesis process. Each row is a temporal assertion about the student.

| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | Unique fact ID |
| `student_id` | TEXT FK | References `students(apple_user_id)` |
| `fact_text` | TEXT NOT NULL | Natural language assertion, e.g. "Pedaling has been a persistent weakness but is improving" |
| `dimension` | TEXT | Nullable -- some facts span dimensions |
| `valid_at` | TEXT NOT NULL | When this fact became true in the world |
| `invalid_at` | TEXT | When this fact stopped being true (null = still active) |
| `trend` | TEXT | One of: improving, stable, declining, new |
| `confidence` | TEXT | One of: high, medium, low |
| `evidence` | TEXT (JSON) | Array of observation IDs that support this fact |
| `source_type` | TEXT NOT NULL | One of: synthesized, student_reported, inferred |
| `created_at` | TEXT NOT NULL | When this fact was recorded in the system |
| `expired_at` | TEXT | When this fact was superseded in the system |

```sql
CREATE TABLE synthesized_facts (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    fact_text TEXT NOT NULL,
    dimension TEXT,
    valid_at TEXT NOT NULL,
    invalid_at TEXT,
    trend TEXT,
    confidence TEXT,
    evidence TEXT,
    source_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expired_at TEXT
);
```

**Bi-temporal fields:** The table tracks two timelines separately:
- **World timeline:** `valid_at` / `invalid_at` -- when was this true about the student?
- **System timeline:** `created_at` / `expired_at` -- when did the system record or supersede this fact?

This separation matters because contradictions are handled by invalidation, not deletion. When "pedaling is a persistent weakness" is contradicted by "pedaling improved over 3 sessions," the old fact gets `invalid_at` set and a new fact is created. The old fact remains in the table for history.

### How Observations Become Facts

**Trigger:** Background process, after a session ends. Triggered by `POST /api/sync` or a dedicated `POST /api/synthesize` endpoint. Never in the `/api/ask` hot path.

**Process:** An LLM call (Groq, Llama 70B) reads the last N observations for a student and the current active facts. It outputs:
- New assertions (facts that were not previously tracked)
- Invalidated assertions (old facts contradicted by new evidence, with `invalid_at` set)
- Updated trends (improving/stable/declining)

**Cadence:** Synthesize after every 3-5 new observations, or at minimum once per session if the session produced observations. The synthesis call is approximately 200 tokens of structured output, completing in ~0.3s on Groq.

**Contradiction handling:** When a new observation contradicts an active fact, the synthesis LLM sets `invalid_at` on the old fact and creates a new one. Old facts are preserved -- never deleted.

---

## Retrieval Strategy

### What Gets Fed to the Subagent

When `/api/ask` is called, the Worker builds the subagent's context map with three D1 queries:

| Query | SQL | Purpose |
|---|---|---|
| Active facts | `SELECT * FROM synthesized_facts WHERE student_id = ? AND invalid_at IS NULL ORDER BY valid_at DESC LIMIT 10` | What the system knows about this student right now |
| Recent observations | `SELECT * FROM observations WHERE student_id = ? ORDER BY created_at DESC LIMIT 5` | What happened recently (short-term context) |
| Student baselines | Already in the `/api/ask` request payload from client | Current dimension scores |

### Why No Vector Search

CrescendAI's domain is narrow enough that structured queries retrieve everything the subagent needs:

- **One user per student model** (no identity resolution)
- **Known ontology** (6 dimensions, pieces, sessions, observations)
- **Structured numerical data** (MuQ scores, not free-form text)
- **Low volume** (dozens of observations per student per month, not thousands)

No graph database, no embedding-based retrieval, no complex memory consolidation pipelines. Simple D1 tables with bi-temporal fields.

### Context Window Budget

The subagent receives a structured context map. Approximate token counts:

| Component | Tokens | Source |
|---|---|---|
| Active synthesized facts (up to 10) | ~300 | D1 query |
| Recent observations (up to 5) | ~250 | D1 query |
| Student baselines (6 dimensions) | ~100 | Request payload |
| Current teaching moment | ~200 | Pipeline output |
| System prompt + instructions | ~500 | Static |
| **Total subagent context** | **~1,350** | Well within Groq limits |

The teacher LLM (stage 2) receives the subagent's output (~200 tokens) plus the system prompt (~500 tokens). Total context stays small.

---

## Research Landscape

### Benchmarks

| Benchmark | Focus | Key Insight |
|---|---|---|
| LongMemEval (ICLR 2025) | 500 questions across 5 abilities | Commercial systems score only 30-70%. Temporal reasoning is hardest. |
| LoCoMo (Snap Research) | 300+ turn conversations, 35 sessions | Most systems fail at multi-session reasoning. |
| MemBench (Oct 2025) | Dynamic contexts 0-100K tokens | Knowledge updates and temporal reasoning are weakest across all systems. |
| AMA-Bench (Feb 2026) | Real agentic trajectories | Directly relevant: CrescendAI deals with machine-generated observations (MuQ scores), not human-agent chat. |

### Systems

| System | Approach | Relevance |
|---|---|---|
| Emergence AI (~86% SOTA) | Episode summarization + structured fact extraction + latent knowledge graph | The pattern that keeps winning -- CrescendAI's synthesis model is a simplified version of this. |
| Zep/Graphiti (~71%) | Bi-temporal knowledge graph | CrescendAI borrows bi-temporal fields without the graph layer. |
| Mem0 | Hybrid vector + graph + KV | General-purpose; overkill for CrescendAI's narrow domain. |
| MAGMA (SOTA on LoCoMo) | Multi-graph (episode + entity + concept) | Outperforms baselines by 18-45%; validates episode-to-fact consolidation. |

### The Winning Pattern

Across the literature, the same architecture keeps winning: episodic capture, structured fact extraction, background consolidation, bi-temporal tracking, contradiction handling via invalidation. CrescendAI adopts this pattern but strips it down to what the domain requires -- no graph database, no vector search, no complex pipelines.

---

## Implementation Sequence

### Phase 1: Observations -- COMPLETE

```
[x] observations table in D1 schema
[x] Condensed reasoning traces stored with each observation
[x] Observations written by /api/ask pipeline
[x] Recent observations retrieved for subagent context
```

Ships with the `/api/ask` pipeline (see `02-pipeline.md`).

### Phase 2: Synthesis -- DEFERRED

```
[ ] synthesized_facts table in D1 schema
[ ] Background synthesis trigger (POST /api/synthesize or via /api/sync)
[ ] Synthesis prompt (Groq, Llama 70B)
[ ] Active facts integrated into subagent context map
[ ] Contradiction handling (invalidation, not deletion)
```

Deferred until the pipeline has real observation data. Premature synthesis on synthetic or sparse data would produce misleading facts.

### Phase 3: Retrieval Optimization

```
[ ] Tune fact retrieval limits (10 facts? 20? depends on real data distribution)
[ ] Piece-specific observation retrieval (filter by piece_context)
[ ] Dimension-specific fact retrieval for targeted feedback
[ ] Evaluate whether synthesis measurably improves subagent output vs. raw observations alone
```

---

## Evaluation

The memory system eval framework lives at `apps/api/evals/memory/` (own pyproject.toml, run with `uv run python -m src.run_all`).

### Two Eval Tracks

**Track 1: CrescendAI Chat Extraction Eval (38 scenarios)**

Domain-specific scenarios testing fact extraction from piano practice conversations. Covers single-dimension, multi-dimension, piece lifecycle, temporal reasoning, engagement, relationships, activities, opinions, context, multi-subject, selectivity, and mixed-domain.

Current results: Recall=1.000, Precision=0.722. All 38 scenarios pass. The lower precision is acceptable — the model extracts more facts than the minimum expected set, which is preferred over missing facts.

4 eval layers:
- **Retrieval** (deterministic): F1=0.94 — does the system retrieve the right facts?
- **Synthesis** (Groq): JSON parse=100%, invalidation=0.94 — does the LLM produce valid structured facts?
- **Temporal** (chronological replay): 1.0 — does the system handle time correctly?
- **Downstream** (A/B + Claude judge): win rate=60% — does memory improve feedback quality?

Key weakness: regex-only fact matching is too brittle for LLM output (recall=0.55, "hallucination"=0.87 are mostly matching failures). Next step: add sentence-transformer cosine similarity fallback (>0.85 threshold).

**Track 2: LoCoMo Benchmark (ACL 2024)**

External benchmark: long-conversation memory via QA pairs. Tests whether the extraction pipeline generalizes beyond piano teaching to multi-session dialogue with 300+ turns across 35 sessions.

| Metric | v1 (baseline) | v6 (current) | Notes |
|---|---|---|---|
| Raw F1 | 0.052 | 0.280 | Standard token-level F1 (LoCoMo official metric) |
| Adjusted F1 | -- | ~0.45 | Scores correct abstention as 1.0 (45/47 adversarial have empty gold) |
| Single-hop | -- | 0.315 | Plateau — failures mostly list-recall and paraphrasing, not retrieval misses |
| Multi-hop | -- | 0.460 | Improved via date normalization and retrieval tuning |
| Temporal | -- | 0.273 | Some questions require counterfactual reasoning beyond stored facts |
| Open-ended | -- | 0.328 | Highly variable; needs more consistent synthesis from QA model |
| Adversarial | -- | ~0.62 adj | Abstention rate varies with prompt wording |
| Facts extracted | 61 | 637 | After semantic dedup (cosine >0.90 via all-MiniLM-L6-v2) |

### Key Findings from LoCoMo

1. **Extraction scope was the #1 bottleneck.** Piano-specific prompt missed 85% of LoCoMo facts. Broadening to general extraction increased facts from 61 to 422.

2. **Speaker attribution matters.** LoCoMo has peer conversations (e.g., Caroline & Melanie), but our pipeline assigns "Student"/"Teacher" roles. 52% of questions asked about the second speaker by name. Bidirectional extraction + name fixup resolved this.

3. **Date format mismatch inflated multi-hop failures.** Model answered "2023-05-08" but gold expected "7 May 2023" — token F1 gave 0. Date normalization boosted multi-hop from 0.052 to 0.420.

4. **Retrieval-synthesis tension.** Tight retrieval (max_facts=50) helps focused questions; loose retrieval (max_facts=150+) helps synthesis questions. No single limit is optimal. Solution: adaptive per-question limits based on question keywords.

5. **Semantic dedup reduces noise.** Cosine >0.90 dedup cut facts from 818 to 637 (-22%), directly improving multi-hop and temporal reasoning by reducing noise.

6. **Single-hop plateau at ~0.33.** Most failures are measurement artifacts (token-F1 on paraphrases), not retrieval failures. Unlikely to improve without metric changes.

7. **Run-to-run variance ~0.02-0.03** on Groq Llama 70B at temperature=0. Makes small improvements hard to distinguish from noise on 1-sample eval.

### Remaining Gaps (target: F1 > 0.65)

- **Multi-sample eval:** Run 3-5 samples to get stable numbers
- **Stronger QA model:** Try Claude Haiku or GPT-4o-mini instead of Llama 70B — may reduce variance and improve synthesis
- **Chain-of-thought QA:** Add explicit reasoning step before answering open-ended questions
- **Extraction cap:** Raise from 5 to 8 per exchange to catch missed facts
- **Semantic dedup threshold:** Current 0.90 is conservative; try 0.85

### Eval Code

| File | Purpose |
|---|---|
| `apps/api/evals/memory/src/run_all.py` | Run all eval layers |
| `apps/api/evals/memory/src/build_chat_scenarios.py` | 38 chat extraction scenarios |
| `apps/api/evals/memory/src/locomo_adapter.py` | LoCoMo benchmark adapter (bidirectional extraction, speaker fixup, date normalization, semantic dedup, adaptive retrieval) |
| `apps/api/src/services/prompts.rs` | Extraction prompt (broadened from piano-specific to general, 10 categories, 5 facts per exchange) |

---

## Open Questions

1. **Synthesis eval:** How do we measure whether synthesized facts improve feedback quality? Plan: A/B test (subagent with raw observations only vs. subagent with synthesized facts). Needs real data first.

2. **Fact granularity:** One fact per dimension per student, or finer-grained (per piece, per section)? Starting coarse (per dimension), will refine with usage data.

3. **Student-reported facts:** Students tell the teacher things ("I have a recital in 3 weeks"). These are facts with temporal validity. Store in `synthesized_facts` with `source_type = 'student_reported'`? Or a separate table?

4. **Positive trajectory tracking:** The research literature focuses on contradiction handling. CrescendAI also needs to track positive trajectories ("dynamics improved steadily over 4 sessions"). The bi-temporal model with the `trend` field handles this, but synthesis prompts need to explicitly surface improvements, not just problems.

5. **When does synthesis become necessary?** At what observation count does the subagent need synthesized facts instead of reading the last N raw observations? Likely 50-100+ observations per student -- months of usage. Until then, raw observation retrieval may be sufficient.
