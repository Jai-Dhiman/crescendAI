# Slice 6c: Student Memory System

See `docs/architecture.md` for the full system architecture.
See `docs/apps/06a-subagent-architecture.md` for the two-stage subagent pipeline (consumer of memory).
See `docs/apps/06-teacher-llm-prompt.md` for the teacher persona prompt.

**Status:** IMPLEMENTED (core pipeline)
**Last verified:** 2026-03-07
**Date:** 2026-03-04
**Notes:** Research and design document for the student memory consolidation system. Separate from the core `/api/ask` pipeline (Slice 06 + 06a). Build after the two-stage pipeline is live with real observations to consume.

**Goal:** Build a memory system that gives the subagent accumulated student context -- not just baselines, but temporal facts about what has been observed, what has changed, and what patterns persist -- so that feedback compounds across sessions.

---

## Background: The Two Clocks

From Koratana's "How to build a context graph" and Marple's "Building the Event Clock":

Every system has a **state clock** (what is true now) and an **event clock** (what happened, in what order, with what reasoning). Most systems only build the state clock. The event clock -- the reasoning connecting observations to actions -- is rarely treated as data.

For CrescendAI:
- **State clock:** Student baselines (exponential moving average per dimension), inferred level, explicit goals. Already implemented in SwiftData + D1.
- **Event clock:** The history of observations, what was flagged, why it was flagged, how the student responded, and what changed over time. Not yet implemented.

The subagent needs both clocks to give good feedback. Without the event clock, it rediscovers patterns from scratch each session.

### Why This Matters for Piano Teaching

A teacher who has worked with a student for months knows:
- "We've talked about pedaling three times. She knows it's an issue. Frame as progress check, not discovery."
- "His dynamics improved dramatically after we worked on the Chopin. That approach worked."
- "She always struggles with the development section of new pieces. That's her pattern."

These are temporal assertions with validity periods -- exactly what the event clock captures.

---

## Landscape: Memory Systems (Researched 2026-03-04)

### Benchmarks

| Benchmark | Focus | Key Insight |
|---|---|---|
| LongMemEval (ICLR 2025) | 500 questions across 5 abilities: extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | Commercial systems score only 30-70%. Temporal reasoning is the hardest category. |
| LoCoMo (Snap Research) | 300+ turn conversations across 35 sessions | Tests very long-term conversational memory. Most systems fail at multi-session reasoning. |
| MemBench (Oct 2025) | Dynamic contexts 0-100K tokens | Knowledge updates and temporal reasoning are the weakest capabilities across all systems. |
| AMA-Bench (Feb 2026) | Real agentic trajectories, not just dialogue | Key gap: existing benchmarks test human-agent chat, but real agents deal with machine-generated observations. Directly relevant to CrescendAI (MuQ scores are machine-generated). |

### Who Is Winning

| System | LongMemEval Score | Approach |
|---|---|---|
| Emergence AI | ~86% (SOTA) | Episode summarization + structured fact extraction + latent knowledge graph + chain-of-thought at retrieval |
| Supermemory | 71-77% | Strong on temporal reasoning (76.69%) |
| Zep/Graphiti | ~71.2% | Bi-temporal knowledge graph, 5.65s/item latency |
| MAGMA (Jan 2026) | SOTA on LoCoMo | Multi-graph (episode + entity + concept), outperforms baselines by 18.6-45.5% |

### Production Systems

| System | Architecture | Best For | Temporal Support |
|---|---|---|---|
| **Zep/Graphiti** | Bi-temporal knowledge graph | Temporal reasoning, contradiction handling | Native: valid_at/invalid_at + created_at/expired_at per fact |
| **Mem0** | Hybrid vector + graph + KV | Fast integration, general-purpose | Limited |
| **LangMem** | Background consolidation into LangGraph | Episodic/semantic/procedural separation | Via episodic timestamps |
| **Letta (MemGPT)** | Explicit agent state blocks, tool-callable memory ops | Stateful long-running agents | Manual |

### The Pattern That Keeps Winning

1. **Episodic capture:** Store what happened (observations, events) lightly summarized
2. **Structured fact extraction:** Subject, relation, object, temporal bounds
3. **Background consolidation:** Async process turns episodes into semantic facts (not in the hot path)
4. **Bi-temporal tracking:** When a fact was true in the world vs. when recorded in the system
5. **Contradiction handling:** New facts invalidate old ones (set invalid_at) rather than overwrite
6. **Hybrid retrieval:** Semantic similarity + recency + importance score (not pure vector search)

---

## Design for CrescendAI

### Why CrescendAI's Domain Is Simpler

The enterprise context graph papers describe systems for organizations with thousands of agents, disparate data sources, and unknown ontology. CrescendAI has:

- **One user per student model** (no identity resolution)
- **One agent** (the subagent pipeline)
- **Known ontology** (6 dimensions, pieces, sessions, observations)
- **Structured numerical data** (MuQ scores, not free-form text)
- **Low volume** (dozens of observations per student per month, not thousands of agent trajectories)

This means: no graph database needed, no node2vec embedding discovery, no complex memory consolidation pipelines. Simple D1 tables with bi-temporal fields and structured queries.

### Data Model

**Observations table** (episode capture -- built with the /api/ask pipeline):

```sql
CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    chunk_index INTEGER,
    dimension TEXT NOT NULL,
    observation_text TEXT NOT NULL,
    elaboration_text TEXT,
    reasoning_trace TEXT,          -- JSON: condensed subagent reasoning
    framing TEXT,                  -- correction / recognition / encouragement / question
    dimension_score REAL,
    student_baseline REAL,
    piece_context TEXT,            -- JSON: composer, title, section, bar_range
    learning_arc TEXT,             -- new / mid-learning / polishing
    created_at TEXT NOT NULL
);
```

**Synthesized facts table** (event clock -- built later by memory system):

```sql
CREATE TABLE synthesized_facts (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    fact_text TEXT NOT NULL,       -- "Pedaling has been a persistent weakness but is improving"
    dimension TEXT,                -- nullable: some facts span dimensions
    valid_at TEXT NOT NULL,        -- when this became true
    invalid_at TEXT,               -- when this stopped being true (null = still active)
    trend TEXT,                    -- improving / stable / declining / new
    confidence TEXT,               -- high / medium / low
    evidence TEXT,                 -- JSON: array of observation IDs that support this fact
    source_type TEXT NOT NULL,     -- synthesized / student_reported / inferred
    created_at TEXT NOT NULL,      -- when this fact was recorded in the system
    expired_at TEXT                -- when this fact was superseded in the system (bi-temporal)
);
```

### Synthesis Strategy

**When:** Background, after session ends. Triggered by `POST /api/sync` or a dedicated `POST /api/synthesize` endpoint. Not in the `/api/ask` hot path.

**How:** LLM call (Groq, Llama 70B) reads the last N observations for a student and the current active facts. Outputs updated facts: new assertions, invalidated old assertions, updated trends.

**Contradiction handling:** When a new observation contradicts an active fact (e.g., "pedaling improved for 3 consecutive sessions" contradicts "pedaling is a persistent weakness"), the synthesis LLM sets `invalid_at` on the old fact and creates a new one. The old fact is preserved for history -- never deleted.

**Cadence:** Synthesize after every 3-5 new observations, or at minimum once per session if the session produced observations. The synthesis call is ~200 tokens of structured output, completing in ~0.3s on Groq.

### Retrieval Strategy for Subagent

When `/api/ask` is called, the Worker builds the subagent's context map with three queries:

1. **Active synthesized facts:** `SELECT * FROM synthesized_facts WHERE student_id = ? AND invalid_at IS NULL ORDER BY valid_at DESC LIMIT 10`
2. **Recent observations:** `SELECT * FROM observations WHERE student_id = ? ORDER BY created_at DESC LIMIT 5`
3. **Student baselines:** Already in the `/api/ask` request payload from iOS

No vector search needed. The domain is narrow enough that structured D1 queries retrieve everything the subagent needs.

---

## Key References

### Papers
- LongMemEval (ICLR 2025): arxiv.org/abs/2410.10813
- LoCoMo (Snap Research): snap-research.github.io/locomo/
- AMA-Bench (Feb 2026): arxiv.org/abs/2602.22769 -- targets agentic trajectories, relevant to machine-generated observations
- MAGMA (Jan 2026): arxiv.org/abs/2601.03236 -- multi-graph memory, SOTA on LoCoMo
- Zep/Graphiti (Jan 2025): arxiv.org/abs/2501.13956 -- bi-temporal knowledge graph, production-grade
- Mem0 (Apr 2025): arxiv.org/abs/2504.19413 -- hybrid architecture survey
- Position: Episodic Memory is the Missing Piece (Feb 2025): arxiv.org/abs/2502.06975
- A-MEM Zettelkasten-inspired memory (Feb 2025): arxiv.org/abs/2502.12110
- AgeMem unified STM/LTM (Jan 2026): arxiv.org/abs/2601.01885

### Concept Sources
- "How to build a context graph" (Animesh Koratana / PlayerZero): Two clocks, agents as informed walkers, context graphs as world models
- "Building the Event Clock" (Kirk Marple / Graphlit): Bi-temporal facts, three-layer model (content/entities/facts), map-first principle

### Production Systems
- Zep/Graphiti: github.com/getzep/graphiti -- best temporal reasoning in production
- Mem0: github.com/mem0ai/mem0 -- most widely used, hybrid architecture
- LangMem: langchain-ai.github.io/langmem/ -- background consolidation pattern
- Emergence AI: emergence.ai/blog/sota-on-longmemeval-with-rag -- SOTA approach details

---

## Implementation Sequence

1. **First (with /api/ask pipeline):** Build the `observations` table. Store condensed traces with each observation. No synthesis yet.
2. **Second (this slice):** Build the `synthesized_facts` table. Implement background synthesis after sessions end. Integrate active facts into the subagent's context map.
3. **Third (iteration):** Tune synthesis prompts with real observation data. Add contradiction handling. Evaluate whether synthesized facts measurably improve subagent output quality vs. raw observation history alone.

---

## Open Questions

1. **Synthesis eval:** How do we measure whether synthesized facts improve feedback quality? A/B test: subagent with raw observations only vs. subagent with synthesized facts. Need real data first.
2. **Fact granularity:** One fact per dimension per student? Or finer-grained (per piece, per section)? Start coarse (per dimension), refine with usage data.
3. **Student-reported facts:** Students tell the teacher things ("I have a recital in 3 weeks"). These are facts with temporal validity. Store them in the same table with source_type = "student_reported"? Or separate?
4. **Positive trajectory tracking:** The papers focus on contradiction handling. CrescendAI also needs to track positive trajectories ("dynamics improved steadily over 4 sessions"). Same bi-temporal model, with trend field.
5. **When does the subagent outgrow raw observations?** At what observation count does the subagent need synthesized facts instead of reading the last N raw observations? Likely 50-100+ observations per student. Months of usage.
