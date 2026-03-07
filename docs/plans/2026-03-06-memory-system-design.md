# Student Memory System Design

**Date:** 2026-03-06
**Status:** Approved
**Builds on:** `docs/apps/06c-memory-system.md` (research), `docs/apps/06a-subagent-architecture.md` (consumer)
**Depends on:** `/api/ask` pipeline (Slice 06/06a) being live with real observations

## Summary

Full LLM synthesis on the Worker after session sync, writing bi-temporal facts and teaching approach records into D1, retrieved via four structured queries before each subagent call, with student-reported facts as first-class citizens in the same table.

The teacher's memory is the product's moat. A teacher who remembers deeply -- what patterns persist, what approaches landed, what the student cares about -- compounds in value over time. This system gives the subagent accumulated context so feedback improves with every session instead of starting from scratch.

## Research Validation

The design was pressure-tested against the top memory systems (2025-2026):

| System | Score | Key Insight | Adopted? |
|---|---|---|---|
| Emergence AI | 86% LongMemEval | Session-level retrieval + CoT reading | Deferred (volume too low) |
| Zep/Graphiti | 71% LongMemEval | Bi-temporal facts with 4 timestamps per edge | Yes -- `valid_at`/`invalid_at` + `created_at`/`expired_at` |
| MAGMA | 0.700 LoCoMo | Intent-aware routing across 4 graph layers | Deferred (known ontology, single query type) |
| Mem0 | ~65-70% LoCoMo | Redundant multi-store, LLM conflict detector | Partial -- LLM synthesis with structured operations |
| LangMem | Not benchmarked | Procedural memory (what approaches worked) | Yes -- `teaching_approaches` table |

CrescendAI's domain is simpler than enterprise context graphs: one user per student model, one agent (the subagent), known ontology (6 dimensions, pieces, sessions), low volume (dozens of observations per month). No graph database, no vector search, no complex memory consolidation pipelines needed.

---

## Data Model

### Observations (episode capture)

Built alongside the `/api/ask` pipeline. Each call produces one row.

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
    framing TEXT NOT NULL,
    dimension_score REAL,
    student_baseline REAL,
    piece_context TEXT,
    learning_arc TEXT,
    created_at TEXT NOT NULL
);
```

- `observation_text`: what the teacher said (1-3 sentences)
- `elaboration_text`: the "tell me more" response (set later if student engages)
- `reasoning_trace`: JSON condensed subagent reasoning (dimension, insight, confidence, framing, scores, piece, learning_arc)
- `framing`: correction / recognition / encouragement / question
- `piece_context`: JSON `{ composer, title, section, bar_range }`
- `learning_arc`: new / mid-learning / polishing

### Synthesized Facts (event clock)

The memory system proper. Bi-temporal facts with validity tracking.

```sql
CREATE TABLE synthesized_facts (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    fact_text TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    dimension TEXT,
    piece_context TEXT,
    valid_at TEXT NOT NULL,
    invalid_at TEXT,
    trend TEXT,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    source_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expired_at TEXT
);
```

- `fact_type`: dimension / approach / arc / student_reported
- `piece_context`: nullable JSON -- facts can be piece-specific or general
- `valid_at` / `invalid_at`: when the fact was true in the world (null `invalid_at` = still active)
- `created_at` / `expired_at`: when the system recorded/superseded this fact (bi-temporal)
- `trend`: improving / stable / declining / new / resolved
- `confidence`: high / medium / low
- `evidence`: JSON array of observation IDs supporting this fact
- `source_type`: synthesized / student_reported / inferred

### Teaching Approaches (procedural memory)

Tracks what feedback approaches work for each student.

```sql
CREATE TABLE teaching_approaches (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    observation_id TEXT NOT NULL REFERENCES observations(id),
    dimension TEXT NOT NULL,
    framing TEXT NOT NULL,
    approach_summary TEXT NOT NULL,
    engaged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
```

- `approach_summary`: one-line description ("specific technique suggestion for pedaling")
- `engaged`: 1 if student tapped "tell me more" (positive signal for this approach)

### Student Memory Meta

Tracks synthesis state per student.

```sql
CREATE TABLE student_memory_meta (
    student_id TEXT PRIMARY KEY REFERENCES students(apple_user_id),
    last_synthesis_at TEXT,
    total_observations INTEGER DEFAULT 0,
    total_facts INTEGER DEFAULT 0
);
```

---

## Synthesis Pipeline

### Trigger

Runs after `POST /api/sync` when new observations exist since last synthesis. Conditions:

- `count(new observations since last_synthesis_at) >= 3`, OR
- Any new observations AND `last_synthesis_at` > 7 days ago

Synthesis runs as `ctx.waitUntil()` -- fire-and-forget after the sync response is sent. Sync latency is unchanged.

### Input

The Worker builds a synthesis prompt with:

1. Current active facts: `WHERE invalid_at IS NULL AND expired_at IS NULL`
2. New observations since last synthesis (with reasoning traces)
3. Teaching approach records since last synthesis (with engagement signals)
4. Student baselines from the `students` table

### Output

Structured JSON with three operations:

```json
{
  "new_facts": [
    {
      "fact_text": "Pedaling has improved steadily over the last 3 sessions on the Chopin Nocturne",
      "fact_type": "dimension",
      "dimension": "pedaling",
      "piece_context": { "composer": "Chopin", "title": "Nocturne Op. 9 No. 2" },
      "trend": "improving",
      "confidence": "high",
      "evidence": ["obs_12", "obs_14", "obs_17"]
    }
  ],
  "invalidated_facts": [
    {
      "fact_id": "fact_05",
      "reason": "Improved for 3 consecutive sessions",
      "invalid_at": "2026-03-06"
    }
  ],
  "unchanged_facts": ["fact_01", "fact_03", "fact_08"]
}
```

The Worker applies these transactionally:
- `new_facts`: INSERT into `synthesized_facts`
- `invalidated_facts`: SET `invalid_at` on old fact, SET `expired_at = now()`, INSERT superseding fact
- `unchanged_facts`: no-op (confirms the LLM reviewed them)

### Approach facts

The synthesis LLM also produces `fact_type = "approach"` facts from teaching approach records:

```json
{
  "fact_text": "Student engages most with specific technique corrections -- 4 of 5 elaboration requests were correction-framed with concrete suggestions",
  "fact_type": "approach",
  "dimension": null,
  "trend": "stable",
  "confidence": "medium",
  "evidence": ["obs_10", "obs_12", "obs_14", "obs_17"]
}
```

### Model

Groq, Llama 3.3 70B. ~500-800 tokens input, ~200-400 tokens output. ~0.3s, ~$0.0005/call.

### Error handling

If synthesis fails (Groq down, malformed output), the system continues with stale facts. The subagent falls back to raw observations + existing facts. Synthesis retries on next sync. No data loss -- observations are already persisted.

---

## Retrieval Strategy

When `/api/ask` is called, the Worker runs four D1 queries before calling the subagent.

### Query 1: Active synthesized facts

```sql
SELECT * FROM synthesized_facts
WHERE student_id = ?
  AND invalid_at IS NULL
  AND expired_at IS NULL
ORDER BY fact_type, valid_at DESC
LIMIT 15
```

### Query 2: Recent observations with engagement

```sql
SELECT o.*, ta.framing AS approach_framing, ta.engaged
FROM observations o
LEFT JOIN teaching_approaches ta ON ta.observation_id = o.id
WHERE o.student_id = ?
ORDER BY o.created_at DESC
LIMIT 5
```

### Query 3: Student baselines

Already in the `/api/ask` request payload from iOS. No query needed.

### Query 4: Piece-specific facts (conditional)

```sql
SELECT * FROM synthesized_facts
WHERE student_id = ?
  AND piece_context IS NOT NULL
  AND json_extract(piece_context, '$.title') = ?
  AND invalid_at IS NULL
  AND expired_at IS NULL
```

Only runs if the student reported a piece. Results deduplicated against Query 1.

### Context map format

Assembled into plain text for the subagent prompt:

```
## Student Memory

### Active Patterns
- [dimension/pedaling, improving, high confidence] Pedaling has improved steadily
  over the last 3 sessions on the Chopin Nocturne (since 2026-02-20)
- [dimension/dynamics, stable, medium confidence] Dynamics remain flat -- student
  doesn't explore dynamic range in lyrical passages (since 2026-02-01)
- [approach, stable, medium confidence] Student engages most with specific technique
  corrections (4/5 elaboration requests were correction-framed)

### Recent Feedback
- Last session: flagged pedaling (correction), student asked for elaboration
- 2 sessions ago: flagged dynamics (recognition), no elaboration

### Current Piece
- Chopin Nocturne Op. 9 No. 2, polishing phase (session 14 on this piece)
- Piece-specific: "Pedaling in the second phrase (bars 20-24) has been a recurring
  issue but trending better"
```

No vector search. No graph traversal. Four structured D1 queries formatted into plain text.

---

## Student-Reported Facts

Students provide context that isn't derived from audio scores: goals, recital dates, piece declarations.

### Entry paths

1. **Check-ins** (already implemented): goal extraction via Workers AI, synced to D1
2. **Chat responses** (future): Worker extracts declarative statements from student text

### Storage

Same `synthesized_facts` table with `source_type = 'student_reported'`:

```json
{
  "fact_text": "Preparing Chopin Nocturne for recital on March 28",
  "fact_type": "arc",
  "valid_at": "2026-03-06",
  "invalid_at": "2026-03-28",
  "source_type": "student_reported"
}
```

Facts with known end dates (recitals, deadlines) have `invalid_at` pre-set. The synthesis LLM sees this and reasons about urgency.

### Auto-expiry

Student-reported facts without explicit end dates stay active until the synthesis LLM invalidates them based on evidence, or until a 90-day TTL. The synthesis prompt includes an instruction to review student-reported facts for staleness.

---

## Integration with Existing Pipeline

### Write path (every `/api/ask`)

After the two-stage pipeline completes:

1. INSERT `observation` row (teacher output + condensed reasoning trace)
2. INSERT `teaching_approach` row (framing, approach summary, `engaged = 0`)

On "tell me more":

3. UPDATE `observations.elaboration_text`
4. UPDATE `teaching_approaches.engaged = 1`

The "tell me more" engagement signal is piggybacked on the existing elaboration request to `/api/ask`.

### Synthesis trigger (after `POST /api/sync`)

Appended to the existing sync flow. After upsert student + insert sessions, check `student_memory_meta` and conditionally run synthesis via `ctx.waitUntil()`.

### Read path (every `/api/ask`)

Before calling the subagent, run the four D1 queries and format the context map. Adds ~5-10ms to `/api/ask` latency.

### iOS changes

None required for the memory system itself. The only addition is sending an engagement signal when "tell me more" is tapped, which can be piggybacked on the existing elaboration request.

### D1 migration

One migration file adds four tables: `observations`, `synthesized_facts`, `teaching_approaches`, `student_memory_meta`. No changes to existing tables.

---

## What We're NOT Building

- **Vector search / embeddings** -- structured D1 queries are sufficient at current volume
- **Graph database** -- one user, one agent, known ontology; D1 tables with foreign keys suffice
- **Session-level retrieval** -- Emergence AI's insight; not needed at dozens of observations/month
- **Intent-aware routing** -- MAGMA's multi-graph approach; subagent reasons over a flat context map
- **System timeline queries** -- `created_at`/`expired_at` are there for auditability, not queried
- **Synthesis eval harness** -- A/B testing (raw observations vs. synthesized facts); needs real data first
- **On-device synthesis** -- all synthesis runs on the Worker

---

## Implementation Sequence

1. **D1 migration:** Add the four tables
2. **Write path:** After `/api/ask`, persist observation + teaching approach rows
3. **Engagement tracking:** Update `engaged` flag on "tell me more"
4. **Synthesis pipeline:** LLM call in `POST /api/sync` via `ctx.waitUntil()`
5. **Read path:** Four D1 queries before subagent call, format context map
6. **Student-reported facts:** Wire check-in goals into `synthesized_facts`
7. **Synthesis prompt iteration:** Tune with real observation data
