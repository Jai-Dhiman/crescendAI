# Design: POST /api/ask -- Two-Stage Teacher Pipeline

**Date:** 2026-03-04
**Status:** Approved
**Implements:** Slice 06 (teacher LLM prompt) + Slice 06a (subagent architecture)
**Deferred:** Slice 06c (memory/synthesized facts), Slice 10 (UI subagent/stage 3)

---

## Summary

Build the `POST /api/ask` endpoint on the Rust Cloudflare Workers API. Two-stage LLM pipeline: Groq (Llama 70B subagent for analysis) followed by Anthropic (Sonnet 4.6 for teacher voice). Stores observations in D1 for future memory system consumption. Includes "Tell me more" elaboration endpoint.

## Providers

- **Stage 1 (Subagent):** Groq direct API, Llama 3.3 70B. ~0.3s latency.
- **Stage 2 (Teacher):** Anthropic direct API, Claude Sonnet 4.6. ~1.0-1.5s latency.
- **Fallback:** Template response if either LLM call fails. No intermediate fallback providers for v1.

API keys: `GROQ_API_KEY` and `ANTHROPIC_API_KEY` in `.dev.vars`.

## Request Flow

```
iOS -> POST /api/ask (JWT auth)
  -> Worker queries D1: last 5 observations for student
  -> Worker builds subagent prompt (system + user with context map)
  -> Groq call: Llama 70B returns JSON analysis + narrative reasoning
  -> Worker builds teacher prompt (system persona + subagent handoff)
  -> Anthropic call: Sonnet 4.6 returns 1-3 sentence observation
  -> Worker post-processes (strip markdown, length check)
  -> Worker stores observation + condensed trace in D1
  -> Return observation to iOS
```

## API Contracts

### POST /api/ask

**Auth:** Bearer JWT (same as /api/sync)

**Request:**
```json
{
  "teaching_moment": {
    "chunk_index": 7,
    "start_offset_sec": 105.0,
    "stop_probability": 0.87,
    "dimension": "pedaling",
    "dimension_score": 0.35,
    "all_scores": {
      "dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
      "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54
    }
  },
  "student": {
    "level": "intermediate",
    "baselines": {
      "dynamics": 0.68, "timing": 0.72, "pedaling": 0.62,
      "articulation": 0.61, "phrasing": 0.65, "interpretation": 0.58
    },
    "goals": "Preparing Chopin Nocturne Op. 9 No. 2 for recital",
    "session_count": 12
  },
  "session": {
    "id": "uuid",
    "duration_min": 18,
    "total_chunks": 72,
    "chunks_above_threshold": 5
  },
  "piece_context": {
    "composer": "Chopin",
    "title": "Nocturne Op. 9 No. 2",
    "section": "second phrase",
    "bar_range": "bars 20-24"
  }
}
```

**Response (200):**
```json
{
  "observation": "The pedaling in the second phrase got away from you -- ...",
  "observation_id": "abc-123",
  "dimension": "pedaling",
  "framing": "correction"
}
```

**Response (503 -- LLM failure, template fallback):**
```json
{
  "observation": "I noticed your pedaling could use some attention in that last section. Try recording yourself and listening back -- sometimes it's hard to hear pedal clarity while you're playing.",
  "observation_id": "abc-123",
  "dimension": "pedaling",
  "framing": "correction",
  "is_fallback": true
}
```

### POST /api/ask/elaborate

**Auth:** Bearer JWT

**Request:**
```json
{
  "observation_id": "abc-123"
}
```

**Response (200):**
```json
{
  "elaboration": "In this Nocturne, the pedaling needs to track the harmonic rhythm...",
  "observation_id": "abc-123"
}
```

Worker fetches the stored observation + reasoning trace from D1, sends to Anthropic Sonnet with the elaboration instruction from Slice 06. No subagent call for elaboration.

## D1 Schema

```sql
CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    session_id TEXT NOT NULL,
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
    is_fallback BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);
```

`reasoning_trace` stores the condensed subagent output as JSON:
```json
{
  "selected_moment": { "chunk_index": 7, "dimension": "pedaling", "bar_range": "bars 20-24" },
  "framing": "correction",
  "learning_arc": "polishing",
  "is_positive": false,
  "narrative": "Pedaling dropped to 0.35, well below baseline 0.62. Blind spot..."
}
```

## Prompt Design

### Stage 1: Subagent (Groq)

**System prompt:** Analyst persona. Structured reasoning over teaching moments. Output JSON + narrative. Follows the 5-step reasoning framework from 06a: learning arc, delta vs history, musical context, selection, framing.

**User prompt:** Teaching moment data + student context + recent observation history (last 5 from D1) + piece context.

**Output:** JSON (selected_moment, framing, learning_arc, is_positive, musical_context) + narrative reasoning paragraph.

### Stage 2: Teacher (Anthropic)

**System prompt:** Teacher persona from Slice 06. Warm, specific, brief, actionable. 1-3 sentences. No scores, no bullets, no jargon.

**User prompt:** Subagent handoff (JSON + narrative) + student level/goals.

**Output:** 1-3 sentences of natural language.

### Post-Processing

- Strip any markdown formatting
- Reject outputs > 500 characters (use template fallback)
- Reject outputs containing numbers/scores (use template fallback)

## New Files

| File | Purpose |
|---|---|
| `src/services/ask.rs` | Orchestrates the two-stage pipeline, D1 queries, observation storage |
| `src/services/llm.rs` | HTTP clients for Groq and Anthropic APIs |
| `src/services/prompts.rs` | Prompt templates (subagent system/user, teacher system/user, elaboration) |

Route additions in `src/server.rs`:
- `POST /api/ask` -> `handle_ask`
- `POST /api/ask/elaborate` -> `handle_elaborate`

## Fallback

If Groq or Anthropic calls fail (timeout, 5xx, parse error), return a template observation:

```
"I noticed your {dimension} could use some attention in that last section.
Try recording yourself and listening back -- sometimes it's hard to hear
{dimension_description} while you're playing."
```

With `is_fallback: true` in the response so iOS can handle it differently if needed.

No intermediate fallback providers (OpenRouter, Workers AI) for v1. Add later if reliability becomes an issue.

## Not Included

- Synthesized facts / memory consolidation (Slice 06c)
- Exercise recommendation (Slice 07+08)
- Streaming responses
- UI subagent / stage 3 (Slice 10)
- Score alignment (bar number mapping)
