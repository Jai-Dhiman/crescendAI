# Session Synthesis: Accumulated Signals, Post-Session Teaching

**Date:** 2026-03-23
**Status:** Design approved
**Scope:** `apps/api/src/practice/session.rs`, `apps/api/src/practice/practice_mode.rs`, `apps/api/src/services/ask.rs`, `apps/web/src/hooks/usePracticeSession.ts`, `apps/web/src/components/RecordingBar.tsx`

## Problem

The current observation system generates real-time LLM feedback during practice sessions, pushing observations to the client via WebSocket as they occur. This creates three problems:

1. **Persistence fragility:** Observations are written to D1 per-occurrence, gated on `conversation_id` being present. If the DO gets evicted during the LLM call, `last_observation_at` and the in-memory observation buffer are lost -- the throttle resets (allowing double-fire) and `finalize_session()` won't have the lost observations for batch write.
2. **Bad teaching model:** Real-time interruptions during practice feel like a notification bot, not a teacher. A real teacher watches the full session, then gives synthesized feedback. The cooldown timer system (30s/90s/150s/180s per mode) doesn't account for what the student is actually doing -- grinding scales, drilling a section, ending abruptly.
3. **Wasted LLM cost without quality signal:** Each STOP-triggered observation runs the full two-stage LLM pipeline (Groq subagent + Anthropic teacher). For a 3-hour session, this generates 30+ individual observations nobody reads directly, at $0.60-1.50 total cost.

## Solution

Replace real-time per-observation delivery with accumulated-signal post-session synthesis.

### Core Model

1. **During listening mode:** The DO silently accumulates structured analytical signals (scores, teaching moments, mode transitions, drilling records, timeline events). No LLM calls, no WS observation pushes. Inference (MuQ + Aria-AMT), STOP classification, teaching moment selection, mode detection, and score following all run as today.
2. **When the user exits listening mode:** A single teacher LLM call synthesizes the full accumulated context into one cohesive teaching response, persisted as a message in the conversation.
3. **The user controls the boundary.** Exiting listening mode is the explicit trigger. No automatic session-end detection needed.

### Data Flow

```
Listening mode ON
  -> Chunks flow in -> MuQ + Aria-AMT inference (parallel, unchanged)
  -> Per-chunk: STOP classify, score follow, bar analysis, mode detect, teaching moment select
  -> Accumulate to SessionAccumulator (replaces observation buffer)
  -> Persist accumulator to DO storage after each chunk (~1ms KV write)
  -> NO LLM calls, NO WS observation pushes

Listening mode OFF (user exits)
  -> Client sends { "type": "session_end" }
  -> synthesize_session() builds structured prompt from accumulator
  -> Single teacher LLM call (Anthropic)
  -> Persist synthesis to D1 messages table
  -> Persist raw accumulated data to D1 observations table
  -> Send synthesis to client via WS
  -> finalize_session() cleanup
```

## SessionAccumulator

Replaces `observations: Vec<ObservationRecord>` and `last_observation_at` on `SessionState`.

```rust
struct SessionAccumulator {
    teaching_moments: Vec<AccumulatedMoment>,
    mode_transitions: Vec<ModeTransitionRecord>,
    drilling_records: Vec<DrillingRecord>,
    timeline: Vec<TimelineEvent>,
}

struct AccumulatedMoment {
    chunk_index: usize,
    dimension: String,
    score: f64,
    baseline: f64,
    deviation: f64,
    is_positive: bool,
    reasoning: String,              // deterministic, from teaching_moment selection
    bar_range: Option<(u32, u32)>,
    analysis_tier: u8,
    timestamp_ms: u64,
    llm_analysis: Option<String>,   // future: per-STOP LLM enrichment (A/B test hook)
}

struct ModeTransitionRecord {
    from: PracticeMode,
    to: PracticeMode,
    chunk_index: usize,
    timestamp_ms: u64,
    dwell_ms: u64,
}

struct DrillingRecord {
    bar_range: Option<(u32, u32)>,
    repetition_count: usize,
    first_scores: [f64; 6],
    final_scores: [f64; 6],
    started_at_chunk: usize,
    ended_at_chunk: usize,
}

struct TimelineEvent {
    chunk_index: usize,
    timestamp_ms: u64,
    has_audio: bool,
}
```

All structs derive `Serialize`/`Deserialize` for DO storage persistence.

## Persistence Strategy

Three layers, replacing the current per-observation D1 write:

### Layer 1: DO durable storage (during session)
After each chunk is processed, persist the full accumulator to `state.storage().put("accumulator", &accumulator)`. The DO's built-in KV survives eviction (~1ms writes). Reload in `ensure_session_state()` if in-memory state is empty.

### Layer 2: D1 write (at synthesis time)
When `synthesize_session()` completes:
- `INSERT INTO messages` -- synthesis text, `message_type='synthesis'`, conversation_id, session_id
- `INSERT INTO observations` -- batch write accumulated moments (raw structured data, for analytics/memory)
- Update `student_memory_meta.total_observations`
- Run memory synthesis if threshold met

### Layer 3: finalize_session() safety net
If the session ends without synthesis (disconnect, alarm timeout):
- Persist accumulator to D1 with `needs_synthesis` flag
- No LLM call (client may be gone)
- On next conversation load, web client detects unsynthesized data and triggers deferred synthesis via `POST /api/practice/synthesize`

### conversation_id requirement
The current code silently skips persistence if `conversation_id` is None. Under the new model, missing `conversation_id` at synthesis time is a hard error -- log `console_error!`, persist accumulator to DO storage with error flag. The web client must always provide `conversation_id` on WS connect.

## Synthesis Call

### Guard: always synthesize if playing detected
No minimum chunk threshold. No canned responses. If the "is the user playing" system detected real playing (AMT detected notes, not just silence/noise), the teacher always speaks. Even a 5-second snippet gets a contextual response from the teacher LLM working with whatever data the accumulator has.

The only skip condition: accumulator is truly empty (listening mode on but no audio chunks arrived, or all chunks were silence with no notes detected).

### Prompt structure
The synthesis prompt gives the teacher structured JSON context:

```json
{
  "session_duration_minutes": 22,
  "chunks_processed": 88,
  "piece": { "composer": "Bach", "title": "Prelude in C Major, WTC1" },
  "practice_pattern": [
    { "mode": "warming", "duration_min": 2, "chunks": 8 },
    { "mode": "drilling", "duration_min": 8, "bar_range": [12, 16], "repetitions": 4,
      "first_scores": { "dynamics": 0.45, "timing": 0.62 },
      "final_scores": { "dynamics": 0.72, "timing": 0.68 } },
    { "mode": "running", "duration_min": 12, "chunks": 48 }
  ],
  "top_moments": [
    { "dimension": "dynamics", "deviation": 0.27, "is_positive": true, "bar_range": [12, 16],
      "reasoning": "Sustained improvement across drilling repetitions" },
    { "dimension": "pedaling", "deviation": -0.31, "is_positive": false, "bar_range": [24, 28],
      "reasoning": "Consistent underperformance vs baseline in running section" }
  ],
  "baselines": { "dynamics": 0.55, "timing": 0.61, "pedaling": 0.58 },
  "student_memory": "Has been working on dynamics in Bach for 3 sessions."
}
```

The teacher interprets this into a natural, cohesive teaching response. No subagent stage -- the structured data IS the analysis; the teacher narrates.

## What Changes

### Removed from hot path
- `try_generate_observation()` -- per-chunk observation gate
- `generate_observation()` -- per-chunk LLM pipeline
- `mode_throttle_allows()` -- cooldown timer
- `ObservationPolicy.suppress` / `min_interval_ms` -- no throttling
- `last_observation_at` on `SessionState`
- Per-observation WS push (`"type": "observation"` event) during session
- Per-observation D1 write to `messages` during session

### Refactored
- `SessionState.observations` -> `SessionState.accumulator: SessionAccumulator`
- `finalize_session()` -> simplified: persist accumulator if unsynthesized, run memory synthesis, close WS
- `persist_observations()` -> writes `AccumulatedMoment` records (structured data, not LLM text)
- `ObservationPolicy` -> keeps `comparative` (synthesis context), drops `suppress`/`min_interval_ms`
- `observation_policy()` -> renamed to `mode_context()`, returns mode metadata for synthesis
- `ensure_session_identity()` -> expanded to `ensure_session_state()`, also reloads accumulator from DO storage

### Kept as-is
- STOP classification, teaching moment selection, mode detector state machine, score following, bar analysis, piece identification, baselines loading, memory synthesis

### New
- `SessionAccumulator` struct + per-chunk accumulation
- `synthesize_session()` -- prompt builder + teacher LLM call + persistence
- DO storage for accumulator (per-chunk write)
- `POST /api/practice/synthesize` endpoint (deferred synthesis recovery)
- `needs_synthesis` flag on session records

### Web client impact
- `usePracticeSession.ts` no longer handles `"observation"` WS events during recording
- Handles new `"synthesis"` event after session end
- `RecordingBar.tsx` no longer shows observation toasts during recording
- Synthesis message appears in chat after listening mode ends

## Edge Cases

### DO eviction during synthesis LLM call
Accumulator already persisted to DO storage. Identity reloads via `ensure_session_state()`. Synthesis call itself is lost. Recovery: `finalize_session()` detects missing synthesis, writes raw data to D1 with `needs_synthesis` flag. Web client triggers deferred synthesis on next conversation load.

### Overnight idle (listening mode left on)
30-minute alarm fires. If chunks were processed: persist accumulator to D1 with `needs_synthesis` flag. If no chunks: clean up silently. Deferred synthesis on next visit.

### Client disconnect (browser tab closed)
WebSocket close event fires. Same as alarm: persist accumulator, flag for deferred synthesis.

### Rapid re-entry
Each listening mode session gets its own accumulator. New session = new DO instance (keyed by session_id). First session's synthesis is independent.

### No piece match
Synthesis works without piece context. Teacher gets scores, mode history, temporal patterns. Feedback is more general without bar-level specificity.

### Multiple moments on same dimension
Accumulator may have 5 dynamics moments across 30 minutes. Synthesis prompt presents `top_moments` sorted by `|deviation|`, deduplicated by dimension (most significant per dimension, with temporal spread).

## Future: Per-STOP LLM Enrichment (A/B Test)

The `llm_analysis: Option<String>` field on `AccumulatedMoment` is the hook. To test whether per-STOP LLM analysis improves synthesis quality:
1. Enable per-STOP LLM calls (subagent only, not full teacher pipeline)
2. Store analysis text in the moment's `llm_analysis` field
3. Synthesis prompt includes these as additional context alongside raw scores
4. Compare synthesis quality: raw-only vs LLM-enriched via eval pipeline

Build for raw-only now. The accumulator format supports enrichment without changing the synthesis interface.

## Testing

### Unit tests
- `SessionAccumulator` accumulation logic
- Synthesis prompt builder: given known accumulator, verify well-formed JSON context
- Guard logic: no notes detected -> no synthesis; notes detected -> always synthesize
- Mode detector: existing tests unchanged

### Integration tests
- Full session simulation: feed N chunks, verify accumulator state, trigger session_end, verify synthesis context
- DO eviction: populate accumulator, clear in-memory, verify reload from DO storage
- Deferred synthesis: simulate disconnect, verify `needs_synthesis` flag, verify HTTP endpoint

### Eval
- Same recorded sessions through old (per-observation) vs new (synthesis) model
- Human eval: coherence, actionability, musical specificity
- Future: raw-only vs LLM-enriched A/B test
