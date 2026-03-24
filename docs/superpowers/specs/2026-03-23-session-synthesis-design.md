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
  -> Client sends { "type": "end_session" } (existing WS protocol, line 270 of session.rs)
  -> Existing end_session handler waits for in-flight chunks
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
    final_scores: [f64; 6],  // NOTE: existing DrillingPassage only tracks first_scores.
                               // New logic needed: snapshot scores on drilling mode exit.
    started_at_chunk: usize,
    ended_at_chunk: usize,
}

// TimelineEvent exists solely for gap detection (silence periods between playing).
// Chunk-to-mode mapping is derived from mode_transitions (transition boundaries)
// cross-referenced with scored_chunks (which carry chunk_index).
struct TimelineEvent {
    chunk_index: usize,
    timestamp_ms: u64,
    has_audio: bool,  // false = silence/noise only (no AMT notes detected)
}
```

All structs derive `Serialize`/`Deserialize` for DO storage persistence.

## Persistence Strategy

Three layers, replacing the current per-observation D1 write:

### Layer 1: DO durable storage (during session)
After each chunk is processed, persist the full accumulator to `state.storage().put("accumulator", &accumulator)`. The DO's built-in KV survives eviction (~1ms writes). Reload in `ensure_session_state()` if in-memory state is empty.

### Layer 2: D1 write (at synthesis time)
When `synthesize_session()` completes:
- `INSERT INTO messages` -- synthesis text, `message_type='synthesis'` (new value, existing column), conversation_id, session_id
- `INSERT INTO observations` -- batch write accumulated moments. Existing columns are repurposed: `observation_text` stores the deterministic `reasoning` string, `dimension_score` maps to `score`, `student_baseline` maps to `baseline`. No schema change needed for this table.
- Update `student_memory_meta.total_observations`
- Run memory synthesis if threshold met

### Layer 3: finalize_session() safety net
If the session ends without synthesis (disconnect, alarm timeout):
- Persist the full serialized accumulator JSON to a new `session_accumulator` column on the `sessions` table (`TEXT`, nullable). **Requires D1 migration:** `ALTER TABLE sessions ADD COLUMN accumulator_json TEXT; ALTER TABLE sessions ADD COLUMN needs_synthesis INTEGER DEFAULT 0;`
- Set `needs_synthesis = 1` on the session row
- No LLM call (client may be gone)
- On next conversation load, the web client queries `GET /api/practice/sessions?needs_synthesis=1&conversation_id=X`. If found, calls `POST /api/practice/synthesize` with the `session_id`. That endpoint reads `accumulator_json` from the `sessions` row, deserializes it into a `SessionAccumulator`, rebuilds the synthesis prompt (loading baselines and student memory from D1), calls the teacher LLM, persists the result to `messages`, and clears the `needs_synthesis` flag.
- If the DO storage has already been cleaned up (DO garbage collected), the D1 `accumulator_json` is the sole recovery source.

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

### LLM model and pipeline
Synthesis uses the Anthropic teacher model directly (same model as the current teacher stage in `ask.rs`). The Groq subagent stage is eliminated -- there is no "fast analysis" step because the structured accumulator data already IS the analysis. This is a new `synthesize_session()` function in `session.rs` (or a new `synthesis.rs` module), NOT a reuse of `handle_ask_inner()` from `ask.rs`. The existing `ask.rs` pipeline remains for the chat path (user asks a question in conversation).

## What Changes

### Removed from hot path (production sessions only -- see Eval Sessions below)
- `try_generate_observation()` -- per-chunk observation gate
- `generate_observation()` -- per-chunk LLM pipeline
- `generate_session_summary()` -- replaced by `synthesize_session()` (the synthesis IS the summary)
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
Accumulator may have 5 dynamics moments across 30 minutes. Selection algorithm for `top_moments` in the synthesis prompt:
1. Group moments by dimension
2. Per dimension: keep the moment with highest `|deviation|` (most notable)
3. If a dimension has both positive and negative moments, keep top-1 of each
4. Cap total `top_moments` at 8 (across all dimensions)
5. Sort final list by chunk_index (chronological order for narrative coherence)

### DO eviction during accumulation (pipeline state loss)
After DO eviction, only the accumulator and identity are reloaded from DO storage. The mode detector resets to Warming, baselines are unloaded, `scored_chunks` is empty, `follower_state` is reset. This means:
- Mode transitions detected after eviction may be spurious (Warming -> X when the student was already in Running)
- Teaching moment selection needs baselines (reloaded from D1 on next chunk, as today)
- scored_chunks loss means teaching moment selection only sees post-eviction chunks
Accept degraded accumulation post-eviction. The pre-eviction accumulator data (moments, transitions, drilling records) is preserved in DO storage. Post-eviction, new moments accumulate from the reconstructed pipeline state. The synthesis prompt will have a gap in mode history but the core teaching moments from before eviction are intact. This is acceptable for beta -- full pipeline state persistence (mode_detector, scored_chunks, follower_state) is deferred as a future improvement.

### Eval sessions
Eval sessions (`eval_chunk` messages, `is_eval_session=true`) depend on per-chunk observation responses with `eval_context` attached. **The per-chunk observation path (`generate_observation`) is preserved for eval sessions only.** Guard: `if self.inner.borrow().is_eval_session { /* old per-chunk path */ } else { /* accumulate */ }`. This keeps the eval infrastructure working while production sessions use the new synthesis model.

### Deferred synthesis: dual execution context
`synthesize_session()` logic must run in two contexts: inside the DO (normal path) and inside a Worker route handler (deferred recovery via `POST /api/practice/synthesize`). Extract the core synthesis logic (prompt building, Anthropic call, D1 persistence) into `synthesis.rs` as free functions that accept `&Env` + `SessionAccumulator` + student context. Both the DO and the Worker route call into the same functions.

### Deployment sequencing
D1 migration (`ALTER TABLE sessions ADD COLUMN accumulator_json TEXT; ADD COLUMN needs_synthesis INTEGER DEFAULT 0;`) must deploy BEFORE the Worker code update. If code deploys first, any disconnect during the rollout window hits a missing column. Sequence: 1) Apply migration (`just migrate-prod`), 2) Deploy Worker (`just deploy-api`).

## Future: Per-STOP LLM Enrichment (A/B Test)

The `llm_analysis: Option<String>` field on `AccumulatedMoment` is the hook. To test whether per-STOP LLM analysis improves synthesis quality:
1. Enable per-STOP LLM calls (subagent only, not full teacher pipeline)
2. Store analysis text in the moment's `llm_analysis` field
3. Synthesis prompt includes these as additional context alongside raw scores
4. Compare synthesis quality: raw-only vs LLM-enriched via eval pipeline

Build for raw-only now. The accumulator format supports enrichment without changing the synthesis interface.

## Implementation Requirements (from autoplan review)

### Error handling for synthesis LLM call
- Timeout: persist accumulator to D1 with `needs_synthesis`, send "Preparing your feedback..." to client, deferred recovery on next load
- Malformed/empty response: log full response, persist raw accumulator, send fallback message with key moments as structured data
- Model refusal: same as malformed
- DO storage write failure: log, continue with in-memory (non-fatal)
- D1 accumulator_json write failure on disconnect: log as data loss event (Sentry), no recovery possible
- Accumulator deserialization failure: clear `needs_synthesis` flag, log schema mismatch

### Auth on deferred synthesis endpoint
`POST /api/practice/synthesize` must verify student JWT and that `student_id` from JWT matches the session's `student_id`. Use existing auth middleware.

### Module structure
Extract synthesis logic into new `apps/api/src/practice/synthesis.rs` (prompt builder, LLM call, persistence). Keep `session.rs` focused on DO lifecycle and chunk processing.

### Observability
- Log accumulator size (moments, transitions, drilling records) at synthesis time
- Log synthesis LLM latency
- Log deferred recovery triggers (alarm, disconnect, reconnect)
- Track via Cloudflare Workers OTLP drain to Sentry

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

---

## Autoplan Review Outputs

### NOT in scope
- Cross-session synthesis (load previous session's synthesis for same piece) -- memory system supports this, wire later
- Streaming synthesis response (SSE/chunked WS) -- synthesis is short, blocking wait is fine for beta
- Per-STOP LLM enrichment A/B test infrastructure -- hook exists (`llm_analysis` field), defer test execution
- Real-time chat tool_use during listening mode -- separate feature, unrelated to observation model
- iOS client changes -- iOS follows web beta

### What already exists
| Sub-problem | Existing Code | Reused? |
|---|---|---|
| STOP classification | `stop::classify()` | Yes, unchanged |
| Teaching moment selection | `teaching_moments::select_teaching_moment()` | Yes, accumulates |
| Mode state machine | `practice_mode.rs` ModeDetector | Yes, output repurposed |
| Score following + bar analysis | `analysis.rs`, `score_follower.rs` | Yes, unchanged |
| Piece identification | `piece_identify.rs` | Yes, unchanged |
| DO identity persistence | `ensure_session_identity()` | Yes, expanded |
| Memory synthesis | `memory.rs` should_synthesize/run_synthesis | Yes, unchanged |
| Anthropic teacher HTTP client | `ask.rs` call_anthropic_teacher() | Yes, reused in synthesis.rs |

### Dream state delta
This plan eliminates the notification-bot model and establishes the accumulate-then-synthesize pattern. After shipping, we are ~60% toward the 12-month ideal (multi-session arcs, adaptive practice plans). The remaining 40% builds directly on the accumulator infrastructure.

### Error & Rescue Registry

| Method/Codepath | Failure | Rescued? | Action | User Sees |
|---|---|---|---|---|
| DO storage put accumulator | Write fail | Y | Log, continue with in-memory | Nothing |
| synthesize_session() Anthropic | Timeout | Y | D1 needs_synthesis, deferred recovery | "Preparing feedback..." then later delivery |
| synthesize_session() Anthropic | Malformed response | Y | Log, fallback with raw moments | Structured key moments |
| synthesize_session() Anthropic | Model refusal | Y | Same as malformed | Same |
| D1 INSERT messages (synthesis) | Write fail | Y | console_error!, WS push already succeeded | Synthesis visible in session but missing from history |
| D1 INSERT sessions (accumulator) | Write fail | N | console_error!, Sentry alert | No deferred recovery possible |
| POST /api/practice/synthesize | Session not found | Y | 404 | "Session data not found" |
| POST /api/practice/synthesize | accumulator_json NULL | Y | Clear flag, 400 | Silent (no feedback) |
| Accumulator deserialization | Schema mismatch | Y | Clear flag, log | No feedback for that session |

### Failure Modes Registry

| Codepath | Failure Mode | Rescued? | Test? | User Sees? | Logged? |
|---|---|---|---|---|---|
| Per-chunk accumulation | DO storage write fail | Y | Needed | Nothing | Y |
| Synthesis LLM call | Timeout (>10s) | Y | Needed | Deferred feedback | Y |
| Synthesis LLM call | Empty/malformed response | Y | Needed | Fallback moments | Y |
| D1 synthesis persist | Write fail | Y | Needed | Missing from history | Y |
| D1 accumulator persist | Write fail on disconnect | N (CRITICAL) | Needed | Lost session data | Y |
| Deferred synthesis | Deserialization fail | Y | Needed | No feedback | Y |
| Deferred synthesis | Auth fail | Y | Needed | 401 | Y |

**1 CRITICAL GAP:** D1 accumulator_json write failure on disconnect means total data loss for that session. Mitigation: this is the belt in a belt-and-suspenders setup (DO storage is primary). Only triggers when DO is already garbage-collected AND D1 write fails simultaneously. Probability: very low. Monitoring via Sentry alert is sufficient.

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Principle | Rationale | Rejected |
|---|-------|----------|-----------|-----------|----------|
| 1 | CEO-S3 | Auth check on /api/practice/synthesize | P1 completeness | New endpoint needs JWT + student_id ownership check | Skip auth |
| 2 | CEO-S2 | Add error handling for synthesis LLM failures | P1+P5 | 4 unhandled error paths (timeout, malformed, refusal, empty) need explicit rescue | Silent failures |
| 3 | CEO-S3 | Rate limit deferred synthesis endpoint | P3 pragmatic | Reuse existing rate limiter, prevent abuse | No rate limit |
| 4 | CEO-S5 | Extract synthesis.rs module | P5 explicit | session.rs at 1813 lines, adding synthesis would exceed 2000 | Keep in session.rs |
| 5 | CEO-S6 | All 8 test gaps required | P1 completeness | New codepaths need unit + integration coverage | Defer tests |
| 6 | CEO-S8 | Add observability (log accumulator size, LLM latency, recovery triggers) | P1 completeness | New codepaths need structured logging | Skip observability |
| 7 | CEO-0bis | Choose Approach A (full rewrite) over B (hold+deliver) or C (hybrid flag) | P1+P5 | B doesn't fix durability/cost; C over-engineers A/B hook | Approach B, C |
| 8 | CEO-0D | Defer cross-session synthesis to TODOS | P3 pragmatic | Memory system supports it, additive scope for later | Add to this plan |
| 9 | CEO-0D | Defer streaming synthesis to TODOS | P3 pragmatic | Response short enough for blocking wait in beta | Add to this plan |
| 10 | CEO-0D | Surface progress indicators as TASTE DECISION | -- | Close call: UX value vs scope creep | Auto-decide either way |
