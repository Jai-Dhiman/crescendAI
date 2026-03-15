# Durable Object Practice Session Orchestration -- Design Spec

**Goal:** Wire the `PracticeSession` Durable Object to orchestrate the full inference-to-observation pipeline: chunk upload notification -> HF inference -> STOP classifier -> teaching moment selection -> LLM observation -> WebSocket push.

**Scope:** `apps/api/src/practice/session.rs` (primary), `apps/api/src/server.rs` (auth pass-through). Does NOT touch `apps/web/`, `apps/ios/`, or `model/`.

**Decisions:**
- DO owns the full pipeline (fetches from R2, calls HF, runs STOP, generates observations)
- Direct Rust function call to `ask::handle_ask_inner()` for LLM observations (bypasses HTTP auth, avoids self-referential fetch overhead)
- Client builds session summary text from observations array (no holistic LLM summary)
- D1 baselines fetched once on first `chunk_ready`, cached for session
- Silent degradation on HF/R2 failures (zeroed scores, Sentry logging); graceful degradation on LLM failures (template fallback observation still delivered)
- Server-side observation throttle: skip LLM call if last observation was < 3 minutes ago (matches client-side throttle, avoids wasteful LLM token spend)
- Alarm + WebSocket close for cleanup (belt and suspenders)
- Ephemeral in-memory state (no DO storage persistence mid-session)

---

## Data Flow

```
Client                    Worker (stateless)           DO (PracticeSession)          External
  |                            |                            |                          |
  |-- POST /practice/start --->|  (validate JWT)            |                          |
  |<-- { sessionId } ---------|                            |                          |
  |                            |                            |                          |
  |-- WS /practice/ws/:id --->|-- internal fetch --------->|                          |
  |                            |   (pass student_id param)  |                          |
  |<-- { type: "connected" } --|<--------------------------|                          |
  |                            |                            |-- D1: get baselines --->|
  |                            |                            |<-- StudentBaselines -----|
  |                            |                            |                          |
  |-- POST /practice/chunk -->|  (validate JWT, store R2)  |                          |
  |<-- { r2Key } -------------|                            |                          |
  |                            |                            |                          |
  |-- WS { chunk_ready } ---->|                            |                          |
  |                            |                     [R2: fetch audio]                 |
  |                            |                     [HF: inference (~1-2s)]           |
  |                            |                     [STOP classifier]                 |
  |<-- { chunk_processed } ---|<--------------------------|                          |
  |                            |                            |                          |
  |                            |              [if STOP triggers teaching moment:]      |
  |                            |                     [direct call ask::handle_ask_inner]|
  |<-- { observation } -------|<--------------------------|                          |
  |                            |                            |                          |
  |-- WS { end_session } ---->|                            |                          |
  |                            |                     [persist observations to D1]      |
  |<-- { session_summary } ---|<--------------------------|                          |
```

- `chunk_processed` is sent immediately after HF inference (scores only)
- `observation` is sent asynchronously after STOP + LLM pipeline
- Both client-side `ObservationThrottle` AND server-side throttle gate observation delivery
- `session_summary` sends `{ type: "session_summary", observations: [...], summary: "" }`. The `summary` field is an empty string -- the client builds summary text from the observations array. The field is kept for backward compatibility with the `PracticeWsEvent` type.

---

## DO State and Lifecycle

### State Transitions

```
Idle -> Connected -> Active -> Summarizing -> Done
         (WS open)   (chunks    (end_session   (persisted,
                      flowing)    received)      WS closed)
```

### In-Memory State (not persisted to DO storage)

| Field | Type | Source |
|-------|------|--------|
| `session_id` | `String` | Extracted from WebSocket URL path |
| `student_id` | `String` | Passed via query param from Worker |
| `baselines` | `Option<StudentBaselines>` | D1 query on first `chunk_ready` |
| `baselines_loaded` | `bool` | Guard for one-time D1 query |
| `scored_chunks` | `Vec<ScoredChunk>` | Accumulated HF inference results |
| `observations` | `Vec<ObservationRecord>` | Generated teaching moments |
| `dim_stats` | `DimStats` | Welford's running stats (existing, uses `HashMap<String, f64>`) |
| `last_observation_at` | `Option<u64>` | Timestamp of last observation delivery (server-side throttle) |

Sessions are ephemeral. If the DO is evicted mid-session, the session is lost. Audio chunks persist in R2 regardless. Recovering mid-session state adds complexity for a rare edge case.

### Alarm

- Set 30-minute alarm on WebSocket connect
- Reset alarm on each `chunk_ready`
- On alarm fire: run `finalize_session()` (same as `end_session`)

### Cleanup (`finalize_session`)

1. Persist observations to D1 (`INSERT INTO observations`)
2. Send `session_summary` via WebSocket (if still open)
3. Close WebSocket
4. DO is garbage collected

---

## chunk_ready Pipeline

When DO receives `chunk_ready { index, r2Key }`:

1. **Fetch audio from R2** -- `env.bucket("CHUNKS").get(r2Key)` -> bytes
2. **Call HF inference** -- POST raw bytes to `HF_INFERENCE_ENDPOINT` with Bearer token
   - Response is nested: extract scores from `response.predictions`
   - On failure: log to Sentry, send `chunk_processed` with zeroed scores, skip steps 4-9
3. **Convert scores** -- HF returns `HashMap<String, f64>`. Convert to `[f64; 6]` using `DIMS_6` order for STOP classifier. Keep the HashMap for DimStats.
4. **Send `chunk_processed`** -- Push `{ type: "chunk_processed", index, scores }` immediately
5. **Update DimStats** -- `dim_stats.update(&scores_hashmap)` (Welford's, uses `HashMap<String, f64>`)
6. **Store ScoredChunk** -- Append `{ chunk_index, scores: scores_array }` to `scored_chunks`
7. **Load baselines (one-time)** -- If `!baselines_loaded`, query D1 for student's historical dimension averages, set `baselines_loaded = true`
8. **Run STOP classifier** -- `stop::classify(&scores_array)` -> `StopResult` (on current chunk only)
9. **If triggered AND baselines loaded AND server-side throttle allows (> 3 min since last observation):**
   a. Run `teaching_moments::select_teaching_moment(scored_chunks, baselines, recent_obs)`
   b. If a moment is selected:
      - Direct Rust call to `ask::handle_ask_inner(&env, student_id, teaching_moment)` (no HTTP, no auth needed)
      - On success: push `{ type: "observation", text, dimension, framing }`
      - On LLM failure: use template fallback from `ask.rs`, push observation anyway
      - Store `ObservationRecord`, update `last_observation_at`
10. **Reset 30-min alarm**

**Note on `select_teaching_moment`:** This function re-runs STOP classification on all accumulated chunks internally. For typical sessions (5-30 min, 20-120 chunks) this is negligible. For sessions > 30 min, the alarm cleanup will fire anyway.

### Timing

Steps 1-4: ~1-2s (HF inference dominates). Client gets scores quickly.
Steps 8-9: ~2-3s additional (LLM call). Observation arrives asynchronously.
Total worst case per chunk: ~4-5s (well within CF Workers' 30s wall-clock limit).

### HF Inference Call

```
POST {HF_INFERENCE_ENDPOINT}
Content-Type: application/octet-stream
Authorization: Bearer {HF_TOKEN}
Body: raw audio bytes (WebM/Opus from R2)

Response (nested):
{
  "predictions": { "dynamics": 0.65, "timing": 0.48, ... },
  "midi_notes": [...],
  "transcription_info": {...},
  "model_info": {...},
  "audio_duration_seconds": 15.0,
  "processing_time_ms": 1234
}
```

The DO extracts `response.predictions` for the 6-dim scores. The `midi_notes` and `transcription_info` fields are available for future score-following (Phase 1c) but are not used in this iteration.

The HF handler (`apps/inference/handler.py`) accepts any audio format and resamples to 24kHz mono internally. The model outputs 6 dimensions directly (not 19) -- the `map_19_to_6` function in `dims.rs` is NOT needed in this path.

### Server-Side Observation Throttle

The DO tracks `last_observation_at` and skips the LLM call if < 3 minutes have passed since the last observation. This mirrors the client-side `ObservationThrottle` and prevents wasteful LLM token spend when STOP fires frequently.

### LLM Call: Direct Function vs HTTP

The DO calls `ask::handle_ask_inner()` directly as a Rust function, not via internal HTTP fetch. This avoids two problems:
- **Auth:** The HTTP endpoint (`handle_ask`) validates JWT. The DO does not have the student's JWT token, only the `student_id` (passed by the Worker). A direct function call bypasses auth entirely since the caller is trusted.
- **Overhead:** Eliminates ~50ms HTTP round-trip overhead.

This requires extracting the core logic of `handle_ask` into a public `handle_ask_inner(env, student_id, teaching_moment) -> AskResult` function that the HTTP handler and the DO can both call.

---

## Auth and Student Identity

### WebSocket Auth Flow

**Current gap:** `server.rs` does NOT validate auth on the WebSocket upgrade path. It routes directly to the DO without checking the JWT. This must be fixed.

**Required changes to `server.rs`:**

1. `POST /api/practice/start` validates JWT cookie, returns `sessionId`
2. Client connects to `WS /api/practice/ws/:sessionId`
3. Worker's `server.rs` validates auth from cookie/header, extracts `student_id` (NEW -- currently missing)
4. Worker passes `student_id` as query param on internal DO fetch (NEW -- currently just passes session_id):

```rust
// In server.rs WebSocket routing (BEFORE -- no auth):
let session_id = path.trim_start_matches("/api/practice/ws/");
let url = format!("https://do.internal/ws/{}", session_id);

// AFTER -- validate auth and pass student_id:
let student_id = verify_auth_from_cookie_or_header(&req, &env)?;
let url = format!("https://do.internal/ws/{}?student_id={}", session_id, student_id);
```

5. DO extracts `student_id` from URL query param in its `fetch` handler

The `student_id` param is trusted because the internal fetch originates from the Worker (not the external client). The external WebSocket URL never includes `student_id`.

### Reconnection

When the client's WebSocket reconnects, it goes through the same `server.rs` path. The Worker re-validates auth. The DO receives a new WebSocket connection but keeps its in-memory state (scores, observations, baselines).

**Multiple connections:** The DO uses `state.accept_web_socket()` which can track multiple WebSockets. On reconnection, the old WebSocket may still be registered. The DO should close the previous WebSocket when a new one connects to the same session, keeping only the latest connection active.

---

## Error Handling

| Step | Failure Mode | Response |
|------|-------------|----------|
| R2 fetch | Bucket unavailable, key missing | Log Sentry, send zeroed `chunk_processed`, skip pipeline |
| HF inference | Timeout, 5xx, malformed response | Log Sentry, send zeroed `chunk_processed`, skip pipeline |
| STOP classifier | Cannot fail (pure math) | N/A |
| Teaching moment selection | Cannot fail (pure logic) | N/A |
| `ask::handle_ask_inner` LLM call | Provider timeout, rate limit | Use template fallback observation, push to client (graceful degradation, not silent) |
| D1 baselines query | DB unavailable | Set empty baselines, teaching moments use session-local DimStats only |
| D1 observation persist | DB unavailable | Log Sentry, observations lost (audio in R2 for retry) |

Principle: never surface infrastructure errors to the student mid-practice. Log to Sentry for operational visibility.

---

## D1 Baselines Query

Query the `observations` table for the student's last N sessions to compute per-dimension averages:

```sql
SELECT dimension, AVG(dimension_score) as avg_score
FROM observations
WHERE student_id = ?
  AND created_at > datetime('now', '-30 days')
GROUP BY dimension
```

**Mapping query results to `StudentBaselines`:** The query returns `{ dimension, avg_score }` rows. Map each row's `dimension` string to the corresponding `StudentBaselines` field. For dimensions with no historical rows (e.g., a student with pedaling data but no phrasing data), use the STOP classifier's training mean for that dimension (`SCALER_MEAN` from `stop.rs`).

If no history exists at all (new student), all baselines default to `SCALER_MEAN`. This means the STOP classifier fires based on absolute quality rather than personal deviation -- reasonable for a first session.

---

## Files Modified

| File | Action | What Changes |
|------|--------|-------------|
| `apps/api/src/practice/session.rs` | Major rewrite | Full pipeline orchestration, alarm, finalize, HF/R2/D1 calls |
| `apps/api/src/server.rs` | Modify | Add auth validation on WS path, pass `student_id` query param to DO |
| `apps/api/src/services/ask.rs` | Modify | Extract `handle_ask_inner(env, student_id, teaching_moment)` from `handle_ask` for direct DO calls |
| `apps/api/src/practice/mod.rs` | No change | Already exports all needed modules |

No new files needed. All supporting services (stop, teaching_moments, ask, dims) already exist.

### Deferred

- **Piece context:** The `/api/ask` pipeline uses `piece_context` for memory lookups and prompt building. The DO does not currently receive piece context from the client. Deferred to a follow-up where the client sends piece context via a WebSocket message (e.g., `{ type: "set_context", piece: "...", section: "..." }`).

---

## Testing

### Unit Tests (Rust `#[cfg(test)]`)

Extract pipeline logic into a pure function:
- Given: scores array, baselines, recent observations, STOP threshold
- Returns: `chunk_processed` event + optional observation event
- Testable without WebSocket, HTTP, or external services

Existing tests provide coverage for STOP classifier (7 tests) and teaching moment selection.

### Integration Test (Manual, Post-Deploy)

1. Sign in on web app, start recording, play piano for 30+ seconds
2. Verify `chunk_processed` events arrive with non-zero scores
3. Stop recording, verify `session_summary` arrives
4. Check D1: observations persisted with correct student_id
5. Check Sentry: no errors during session

### Not Tested This Iteration

- WebSocket reconnection edge cases (manual QA)
- Alarm-based cleanup (set alarm to 60s in dev to verify)
- Concurrent WebSocket connections to same DO
