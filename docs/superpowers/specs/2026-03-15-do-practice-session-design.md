# Durable Object Practice Session Orchestration -- Design Spec

**Goal:** Wire the `PracticeSession` Durable Object to orchestrate the full inference-to-observation pipeline: chunk upload notification -> HF inference -> STOP classifier -> teaching moment selection -> LLM observation -> WebSocket push.

**Scope:** `apps/api/src/practice/session.rs` (primary), `apps/api/src/server.rs` (auth pass-through). Does NOT touch `apps/web/`, `apps/ios/`, or `model/`.

**Decisions:**
- DO owns the full pipeline (fetches from R2, calls HF, runs STOP, calls `/api/ask`)
- Internal fetch to `/api/ask` for LLM observations (reuses existing two-stage pipeline)
- Client builds session summary text from observations array (no holistic LLM summary)
- D1 baselines fetched once on first `chunk_ready`, cached for session
- Silent degradation on inference failures (zeroed scores, Sentry logging, no client-facing errors)
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
  |                            |                     [internal fetch /api/ask (~2s)]   |
  |<-- { observation } -------|<--------------------------|                          |
  |                            |                            |                          |
  |-- WS { end_session } ---->|                            |                          |
  |                            |                     [persist observations to D1]      |
  |<-- { session_summary } ---|<--------------------------|                          |
```

- `chunk_processed` is sent immediately after HF inference (scores only)
- `observation` is sent asynchronously after STOP + LLM pipeline
- Client-side `ObservationThrottle` gates delivery to 3-minute windows
- `session_summary` sends accumulated observations; client builds summary text

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
| `dim_stats` | `DimStats` | Welford's running stats (existing) |

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
   - Response: `{ dynamics, timing, pedaling, articulation, phrasing, interpretation }`
   - On failure: log to Sentry, send `chunk_processed` with zeroed scores, skip steps 4-8
3. **Send `chunk_processed`** -- Push `{ type: "chunk_processed", index, scores }` immediately
4. **Update DimStats** -- Welford's online algorithm with new scores
5. **Store ScoredChunk** -- Append `{ chunk_index, scores }` to `scored_chunks`
6. **Load baselines (one-time)** -- If `!baselines_loaded`, query D1 for student's historical dimension averages, set `baselines_loaded = true`
7. **Run STOP classifier** -- `stop::classify(&scores_array)` -> `StopResult`
8. **If triggered AND baselines loaded:**
   a. Run `teaching_moments::select_teaching_moment(scored_chunks, baselines, recent_obs)`
   b. If a moment is selected:
      - Internal fetch `POST /api/ask` with teaching moment payload
      - On success: push `{ type: "observation", text, dimension, framing }`
      - On LLM failure: use template fallback from `ask.rs`, push observation anyway
      - Store `ObservationRecord`
9. **Reset 30-min alarm**

### Timing

Steps 1-3: ~1-2s (HF inference dominates). Client gets scores quickly.
Steps 7-8: ~2-3s additional (LLM call). Observation arrives asynchronously.
Total worst case per chunk: ~4-5s (well within CF Workers' 30s wall-clock limit).

### HF Inference Call

```
POST {HF_INFERENCE_ENDPOINT}
Content-Type: application/octet-stream
Authorization: Bearer {HF_TOKEN}
Body: raw audio bytes (WebM/Opus from R2)

Response: { "dynamics": 0.65, "timing": 0.48, ... }
```

The HF handler (`apps/inference/handler.py`) accepts any audio format and resamples to 24kHz mono internally.

---

## Auth and Student Identity

### WebSocket Auth Flow

1. `POST /api/practice/start` validates JWT cookie, returns `sessionId`
2. Client connects to `WS /api/practice/ws/:sessionId`
3. Worker's `server.rs` validates auth from cookie, extracts `student_id`
4. Worker passes `student_id` as query param on internal DO fetch:

```rust
// In server.rs WebSocket routing:
let student_id = verify_auth_from_cookie(&req, &env)?;
let url = format!("https://do.internal/ws/{}?student_id={}", session_id, student_id);
```

5. DO extracts `student_id` from URL in its `fetch` handler

The `student_id` param is trusted because the internal fetch originates from the Worker (not the external client).

### Reconnection

When the client's WebSocket reconnects, it goes through the same `server.rs` path. The Worker re-validates auth. The DO receives a new WebSocket connection but keeps its in-memory state (scores, observations, baselines).

---

## Error Handling

| Step | Failure Mode | Response |
|------|-------------|----------|
| R2 fetch | Bucket unavailable, key missing | Log Sentry, send zeroed `chunk_processed`, skip pipeline |
| HF inference | Timeout, 5xx, malformed response | Log Sentry, send zeroed `chunk_processed`, skip pipeline |
| STOP classifier | Cannot fail (pure math) | N/A |
| Teaching moment selection | Cannot fail (pure logic) | N/A |
| `/api/ask` LLM call | Provider timeout, rate limit | Use template fallback observation, push to client |
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

If no history exists (new student), baselines default to the STOP classifier's training means (`SCALER_MEAN` from `stop.rs`). This means the STOP classifier fires based on absolute quality rather than personal deviation -- reasonable for a first session.

---

## Files Modified

| File | Action | What Changes |
|------|--------|-------------|
| `apps/api/src/practice/session.rs` | Major rewrite | Full pipeline orchestration, alarm, finalize, HF/R2/D1 calls |
| `apps/api/src/server.rs` | Modify | Pass `student_id` query param on DO WebSocket routing, validate auth on WS path |
| `apps/api/src/practice/mod.rs` | No change | Already exports all needed modules |

No new files needed. All supporting services (stop, teaching_moments, ask, dims) already exist.

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
