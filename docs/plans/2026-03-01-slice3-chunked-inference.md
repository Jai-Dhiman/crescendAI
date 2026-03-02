# Slice 3: Background Chunked MuQ Inference

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Process audio chunks through MuQ inference as they arrive during a practice session, accumulating 6-dimension scores in a session analysis buffer.

**Architecture:** Backend receives chunks from the iOS app, stores audio in R2, calls the existing HF inference endpoint per chunk, and accumulates results in a session buffer (D1). By the time the student asks "how was that?", most or all chunks have been analyzed.

**Tech Stack:** Rust/Axum (Cloudflare Workers), existing HF MuQ endpoint, D1 for session state, R2 for audio storage

---

## Context

Slice 2 produces 15-second audio chunks every 15 seconds. This slice processes them. The existing HF inference endpoint handles a single audio file and returns 19-dimension scores (mapped to 6 composite dimensions). The new requirement is handling a stream of chunks per session, pipelining inference, and accumulating results.

## Design

### Session + Chunk API

**New endpoints:**

```
POST /api/sessions
  -> Creates a new practice session
  <- { session_id, created_at }

POST /api/sessions/{session_id}/chunks
  Content-Type: multipart/form-data (audio file)
  Headers: X-Chunk-Index: 0, X-Chunk-Offset: 0.0
  -> Stores chunk, triggers inference
  <- { chunk_id, status: "processing" }

GET /api/sessions/{session_id}/status
  -> Returns session state with all chunk results so far
  <- { session_id, chunks: [{ index, status, dimensions, teaching_moment_score }] }

POST /api/sessions/{session_id}/ask
  Body: { question: "how was that?" }
  -> Triggers the priority logic + teacher LLM (Slices 4+6)
  <- { observation: "..." }
```

### Inference Pipeline Per Chunk

1. Chunk audio uploaded by iOS app arrives at Workers endpoint
2. Store audio in R2: `sessions/{session_id}/chunks/{chunk_index}.aac`
3. Construct absolute URL for the audio file
4. Call HF inference endpoint (existing) with the audio URL
5. Receive 19-dimension scores
6. Map to 6 composite dimensions (same mapping already in web app)
7. Store result in D1: `session_chunks` table
8. Return success to iOS app (or error if inference failed)

### D1 Schema

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,  -- nullable until auth exists (Slice 5)
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_chunks INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'  -- active, ended, abandoned
);

CREATE TABLE session_chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    chunk_index INTEGER NOT NULL,
    start_offset_sec REAL NOT NULL,
    duration_sec REAL NOT NULL,
    r2_key TEXT NOT NULL,
    inference_status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    dynamics REAL,
    timing REAL,
    pedaling REAL,
    articulation REAL,
    phrasing REAL,
    interpretation REAL,
    teaching_moment_score REAL,  -- null until Slice 4
    processing_time_ms INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, chunk_index)
);

CREATE INDEX idx_chunks_session ON session_chunks(session_id, chunk_index);
```

### Concurrency and Pipelining

- Chunks arrive every 15 seconds
- HF inference takes ~2-3 seconds per 15s chunk on GPU
- Pipeline: chunk N is being inferred while chunk N+1 is being uploaded
- No queuing problem -- inference is faster than chunk arrival rate
- If HF endpoint is slow or down: mark chunk as failed, continue accepting chunks. Retry failed chunks when endpoint recovers.

### Dimension Mapping (19 -> 6)

Reuse the existing mapping from the web app:

```
dynamics    = mean(dynamics_range, projection)
timing      = mean(rhythmic_timing, tempo_control)
pedaling    = mean(pedal_use, pedal_clarity)
articulation = mean(note_articulation, touch_sensitivity)
phrasing    = mean(use_of_space, dramatic_arc)
interpretation = mean(emotional_expression, creative_imagination, interpretive_depth, overall_interpretation)
```

(Exact mapping may differ -- check `apps/web/src/server.rs` for current composite logic.)

### What This Slice Does NOT Include

- Teaching moment scoring (Slice 4 adds `teaching_moment_score` to chunks)
- The "how was that?" response logic (Slice 6)
- User authentication (Slice 5)
- iOS app changes (Slice 2 handles the client side)

### Tasks

**Task 1: Add D1 migration for sessions and chunks**
- Create migration file with sessions + session_chunks tables
- Run migration on D1

**Task 2: Implement session creation endpoint**
- `POST /api/sessions` creates a session record
- Returns session_id

**Task 3: Implement chunk ingestion endpoint**
- `POST /api/sessions/{session_id}/chunks` receives audio
- Stores in R2
- Calls HF inference endpoint
- Maps 19 dims to 6 composite dims
- Stores result in D1
- Returns chunk_id + status

**Task 4: Implement session status endpoint**
- `GET /api/sessions/{session_id}/status` returns all chunk results
- Used by iOS app to display real-time analysis state

**Task 5: Handle inference failures**
- If HF endpoint returns error: mark chunk as failed
- Background retry logic (or retry on next status poll)
- Never block chunk ingestion on inference failure

### Open Questions

1. Should inference be synchronous (chunk endpoint waits for HF response) or async (returns immediately, iOS polls for results)? Sync is simpler but adds latency to the upload response. Async is more resilient.
2. R2 storage cost for session audio: at ~120KB per chunk, a 1-hour session is ~29MB. At 100 users doing 1 hour/day, that's ~87GB/month. Acceptable?
3. Should old session audio be garbage-collected after some retention period?
