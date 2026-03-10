# CrescendAI System Architecture

**Date:** 2026-03-02
**Status:** Approved, pending implementation plan

## Overview

CrescendAI is a multi-platform (iOS + web) practice companion for pianists. It listens to the student play, identifies teaching moments, and gives specific observations -- the kind a teacher would make after hearing them play.

Two platform paths share one backend:
- **iOS (on-device-first):** Audio inference, teaching moment detection, and the student model all run locally on the iPhone. The only cloud dependency is the teacher LLM call and data sync.
- **Web (cloud-based):** Browser captures audio via MediaRecorder, uploads 15s chunks to the API, which runs inference via the HF endpoint and pushes real-time observations back over WebSocket. A chat interface lets the student discuss sessions with the teacher.

## System Diagram

### iOS Path (on-device inference)

```
+--------------------------------------------------------------+
|                    iPhone (iOS 17+)                           |
|                                                              |
|  +----------+   +----------+   +------------------------+   |
|  | AVAudio  |-->| Ring     |-->| Core ML MuQ            |   |
|  | Engine   |   | Buffer   |   | (6-dim scores per      |   |
|  | (24kHz   |   | (15s     |   |  15s chunk, ~1-2s)     |   |
|  |  mono)   |   |  chunks) |   +----------+-------------+   |
|  +----------+   +----------+              |                  |
|                                           v                  |
|                              +---------------------+         |
|                              | On-Device Pipeline   |         |
|                              |  - STOP classifier   |         |
|                              |    (6-weight LR)     |         |
|                              |  - Teaching moment   |         |
|                              |    selection          |         |
|                              |  - Blind spot detect  |         |
|                              +----------+----------+         |
|                                         |                    |
|  +----------------------------------------------------+     |
|  |                    SwiftData (local)                 |     |
|  |  Student { baselines, level, goals, repertoire }    |     |
|  |  Session { chunks[], moments[], duration }          |     |
|  |  Exercise { title, instructions, dimensions }       |     |
|  +------------------------+---------------------------+     |
|                           | background sync                  |
|  "How was that?" -------->+                                  |
+---------------------------+----------------------------------+
                            | HTTPS
                            v
                    [Shared Backend]
```

### Web Path (cloud inference)

```
+--------------------------------------------------------------+
|                    Browser (crescend.ai)                      |
|                                                              |
|  +-------------+   +---------------+   +----------------+   |
|  | MediaRec-   |-->| 15s Opus/WebM |-->| Upload to      |   |
|  | order       |   | chunks        |   | POST /api/     |   |
|  | (getUserMe- |   |               |   | practice/chunk |   |
|  |  dia)       |   +---------------+   +-------+--------+   |
|  +-------------+                                |            |
|                                                 |            |
|  +-------------+   +---------------+            |            |
|  | AudioContext|-->| Waveform      |            |            |
|  | AnalyserNode|   | Visualizer    |            |            |
|  +-------------+   +---------------+            |            |
|                                                 |            |
|  +----------------------------------------------------+     |
|  |  Chat Interface (TanStack Start)               |     |
|  |  - Streaming LLM responses                     |     |
|  |  - Recording overlay with observation toasts    |     |
|  |  - Session summary posted to chat               |     |
|  |  - Conversation history                         |     |
|  +------------------------+---------------------------+     |
|                           | WebSocket + HTTPS                |
+---------------------------+----------------------------------+
                            |
                            v
                    [Shared Backend]
```

### Shared Backend

```
           +------------------------------------+
           |   Cloudflare Workers (single app)   |
           |                                     |
           |  POST /api/ask                      |
           |    -> builds teacher prompt          |
           |    -> calls LLM providers            |
           |    -> returns observation text        |
           |                                     |
           |  POST /api/practice/start           |
           |    -> creates Durable Object session |
           |  POST /api/practice/chunk            |
           |    -> uploads audio, triggers infer  |
           |  WS /api/practice/ws/:sessionId      |
           |    -> real-time observations          |
           |                                     |
           |  POST /api/chat/send                 |
           |    -> streaming teacher chat          |
           |                                     |
           |  POST /api/sync                      |
           |    -> receives student model delta    |
           |    -> upserts to D1                  |
           |    -> returns exercise updates        |
           |                                     |
           |  POST /api/auth/apple                |
           |    -> validates Apple ID token        |
           |    -> issues session JWT              |
           |                                     |
           |  Bindings:                           |
           |    D1 -> students, sessions, exercises|
           |    KV -> JWTs, rate limits            |
           |    R2 -> audio chunks (web)           |
           |    DO -> practice sessions (web)      |
           |                                     |
           +-------+----------------+----------+
                   | HTTPS           | HTTPS
                   v                 v
          +----------------+  +----------------+
          |   Groq API     |  | Anthropic API  |
          |  (Llama 70B)   |  | (Sonnet 4.6)   |
          |  Subagent +    |  | Teacher LLM    |
          |  UI subagent   |  |                |
          +----------------+  +----------------+
                   |
          +----------------+  +----------------+
          | OpenRouter     |  | HF Inference   |
          | (fallback)     |  | (MuQ, web path)|
          +----------------+  +----------------+
```

## What Runs Where

| Component | iOS | Web |
|---|---|---|
| Audio capture + chunking | AVAudioEngine (on-device, no network) | MediaRecorder + chunk upload (network) |
| MuQ inference (6 dims) | Core ML (on-device, no network) | HF Inference Endpoint (cloud) |
| STOP classifier | On-device (6 floats + bias) | Heuristic in Durable Object (cloud) |
| Teaching moment selection | On-device | Durable Object (cloud) |
| Blind spot detection | On-device | Durable Object (cloud) |
| Student model + history | SwiftData (local-first) | D1 (cloud) |
| Real-time observations | N/A (on-demand "how was that?") | WebSocket push during recording |
| Chat interface | N/A (planned) | TanStack Start (streaming SSE) |
| Data sync/backup | Background async to D1 | Direct D1 access |
| Auth | Sign in with Apple (native) | Sign in with Apple (JS SDK popup) |
| Teacher LLM observation | Workers -> LLM providers | Workers -> LLM providers |

## Key Design Decisions

### On-Device Inference (Core ML)

MuQ runs on the iPhone's Neural Engine via Core ML. The finetuned model outputs 6 dimensions directly (dynamics, timing, pedaling, articulation, phrasing, interpretation).

**Why on-device:**

- Zero inference cost at any scale
- Zero network dependency during practice
- Lower latency (no upload, no round-trip)
- Privacy: audio never leaves the device for scoring

**Model delivery:**

- ~315MB quantized (INT8) or ~630MB (FP16)
- Downloaded on first launch (too large for App Store binary)
- Apple's On Demand Resources (ODR) or direct download from R2

**Fallback:** If Core ML conversion fails or quality degrades, the existing HuggingFace inference endpoint remains available. The iOS app has a feature flag (`useOnDeviceInference`) that switches between Core ML and cloud inference. Cloud path: upload chunk to `POST /api/inference`, Worker forwards to HF endpoint, returns scores.

### Local-First Data (SwiftData + D1 Sync)

On iOS, all data lives on-device first in SwiftData. The phone is the source of truth for student data and sessions. D1 stores copies for backup and cross-platform access. On web, D1 is the primary data store -- the web app reads and writes directly through the API.

**Why local-first (iOS):**

- Practice sessions work without internet (except LLM call)
- Fast reads (no network latency for student model lookups)
- D1 enables cross-platform access to the same data (web, future Android)

**Sync protocol:** Simple, conflict-free. The phone pushes deltas (new sessions, updated baselines) to D1 after each session. The only server-to-client data is exercise updates. No conflict resolution needed because the phone is authoritative.

### Sign in with Apple

Native one-tap auth. Apple provides a stable user ID and relay email. The Workers backend validates the Apple ID token and issues a session JWT stored in iOS Keychain.

**Why Sign in with Apple:**

- Zero-friction auth (one tap)
- Required by App Store if you offer account-based features
- Provides stable cross-device identity
- Captures relay email for future communication

### LLM Providers (Multi-Provider)

The pipeline uses direct provider APIs optimized per stage rather than routing everything through a single gateway. See `docs/apps/11-teacher-voice-finetuning.md` for the full provider rationale and future fine-tuning strategy.

**Stage 1 (Subagent):** Groq direct API, Llama 3.3 70B or Llama 4 Maverick. Groq's LPU runs Llama 70B at 450-800 tok/s -- the subagent completes in ~0.3s. Cost: ~$0.50-0.60/M tokens.

**Stage 2 (Teacher):** Anthropic direct API, Claude Sonnet 4.6. Eliminates the OpenRouter routing hop. Prompt caching keeps the teacher persona prefix cached across all requests. Cost: $3.00 input / $15.00 output per M tokens.

**Stage 3 (UI Subagent, optional):** Same as Stage 1 (Groq).

**Fallback:** OpenRouter as a fallback gateway if either direct provider is down. Emergency fallback: Cloudflare Workers AI (Llama 3.1 70B), co-located with Workers.

**Why multi-provider over OpenRouter-only:**

- ~0.3-0.5s latency savings (no routing hop)
- Native prompt caching with Anthropic API
- Groq's LPU gives 3-5x faster subagent inference vs. GPU-based providers
- OpenRouter remains available as fallback routing layer

### Model Accuracy

The finetuned MuQ model is approximately 80% accurate on pairwise rankings (R2 ~0.5) with current training data. Even expert piano teachers disagree roughly 20% of the time on what constitutes better playing. The 6 dimension scores are useful signals, not ground truth. The system's value is in the subagent analysis and teacher delivery, not raw scores. This shapes how prompts treat dimension scores: as evidence to reason over, not verdicts to report.

### Cloudflare Workers (Thin Backend)

The backend is a single Cloudflare Workers application. It handles:

- Auth (Apple ID token validation, JWT issuance)
- LLM proxy (holds OpenRouter API key, builds prompts, calls LLM)
- Data sync (receives student/session data from iOS, stores in D1)
- Exercise serving (serves curated exercises from D1)

**Why Cloudflare Workers:**

- Free tier covers everything at early scale ($0-5/mo)
- D1 (SQLite) is sufficient for the data model
- KV for JWT storage and rate limiting
- R2 available if audio storage is needed later
- Already have a Cloudflare account and domain (crescend.ai)

## On-Device Pipeline Detail

### Audio Capture

AVAudioEngine captures audio at 24kHz mono (MuQ's native sample rate). A circular ring buffer holds the last 5 minutes of PCM samples (~29MB). A background timer fires every 15 seconds, extracts the latest chunk, and feeds it to Core ML.

Background audio mode (`UIBackgroundModes: audio`) keeps recording when the screen is off.

### Inference

Each 15-second chunk is processed by the Core ML MuQ model:

- Input: raw waveform or mel spectrogram (depending on conversion approach)
- Output: 6 dimension scores (Float, 0-1 range)
- Latency: ~1-2 seconds on A16+ (iPhone 14+)

The Core ML model package includes MuQ's audio preprocessing (resampling, feature extraction), either baked into the model via coremltools tracing or reimplemented in Swift using Accelerate.

### STOP Classification

A logistic regression with 6 weights + bias, hardcoded in Swift. Extracts from the trained sklearn model in `model/src/masterclass_experiments/`. Predicts "would a teacher stop here?" for each chunk.

### Teaching Moment Selection

When "how was that?" is triggered:

1. Collect all chunks from the current session (or since last query)
2. Filter: STOP probability > 0.5
3. Rank by STOP probability descending
4. Select top-1 chunk
5. Identify blind-spot dimension (deviation from student baseline, or lowest score on cold start)

### Student Model Update

After each session ends:

- Compute session averages per dimension
- Update baselines via exponential moving average (alpha=0.3)
- Infer level from scores + repertoire
- Sync to D1 in background

## Data Model

### SwiftData (on-device)

```swift
@Model class Student {
    var appleUserId: String
    var email: String?
    var inferredLevel: String?    // beginner / intermediate / advanced
    var baselineDynamics: Double?
    var baselineTiming: Double?
    var baselinePedaling: Double?
    var baselineArticulation: Double?
    var baselinePhrasing: Double?
    var baselineInterpretation: Double?
    var baselineSessionCount: Int
    var explicitGoals: String?    // JSON
    var lastSyncedAt: Date?
    var sessions: [PracticeSession]
}

@Model class PracticeSession {
    var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var chunks: [ChunkResult]
    var observations: [Observation]
    var synced: Bool
}

@Model class ChunkResult {
    var index: Int
    var startOffset: TimeInterval
    var duration: TimeInterval
    var dynamics: Double
    var timing: Double
    var pedaling: Double
    var articulation: Double
    var phrasing: Double
    var interpretation: Double
    var stopProbability: Double
}

@Model class Observation {
    var chunkIndex: Int
    var dimension: String
    var text: String
    var elaboration: String?
    var createdAt: Date
}
```

### D1 (cloud sync)

```sql
CREATE TABLE students (
    apple_user_id TEXT PRIMARY KEY,
    email TEXT,
    inferred_level TEXT,
    baseline_dynamics REAL,
    baseline_timing REAL,
    baseline_pedaling REAL,
    baseline_articulation REAL,
    baseline_phrasing REAL,
    baseline_interpretation REAL,
    baseline_session_count INTEGER DEFAULT 0,
    explicit_goals TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    avg_dynamics REAL,
    avg_timing REAL,
    avg_pedaling REAL,
    avg_articulation REAL,
    avg_phrasing REAL,
    avg_interpretation REAL,
    observations_json TEXT,
    chunks_summary_json TEXT
);

CREATE TABLE exercises (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    target_dimensions TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    category TEXT NOT NULL,
    repertoire_tags TEXT,
    source TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE student_exercises (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    session_id TEXT REFERENCES sessions(id),
    assigned_at TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    response TEXT,
    dimension_before_json TEXT,
    dimension_after_json TEXT,
    times_assigned INTEGER DEFAULT 1,
    UNIQUE(student_id, exercise_id, session_id)
);
```

## Sync Protocol

```
iOS App                              Workers + D1
   |                                      |
   |  POST /api/sync                      |
   |  Authorization: Bearer <JWT>         |
   |  {                                   |
   |    student: { baselines, goals },    |
   |    newSessions: [ ... ],             |
   |    lastSyncTimestamp: "..."           |
   |  }                                   |
   |  ------------------------------>     |
   |                                      |  Upsert student
   |                                      |  Insert new sessions
   |                                      |
   |  <------------------------------     |
   |  {                                   |
   |    exerciseUpdates: [ ... ],         |  New/updated exercises
   |    syncTimestamp: "..."               |  since lastSyncTimestamp
   |  }                                   |
```

**When sync happens:**

- After each session ends
- On app launch (if last sync > 1 hour ago)
- Piggybacked on "ask" calls

**Conflict resolution:** None needed. The phone is authoritative for student and session data. The server is authoritative for exercises.

## The "Ask" Flow

The core interaction, end to end:

1. Student taps "How was that?"
2. iOS collects all chunks since session start (or last query)
3. On-device: filters by STOP > 0.5, picks top teaching moment, identifies dimension
4. iOS sends structured context to `POST /api/ask`:

```json
{
    "teaching_moment": {
        "chunk_index": 7,
        "stop_probability": 0.87,
        "dimension": "pedaling",
        "dimension_score": 0.35,
        "all_scores": { "dynamics": 0.65, "timing": 0.71, "pedaling": 0.35, "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54 }
    },
    "student": {
        "level": "intermediate",
        "baselines": { "pedaling": 0.62 },
        "goals": "Preparing Chopin Nocturne for recital",
        "session_count": 12
    },
    "session": {
        "duration_min": 18,
        "total_chunks": 72,
        "chunks_above_threshold": 5
    }
}
```

1. Worker builds teacher prompt (system + user), calls OpenRouter
2. OpenRouter returns observation text
3. Worker returns observation to iOS
4. iOS displays 1-3 sentences on screen

**Latency target:** <3 seconds from tap to observation. Teaching moment selection is instant (on-device). LLM call is ~2 seconds total.

## Teacher LLM Prompt

The "Ask" flow uses a two-stage subagent pipeline (see `docs/apps/06a-subagent-architecture.md`): a fast analysis subagent (Haiku/Flash-class) reasons about which moment matters most and why, then a quality teacher LLM (Sonnet/GPT-4o-class) generates the observation. The prompt templates are in `docs/apps/06-teacher-llm-prompt.md`.

Using OpenRouter, models can be changed per-request and per-stage. The subagent and teacher LLM are separate API calls with independent model selection.

## Implementation Slices

The 00-09 docs in `docs/` define the implementation slices. Updated to reflect this architecture:

| Slice | Doc | What It Covers |
|---|---|---|
| 00 | 00-practice-companion.md | Product spec (updated) |
| 01 | 01-phone-audio-validation.md | Validate MuQ on phone audio + Core ML feasibility |
| 02 | 02-ios-audio-capture.md | AVAudioEngine, ring buffer, chunking, background mode |
| 03 | 03-chunked-inference-pipeline.md | Core ML MuQ conversion, on-device pipeline, session buffer |
| 04 | 04-teaching-moment-detection.md | STOP classifier in Swift, blind spot detection |
| 05 | 05-student-model-and-auth.md | Sign in with Apple, SwiftData, D1 sync |
| 06 | 06-teacher-llm-prompt.md | Teacher persona prompt, output handling (stage 2 of pipeline) |
| 06a | 06a-subagent-architecture.md | Two-stage subagent, synthesized facts, reasoning framework |
| 07 | 07-exercise-database.md | D1 exercises, sync to device, LLM-generated exercises |
| 11 | 11-teacher-voice-finetuning.md | Teacher voice fine-tuning strategy, provider architecture (Groq + Anthropic) |
| 08 | 08-focus-mode.md | Guided practice mode with targeted exercises |
| 09 | 09-ios-frontend.md | SwiftUI screens: Practice, Observation, Review, Focus |
| 10 | 10-on-demand-ui.md | Chat-first interface with on-demand interactive components |

## Observability

Error tracking via Sentry across all three surfaces:

- **iOS:** `sentry-cocoa` SPM -- crash reporting, error capture with breadcrumbs, MetricKit integration
- **Web (client):** `@sentry/react` -- React ErrorBoundary integration, API error capture, WebSocket error capture
- **Web (SSR) + API Worker:** Cloudflare Workers Observability OTLP drain to Sentry -- captures invocation traces, `console_error!` output, and panics without any SDK in the Rust/WASM binary

Cloudflare Workers built-in analytics covers API health and latency metrics. Performance monitoring via Sentry at 10% sample rate for beta.

## Future Considerations

- **Android:** D1 sync layer is platform-agnostic. Add Android client that talks to the same Workers API. Auth: add Google sign-in alongside Apple.
- **On-device LLM:** Apple Foundation Models (iOS 18.2+) or bundled Llama 3.2 3B could replace the cloud LLM call for fully offline operation.
- **Audio playback:** R2-stored chunks (web path) could enable session playback and re-analysis.
