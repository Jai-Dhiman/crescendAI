# CrescendAI System Architecture

**Date:** 2026-03-02
**Status:** Approved, pending implementation plan

## Overview

CrescendAI is an iOS-first practice companion for pianists. The phone sits on the piano, listens continuously, and gives one specific teaching observation when the student asks "how was that?"

The architecture is on-device-first: audio inference, teaching moment detection, and the student model all run locally on the iPhone. The only cloud dependency is a thin API for the teacher LLM call and data sync.

## System Diagram

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
           +------------------------------------+
           |   Cloudflare Workers (single app)   |
           |                                     |
           |  POST /api/ask                      |
           |    -> builds teacher prompt          |
           |    -> calls OpenRouter (any LLM)     |
           |    -> returns observation text        |
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
           |  GET /api/exercises                  |
           |    -> serves from D1                  |
           |    -> versioned, iOS caches locally   |
           |                                     |
           |  Bindings:                           |
           |    D1 -> students, sessions, exercises|
           |    KV -> JWTs, rate limits            |
           |                                     |
           +----------------+-------------------+
                            | HTTPS
                            v
                   +------------------+
                   |   OpenRouter     |
                   |   (any LLM)      |
                   +------------------+
```

## What Runs Where

| Component | Location | Network Required? |
|---|---|---|
| Audio capture + chunking | iPhone (AVAudioEngine) | No |
| MuQ inference (6 dims) | iPhone (Core ML) | No |
| STOP classifier | iPhone (6 floats + bias) | No |
| Teaching moment selection | iPhone | No |
| Blind spot detection | iPhone | No |
| Student model + history | iPhone (SwiftData) | No |
| Exercise database | iPhone (cached from D1) | Initial download only |
| Data sync/backup | Cloudflare Workers + D1 | Background, async |
| Auth | Sign in with Apple + Workers JWT | On login only |
| Teacher LLM observation | Workers -> OpenRouter | Yes (only on "ask") |

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

All data lives on-device first in SwiftData. The phone is the source of truth for student data and sessions. D1 stores copies for backup and future cross-platform access.

**Why local-first:**
- Practice sessions work without internet (except LLM call)
- Fast reads (no network latency for student model lookups)
- D1 enables future Android/web access to the same data

**Sync protocol:** Simple, conflict-free. The phone pushes deltas (new sessions, updated baselines) to D1 after each session. The only server-to-client data is exercise updates. No conflict resolution needed because the phone is authoritative.

### Sign in with Apple

Native one-tap auth. Apple provides a stable user ID and relay email. The Workers backend validates the Apple ID token and issues a session JWT stored in iOS Keychain.

**Why Sign in with Apple:**
- Zero-friction auth (one tap)
- Required by App Store if you offer account-based features
- Provides stable cross-device identity
- Captures relay email for future communication

### OpenRouter for LLM

The teacher LLM call goes through OpenRouter, which provides a unified API to Claude, GPT-4, Llama, Gemini, and others. Switching models is a string change.

**Why OpenRouter:**
- Model-agnostic: A/B test any model without code changes
- Single API key manages multiple providers
- Fallback routing: if one provider is down, try another

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

5. Worker builds teacher prompt (system + user), calls OpenRouter
6. OpenRouter returns observation text
7. Worker returns observation to iOS
8. iOS displays 1-3 sentences on screen

**Latency target:** <3 seconds from tap to observation. Teaching moment selection is instant (on-device). LLM call is ~2 seconds total.

## Teacher LLM Prompt

The "Ask" flow uses a two-stage subagent pipeline (see `docs/06a-subagent-architecture.md`): a fast analysis subagent (Haiku/Flash-class) reasons about which moment matters most and why, then a quality teacher LLM (Sonnet/GPT-4o-class) generates the observation. The prompt templates are in `docs/06-teacher-llm-prompt.md`.

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
| 08 | 08-focus-mode.md | Guided practice mode with targeted exercises |
| 09 | 09-ios-frontend.md | SwiftUI screens: Practice, Observation, Review, Focus |
| 10 | 10-on-demand-ui.md | Chat-first interface with on-demand interactive components |

## Future Considerations

- **Android:** D1 sync layer is platform-agnostic. Add Android client that talks to the same Workers API. Auth: add Google sign-in alongside Apple.
- **Web dashboard:** D1 data is accessible from Workers. Build a simple HTML dashboard for session review on desktop.
- **On-device LLM:** Apple Foundation Models (iOS 18.2+) or bundled Llama 3.2 3B could replace the cloud LLM call for fully offline operation.
- **Audio storage:** If needed (playback, cloud fallback inference), store chunks in R2 via the sync endpoint.
