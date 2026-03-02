# Slice 5: Student Model + Auth

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add user authentication and a persistent student model that tracks dimension profiles, repertoire, goals, and session history across practice sessions.

**Architecture:** Sign in with Apple for auth. Student model stored locally in SwiftData, synced to Cloudflare D1 for backup and cross-platform readiness. No onboarding form -- the model builds from observation and occasional check-ins.

**Tech Stack:** Swift, SwiftData, Sign in with Apple, Cloudflare Workers (sync API), D1

---

## Context

The practice companion needs persistent identity to build a student model over time. Without auth, every session is anonymous and cold-start. The student model enables: blind spot detection from historical patterns, focus mode recommendations, progress tracking, and the relational "teacher who knows you" experience.

## Design

### Authentication

**Sign in with Apple** (one-tap, zero friction):

1. Student taps "Sign in with Apple" on first launch (one tap)
2. Apple provides a stable user ID + relay email
3. iOS sends Apple identity token to Workers: `POST /api/auth/apple`
4. Worker validates the token with Apple's servers, creates/finds student record in D1
5. Worker issues a session JWT, stored in iOS Keychain
6. JWT sent with all API requests (sync, ask)
7. JWT expires after 30 days, refreshed on each app launch

**Why Sign in with Apple:**

- Zero friction (one tap, no email entry, no password)
- Required by App Store for apps that offer account-based features
- Provides stable cross-device identity
- Relay email captured for future communication

### SwiftData Models (Local, On-Device)

The student model lives on-device as the source of truth. See `docs/architecture.md` for the full SwiftData model definitions.

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

### D1 Schema (Cloud Sync)

Same structure as SwiftData, stored in SQL. Uses `apple_user_id` as primary key. D1 stores copies for backup and future cross-platform access.

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

CREATE TABLE student_check_ins (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(apple_user_id),
    session_id TEXT REFERENCES sessions(id),
    question TEXT NOT NULL,
    answer TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_sessions_student ON sessions(student_id, started_at);
CREATE INDEX idx_checkins_student ON student_check_ins(student_id);
```

### Baseline Update Logic

Runs on-device in Swift after each session ends (not in a backend service). The updated baselines are synced to D1 afterward.

```python
# Exponential moving average with configurable alpha
# (pseudocode -- implemented in Swift on-device)
alpha = 0.3  # weight of new session vs. history (higher = more responsive)

for dim in ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation']:
    session_avg = mean(chunk[dim] for chunk in session.chunks)
    if student.baseline[dim] is None:
        student.baseline[dim] = session_avg  # first session
    else:
        student.baseline[dim] = alpha * session_avg + (1 - alpha) * student.baseline[dim]

student.baseline_session_count += 1
```

The alpha value means:

- Recent sessions weigh more (captures improvement)
- Old patterns still influence (stable baseline)
- After ~5 sessions, the baseline is reliable

### Level Inference

Infer student level from:

- **Repertoire difficulty**: Pieces mapped to approximate grade levels (RCM, ABRSM, Henle)
- **Dimension scores**: Average across dimensions correlates with skill level
- **Session behavior**: Advanced students drill specific passages; beginners play through more

Simple heuristic for V1:

- Score < 0.3 average AND beginner repertoire -> beginner
- Score 0.3-0.6 AND intermediate repertoire -> intermediate
- Score > 0.6 AND advanced repertoire -> advanced

### Check-In Logic

Triggered at session end (not during practice -- never interrupt):

- Only after session 3+ (let the system observe first)
- Max one check-in per session
- Only when the system has a genuine observation to ground the question

**Check-in templates:**

- "I've noticed you've been working on [piece] a lot -- are you preparing for something?"
- "Your [dimension] has been improving over the last few sessions. Is that something you've been focusing on?"
- "Is there anything specific you'd like me to pay attention to in your playing?"

**Trigger rules:**

- Repertoire check-in: triggered when same piece appears in 3+ sessions
- Progress check-in: triggered when a dimension improves by >0.1 over 3 sessions
- Open-ended: triggered randomly (10% chance) after session 5+

### Explicit Input Handling

Student can tell the teacher things via the chat interface:

- "I have a recital on June 15"
- "I'm working on Chopin Ballade No. 1"
- "I want to focus on my pedaling"

These are parsed by the LLM and stored in `explicit_goals` JSON. Explicit context overrides inferred context for all downstream decisions (teaching moment selection, focus mode triggers, check-in questions).

### What This Slice Does NOT Include

- Focus mode (Slice 8)
- Exercise tracking (Slice 7)
- Teacher LLM prompt (Slice 6)
- Piece identification from audio (future work)

### Tasks

**Task 1: SwiftData model definitions**

- Define `Student`, `PracticeSession`, `ChunkResult`, `Observation` @Model classes
- Set up SwiftData container and migration strategy
- Verify persistence across app launches

**Task 2: Sign in with Apple integration**

- Add Sign in with Apple capability to the Xcode project
- Implement `ASAuthorizationController` flow
- Send Apple identity token to Workers for validation
- Store session JWT in iOS Keychain
- Auto-attach JWT to all API requests

**Task 3: Workers auth endpoint**

- `POST /api/auth/apple` - validate Apple identity token with Apple's servers
- Create or find student record in D1 by `apple_user_id`
- Issue session JWT (30-day expiry)
- JWT middleware for protected endpoints

**Task 4: Sync protocol implementation**

- iOS sends student model deltas to `POST /api/sync` after each session
- Worker upserts student and session data to D1
- Worker returns exercise updates in response
- Handle offline gracefully (queue syncs, retry on connectivity)

**Task 5: Baseline update logic in Swift**

- After session ends, compute session averages per dimension on-device
- Update student baseline with exponential moving average
- Update inferred_level
- Trigger sync to D1 in background

**Task 6: Check-in system**

- Implement trigger rules (repertoire, progress, open-ended)
- Generate check-in question at session end
- Store question + answer in student_check_ins
- Surface check-in in iOS app post-session

**Task 7: Explicit input parsing**

- When student sends a message via chat that contains goals/context
- LLM extracts structured data (date, piece, focus area)
- Store in explicit_goals JSON on student record

### Open Questions

1. Should the sync endpoint be called after every session, or batched (e.g., on app launch + every N sessions)?
2. Should JWT be a Cloudflare Workers JWT or a standard JWT library? Workers has built-in crypto.
3. How to handle the student telling the system something vs. asking "how was that?" -- same input channel, different intent. LLM-based intent classification?
