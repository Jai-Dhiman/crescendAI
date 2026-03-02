# Slice 5: Student Model + Auth

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add user authentication and a persistent student model that tracks dimension profiles, repertoire, goals, and session history across practice sessions.

**Architecture:** Email magic link auth via Cloudflare Workers + D1. Student model stored in D1, updated after each session. No onboarding form -- the model builds from observation and occasional check-ins.

**Tech Stack:** Cloudflare Workers (auth endpoints), D1 (student model storage), iOS Keychain (token storage)

---

## Context

The practice companion needs persistent identity to build a student model over time. Without auth, every session is anonymous and cold-start. The student model enables: blind spot detection from historical patterns, focus mode recommendations, progress tracking, and the relational "teacher who knows you" experience.

## Design

### Authentication

**Email magic link** (simplest, no password management):
1. Student enters email
2. Backend generates a one-time token, stores it in KV with 15-min TTL
3. Sends email with login link (via Cloudflare Email Workers or a third-party like Resend)
4. Student clicks link, backend validates token, issues a session JWT
5. JWT stored in iOS Keychain, sent with all API requests
6. JWT expires after 30 days, refresh on each session start

**Why magic link:**
- No password to manage
- Works on mobile (student checks email, taps link, returns to app)
- Simple to implement
- Appropriate for a practice tool (not handling sensitive data)

### D1 Schema

```sql
CREATE TABLE students (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    last_session_at TEXT,

    -- Inferred profile (updated after each session)
    inferred_level TEXT,              -- beginner / intermediate / advanced
    primary_repertoire_tags TEXT,     -- JSON array: ["Chopin", "Bach", "romantic"]

    -- Dimension baselines (rolling averages)
    baseline_dynamics REAL,
    baseline_timing REAL,
    baseline_pedaling REAL,
    baseline_articulation REAL,
    baseline_phrasing REAL,
    baseline_interpretation REAL,
    baseline_session_count INTEGER DEFAULT 0,  -- how many sessions feed the baseline

    -- Explicit context (student-provided)
    explicit_goals TEXT,              -- JSON: {"recital_date": "2026-06-15", "focus": "Chopin interpretation"}
    explicit_level TEXT,              -- overrides inferred_level if set
    explicit_notes TEXT               -- free-form notes from check-in answers
);

CREATE TABLE student_sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_min REAL,
    total_chunks INTEGER,
    -- Session-level dimension averages
    avg_dynamics REAL,
    avg_timing REAL,
    avg_pedaling REAL,
    avg_articulation REAL,
    avg_phrasing REAL,
    avg_interpretation REAL,
    -- Teaching moments surfaced
    moments_surfaced INTEGER DEFAULT 0,
    -- Pieces worked on (inferred or stated)
    pieces_json TEXT,                 -- JSON array
    notes TEXT                        -- session-level notes
);

CREATE TABLE student_check_ins (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(id),
    session_id TEXT REFERENCES student_sessions(id),
    question TEXT NOT NULL,           -- what the system asked
    answer TEXT,                      -- what the student said (null if skipped)
    created_at TEXT NOT NULL
);

CREATE INDEX idx_sessions_student ON student_sessions(student_id, started_at);
CREATE INDEX idx_checkins_student ON student_check_ins(student_id);
```

### Baseline Update Logic

After each session ends:

```python
# Exponential moving average with configurable alpha
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

**Task 1: D1 migration for student tables**
- Create students, student_sessions, student_check_ins tables
- Run migration

**Task 2: Magic link auth endpoints**
- `POST /api/auth/request` - send magic link email
- `GET /api/auth/verify?token=...` - validate token, issue JWT
- JWT middleware for protected endpoints
- Email sending via Resend or Cloudflare Email Workers

**Task 3: iOS auth flow**
- Login screen with email input
- Deep link handler for magic link callback
- JWT storage in Keychain
- Auto-attach JWT to all API requests

**Task 4: Baseline update logic**
- After session ends, compute session averages per dimension
- Update student baseline with exponential moving average
- Update inferred_level

**Task 5: Check-in system**
- Implement trigger rules (repertoire, progress, open-ended)
- Generate check-in question at session end
- Store question + answer in student_check_ins
- Surface check-in in iOS app post-session

**Task 6: Explicit input parsing**
- When student sends a message via chat that contains goals/context
- LLM extracts structured data (date, piece, focus area)
- Store in explicit_goals JSON on student record

### Open Questions

1. Email service: Resend ($0/month for <100 emails) vs. Cloudflare Email Workers (free but more setup)?
2. Should JWT be a Cloudflare Workers JWT or a standard JWT library? Workers has built-in crypto.
3. How to handle the student telling the system something vs. asking "how was that?" -- same input channel, different intent. LLM-based intent classification?
