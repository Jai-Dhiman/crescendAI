# Product Vision: A Teacher for Every Pianist

What CrescendAI is building, for whom, and what the ideal experience feels like. This is the product north star for the apps and delivery layer -- the "why" and "what," not the "how." For the technical pipeline, see `02-pipeline.md`. For UI component details, see `05-ui-system.md`.

> **Status (2026-03-14):** Vision document. Core interaction loop is implemented on both platforms (iOS and web). Chat interface, continuous audio capture, cloud inference, and two-stage subagent pipeline are functional. Student model, exercise database, focus mode, and on-demand UI components are designed but not yet shipped.

---

## The Core Insight

Every existing piano learning app (Simply Piano, Flowkey, Piano Marvel) answers the same question: "Did you play the right notes?" They use MIDI to check pitch and rhythm against a score. This is a solved problem and not a particularly interesting one -- most intermediate pianists already know when they hit a wrong note.

The question no app answers: **"How does it sound?"**

A piano teacher sitting in the room hears dynamics, phrasing, pedaling, articulation -- the musical dimensions that separate mechanical note reproduction from actual music-making. And critically, a good teacher does not dump a 19-dimension report card on the student. They pick *one thing* -- the most important thing the student probably cannot hear from the player's seat -- and they say it clearly, specifically, in a way the student can act on immediately.

CrescendAI is that teacher. Not a score checker. Not a report card generator. A practice companion that listens to how you play and tells you the one thing that matters most right now.

**Product thesis:** One useful observation about a blind spot is worth more than a hundred dimension scores.

---

## Target Users

### Sarah: The Self-Learner (B2C, primary)

Adult pianist, intermediate level (ABRSM 4-7). Returned to piano after a gap, or self-taught from YouTube. Practices 3-5 times per week, 20-45 minutes per session. Has no teacher -- either by choice (cost, scheduling, anxiety) or by circumstance (rural, busy schedule).

Sarah knows she is "missing something" but cannot identify what. She watches masterclass videos and thinks "that sounds so different from me" without understanding why. She does not want gamification, badges, or to be told she played 94% of notes correctly. She wants someone knowledgeable to listen and tell her what to work on.

**What Sarah needs:**
- Feedback on musical expression, not note accuracy
- Specific, actionable observations ("the crescendo peaked too early") not vague encouragement ("nice dynamics!")
- A system that learns her tendencies over time
- Minimal interface -- her focus should be on playing, not the app

### Future: Educators (B2B)

Music teachers who want to extend their reach between weekly lessons. The teacher assigns repertoire; CrescendAI monitors daily practice and surfaces the most important patterns for the teacher to review. The teacher's time is spent on interpretation and artistry, not catching the same pedaling issue for the fourth week.

### Future: Institutions (B2B)

Conservatories and university music programs that need scalable practice monitoring across dozens of students. Aggregate trends, practice engagement, dimension trajectories per student.

---

## The Ideal Practice Session

What it feels like from the student's perspective:

**1. Sit down, open the app, start practicing.**
The screen is minimal. Maybe a session timer. The app is listening, but the student's attention is on the music, not the screen. The phone sits on the music stand or beside the piano.

**2. Play for a while.**
Behind the scenes, audio streams to the cloud in 15-second chunks. Each chunk gets MuQ inference (6 dimensions) and a STOP classifier score. Results accumulate in a session buffer. The student sees none of this.

**3. Pause. "How was that?"**
The student taps a button or asks aloud. The system reviews *everything* played since the session started (or since the last query) -- not just the last 30 seconds. It identifies the top teaching moments across the full session.

**4. One observation appears.**
> "In that last run-through of the Nocturne, the left hand crescendo in bars 20-24 swallowed the right hand melody. Try bringing the LH down to pp and letting the RH sing above it."

Specific. Localized (bars 20-24). Actionable (bring LH down to pp). Grounded in what the model actually heard.

**5. Optionally, go deeper.**
The student can tap "Tell me more" for additional context -- a score highlight showing the passage, a reference recording comparison, or a targeted exercise. Or they can just go back to practicing.

**6. Session ends.**
Observations accumulate as a session log. The student can review them later. Over multiple sessions, the system builds a picture of the student's tendencies, tracks dimension trajectories, and adapts what it chooses to say.

**Latency target:** Under 5 seconds from "how was that?" to observation on screen. Because inference runs in the background during playing, most analysis is already done when the student asks.

---

## UX Principles

### Chat-first

The primary interface is a conversation. The student asks, the teacher responds. This is not a dashboard with charts and scores -- it is a dialogue. Rich components (score highlights, exercises, reference clips) appear as inline cards within the chat when the teacher decides the student needs more than text. See `05-ui-system.md` for component details.

### One observation at a time

The system hears many things. It picks ONE to say. This is the hardest design constraint and the most important one. A real teacher does not list every issue they noticed -- they prioritize. The teaching moment selection pipeline (see `02-pipeline.md`) exists to enforce this discipline.

### "How was that?"

The student initiates. The system does not interrupt practice with unsolicited feedback. The sole exception is rare, genuine check-ins ("I notice you've been repeating bars 12-16 for a while -- is there something specific you're working through?"), limited to at most once per session, and only when the system has a real observation to offer.

**Platform interaction models differ.** On iOS, the student taps "How was that?" to request feedback on-demand -- the phone sits on the music stand and the student's attention is on the music. On web, observations are pushed via WebSocket during recording, throttled to at most one observation per 3-minute window (see `02-pipeline.md`). The web model is semi-continuous: the system accumulates candidates and surfaces the top-1 per window, but the student can also ask "how was that?" at any time to get an immediate response. Both platforms share the same backend pipeline and the same "one observation at a time" discipline.

### Progressive disclosure

Default: one sentence of text. Tap for more: additional context, a score view, an exercise. The student controls the depth. Power users who want to see dimension trajectories over time can find them, but they are never the default view.

### Text-first, rich when it helps

Most observations are text. Sometimes the teacher needs to *show* something -- a passage on the score, a keyboard diagram, a dynamics curve. The teacher LLM declares a modality ("the student needs to see this passage"), and the UI renders the appropriate component. Text is always present; rich components are additive.

### Serious, adult, restrained

No gamification. No streaks. No confetti. No "Great job!" after every run-through. The design language is closer to Oura or Arc than Duolingo. Premium, minimal, respectful of the student's intelligence. The app earns trust by being specific and honest, not by being encouraging.

---

## Core Interaction Loop

The fundamental cycle that everything else builds on:

```
CAPTURE                    ANALYZE                     OBSERVE

Student plays     --->     Cloud inference      --->    One observation
(continuous audio)         (MuQ 6-dim scores)           (specific, grounded)
                           (STOP classifier)
                           (teaching moment
                            selection)

      ^                                                      |
      |                                                      |
      +--------- Student returns to practicing <-------------+
```

Three stages, one principle: the system does significant work to produce minimal output. The ratio of analysis to output is deliberately lopsided -- dozens of scored chunks, multiple candidate teaching moments, a full reasoning trace through the subagent -- all to produce one or two sentences.

### What feeds each stage

| Stage | Inputs | Output |
|-------|--------|--------|
| Capture | Microphone audio (AVAudioEngine on iOS, MediaRecorder on web) | 15s WAV chunks uploaded to API |
| Analyze | Audio chunks, student model, piece context, session history | Scored chunks + ranked teaching moments |
| Observe | Top teaching moment + student context + musical context | Natural language observation (+ optional rich component) |

### What's built vs. planned

| Component | Status | Notes |
|-----------|--------|-------|
| Audio capture (iOS) | COMPLETE | AVAudioEngine + ring buffer + chunking |
| Audio capture (web) | COMPLETE | MediaRecorder + WebSocket streaming |
| Cloud inference (MuQ) | DEPLOYED | A1-Max 4-fold ensemble, HF endpoint |
| STOP classifier | DESIGNED | Cloud worker, not yet deployed |
| Teaching moment selection | DESIGNED | Rules-based first, learned model later |
| Two-stage subagent | IMPLEMENTED | Groq/Llama analysis + Anthropic/Sonnet teacher |
| Student model | PARTIAL | SwiftData models built, synthesis not yet live |
| Chat interface (iOS) | PARTIAL | Basic session screen |
| Chat interface (web) | IN PROGRESS | Chat + recording + real-time observations |

---

## Platform Strategy

### iOS: The Primary Practice Companion

Native app (SwiftUI). Phone on the music stand. Optimized for the practice session use case -- minimal interaction during playing, quick "how was that?" between runs. SwiftData for local-first persistence; D1 sync for cross-device backup.

iOS is the primary platform because that is where the phone-on-the-piano use case lives. Audio capture via AVAudioEngine gives precise control over the recording pipeline.

### Web: The Accessible Companion

TanStack Start app at crescend.ai. Same cloud inference pipeline, same teacher, same student model (synced via D1). Lower barrier to entry -- no app install required. Better for post-session review and longer chat interactions.

The web app is a full practice companion, not a stripped-down dashboard. It captures audio via the browser (MediaRecorder API), streams to the same inference endpoint, and delivers observations in the same chat interface.

### Cloud Inference for Both

All MuQ inference runs on the HF endpoint (A1-Max 4-fold ensemble). No on-device ML. This is a deliberate choice: the model is too large for on-device, and cloud inference lets both platforms share identical scoring. The cost is latency and a network dependency; the benefit is consistency, simpler updates, and no Core ML conversion headaches.

### Auth

Sign in with Apple across both platforms. Single identity, synced student model.

---

## The Student Model

The student model is what turns CrescendAI from a stateless evaluator into a practice companion. It is built through observation and conversation -- no onboarding quiz.

**What it tracks:**

- **Dimension profile:** 6-dimension trajectory over time. Not a single score -- a trend. Used for blind spot detection (normally fine but dipped today) vs. known weakness (consistently low).
- **Repertoire history:** What pieces the student works on, how often, inferred from audio or stated by the student.
- **Learning arc per piece:** New / mid-learning / polishing. Feedback intensity adapts to phase -- encouragement early, precision later.
- **Practice habits:** Session length, repetition patterns, warm-up behavior.
- **Teaching moment history:** What was flagged, what the student engaged with vs. ignored. Prevents saying the same thing repeatedly.
- **Explicit context:** The student can tell the teacher things directly ("I have a recital in 3 weeks," "I just started this piece"). Explicit context overrides inferred context when they conflict.

**Learning curve:**

| Sessions | What the system knows | Feedback quality |
|----------|----------------------|-----------------|
| 1 | Nothing. Infers level from repertoire difficulty and dimension scores. | Useful but generic. |
| 3-5 | Dimension patterns emerge. Can distinguish "always weak" from "new problem." | Blind spot detection kicks in. |
| 10+ | Trajectory trends, repertoire breadth, practice habits, teaching moment engagement. | Personalized, adapted to learning arc. |

**Not included:** No gamification, no streaks, no user-to-user comparison, no unsolicited progress reports.

---

## Open Questions

| Question | Current Status | Notes |
|----------|---------------|-------|
| Phone audio quality | PSEUDO-VALIDATED | YouTube AMT test (79.9% agreement on mediocre recordings) serves as proxy. Formal paired recordings remain nice-to-have. |
| Teaching moment scoring | DESIGNED | Start rules-based (dimension outlier detection + STOP classifier). Learned model later when we have engagement data. |
| Piece identification | USER-ASSISTED | Student says what they are working on. Automatic identification (audio fingerprinting) is explicitly out of scope. |
| Exercise rendering | OPEN | MusicXML to notation in mobile browser. Candidates: VexFlow, OpenSheetMusicDisplay. |
| Continuous inference cost | OPEN | Background MuQ inference on every 15s chunk. Per-session cost on HF endpoints needs measurement at scale. |
| Voice input | DEFERRED | "How was that?" via voice is natural but adds speech recognition complexity. Tap-first, voice later. |

---

## What's NOT in Scope

| Item | Rationale |
|------|-----------|
| Note accuracy checking (MIDI-based) | Solved problem. Not our differentiator. |
| Teacher voice fine-tuning | Out of scope. The harness (teaching moment selection, score alignment, student context) matters more than the voice. Claude with rich context is sufficient. |
| On-device inference (Core ML) | Cloud-only is correct for foreseeable future. Model too large, and consistency across platforms matters more than offline support. |
| Gamification / social features | Streaks, badges, leaderboards. Incompatible with the serious, adult design language. |
| Multi-instrument support | Entire pipeline is piano-specific (MuQ, taxonomy, exercises). |
| Video analysis (hand position, posture) | Separate modality, separate research problem. |
| Automatic piece identification | Separate ML problem, limited ROI. User-assisted is sufficient. |
| Unsolicited real-time interruptions | The student initiates feedback. The system does not stop the student mid-phrase. |
