# Product Vision: A Teacher for Every Pianist

What CrescendAI is building, for whom, and what the ideal experience feels like. This is the product north star for the apps and delivery layer -- the "why" and "what," not the "how." For the technical pipeline, see `02-pipeline.md`. For UI component details, see `05-ui-system.md`.

> **Status (2026-03-19):** Vision document. CEO review (2026-03-19) established: web-first platform strategy, zero-config first session, unified artifact system, session intelligence via DO state machine, tiered monetization. Core interaction loop IMPLEMENTED on web. Web beta scope defined (~4 weeks).

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

**1. Open the app, sign in, play anything.**
No onboarding form. No piece selection. "Play anything. I'm listening." The screen shows the chat and a record button. The student taps record and plays.

**2. The system listens and identifies.**
Behind the scenes: 15-second chunks stream to the cloud. AMT transcribes MIDI. The system fingerprints against the 242-piece score library. If matched, score context activates silently. If not, the system delivers audio-quality observations without bar numbers and asks "What piece is this?" after the first observation.

**3. First observation in under 60 seconds.**
The session brain (Durable Object) detects the student is warming up and waits for enough material. After 2-3 chunks, the first observation appears as a toast during recording and as a message in chat:
> "I notice you're playing Clair de Lune. In the opening measures, your pedaling is creating a nice wash but blurring the bass line changes. Try lifting briefly at each bass note."

**4. Session adapts to practice mode.**
The session brain detects what the student is doing: warming up (sparse observations), drilling a passage (compare repetitions), running through (observe at phrase boundaries), winding down (no new observations). Observation pacing adapts automatically.

**5. Exercises as artifacts in chat.**
When the teacher observes something worth practicing, an exercise artifact appears inline in the chat. The student can expand it to full screen, play the exercise, and the system evaluates their attempt -- all without leaving the conversation.

**6. Session closes with synthesis.**
When the student stops recording (or after extended silence), the system synthesizes: "Today you worked on Clair de Lune for 22 minutes. 4 observations. Your pedaling improved. Next time, try the left hand voicing in bars 40-48." Memory persists for the next session.

**Latency target:** Under 5 seconds from "how was that?" to observation on screen. Because inference runs in the background during playing, most analysis is already done when the student asks.

---

## UX Principles

### Chat-first

The primary interface is a conversation. The student asks, the teacher responds. This is not a dashboard with charts and scores -- it is a dialogue. Rich components (score highlights, exercises, reference clips) appear as inline cards within the chat when the teacher decides the student needs more than text. See `05-ui-system.md` for component details.

### One observation at a time

The system hears many things. It picks ONE to say. This is the hardest design constraint and the most important one. A real teacher does not list every issue they noticed -- they prioritize. The teaching moment selection pipeline (see `02-pipeline.md`) exists to enforce this discipline.

### "How was that?"

The system observes during practice, paced by the session brain's practice mode detection (warming up: sparse, drilling: comparative, running through: at phrase boundaries, winding down: silent). The student can also ask "how was that?" at any time for an immediate response. Observation pacing replaces the old "student initiates only" model -- but the discipline of "one observation at a time" remains.

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
| STOP classifier | COMPLETE | Cloud worker, 6-weight logistic regression, AUC 0.845 |
| Teaching moment selection | COMPLETE | STOP + blind-spot + positive moments + dedup |
| Two-stage subagent | COMPLETE | Groq/Llama analysis + Anthropic/Sonnet teacher, AI Gateway |
| Session synthesis | COMPLETE | Alarm-triggered, all exit paths, deferred recovery |
| Zero-config piece ID | COMPLETE | N-gram + rerank + DTW (pending AMT container deploy) |
| Artifact system | COMPLETE | Unified container, Anthropic tool_use (exercise type) |
| Student model | PARTIAL | SwiftData models built, synthesized facts COMPLETE |
| Chat interface (iOS) | PARTIAL | Basic session screen |
| Chat interface (web) | COMPLETE | Chat + recording + real-time observations + synthesis |

---

## Platform Strategy

### Web: The Primary Practice Companion (Beta First)

TanStack Start app at crescend.ai. Web ships first because it's ~90% complete, fastest to iterate (no App Store review), and shareable via URL (growth). Full practice companion: audio capture, real-time observations via WebSocket, chat with artifacts, session arc. Laptop on music stand or desk beside piano.

### iOS: The Native Experience (Follows Web Beta)

Native app (SwiftUI). Phone on the music stand. Audio capture and auth are complete; inference client needs cloud wiring. iOS ships after web beta validates the product. Better audio quality (AVAudioEngine vs MediaRecorder), App Store discovery, push notifications.

### Cloud Inference for Both

All MuQ inference runs on the HF endpoint (A1-Max 4-fold ensemble). No on-device ML. This is a deliberate choice: the model is too large for on-device, and cloud inference lets both platforms share identical scoring. The cost is latency and a network dependency; the benefit is consistency, simpler updates, and no Core ML conversion headaches.

### Auth

Sign in with Apple and Google Sign In across both platforms. Both auth flows are complete on the API. Web uses Apple JS SDK and Google Identity Services.

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
| Piece identification | AMT FINGERPRINT | Auto-detect via AMT MIDI fingerprint against 242-piece score library. Graceful degradation for unknown pieces. |
| Exercise rendering | OPEN | MusicXML to notation in mobile browser. Candidates: VexFlow, OpenSheetMusicDisplay. |
| Continuous inference cost | OPEN | Background MuQ inference on every 15s chunk. Per-session cost on HF endpoints needs measurement at scale. |
| Artifact tool use pattern | OPEN | Teacher LLM declares artifacts via tool use or self-hosted MCP. Pattern needs research. |
| Session brain tuning | OPEN | Practice mode detection thresholds (DTW similarity for drilling, silence duration for winding down) need calibration with real sessions. |

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
| Excessive observation frequency | Observation pacing via session brain prevents overload. System observes at natural boundaries, not continuously. |
