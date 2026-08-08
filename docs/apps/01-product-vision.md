# Product Vision: A Teacher for Every Pianist

What CrescendAI is building, for whom, and what the ideal experience feels like. This is the product north star for the apps and delivery layer -- the "why" and "what," not the "how." For the technical pipeline, see `02-pipeline.md`. For UI component details, see `05-ui-system.md`.

> **Status (2026-08-04):** Vision document, revised for the score-first redesign (epic #154). The 2026-03-19 chat-first interaction model is superseded: feedback now attaches to the score as marks, delivery is silent-while-playing and boundary-gated, and chat survives only as passage-scoped ask threads. Web-first platform strategy, DO session intelligence, and tiered monetization stand. Build state lives in #154's sub-issues.

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

**1. Open the app, tap your piece, the score appears.**
Home shows your repertoire. Tap Clair de Lune and the sheet music fills the screen — the app *is* the music stand. Tap record and play. (A "just play" path exists too: record without picking a piece; if the system recognizes it, a quiet confirm chip names its guess.)

**2. While you play: silence.**
The app says nothing during playing. No toasts, no messages, no moving cursor. Audio streams to the cloud in the background — Transkun transcribes, alignment maps chunks to bars, analysis runs — so feedback is ready the moment you pause, but nothing is delivered mid-phrase.

**3. When you pause, a pencil mark.**
After ~20 seconds of silence, at most one mark lands on the score at the bars in question — like a teacher penciling your music while you catch your breath. Glance down, see it, play on. It never demands a response. Marks fire only when something deviates from *your own* baseline, persistently — not on every imperfection.

**4. Stop, and the score becomes the review.**
Stop recording (or let 60 seconds of silence end the session softly — one tap resumes) and the score you were just reading transitions into the session review: one teacher sentence as the verdict, the session's marks consolidated, and each mark expandable to hear your own playing of exactly those bars, cursor tracking the notation.

**5. Work on it, or ask about it.**
From any mark: "work on this" generates a bar-bounded drill of the flagged passage, with variants shaped by your habits. "Ask about this" opens a conversation scoped to that passage — the one place dialogue lives.

**6. One thing carries forward.**
The review ends with a single next-session focus ("start with bars 40-44"). Next session, the verdict closes the loop: "That bass-blur from Tuesday is gone." Over weeks, the piece page shows your score with marks fading as passages improve — the semester's pencil marks, aging honestly.

**Latency posture:** capture streams continuously so analysis is done by the time you pause or stop; marks land within seconds of a pause, the review within seconds of stopping. Nothing is ever delivered while you play.

---

## UX Principles

### Score-first

Feedback attaches to the music, not to a message stream. The score (or the session timeline, when no score is available) is the canvas; feedback is marks on it — the way a real teacher pencils your music. Prose exists at exactly three points: the session verdict, the carry-forward, and on-demand passage explanations. Conversation survives only as passage-scoped ask threads. See `05-ui-system.md`.

This replaces the original chat-first principle, on evidence (recorded in #154): concurrent verbal feedback during play splits attention and undermines the self-error-detection that distinguishes strong pianists; direct manipulation beats conversation for iterate-and-compare loops; and the best-loved analog for post-practice analysis (chess.com's Game Review) is artifact-anchored, not conversational.

### Silent while playing, present at boundaries

The app never speaks during playing. Marks appear only during pauses, at most one per pause, gated on deviation from the student's own baseline (bandwidth feedback — the best-validated mechanism in the motor-learning literature). A real teacher corrects infrequently and selectively; so does this one.

### One mark at a time

The system hears many things. It surfaces ONE per pause, and one verdict plus one carry-forward per session. This remains the hardest and most important design constraint.

### Progressive disclosure

Default: a mark. Tap for more: the evidence in words, your own audio of those bars, a drill, a conversation. The student controls the depth. Dimension trajectories exist as direction-over-time statements in drill-down — never numbers, never the default view.

### The system may fall silent, never guess

Wrong bar numbers are never shown (anchors degrade to timestamps, which are always true); low-confidence transcription spans are no-comment zones; a failed review says so loudly. Trust is the product's core asset, and a wrong claim costs more than a missing one.

### Serious, adult, restrained

No gamification. No streaks. No confetti. No "Great job!" after every run-through. The design language is closer to Oura or Arc than Duolingo. Premium, minimal, respectful of the student's intelligence. The app earns trust by being specific and honest, not by being encouraging.

---

## Core Interaction Loop

The fundamental cycle that everything else builds on:

```
CAPTURE                    ANALYZE                      MARK

Student plays     --->     Transkun AMT -> MoonBeam ---> At most one mark
(continuous audio,         + MPM evidence features       per pause, on the
 app silent)               (baseline bandwidth gate)     score or timeline

      ^                                                      |
      |                                                      v
      +--- Student plays on <--- glances at the mark --------+

STOP  --->  the score becomes the review: one verdict, the session's
            marks, own-audio playback per mark, ONE carry-forward
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
| Audio capture (iOS) | COMPLETE, carried forward | AVAudioEngine + ring buffer + chunking |
| Audio capture (web) | COMPLETE, carried forward | MediaRecorder + WebSocket streaming (capture transport only) |
| Transkun AMT + alignment | CARRIED FORWARD | #125 adoption; offline chroma-DTW; open-set piece-ID gate (#26) |
| MoonBeam scoring + MPM extraction | NOT STARTED | #162 — replaces the retired MuQ HF-endpoint path |
| Mark system + canvases | COMPLETE | #157 — the central contract |
| Practice mode (music stand) | COMPLETE, wiring gaps remain | #158; piece-ladder `user-picked`/`confirm-chip` rungs unreachable from a live Record session until #160 wires `AppChat`/`usePracticeSession`; marks are WS-injected, not pipeline-fed, until #162 |
| Session review | NOT STARTED | #159 — promotes shipped play_passage machinery |
| Student baseline (dual EWMA) | NOT STARTED | #163 — supersedes the alpha=0.3 EMA baseline |
| Session synthesis (V6 harness) | COMPLETE, output contract changing | verdict + marks + carry-forward (#162) |
| Exercise / segment-loop machinery | COMPLETE, being promoted | from chat cards to first-class drills (#159) |
| Chat interface | BEING REMOVED | #161/#164 — passage-scoped asks replace it |

---

## Platform Strategy

Platform strategy: see docs/architecture.md (CEO review 2026-03-19).

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
| Teaching moment scoring | IMPLEMENTED (rules-based) | Worst-dimension `deviation < 0` gate + blind-spot ranking + positive-moment fallback. A learned model may be revisited when we have engagement data. |
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
| Teacher voice fine-tuning | Out of scope. The harness (teaching moment selection, score alignment, student context) matters more than the voice. The teacher LLM with rich context is sufficient. |
| On-device inference (Core ML) | Cloud-only is correct for foreseeable future. Model too large, and consistency across platforms matters more than offline support. |
| Gamification / social features | Streaks, badges, leaderboards. Incompatible with the serious, adult design language. |
| Multi-instrument support | Entire pipeline is piano-specific (MuQ, taxonomy, exercises). |
| Video analysis (hand position, posture) | Separate modality, separate research problem. |
| Excessive observation frequency | Observation pacing via session brain prevents overload. System observes at natural boundaries, not continuously. |
