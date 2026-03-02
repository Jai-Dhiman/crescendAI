# Practice Companion -- Product Redesign

**Date:** 2026-03-01
**Status:** Design approved, pending implementation plan

## The Problem

The current product is transactional: upload a recording, receive a report card (19-dimension radar chart, LLM-generated paragraphs with pedagogy citations, category scores). This does not match how pianists actually want feedback. The RAG pipeline retrieves from a 7-chunk pedagogy database -- decorative, not useful.

What a pianist actually wants: "What's the one thing that sounds off that I can't hear myself?"

## The Product

A practice session companion. Phone on the piano, always listening. The student plays. When they ask "how was that?" the teacher responds with one specific, grounded observation about a blind spot -- something the student cannot self-diagnose from the player's seat.

Over time, the teacher gets to know the student: their level, their goals, their tendencies. It can suggest focused practice sessions targeting weak dimensions through exercises drawn from the student's own repertoire.

### Core Interaction

1. Student opens app, starts a practice session
2. App captures audio continuously, chunks it (15-30s segments), runs MuQ inference in the background
3. Student plays. Screen is minimal -- the focus is on playing, not the app
4. Student pauses, asks "how was that?" (voice or tap)
5. System reviews ALL analyzed chunks from the session, identifies top teaching moments
6. One specific observation appears on screen:
   > "The crescendo in the second phrase peaked too early -- the sforzando didn't land. Try holding back the build longer."
7. Student can tap "Tell me more" or return to practicing
8. Observations accumulate as a session log for post-practice review

### What Is NOT on Screen

No radar charts. No 19-dimension breakdowns. No citation footnotes. No "Sound Quality: 0.72." The interface is as minimal as a tuner -- serious, adult, restrained. Think Oura or Headspace: premium tools with craft, not toys or bare prototypes.

### Input / Output Modality

- **Input:** Text or voice (student's hands are on the piano)
- **Output:** Text and visual (score snippets, highlighted passages). Voice output as a future option.

## Architecture

### 1. Audio Pipeline

**During practice:**
- Continuous audio capture via MediaRecorder API (mobile web) or AVAudioRecorder (iOS later)
- Audio chunked into 15-30s segments
- Each chunk sent to MuQ inference endpoint in the background
- Returns 6-dimension scores per chunk (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Teaching moment model scores each chunk: "would a teacher stop here?"
- Results accumulate in a session analysis buffer

**When asked "how was that?":**
- System reviews the full session buffer (all chunks since session start or last query)
- Identifies top teaching moments across the entire session
- Does NOT grab just the last 30 seconds -- analyzes everything played

**Latency target:** Under 5 seconds from "how was that?" to observation on screen. Background inference means most analysis is already done when the student asks.

### 2. Teaching Moment Identification (Priority/Filtering Layer)

The model hears many things. It picks ONE to say.

**Teaching moment scoring:**
- Draws from masterclass "stop moment" research (masterclass_experiments/)
- Each chunk gets a "would a teacher stop here?" score
- Scoring considers: deviation from student's baseline, dimension salience for the piece, perceptual blind-spot probability

**Blind spot detection:**
- Compares dimension scores against the student's historical pattern
- A dimension that is consistently weak is KNOWN to the student (not a blind spot)
- A dimension that is normally fine but dipped today IS a blind spot
- Some dimensions are inherently harder to self-diagnose from the player's seat (voicing, balance, tone color) -- these get a blind-spot prior

**Output:** The top 1-3 teaching moments from the session, each tagged with: which chunk, which dimension, what was surprising, and why it matters.

### 3. Student Model

Built through observation and conversation. No onboarding form.

**What the model tracks:**
- **Repertoire history:** Pieces worked on, frequency, inferred from audio or student input
- **Dimension profile:** 6-dimension trajectory over time. Not a single score -- a trend.
- **Practice habits:** Session length, repetition patterns, warm-up behavior
- **Teaching moment history:** What was flagged, what the student engaged with vs. ignored
- **Explicit context:** Student can tell the teacher things directly:
  - "I have a recital in 3 weeks"
  - "I'm struggling with this passage"
  - "I want to work on my Chopin interpretation"
  - "I just started this piece"
  - Explicit context overrides inferred context when they conflict

**Learning curve:**
- **Session 1:** No history. Infers level from repertoire difficulty and dimension scores. Useful from day one but generic.
- **Sessions 3-5:** Dimension patterns emerge. Blind spot detection improves. System starts distinguishing "always weak" from "new problem."
- **Ongoing:** Occasional check-ins (max once per session, only with genuine observation): "I notice you've been spending a lot of time on the development section -- is there something specific you're working through?"

**Not included:** No gamification, no streaks, no user-to-user comparison, no unsolicited progress reports.

### 4. Teacher LLM

**What it receives (structured context):**
- Top teaching moment(s): chunk location, dimension, score, surprise factor
- Student model: level, dimension trajectory, explicit goals, session history
- Piece context (if identified): composer, style, what the score demands at that point
- Session context: duration, what was worked on, repetition patterns

**What it outputs:**
- One observation in natural language. Specific, localized, actionable.
- NOT: "Your dynamics could use work."
- YES: "In that last run-through, the crescendo in the second phrase fell flat -- it peaked too early and the sforzando didn't land. Try holding back the build longer."

**No RAG.** The teacher's pedagogical knowledge comes from the LLM's training data. The value is in the structured input and prompt design, not retrieval from a small database. RAG may be revisited later if feedback quality plateaus.

**Model choice:** General-purpose LLM (Claude, GPT-4) with a carefully crafted teacher system prompt. Intelligence lives in prompt design + structured input.

### 5. Exercise Database + Focus Mode

**The exercise database replaces RAG.** Instead of retrieving text quotes from books, the system draws from a structured, queryable database of musical exercises.

**Exercise record schema:**

| Field | Type | Purpose |
|---|---|---|
| id | UUID | Primary key |
| musical_content | TEXT (MusicXML/Lilypond) | Renderable as notation, playable as audio |
| target_dimensions | TEXT[] | Which of the 6 dimensions this trains |
| difficulty_tier | ENUM | beginner / intermediate / advanced |
| repertoire_tags | TEXT[] | Related pieces, composers, techniques |
| instructions | TEXT | What to focus on, how to practice |
| variants | JSON | Same exercise at different tempos, dynamics, keys |

**Student-exercise tracking:**

| Field | Type | Purpose |
|---|---|---|
| student_id | UUID | FK to student |
| exercise_id | UUID | FK to exercise |
| assigned_date | TIMESTAMP | When presented |
| completed | BOOLEAN | Did they engage? |
| student_response | ENUM | positive / neutral / negative / skipped |
| dimension_before | FLOAT[] | Scores before exercise |
| dimension_after | FLOAT[] | Scores after exercise |
| times_assigned | INT | Prevent repetition fatigue |

**Focus mode flow:**
1. System identifies a dimension consistently flagged across sessions (e.g., dynamics)
2. Suggests: "I've noticed dynamics keep coming up. Want to do a focused session?"
3. Student agrees
4. System queries exercise DB: dimension=dynamics, level=student's tier, not previously completed, preferably related to current repertoire
5. LLM also generates a custom exercise adapted from the student's actual passage: "Take bars 20-24, isolate the LH, practice at three dynamic levels: pp, mf, ff."
6. Exercise displayed: rendered notation + instructions + optional playback
7. Student attempts. MuQ evaluates on the focus dimension. Feedback loop.

**Exercise sources:**
- **Curated (foundation):** Seeded by Jai from standard methods (Hanon, Czerny, Burgmuller) + custom exercises. Quality controlled, correct notation.
- **LLM-generated (adaptive):** Custom exercises that reference the student's specific passage. The LLM generates instructions and context; it does not hallucinate notation. Notation for custom exercises references existing score content or uses simple patterns.

### 6. Infrastructure

**Reused from current system:**
- MuQ inference endpoint on HuggingFace (same model, called more frequently per session)
- 6-dimension composite labels (already built)
- Teaching moment concepts from masterclass experiments
- Cloudflare infrastructure: D1 (student model, exercise DB), R2 (session audio), Workers (API)

**New components:**
- Continuous audio capture + chunked inference pipeline
- Teaching moment scoring model
- Priority/filtering logic
- Student model schema + persistence layer
- Exercise database schema + seed data
- Exercise rendering (MusicXML/Lilypond to notation display)
- Focus mode API flow
- New frontend (practice companion UI)

## What This Replaces

| Current | New |
|---|---|
| File upload (async) | Continuous capture (streaming) |
| Single MuQ call per recording | Pipelined, chunked, background inference |
| 19 raw dimensions | 6 teacher-grounded dimensions |
| RAG + LLM report card with citations | Priority filter + LLM, one observation |
| Anonymous, ephemeral sessions | Persistent student model per user |
| No exercises | Curated DB + LLM-generated exercises |
| Radar chart + paragraphs | Minimal, premium practice companion UI |

## Open Questions for Implementation

1. **Phone audio validation:** MuQ was trained on Pianoteq. Does it produce meaningful results on phone recordings? Must validate before building the full companion around it.
2. **Teaching moment model:** The masterclass stop-moment research provides the concept. What's the simplest scoring function to start with? Rules-based (dimension outlier detection) vs. learned model?
3. **Piece identification:** How does the system know what piece is being played? User-assisted first (student says what they're working on), automatic later.
4. **Exercise rendering:** MusicXML to notation in a mobile web browser. What library? (VexFlow, OpenSheetMusicDisplay, OSMD?)
5. **Continuous inference cost:** Background MuQ inference on every 15-30s chunk. What's the per-session cost on HF endpoints? Is this sustainable at scale?
6. **Auth:** User accounts are now required (student model needs persistence). Simplest approach for V1?
