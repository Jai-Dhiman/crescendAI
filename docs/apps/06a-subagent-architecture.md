# Slice 6a: Two-Stage Subagent Architecture

See `docs/architecture.md` for the full system architecture.
See `docs/apps/06-teacher-llm-prompt.md` for the teacher persona prompt and output handling (still used as stage 2).

**Status:** IMPLEMENTED (core pipeline)
**Last verified:** 2026-03-05
**Date:** 2026-03-03
**Notes:** Core two-stage pipeline implemented in `apps/api/src/services/ask.rs`. Provider: Groq (Llama 3.3 70B) for subagent, Anthropic (Sonnet 4.6) for teacher. Synthesized facts and memory consolidation deferred to Slice 06c.

**Goal:** Replace the single LLM call in the "Ask" flow with a two-stage pipeline: a fast analysis subagent that reasons about what to say, followed by a quality teacher LLM that says it.

**Tech Stack:** Cloudflare Workers, OpenRouter (model-agnostic LLM API)

---

## Why Two Stages

The original Slice 06 design sends structured data (teaching moment, student model, session context) directly to one LLM, which must simultaneously analyze what matters AND generate a natural observation. This conflates two distinct tasks that benefit from different models and different prompting strategies.

The two-stage design separates analysis from delivery:

- **Stage 1 (Subagent):** Fast, cheap model (Haiku/Flash-class). Receives filtered teaching moments + student context. Reasons about which moment matters most and why. Outputs structured JSON + narrative reasoning.
- **Stage 2 (Teacher LLM):** Quality model (Sonnet/GPT-4o-class). Receives the subagent's analysis. Generates the warm, natural, 1-3 sentence observation the student sees.

This mirrors how Claude Code handles complex tasks: the main agent (Opus) delegates analysis to Explore agents (Haiku) via prepared handoff messages, then uses the results. The subagent does the legwork; the teacher provides the voice.

### Model Accuracy Context

The MuQ model is approximately 80% accurate on pairwise rankings (R2 ~0.5) with current training data. Even expert piano teachers disagree roughly 20% of the time on what constitutes better playing. The 6 dimension scores are useful signals, not ground truth.

The system's differentiator is not the raw scores. It is the subagent's reasoning about what those scores mean for this student at this moment, combined with the teacher LLM's ability to deliver that insight naturally. The scores are inputs to a reasoning pipeline, not a report card.

---

## The Two Clocks

The student model tracks two temporal dimensions, a concept drawn from context graph theory (see `docs/references/Howtobuildacontextgraph.md` and `docs/references/BuildingTheEventClock.md`):

**State clock** -- what is true right now:
- Baselines per dimension (exponential moving average)
- Inferred level (beginner/intermediate/advanced)
- Explicit goals, current repertoire

**Event clock** -- what happened, and why:
- Condensed reasoning traces per observation (dimension + key insight + confidence)
- Synthesized facts derived from accumulated traces
- Learning arc per piece

Most student model systems only build the state clock. The event clock is what enables compounding intelligence: the subagent doesn't re-derive patterns from raw session data each time. It reads synthesized facts that encode accumulated reasoning.

### Synthesized Facts

Raw condensed traces accumulate per observation:
- "Session 3: pedaling flagged (0.35), blind spot, high confidence"
- "Session 5: pedaling flagged (0.38), blind spot, medium confidence"
- "Session 8: pedaling improved (0.52), positive moment"

Periodically, the system synthesizes temporal assertions from these traces:

```
{
    "fact": "Pedaling has been a persistent area for growth but is improving",
    "dimension": "pedaling",
    "valid_at": "2026-02-20",   // session 3
    "invalid_at": null,          // still active
    "evidence": "Flagged 3 of 5 sessions. Score improved from 0.35 to 0.52 (+0.17).",
    "trend": "improving",
    "confidence": "high"
}
```

The subagent consumes synthesized facts, not raw traces. This is the map-first principle: build the context map before agents reason over it. Don't make the subagent discover patterns from scratch each call.

---

## Subagent Reasoning Framework

When "how was that?" is triggered, the subagent receives the on-device filtered moments (top 3-5 with STOP > threshold) plus the student's context map (baselines, synthesized facts, learning arc, goals). It reasons through five steps:

### 1. Where are they? (Learning Arc)

The student's familiarity with the current piece changes what feedback is appropriate:

- **New to this piece (sessions 1-3):** Prioritize encouragement. Ignore detail problems. The student expects mistakes. "You're getting the note patterns down" is more useful than "your pedaling needs work."
- **Mid-learning (sessions 4-10):** Focus on structural issues -- sections, transitions, rhythmic patterns. The student is building fluency.
- **Polishing (sessions 10+):** Focus on expression -- dynamics, phrasing, interpretation, pedaling nuance. The student can handle precision feedback.

Learning arc is tracked per piece, inferred from session count where the same piece appears. Student can also declare it ("I just started this piece" vs "I'm preparing this for a recital").

### 2. What changed? (Delta vs History)

Compare the current session's dimension scores against synthesized facts:

- **Improved:** Consider a positive observation. "Your timing in the fast section is much more even this week." Positive moments are legitimate teaching observations, not participation trophies.
- **Regressed from baseline:** Check if this is a blind spot (usually strong, dipped today) or fatigue (end of session). Blind spots are high-value observations.
- **Stable weakness:** The student likely knows about this already. Frame as a progress check, not a discovery. "I know we've talked about pedaling before -- here's something specific to try."

### 3. What matters for this music? (Musical Context)

Piece, composer, and style weight which dimensions matter most:

- **Chopin:** Pedaling and phrasing are paramount. Romantic rubato and singing tone.
- **Bach:** Articulation and timing are paramount. Clarity, voice independence.
- **Beethoven:** Dynamics and interpretation. Structural contrasts, dramatic range.
- **Debussy:** Pedaling and interpretation. Color, atmosphere, impressionistic voicing.

"Pedaling score 0.35 in a Chopin Nocturne" is a serious issue. "Pedaling score 0.35 in a Bach Invention" might be appropriate (less pedal is often correct). The subagent must contextualize scores against musical expectations.

### 4. What's the one thing? (Selection)

Re-rank the filtered moments considering all of the above: learning arc, delta, musical context, blind-spot prior. Pick the moment with the highest leverage -- what will move the needle most for this student right now.

The subagent may select differently from the on-device ranking. On-device uses STOP probability alone. The subagent adds student history, musical context, and learning arc.

### 5. What's the framing? (Correction / Recognition / Encouragement / Question)

The framing decision is part of the analysis, not left to the teacher LLM to figure out. The subagent explicitly outputs one of:

- **Correction:** "Point out the pedaling issue in bar 7 and suggest lifting on beat 3."
- **Recognition:** "Acknowledge that dynamics are noticeably better this session."
- **Encouragement:** "The notes are coming together. Don't worry about expression yet."
- **Question:** "They've been working on the development section a lot -- ask if there's something specific they're working through."

This prevents the teacher LLM from defaulting to critique mode (which it will if only given problems to discuss).

---

## Context Inputs to the Subagent

The Worker builds the subagent's context from three sources:

### From iOS (per request)
- Filtered teaching moments (top 3-5 chunks with STOP > threshold)
- Each chunk: 6 dimension scores, STOP probability, chunk index, start offset
- Current piece (if student-reported): composer, title, section label, approximate bar range
- Session metadata: duration, total chunks, chunks above threshold

### From SwiftData (via iOS request payload)
- Student baselines per dimension
- Inferred level
- Explicit goals
- Learning arc for current piece (session count on this piece)

### From D1 (Worker queries before subagent call)
- Last 5-10 synthesized facts for this student
- Recent observation history (last 3-5 observations with condensed traces)

### Section/Passage Awareness

People practice section-by-section. The Fantaisie-Impromptu learner spent weeks on individual sections before connecting them (see `docs/references/amateur-learns-fantaisie-impromptu.md`). Teaching moments should reference musical structure when available:

- "The transition between sections A and B" rather than "chunk 7"
- "In the development section" rather than "at 1:45"
- "Bars 20-24" rather than "at 5:00"

Initially student-reported: the student tells the app what piece and section they're working on. The iOS app maps this to approximate bar ranges for the chunks in that time window.

### Score Alignment

Connect MuQ chunk timestamps to bar/measure numbers in the score. This lets the teacher say "at bar 7, the dynamics have no range" instead of "at 0:04, the dynamics have no range." The former sounds like a teacher; the latter sounds like a machine.

**V1 (student-reported):** Student selects piece and starting bar. System maps chunk timestamps to approximate bar numbers using the piece's tempo.

**V2 (automated):** Score-following via audio fingerprinting or onset detection aligned to a reference score. Future work.

---

## Positive Teaching Moments

The STOP classifier and blind-spot detection are designed to find problems. But a real teacher notices both what's wrong and what's improved. The Fantaisie-Impromptu learner's most motivating moments were breakthroughs -- "that moment where something actually lands is just incredibly satisfying."

The subagent can flag positive moments when:
- A dimension score is significantly above baseline for the first time
- A previously flagged weakness shows measurable improvement
- A passage that was problematic last session is now clean
- Overall session quality is notably higher than recent history

Positive observations are real teaching, not participation trophies:
- "Your pedaling in the second phrase has gotten much smoother -- you're catching the harmonic changes now."
- "The polyrhythm in the opening is landing consistently. That's a big step."

The subagent's framing decision (step 5 of the reasoning framework) determines when to use positive framing vs. correction.

---

## Condensed Reasoning Traces

Each observation persists a condensed trace alongside the observation text:

```
{
    "dimension": "pedaling",
    "insight": "Blind spot -- usually a strength but dropped significantly this session",
    "confidence": "high",
    "framing": "correction",
    "scores": { "pedaling": 0.35, "baseline": 0.62 },
    "piece": "Chopin Nocturne Op. 9 No. 2",
    "learning_arc": "polishing"
}
```

This is stored in the `Observation` model (or a new `reasoning_trace` field on it). These traces are the raw material from which synthesized facts are periodically generated.

**What to persist:** dimension, key insight (one sentence), confidence level, framing decision, relevant scores, piece context, learning arc position.

**What NOT to persist:** The full subagent narrative, raw chunk data, the teacher LLM's output (that's already stored as the observation text).

---

## Handoff Format

The subagent outputs both structured JSON and a narrative reasoning summary. The teacher LLM receives both.

### Structured output (JSON)

```json
{
    "selected_moment": {
        "chunk_index": 7,
        "dimension": "pedaling",
        "dimension_score": 0.35,
        "student_baseline": 0.62,
        "bar_range": "bars 20-24",
        "section_label": "second phrase"
    },
    "framing": "correction",
    "learning_arc": "polishing",
    "is_positive": false,
    "musical_context": "Chopin Nocturne -- pedaling is critical for the romantic legato sound"
}
```

### Narrative reasoning (for the teacher LLM)

```
The most notable moment was in the second phrase (bars 20-24) where pedaling dropped
to 0.35, well below this student's usual 0.62. This is a blind spot -- their pedaling
is normally a strength. Given they're in the polishing phase with this Nocturne,
and pedaling is central to Chopin's sound, this is high-leverage feedback.
Recommend framing as a discovery: point out what happened and suggest a specific
pedaling technique for the harmonic changes in those bars.
```

The teacher LLM (stage 2) uses `docs/apps/06-teacher-llm-prompt.md`'s system prompt for persona and tone. Its user prompt includes the handoff above instead of the raw session data.

---

## "Tell Me More" Flow

When the student taps "Tell me more" after the initial observation:

1. Skip the subagent entirely -- the moment is already selected and analyzed.
2. Send the original handoff (subagent analysis + initial observation) to the teacher LLM with an elaboration instruction.
3. The teacher LLM elaborates: why this matters for the piece, a specific practice technique, what "fixed" would sound like.

This matches the Claude Code pattern: follow-up questions use existing context, not a new Explore agent call. The teacher LLM's system prompt prefix stays cached across turns (per prompt caching principles below).

---

## Prompt Caching Principles

From `docs/references/LessonsfromBuildingClaudeCodePromptCachingIsEverything.md`:

### Static content first, dynamic content last

The teacher LLM's system prompt (persona, tone rules, output constraints) is identical across all students and sessions. This gets cached at the API level. Dynamic content (subagent analysis, student context) goes in the user message, last.

```
[CACHED] System prompt: teacher persona (same for every request)
[CACHED] Tools/output format constraints
[DYNAMIC] User message: subagent handoff + student context
```

### Don't switch models mid-conversation

For "Tell me more" follow-ups, keep the same teacher model. The system prompt prefix stays cached. Switching to a different model would invalidate the cache and cost more than keeping the quality model.

### Subagents for model switching

The two-stage design naturally follows this principle. The subagent (Haiku/Flash) and teacher (Sonnet/GPT-4o) are separate API calls with separate caches. No mid-conversation model switching.

### Model tiering

| Stage | Model class | Why |
|---|---|---|
| Subagent (analysis) | Haiku / Gemini Flash | Fast (~0.5s), cheap. Analysis is structured reasoning, not creative writing. |
| Teacher (observation) | Sonnet / GPT-4o | Quality voice. Natural tone, nuanced persona following. |

Total latency budget: ~0.5s (subagent) + ~1.5s (teacher) = ~2s, within the <3s target.

---

## Map-First Principle

From `docs/references/BuildingTheEventClock.md`: "You can't wait for thousands of agent runs to discover ontology. Build the map first, then agents walk it effectively."

The subagent needs a pre-built context map before it starts reasoning:

1. **Resolved entities:** The student, their pieces, their dimensions, their goals. Not discovered from scratch.
2. **Temporal state:** Synthesized facts with validity periods. "Pedaling has been weak for 2 weeks" is pre-computed, not re-derived.
3. **Relationships:** Which pieces the student is working on, how long they've been on each, what dimensions were flagged for each.

The Worker builds this map from D1 before calling the subagent. The subagent walks the map to make its reasoning decision. It doesn't fight the identity-resolution problem anew each call.

---

## Updated "Ask" Flow

```
iOS                              Worker (/api/ask)                   OpenRouter
 |                                    |                                   |
 |  POST /api/ask                     |                                   |
 |  { filtered_moments[3-5],          |                                   |
 |    student_context,                |                                   |
 |    piece_context }                 |                                   |
 |  --------------------------------> |                                   |
 |                                    |                                   |
 |                                    |  1. Query D1: synthesized facts   |
 |                                    |     + recent observation history   |
 |                                    |                                   |
 |                                    |  2. Build subagent prompt          |
 |                                    |     (system: analyst persona)      |
 |                                    |     (user: moments + context map)  |
 |                                    |  --- Haiku/Flash call -----------> |
 |                                    |  <-- analysis (JSON + narrative) - |
 |                                    |                                   |
 |                                    |  3. Build teacher prompt           |
 |                                    |     (system: teacher persona)      |
 |                                    |     (user: subagent handoff)       |
 |                                    |  --- Sonnet/GPT-4o call ---------> |
 |                                    |  <-- observation (1-3 sentences) - |
 |                                    |                                   |
 |                                    |  4. Store condensed trace in D1    |
 |                                    |  5. Return observation to iOS      |
 |  <-------------------------------- |                                   |
 |                                    |                                   |
 |  "Tell me more"                    |                                   |
 |  --------------------------------> |                                   |
 |                                    |  Skip subagent. Send original     |
 |                                    |  handoff + observation to teacher  |
 |                                    |  with elaboration instruction.     |
 |                                    |  --- Sonnet/GPT-4o call ---------> |
 |                                    |  <-- elaboration (2-4 sentences) - |
 |  <-------------------------------- |                                   |
```

---

## On-Demand UI Extension

The two-stage pipeline extends to three stages when the teacher decides the student needs more than text. See `docs/apps/10-on-demand-ui.md` for the full design:

- **Stage 2 (Teacher LLM) extended:** Also declares a modality -- `text_only`, `score_highlight`, `keyboard_guide`, `exercise_set`, or `reference_browser`.
- **Stage 3 (UI Subagent, Haiku/Flash):** Configures the specific component from a pre-built SwiftUI library. Only invoked when the teacher requests a non-text modality.
- **Components render as inline cards** in the chat interface, like iMessage rich content.

---

## Open Questions

1. **Synthesis cadence:** When does the system synthesize facts from raw traces? After every N sessions? On a background timer? On-demand when the subagent needs them? Trade-off: more frequent = fresher context, but more D1 writes and LLM calls for synthesis.

2. **Contradictory facts:** What happens when a synthesized fact ("pedaling is a persistent weakness") is contradicted by new evidence ("pedaling has been fine for 3 sessions")? Invalidate the old fact (set `invalid_at`) and synthesize a new one? Or let the subagent reason over both?

3. **Positive/corrective ratio:** How often should the subagent choose a positive observation over a corrective one? Always when improvement is significant? Or biased toward correction since that's what the student is paying for? A real teacher probably does 70% correction, 30% positive -- but it varies by student personality and learning phase.

4. **Low-confidence scores:** When the MuQ model is uncertain (middle-range scores, no clear outlier), should the subagent flag a teaching moment at all? Or say "sounded good, keep going"? The system should not fabricate problems to fill silence.

5. **Subagent prompt iteration:** The subagent's reasoning framework is described here but needs to be tested with synthetic teaching moment data. The prompt will require iteration to balance thoroughness with speed (Haiku needs concise prompts).

6. **Score alignment accuracy:** With student-reported piece and bar range, how accurate can timestamp-to-bar mapping be? Tempo changes, rubato, and pauses introduce error. May need to show "approximately bars 20-24" rather than precise bar numbers.
