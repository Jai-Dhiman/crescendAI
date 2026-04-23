# Audio-to-Observation Pipeline

The complete path from microphone to teaching observation. This is the technical heart of the system -- how audio becomes actionable feedback.

> **Status (2026-03-28):**
> - IMPLEMENTED: Two-stage LLM pipeline (subagent + teacher), HF inference endpoint (A1-Max 4-fold ensemble + AMT + pedal CC64), STOP classifier, teaching moment selection, blind-spot detection, score following (DTW), bar-aligned analysis, synthesized facts, exercise endpoints (25 curated), session brain state machine (DO practice mode detection + state persistence), observation pacing (mode-aware), zero-config piece ID (N-gram + rerank + DTW, merged, pending AMT container deploy), artifact declaration via Anthropic tool_use (tool_choice: auto), session synthesis (alarm-triggered, all exit paths, deferred recovery), AI Gateway (Anthropic + Groq + Workers AI)
> - NOT STARTED: Passage repetition detection
> - **Code:** `apps/api/src/services/ask.rs` (pipeline), `apps/api/src/services/prompts.rs` (teacher persona), `apps/api/src/practice/session.rs` (DO session)
> - **Model details:** `model/03-encoders.md`
> - **Student context:** `03-memory-system.md`
> - **UI delivery:** `05-ui-system.md` (unified artifact system)

---

## Pipeline Overview

```
                          iOS                              Web
                    +--------------+               +----------------+
                    | AVAudioEngine|               | MediaRecorder  |
                    | 24kHz mono   |               | getUserMedia   |
                    | ring buffer  |               | Opus/WebM      |
                    +------+-------+               +-------+--------+
                           |                               |
                     15s chunks                       15s chunks
                           |                               |
                           v                               v
              POST /api/practice/chunk        POST /api/practice/chunk
                           |                               |
                           +--------->   <-----------------+
                                    |
                                    v
                   +--------------------------------+
                   |  Stage 1: Cloud Inference      |
                   |  HF Endpoint (A1-Max ensemble) |
                   |  15s audio -> 6-dim scores     |
                   +---------------+----------------+
                                   |
                                   v
                   +--------------------------------+
                   |  Stage 2: STOP Classification  |
                   |  Logistic regression on scores |
                   |  "Would a teacher stop here?"  |
                   +---------------+----------------+
                                   |
                                   v
                   +--------------------------------+
                   |  Stage 3: Teaching Moment      |
                   |  Selection                     |
                   |  Top chunk + blind-spot dim    |
                   +---------------+----------------+
                                   |
                                   v
                   +--------------------------------+
                   |  Stage 4a: Subagent            |
                   |  Groq / Llama 3.3 70B (~0.3s)  |
                   |  5-step reasoning -> JSON      |
                   +---------------+----------------+
                                   |
                                   v
                   +--------------------------------+
                   |  Stage 4b: Teacher LLM         |
                   |  Anthropic / Sonnet 4.6 (~1.5s)|
                   |  Warm, natural 1-3 sentences   |
                   +---------------+----------------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
              iOS: on-demand               Web: real-time
              "How was that?"              WebSocket push
              POST /api/ask                observation toasts
```

**Interaction models differ by platform.** On iOS, the student plays and then taps "How was that?" to request an observation on-demand. On web, the pipeline runs continuously during recording: each chunk is scored as it arrives, and observations are pushed back via WebSocket as real-time toasts in the chat interface. Both paths share the same backend stages.

The stages below describe the **current implementation**. The target end-state is an agent loop rather than a linear pipeline -- see "Target: Agent Loop" below.

---

## Target: Agent Loop

The linear pipeline above accurately describes what is shipping. The harness target -- see `docs/harness.md` -- is an agent loop that reuses the same signal layer (Stages 1-3) but replaces Stages 4a/4b with a loop that loads skills on demand, calls tools, and produces artifacts.

### Shape

```
ENTRY CONDITION (hook)
  OnStop | OnPieceDetected | OnBarRegression | OnSessionEnd | OnWeeklyReview
        |
        v
  LOAD RELEVANT SKILLS (deferred, signal-triggered)
  Only skills whose YAML triggers match current signals enter context.
  Reference: wiki "Agent Harnesses" -- thin harness + ToolSearch primitive.
        |
        v
  AGENT LOOP
  |- read tools: get_bar_analysis, fetch_similar_past_issue, ...
  |- action tools: assign_segment_loop, render_annotation, schedule_followup_interrupt, ...
  |- produce artifacts (NLAH durable outputs, addressable by later skills)
  |- loop terminates when post-conditions on the triggering hook are met
        |
        v
  ARTIFACTS PERSISTED
  Exercises, annotations, observations, synthesized facts -- all carry
  evidence chains back to Signals (see docs/apps/03-memory-system.md Layer 3).
```

### How This Differs From the Current Pipeline

- **Skills replace the monolithic teacher prompt.** 8-12 markdown files in `docs/harness/skills/` each represent one atomic pedagogical move (voicing diagnosis, pedal triage, etc.). See `docs/harness/skills/README.md`. Wiki reference: *Skill Design* (atomic RL + concrete testable steps).
- **Chat and session-exit use the same loop.** The current chat path has no tool_use; the observation path has tool_use for exercises only. In the target, both enter the same loop with different hook conditions.
- **Action tools are first-class.** The loop can call `assign_segment_loop(bars=12-16, required_correct=3)` to restructure the student's next practice block, not just describe what they did. This is the answer to the Score Following wiki's empirical finding that 90% of home practice is start-to-finish playthrough -- a passive report is insufficient.
- **Contracts are inspectable.** Each skill declares pre/post-conditions in its markdown. The harness detects silent degradation when a post-condition fails, rather than shipping unnoticed bad output.

### What This Does Not Change

- Stages 1-3 (audio capture, cloud inference, STOP classification) are unchanged. The agent loop consumes the same signals.
- The DO-held session accumulator is unchanged (V3 adds sawtooth compaction later).
- The AI Gateway routing and provider mix (Groq subagent + Sonnet teacher) is unchanged; the loop is provider-agnostic, so swapping Sonnet -> Qwen finetune is a runtime decision.

### Writes Stay Single-Threaded

From the Mahler wiki's *Multi-Agents: What's Actually Working*: multi-agent systems work when additional agents contribute intelligence, not actions. The current Groq (analysis) + Sonnet (delivery) split already respects this: Groq does reasoning, Sonnet writes. As V5 skills come online, the constraint tightens: a compound may dispatch many molecules for analysis in parallel, but the compound writes **one** teacher-facing artifact. Skills do not parallel-speak to the student.

### Event Hooks vs Middleware Hooks

Hooks split into two kinds, a distinction from *The runtime behind production deep agents*:

- **Event hooks** fire on external practice signals (`OnStop`, `OnPieceDetected`, `OnBarRegression`, `OnSessionEnd`, `OnWeeklyReview`). Each event hook maps to one compound in `docs/harness/skills/compounds/`.
- **Middleware hooks** wrap every model invocation: `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`. These are runtime primitives, not skill logic. They handle:
  - PII redaction (`before_model`)
  - Tool-call limits and permission gating for action tools (`wrap_tool_call`)
  - Model retries (`wrap_model_call`)
  - Human-in-the-loop gates (`wrap_tool_call`)
  - Online eval / review-agent scoring (`after_model`) -- see V4 production review agent

The production review agent is a specific `after_model` middleware: given only the synthesis output + student baselines + rubric (no raw signals, no session accumulator), it rates coherence and flags drift. From *Multi-Agents: What's Actually Working*: the review agent performs better **without** shared context -- forced to reason from the implementation backward, shorter context yields higher attention quality (Context Rot).

### Capability-Router Across Providers

From *Multi-Agents: What's Actually Working*: the Groq + Sonnet (+ eventually Qwen) mix is a capability router, not a difficulty escalator. Each model handles the sub-task it is best at. This frames the future Sonnet -> Qwen swap: Qwen handles teacher-voice molecules where the finetune pays off; Sonnet or Groq handle analytical atoms where Qwen underperforms. The article also acknowledges that a meaningfully weaker primary calling out to a stronger model remains an open training problem; relevant to the Qwen 27B gating criteria in `docs/plans/` (strong model for core reasoning; router for sub-tasks).

### Sequencing

See `docs/harness.md` priority stack. Skills decomposition (V5) lands before the loop architecture (V6), because the loop's deferred-loading primitive has nothing to load without skill files. The three-tier skill structure (atoms / molecules / compounds) also defines what event hooks dispatch to: one compound per event hook.

---

## Stage 1: Audio Capture and Chunking

### iOS (AVAudioEngine)

AVAudioEngine captures audio at 24kHz mono (MuQ's native sample rate). A circular ring buffer holds the last 5 minutes of PCM samples (~29MB). A background timer fires every 15 seconds, extracts the latest chunk, and uploads it to `POST /api/practice/chunk`.

Background audio mode (`UIBackgroundModes: audio`) keeps recording when the screen is off.

**Status:** IMPLEMENTED (audio capture, ring buffer, chunking). Cloud inference client is stub code ready for API integration.

### Web (MediaRecorder)

The browser captures audio via `getUserMedia` and `MediaRecorder`, producing 15-second Opus/WebM chunks. Each chunk is uploaded to `POST /api/practice/chunk`. An `AudioContext` `AnalyserNode` drives the waveform visualizer during recording.

The web path uses Cloudflare Durable Objects to manage practice session state. `POST /api/practice/start` creates a Durable Object session; `WS /api/practice/ws/:sessionId` opens a WebSocket connection for real-time observation delivery. Audio chunks are stored in R2 for potential session playback.

**Status:** IN PROGRESS (recording, chunk upload, WebSocket connection, chat interface).

#### Durable Object Session Lifecycle

Each web practice session is backed by a Cloudflare Durable Object that manages session state. The DO holds scored chunks, teaching moment candidates, and WebSocket connections.

**Creation:** `POST /api/practice/start` creates a DO with a deterministic ID: `session:{student_id}:{timestamp_ms}`. Initial state:

```json
{
    "session_id": "...",
    "student_id": "...",
    "mode": "regular",
    "target_dimension": null,
    "started_at": "...",
    "chunks": [],
    "candidates": [],
    "observations_delivered": 0,
    "last_observation_at": null
}
```

**Chunk accumulation:** When `POST /api/practice/chunk` arrives, the worker forwards audio to HF inference, runs STOP classification on the returned scores, and sends the scored chunk to the DO via internal fetch. The DO appends to its `chunks` array and updates `candidates` if STOP > threshold.

**WebSocket:** `WS /api/practice/ws/:sessionId` opens a WebSocket connection to the DO. The DO pushes observations to connected clients when the throttle window allows. Multiple WebSocket connections to the same DO are supported (e.g., reconnection before old connection times out).

**Reconnection:** If the WebSocket drops, the client reconnects to the same session ID. The DO preserves all state. On reconnect, the DO sends any observations that were generated while disconnected.

**Cleanup:** DOs are cleaned up via an alarm. When a session receives no new chunks for 30 minutes, the DO writes session summary data to D1 and deletes its in-memory state. Cloudflare garbage-collects DOs with no stored state.

**Cost:** DOs are billed per request + per 128MB-second of wall-clock duration. A 30-minute session with 120 chunk events + WebSocket keepalives costs approximately $0.001-0.005 in DO charges.

### Platform Comparison

| Capability | iOS | Web |
|---|---|---|
| Audio format | PCM 24kHz mono | Opus/WebM |
| Chunk delivery | HTTPS upload | HTTPS upload |
| Session state | SwiftData (local-first) | Durable Object (cloud) |
| Observation trigger | On-demand ("How was that?") | Real-time (WebSocket push) |
| Offline recording | Yes (scores arrive on reconnect) | No |

### Silence Detection

Both platforms apply a silence gate before uploading chunks. If a chunk's RMS energy falls below a threshold (e.g., -40dB), the chunk is skipped -- no upload, no inference cost.

- **iOS:** Computed from the PCM ring buffer. Trivial RMS check on raw samples.
- **Web:** The `AnalyserNode` (already running for the waveform visualizer) provides frequency data. RMS computed from time-domain data.

Silence detection reduces inference cost by ~20-30% in a typical session (page turns, pauses, adjustments). It also prevents nonsensical MuQ scores on silent chunks from entering the STOP classifier.

---

## Stage 2: Cloud Inference

Each 15-second audio chunk is forwarded by the API worker to the HuggingFace inference endpoint.

### HF Endpoint (A1-Max 4-Fold Ensemble)

```
Input:  15s audio (Opus/WebM or PCM) at 24kHz mono
          |
          v
MuQ backbone (frozen, pretrained on 160K hours)
  + LoRA rank-32 adapters (layers 7-12, <1% of params)
  + Attention pooling -> Shared encoder -> Per-dim ranking heads
          |
          v
Output: 6 dimension scores (Float, 0.0-1.0)
        [dynamics, timing, pedaling, articulation, phrasing, interpretation]
```

The deployed model averages predictions across all 4 fold models (piece-stratified cross-validation). Calibrated against MAESTRO reference performances (`model/data/maestro_cache/calibration_stats.json`).

### Model Performance

| Metric | Value |
|--------|-------|
| Pairwise accuracy | 80.8% (4-fold ensemble) |
| R2 | 0.50 |
| Robustness (score drop) | 0.08% |
| Latency | ~1-2s (inference + network round-trip) |

**Model accuracy context:** Even expert piano teachers disagree roughly 20% of the time on what constitutes better playing. The 6 dimension scores are useful signals, not ground truth. The system's differentiator is the analysis pipeline's reasoning about what those scores mean for *this student at this moment*, not the raw numbers. Scores are inputs to a reasoning pipeline, not a report card.

**Audio format handling:** The HF inference handler uses `librosa.load()` which accepts any common audio format (PCM, WAV, Opus/WebM, MP3). The API worker forwards raw audio bytes from either platform without conversion. No format normalization is needed in the worker.

For full architecture details, training results, and per-dimension breakdown, see `model/03-encoders.md`.

---

## Stage 3: Teaching Moment Selection

Given a session of scored chunks, determine *which* chunk is a teaching moment and *which* dimension to surface. Runs in the cloud worker (or Durable Object for web sessions).

### STOP Classifier

A logistic regression trained on 1,707 labeled teaching moments from masterclass transcripts. Predicts "would a teacher actually stop the student here?" for each chunk.

**Two deployment options:**

| Option | Where it runs | Accuracy (AUC) | Trade-off |
|--------|---------------|-----------------|-----------|
| A: MuQ embeddings | HF endpoint (additional output head) | 0.936 | Highest accuracy; requires modifying inference endpoint |
| B: 6-dim composite scores | Cloud worker (Rust/WASM) | 0.845 | Simple to deploy; 9% AUC gap |

**Decision: Start with Option B.** The classifier is trivial -- `sigmoid(dot(weights, scores) + bias)` with 6 weights extracted from the trained sklearn model in `model/src/masterclass_experiments/models.py`. Upgrade to Option A if the accuracy gap matters in practice.

```rust
// Option B: logistic regression in cloud worker
fn stop_probability(scores: &[f64; 6], weights: &[f64; 6], bias: f64) -> f64 {
    let logit: f64 = scores.iter().zip(weights).map(|(s, w)| s * w).sum::<f64>() + bias;
    1.0 / (1.0 + (-logit).exp())
}
```

### Selection Algorithm

Triggered when the student asks "how was that?" (iOS) or continuously during recording (web):

1. Collect all chunks from the current session (or since last query)
2. Each chunk has: 6-dim scores + STOP probability
3. **Filter:** Only consider chunks with STOP probability > 0.5
4. **Rank:** Sort by STOP probability descending
5. **Select:** Take the top-1 chunk as the teaching moment
6. **Identify dimension:** Within the selected chunk, find the blind-spot dimension

### Web Observation Throttle

On the web path, teaching moment selection runs continuously as chunks arrive. Without rate limiting, a 3-minute passage could generate multiple unsolicited observations -- violating the "one observation at a time" principle from `01-product-vision.md`.

**Throttle rule:** Accumulate teaching moment candidates during recording. Push the top-1 observation per time window (default: 3 minutes). Between windows, new candidates replace lower-ranked ones in the buffer but are not delivered until the next window opens.

**Configuration:**
- `observation_window_sec`: minimum seconds between WebSocket observation pushes (default: 180)
- `min_chunks_before_first`: minimum chunks scored before first observation can fire (default: 4, i.e., 1 minute of playing)

**"How was that?" override:** If the student explicitly asks (via the chat interface on web), the throttle is bypassed and the current top candidate is delivered immediately, same as the iOS on-demand path.

The iOS path is inherently throttled: the student must tap "How was that?" to receive an observation. No additional rate limiting is needed.

### Blind-Spot Dimension Selection

Determines *which* dimension to talk about within the selected chunk.

**With student history (warm):**
- Compare each dimension to the student's rolling average (exponential moving average, alpha=0.3)
- The most surprising negative deviation = the blind spot
- "Surprising" means the student is usually fine here, but it dipped

**Without student history (cold start):**
- Use the STOP classifier's feature importance (logistic regression coefficients)
- The dimension that contributed most to the STOP prediction for this chunk
- Alternatively: the dimension with the lowest absolute score

**Tie-breaking -- blind-spot prior:**

Prefer dimensions that are harder to self-diagnose from the player's seat:

```
voicing/balance > pedaling > phrasing > timing > dynamics > articulation
```

Dynamics and articulation are easier to feel while playing; voicing and pedaling effects are harder to hear from the bench. This ordering is a hypothesis -- validate with real user testing.

### Positive Teaching Moments

The pipeline also flags improvements and breakthroughs, not just problems. A chunk is a positive candidate when:

- A dimension score is significantly above baseline for the first time
- A previously flagged weakness shows measurable improvement
- A passage that was problematic last session is now clean
- Overall session quality is notably higher than recent history

The pipeline tags positive candidates alongside STOP-flagged chunks. The subagent (Stage 4a) decides whether to use a positive or corrective framing.

### No-Candidates Fallback

When no chunks pass the STOP threshold (probability > 0.5), the student played well and there is nothing to correct. Rather than returning silence or a generic "sounded good," the pipeline falls back to positive moment detection:

1. Find the dimension with the highest score across all session chunks
2. If a student baseline exists, find the dimension with the largest positive deviation from baseline
3. Construct a positive teaching moment with `is_positive: true`
4. The subagent runs with positive framing context, producing recognition or encouragement

This ensures the student always receives a meaningful response when they ask "how was that?" -- either a corrective observation (STOP fired) or a genuine positive observation (nothing to correct, but something good to acknowledge).

If the session has fewer than 2 scored chunks (e.g., student asked after 15 seconds), return a brief "I need a bit more to listen to -- keep playing and ask me again" message without invoking the subagent.

### Output to Stage 4

```json
{
    "teaching_moment": {
        "chunk_index": 7,
        "start_offset_sec": 105.0,
        "stop_probability": 0.87,
        "dimension": "pedaling",
        "dimension_score": 0.35,
        "student_baseline": 0.62,
        "deviation": -0.27,
        "section_label": "second phrase",
        "bar_range": "bars 20-24",
        "is_positive": false
    },
    "session_summary": {
        "total_chunks": 12,
        "chunks_above_threshold": 3,
        "dominant_weak_dimension": "pedaling",
        "session_duration_min": 3.0
    }
}
```

The `section_label`, `bar_range`, and `is_positive` fields are optional -- populated when piece/section info is available. See Score Alignment below.

---

## Stage 4: Two-Stage Analysis Pipeline

### Why Two Stages

The original design sent structured data directly to one LLM, which had to simultaneously *analyze what matters* AND *generate a natural observation*. These are distinct tasks that benefit from different models and prompting strategies.

The two-stage design separates analysis from delivery:

| Stage | Role | Model | Latency | Cost |
|-------|------|-------|---------|------|
| 4a: Subagent | Structured reasoning about what to say | Groq / Llama 3.3 70B | ~0.3s | ~$0.50-0.60/M tokens |
| 4b: Teacher | Natural, warm delivery of the observation | Anthropic / Sonnet 4.6 | ~1.5s | $3.00 input / $15.00 output per M tokens |

This mirrors how Claude Code handles complex tasks: the main agent delegates analysis to Explore agents via prepared handoff messages, then uses the results. The subagent does the legwork; the teacher provides the voice.

### Stage 4a: Subagent (Groq / Llama 3.3 70B)

The subagent receives cloud-filtered moments (top 3-5 with STOP > threshold) plus the student's context map (baselines, synthesized facts, learning arc, goals). It reasons through five steps.

#### Context Inputs

The Worker builds the subagent's context from three sources:

**From the client (per request):**
- Filtered teaching moments (top 3-5 chunks with STOP > threshold)
- Each chunk: 6 dimension scores, STOP probability, chunk index, start offset
- Current piece (if student-reported): composer, title, section label, approximate bar range
- Session metadata: duration, total chunks, chunks above threshold

**From local storage (via client request payload on iOS, direct D1 on web):**
- Student baselines per dimension
- Inferred level
- Explicit goals
- Learning arc for current piece (session count on this piece)

**From D1 (Worker queries before subagent call):**
- Last 5-10 synthesized facts for this student (see `03-memory-system.md`)
- Recent observation history (last 3-5 observations with condensed traces)

This follows the **map-first principle**: build the context map before agents reason over it. The subagent walks the map to make its decision -- it does not re-derive patterns from raw session data each call.

#### Five-Step Reasoning Framework

**1. Where are they? (Learning Arc)**

The student's familiarity with the current piece changes what feedback is appropriate:

- **New to this piece (sessions 1-3):** Prioritize encouragement. Ignore detail problems. "You're getting the note patterns down" is more useful than "your pedaling needs work."
- **Mid-learning (sessions 4-10):** Focus on structural issues -- sections, transitions, rhythmic patterns.
- **Polishing (sessions 10+):** Focus on expression -- dynamics, phrasing, interpretation, pedaling nuance.

Learning arc is tracked per piece, inferred from session count. Student can also declare it ("I just started this piece" vs "I'm preparing this for a recital").

**2. What changed? (Delta vs History)**

Compare current scores against synthesized facts:

- **Improved:** Consider a positive observation. Positive moments are legitimate teaching, not participation trophies.
- **Regressed from baseline:** Check if blind spot (usually strong, dipped today) or fatigue (end of session).
- **Stable weakness:** Student likely knows. Frame as a progress check, not a discovery. "I know we've talked about pedaling before -- here's something specific to try."

**3. What matters for this music? (Musical Context)**

Piece, composer, and style weight which dimensions matter most:

| Style | Critical dimensions | Why |
|-------|--------------------|----|
| Chopin | Pedaling, phrasing | Romantic rubato, singing tone |
| Bach | Articulation, timing | Clarity, voice independence |
| Beethoven | Dynamics, interpretation | Structural contrasts, dramatic range |
| Debussy | Pedaling, interpretation | Color, atmosphere, impressionistic voicing |

"Pedaling score 0.35 in a Chopin Nocturne" is a serious issue. "Pedaling score 0.35 in a Bach Invention" might be appropriate (less pedal is often correct). The subagent contextualizes scores against musical expectations.

**4. What's the one thing? (Selection)**

Re-rank the filtered moments considering learning arc, delta, musical context, and blind-spot prior. Pick the moment with the highest leverage -- what will move the needle most for this student right now.

The subagent may select differently from the initial cloud worker ranking. The cloud worker uses STOP probability alone. The subagent adds student history, musical context, and learning arc.

**5. What's the framing? (Correction / Recognition / Encouragement / Question)**

The framing decision is part of the analysis, not left to the teacher LLM. The subagent explicitly outputs one of:

- **Correction:** "Point out the pedaling issue in bar 7 and suggest lifting on beat 3."
- **Recognition:** "Acknowledge that dynamics are noticeably better this session."
- **Encouragement:** "The notes are coming together. Don't worry about expression yet."
- **Question:** "They've been working on the development section a lot -- ask if there's something specific they're working through."

This prevents the teacher LLM from defaulting to critique mode.

#### Subagent Output (Handoff Format)

The subagent outputs both structured JSON and a narrative reasoning summary. The teacher LLM receives both.

**Structured output:**

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

**Narrative reasoning:**

```
The most notable moment was in the second phrase (bars 20-24) where pedaling dropped
to 0.35, well below this student's usual 0.62. This is a blind spot -- their pedaling
is normally a strength. Given they're in the polishing phase with this Nocturne,
and pedaling is central to Chopin's sound, this is high-leverage feedback.
Recommend framing as a discovery: point out what happened and suggest a specific
pedaling technique for the harmonic changes in those bars.
```

#### Condensed Reasoning Traces

Each observation persists a condensed trace alongside the observation text:

```json
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

These traces are the raw material from which synthesized facts are periodically generated (see `03-memory-system.md` for the two-clock model and synthesis cadence).

**What to persist:** dimension, key insight (one sentence), confidence level, framing decision, relevant scores, piece context, learning arc position.

**What NOT to persist:** The full subagent narrative, raw chunk data, the teacher LLM's output (already stored as the observation text).

### Stage 4b: Teacher LLM (Anthropic / Sonnet 4.6)

The teacher receives the subagent's handoff (structured JSON + narrative reasoning) and generates the observation the student actually sees.

#### System Prompt (Teacher Persona)

```
You are a piano teacher who has been listening to your student practice. You have
years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

Your role is to give ONE specific observation about what you just heard. Not a report.
Not a lesson plan. One thing -- the thing the student most needs to hear right now.

How you speak:
- Specific and grounded: reference the exact musical moment, not generalities
- Natural and warm: you're talking to a student you know, not writing a review
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Brief: 1-3 sentences. A teacher's aside, not a lecture.

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting
```

#### User Prompt

The user prompt includes the subagent handoff (structured JSON + narrative reasoning) plus student context. Dynamic context blocks are built based on available information:

**Session context (always present):**

```
Session duration: {duration_min} minutes
Chunks analyzed: {total_chunks}
Teaching moments found: {moments_above_threshold} (this was the most important)
```

**Student context (cold start):**

```
This is a new student. I don't know their history yet.
Repertoire suggests {inferred_level} level.
No baseline to compare against -- assess based on absolute quality.
```

**Student context (warm, sessions 3+):**

```
Student level: {level}
Current goals: {explicit_goals}
Working on: {pieces}
Dimension baselines (rolling average over {session_count} sessions):
  Dynamics: {baseline_dynamics:.2f}, Timing: {baseline_timing:.2f}, ...
This session's {dimension} ({dimension_score:.2f}) is {deviation_description}
  their usual ({student_baseline:.2f}).
```

Additional framing hints are injected when the student model warrants them:

- If dimension is consistently weak: "Note: {dimension} has been a persistent area for growth -- frame as progress check, not discovery."
- If dimension usually strong: "Note: {dimension} is usually a strength. This dip is likely a blind spot."

#### Output Handling

**Expected output:** 1-3 sentences of natural language. No formatting, no bullets, no scores.

**Post-processing:**
- Strip any markdown formatting the LLM adds
- Reject outputs longer than 500 characters (re-prompt with "shorter")
- Reject outputs that contain numbers/scores (re-prompt with "no scores, just describe what you heard")

**Fallback:** If LLM call fails, use a template:

```
"I noticed your {dimension} could use some attention in that last section.
Try recording yourself and listening back -- sometimes it's hard to hear
{dimension_description} while you're playing."
```

#### "Tell Me More" Flow

When the student taps "Tell me more" after the initial observation:

1. Skip the subagent entirely -- the moment is already selected and analyzed.
2. Send the original handoff (subagent analysis + initial observation) to the teacher LLM with an elaboration instruction:

```
The student wants to know more about this. Elaborate with:
1. Why this matters for this piece/style
2. A specific practice technique they can try right now
3. What "fixed" would sound/feel like

Still conversational. 2-4 sentences.
```

3. Keep the same teacher model -- the system prompt prefix stays cached across turns.

#### Artifact Declaration

The teacher LLM declares artifacts via tool use when a rich component would add pedagogical value. See `05-ui-system.md` for the unified artifact container system. The API returns both observation text and artifact config (if any) to the client.

The old three-stage pipeline (analysis subagent + teacher + UI subagent) is replaced by a two-stage pipeline where the teacher LLM has artifact tools available. This eliminates the UI subagent stage and ~0.3-0.5s of latency.

### Score Alignment

Connect MuQ chunk timestamps to bar/measure numbers in the score. This lets the teacher say "at bar 7, the dynamics have no range" instead of "at 0:04, the dynamics have no range."

**Current approach (CEO review 2026-03-19):** AMT transcribes each chunk to MIDI. The MIDI fingerprint is matched against the 242-piece score library via `piece_match.rs` (bigram Dice similarity). If matched, `score_follower.rs` aligns the performance MIDI to the score via onset+pitch DTW, producing bar numbers. This runs automatically -- no student input required.

**Graceful degradation:** If no piece match is found, observations use audio-quality language without bar numbers ("your pedaling is blurring harmonic changes" instead of "in bars 5-8, your pedaling..."). The system asks "What piece is this?" AFTER the first observation, not before. Piece identification enriches but never gates.

**Score library:** 242 ASAP pieces. Demand tracking via `piece_requests` table logs unmatched fingerprints for catalog expansion.

### Pipeline Modes

The pipeline operates in two modes. The mode is set by the client at session start and determines how teaching moment selection and feedback delivery behave.

#### Regular Mode (default)

Standard pipeline behavior as described above. All 6 dimensions are considered. Teaching moment selection picks the top-1 chunk and the blind-spot dimension. The subagent reasons across the full context.

#### Focus Mode

Activated when the student enters a focused practice session targeting a specific dimension (see `04-exercises.md`). The pipeline changes in three ways:

1. **Dimension weighting:** Only the target dimension is considered for teaching moment selection. Non-target dimensions are suppressed from the subagent context.

2. **Non-target exception:** If a non-target dimension has STOP probability > 0.95, it is surfaced anyway. This prevents ignoring severe issues (e.g., wildly wrong notes while working on pedaling).

3. **Before/after tracking:** Each exercise attempt records the target dimension score. The pipeline maintains a per-exercise score history in the session state, enabling the teacher to compare improvement across attempts.

The mode is communicated via the `/api/ask` request payload (`mode: "regular" | "focus"`) or via the Durable Object session state on the web path. The subagent system prompt includes a mode-specific preamble:

- **Regular:** "Consider all dimensions. Pick the one that matters most."
- **Focus:** "The student is focused on {target_dimension}. Evaluate their progress on this dimension. Only mention other dimensions if something is severely off."

---

## Provider Architecture

### Multi-Provider Routing

The pipeline uses direct provider APIs optimized per stage rather than routing everything through a single gateway.

```
                    +----------------+
                    |  Groq API      |  Stage 4a: Subagent
                    |  Llama 3.3 70B |  ~450-800 tok/s on LPU
                    |  ~0.3s         |  Stage 4c: UI subagent (optional)
                    +----------------+

                    +----------------+
                    |  Anthropic API |  Stage 4b: Teacher
                    |  Sonnet 4.6    |  Prompt caching (persona prefix)
                    |  ~1.5s         |
                    +----------------+

                    +----------------+
                    |  OpenRouter    |  Fallback gateway
                    |  (any model)   |  If either direct provider is down
                    +----------------+

                    +----------------+
                    |  Workers AI    |  Emergency fallback
                    |  Llama 3.1 70B |  Co-located with Workers, free
                    +----------------+
```

**Why multi-provider over OpenRouter-only:**
- ~0.3-0.5s latency savings (no routing hop)
- Native prompt caching with Anthropic API
- Groq's LPU gives 3-5x faster subagent inference vs GPU-based providers
- OpenRouter remains available as fallback routing layer

### Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Audio upload + inference | ~1-2s | HF endpoint round-trip |
| STOP classification | <1ms | 6-weight logistic regression |
| Subagent (Groq) | ~0.3s | 450-800 tok/s on LPU |
| Teacher (Anthropic) | ~1.5s | Prompt caching reduces input cost |
| **Total (Stages 4a+4b)** | **<2s** | Within <3s user-facing target |

### Prompt Caching Strategy

Static content first, dynamic content last:

```
[CACHED] System prompt: teacher persona (same for every request)
[CACHED] Tools/output format constraints
[DYNAMIC] User message: subagent handoff + student context
```

For "Tell me more" follow-ups, keep the same teacher model. The system prompt prefix stays cached. Switching models would invalidate the cache.

---

## Updated "Ask" Flow

```
Client                           Worker (/api/ask)                   LLM Providers
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
 |                                    |  --- Groq (Llama 70B) ----------> |
 |                                    |  <-- analysis (JSON + narrative) - |
 |                                    |                                   |
 |                                    |  3. Build teacher prompt           |
 |                                    |     (system: teacher persona)      |
 |                                    |     (user: subagent handoff)       |
 |                                    |  --- Anthropic (Sonnet 4.6) ----> |
 |                                    |  <-- observation (1-3 sentences) - |
 |                                    |                                   |
 |                                    |  4. Store condensed trace in D1    |
 |                                    |  5. Return observation to client   |
 |  <-------------------------------- |                                   |
 |                                    |                                   |
 |  "Tell me more"                    |                                   |
 |  --------------------------------> |                                   |
 |                                    |  Skip subagent. Send original     |
 |                                    |  handoff + observation to teacher  |
 |                                    |  with elaboration instruction.     |
 |                                    |  --- Anthropic (Sonnet 4.6) ----> |
 |                                    |  <-- elaboration (2-4 sentences) - |
 |  <-------------------------------- |                                   |
```

### Request Payload Schema

**`POST /api/ask`**

```json
{
    "teaching_moments": [
        {
            "chunk_index": 7,
            "start_offset_sec": 105.0,
            "stop_probability": 0.87,
            "scores": {
                "dynamics": 0.72,
                "timing": 0.65,
                "pedaling": 0.35,
                "articulation": 0.58,
                "phrasing": 0.61,
                "interpretation": 0.49
            }
        }
    ],
    "student_context": {
        "student_id": "apple_user_id_here",
        "level": "intermediate",
        "goals": ["Improve pedaling transitions", "Prepare for recital"],
        "baselines": {
            "dynamics": 0.68,
            "timing": 0.71,
            "pedaling": 0.62,
            "articulation": 0.60,
            "phrasing": 0.55,
            "interpretation": 0.58
        },
        "session_count": 12
    },
    "piece_context": {
        "piece_id": "chopin.nocturnes_op_9.2",
        "composer": "Chopin",
        "title": "Nocturne Op. 9 No. 2",
        "section": "second phrase",
        "bar_range": "bars 20-24",
        "learning_arc": "polishing"
    },
    "session": {
        "session_id": "sess_abc123",
        "duration_min": 3.0,
        "total_chunks": 12,
        "chunks_above_threshold": 3
    },
    "mode": "regular",
    "target_dimension": null
}
```

**Required fields:** `teaching_moments` (at least 1), `student_context.student_id`, `session.session_id`.

**Optional fields:** `piece_context` (omitted if student hasn't identified what they're playing), `student_context.baselines` (omitted for cold-start students), `mode` (defaults to `"regular"`), `target_dimension` (required when `mode` is `"focus"`).

The Worker enriches this payload with D1 data (synthesized facts, recent observations, exercise history) before constructing the subagent prompt.

---

## Deployment and Operations

### Cloudflare Workers (api.crescend.ai)

The backend is a single Cloudflare Workers application (Rust/WASM). Bindings:

| Binding | Purpose |
|---------|---------|
| D1 | Students, sessions, exercises, synthesized facts |
| KV | JWTs, rate limits |
| R2 | Audio chunks (web path) |
| DO | Practice sessions (web path, WebSocket state) |

### Key API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/ask` | Two-stage pipeline: builds teacher prompt, calls LLM providers, returns observation |
| `POST /api/practice/start` | Creates Durable Object session (web) |
| `POST /api/practice/chunk` | Uploads audio chunk, triggers HF inference |
| `WS /api/practice/ws/:sessionId` | Real-time observation delivery via WebSocket (web) |
| `POST /api/chat/send` | Streaming teacher chat (web) |
| `POST /api/auth/apple` | Validates Apple ID token, issues session JWT |
| `POST /api/sync` | Receives student model delta from iOS, upserts to D1 |

### HF Inference Endpoint

- **Model:** A1-Max 4-fold ensemble
- **Handler:** `apps/inference/handler.py`
- **Input:** 15-second audio at 24kHz mono
- **Output:** 6 dimension scores (0.0-1.0)

---

## Open Questions

1. **STOP classifier generalization.** The classifier was trained on masterclass audio (professional students, concert pianos). Will it generalize to beginner/intermediate students on upright pianos recorded with phones? Likely needs recalibration.

2. **Minimum STOP threshold.** Should there be a floor below which the system says "sounded good, keep going" instead of always finding something to critique? The system should not fabricate problems to fill silence.

3. **Low-confidence scores.** When the MuQ model is uncertain (middle-range scores, no clear outlier), should the subagent flag a teaching moment at all?

4. **Positive/corrective ratio.** How often should the subagent choose a positive observation over a corrective one? A real teacher probably does 70% correction, 30% positive -- but it varies by student personality and learning phase.

5. **Blind-spot prior validation.** The ordering (voicing > pedaling > phrasing > timing > dynamics > articulation) is a hypothesis. Validate with real user testing.

6. **Synthesis cadence.** When does the system synthesize facts from raw traces? After every N sessions? On a background timer? On-demand? More frequent = fresher context, but more D1 writes and LLM calls. See `03-memory-system.md`.

7. **Score alignment accuracy.** With student-reported piece and bar range, how accurate can timestamp-to-bar mapping be? Tempo changes, rubato, and pauses introduce error.

8. **Subagent prompt iteration.** The five-step reasoning framework needs testing with synthetic teaching moment data. Llama 70B on Groq needs concise prompts.

9. **A/B testing within model tiers.** Groq/Llama for subagent and Anthropic/Sonnet for teacher are decided, but A/B testing across alternative models within each tier remains open.

---

## Pipeline Evaluation

The pipeline layer (STOP classification, teaching moment selection, subagent reasoning, teacher output) requires its own eval framework. All evals live in `apps/evals/` -- pipeline evals in `apps/evals/pipeline/`, memory evals in `apps/evals/memory/`, model quality evals in `apps/evals/model/`.

### Teaching Moment Selection Eval

**Goal:** Given a synthetic session (sequence of scored chunks), does the selection algorithm pick the right chunk and dimension?

**Approach:**
- Construct 20-30 synthetic sessions with known "best teaching moment" (human-labeled or expert-authored)
- Each session: 8-15 chunks with realistic 6-dim score patterns (some with STOP-worthy drops, some clean)
- Measure: selection accuracy (did it pick the right chunk?), dimension accuracy (did it pick the right dimension?), positive detection rate (did it correctly identify improvements?)

**Metrics:**
- Selection accuracy >= 80% (correct chunk in top-1)
- Dimension accuracy >= 70% (correct blind-spot dimension)
- Positive detection: recall >= 90% (don't miss genuine improvements)

### Subagent Reasoning Eval

**Goal:** Given a teaching moment + student context, does the subagent produce appropriate analysis?

**Approach:**
- 15-20 scenario-based test cases covering the 5-step reasoning framework
- Test each step independently: learning arc classification, delta detection, musical context weighting, moment selection, framing decision
- Use an LLM judge (Claude) to evaluate subagent output against expected reasoning

**Key scenarios:**
- Cold start (no history) vs warm (10+ sessions)
- Positive framing when scores improve
- Stable weakness (pedaling flagged 3 times) -> "progress check" framing, not "discovery"
- Musical context: pedaling in Chopin vs Bach (different weights)
- No-candidates fallback: positive moment generation

### Teacher Output Post-Processing

**Goal:** Verify that post-processing rules catch invalid outputs.

**Approach:** Unit tests (no LLM needed):
- Input > 500 chars -> rejected, re-prompted
- Input contains numbers/scores (e.g., "your pedaling is 0.35") -> rejected, re-prompted
- Input contains markdown formatting -> stripped
- LLM call failure -> template fallback produced

### Provider Failover Eval

**Goal:** Verify that the fallback chain (Groq -> OpenRouter -> Workers AI) produces acceptable output.

**Approach:**
- Mock primary provider failure (Groq timeout, Anthropic 429)
- Verify fallback activates within 2s
- Verify fallback output passes the same post-processing rules
- Measure quality degradation: LLM judge comparison of primary vs fallback output on 10 scenarios

---

## Cost Budget

Per-session cost estimates for a 30-minute practice session (120 chunks at 15s intervals, assuming ~20% silence skipped = 96 scored chunks, 3-5 observations delivered).

| Component | Per-unit cost | Units/session | Session cost |
|---|---|---|---|
| HF inference (A1-Max endpoint) | ~$0.003-0.005/call | 96 chunks | $0.29-0.48 |
| STOP classification | $0 (in-worker computation) | 96 chunks | $0 |
| Groq subagent (Llama 70B) | ~$0.0004/call (~500 input + 200 output tokens) | 3-5 observations | $0.001-0.002 |
| Anthropic teacher (Sonnet 4.6) | ~$0.004/call (~700 input + 100 output tokens) | 3-5 observations | $0.01-0.02 |
| Durable Object (web path) | ~$0.001-0.005/session | 1 | $0.001-0.005 |
| D1 reads/writes | ~$0.001/session | ~10 queries | $0.001 |
| **Total per session** | | | **$0.30-0.51** |

**HF inference dominates** (~90% of cost). Key optimizations:
- **Silence detection** (Issue: already specified in Stage 1) reduces chunks by ~20%
- **Adaptive chunk frequency** (e.g., 30s chunks during warm-up, 15s during focused playing) could halve inference cost for long sessions
- **HF endpoint autoscaling** (scale-to-zero when no sessions active) eliminates idle cost

At 1,000 daily sessions: ~$300-510/day ($9K-15K/month) in inference costs. LLM costs are negligible by comparison (~$15-20/day).

These estimates are rough and depend on HF endpoint instance type (GPU-hours pricing varies). Measure with real traffic before optimizing.

**Inference cost reduction target (CEO review):** Current ~$6/session (at dedicated endpoint pricing) must reach ~$1/session for tiered pricing (Free/$5/$20/$50) to work. Optimization path: silence gating (-25%), single fused model v2 (-50%), passage-level caching (-20%), serverless inference (-40%). This is part of the model v2 track, not the apps beta.

---

## Key Decisions Log

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Two-stage pipeline | Subagent + Teacher | Separates analysis (fast/cheap) from delivery (quality voice). Different tasks need different models and prompts. |
| STOP classifier deployment | Option B first (6-dim scores in worker) | Simplest path. 0.845 AUC is sufficient to start. Upgrade to Option A (0.936 AUC, MuQ embeddings) if gap matters. |
| Subagent provider | Groq (Llama 3.3 70B) | LPU runs Llama 70B at 450-800 tok/s. Subagent completes in ~0.3s. |
| Teacher provider | Anthropic (Sonnet 4.6) | Best at following nuanced persona instructions. Native prompt caching for system prompt. |
| Fallback chain | OpenRouter, then Workers AI | OpenRouter as routing fallback, Workers AI (Llama 3.1 70B) as zero-dependency emergency fallback. |
| Score alignment | AMT fingerprint + DTW (automated) | Replaces student-reported. AMT transcribes, fingerprint matches, DTW aligns to score. No student input required. Graceful degradation for unknown pieces. |
| Framing as subagent output | Explicit framing decision in JSON | Prevents teacher LLM from defaulting to critique mode when only given problems. |
| Scores as reasoning inputs | Not a report card | Model is ~80% pairwise accurate. Value is in analysis + delivery, not raw scores. |
| Blind-spot prior | voicing > pedaling > phrasing > timing > dynamics > articulation | Dimensions harder to self-diagnose from the bench get priority in tie-breaking. |
| Positive teaching moments | Subagent decides framing | Pipeline flags both improvements and problems. Positive observations are real teaching. |
| Session brain | DO as session intelligence host | Practice mode state machine (warming/drilling/running/winding). Observation pacing adapts to mode. Single-threaded DO is ideal for temporal session state. |
| Artifact declaration | Teacher LLM tool use (replacing UI subagent) | Eliminates the third LLM stage. Teacher decides when artifacts are warranted and calls tools to generate configs. Pattern (Anthropic tool_use vs MCP) TBD. |
| Piece identification | AMT fingerprint (replacing student-reported) | Zero-config first session. Auto-detect via MIDI fingerprint against 242-piece library. Graceful degradation for unknown pieces. |
| Platform priority | Web-first | Web beta ships first (~4 weeks). iOS follows after validation. |
