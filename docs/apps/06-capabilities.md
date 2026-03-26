# Capabilities

> Status: 2026-03-26 -- Seven core capabilities defined. Quality targets set for beta and north star. All implemented, none fully validated.

CrescendAI's teaching pipeline is built from seven capabilities. Each converts raw audio into progressively richer understanding of what the student played and what they should work on. They compose into a single user-facing experience: the student plays, the system listens, and at session end the teacher gives one cohesive, specific, actionable response.

This document defines what each capability does, what "good" looks like quantitatively, and how they connect. It is the quality contract for the product.

---

## 1. Piece Identification

### What It Does

The student opens the app and plays. Within 60 seconds, the system silently identifies the piece from audio alone -- no "what are you playing?" prompt. Score context activates, bar references become available, and the teacher can say "bars 20-24" instead of "that middle section."

### Inputs / Outputs

- **In:** AMT-transcribed MIDI notes (from Aria-AMT endpoint), N-gram index + rerank features (R2 bucket), 242-piece score library
- **Out:** `PieceIdentification { piece_id, confidence, method, notes_consumed }` or `None`
- **Depends on:** AMT transcription quality, score library coverage

### How It Works

Three-stage pipeline:
1. **N-gram recall:** Extract pitch N-grams from incoming MIDI, query against pre-computed index. Returns candidate set with recall scores.
2. **Statistical rerank:** Score candidates using pitch distribution, interval histogram, and rhythmic features. Produces ranked list with confidence.
3. **DTW confirmation:** Align top candidate's score MIDI against performance MIDI using dynamic time warping. Confirms or rejects match.

Notes accumulate across chunks. Identification can trigger as early as chunk 2 (~30s) or as late as chunk 6 (~90s) depending on piece distinctiveness.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Top-1 accuracy (in-library pieces) | > 80% | > 95% |
| Notes to identify | < 300 (~45-60s) | < 100 (~15-20s) |
| False positive rate (confidence > 0.8) | < 10% | < 2% |
| Out-of-library handling | No false match | Suggests adding piece |

### Current State

Code complete on `feat/zero-config-piece-id`, pending deploy. Never evaluated at scale against real student recordings. The 242-piece library covers ASAP dataset; coverage of common student repertoire (Fur Elise, Clair de Lune, Bach Inventions) is good, but long-tail pieces will miss.

### Key Files

- `apps/api/src/practice/piece_identify.rs` -- N-gram + rerank + DTW pipeline
- `apps/api/src/practice/score_context.rs` -- Score/reference loading from R2
- `apps/inference/amt_handler.py` -- Aria-AMT transcription endpoint

---

## 2. STOP Classification

### What It Does

Every 15-second chunk, the system asks: "Would a real teacher stop the student here to talk about something?" STOP is the gate -- if it doesn't trigger, no teaching moment is selected, no feedback is generated. It must trigger on genuinely problematic passages and stay quiet when the student is playing fine.

### Inputs / Outputs

- **In:** 6-dim MuQ scores `[dynamics, timing, pedaling, articulation, phrasing, interpretation]` for one chunk
- **Out:** `StopResult { probability, triggered, top_dimension, top_deviation }`
- **Depends on:** MuQ model quality (scores must be meaningful for STOP to work)

### How It Works

Stateless logistic regression. Each chunk's 6 scores are z-normalized against training means, multiplied by learned weights, summed with bias, and passed through sigmoid. No temporal history, no previous-chunk memory.

Weight interpretation:
- Negative weights (dynamics, pedaling, interpretation): low score triggers STOP (musicality problems)
- Positive weights (timing, articulation, phrasing): high score triggers STOP (technically accurate but musically flat)

Threshold: 0.5 (default). Trained on 1,699 masterclass segments with balanced class weights.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Skill discrimination (Cohen's d, adjacent buckets) | > 0.5 | > 0.8 |
| Spearman rho (STOP rate vs skill level) | > 0.3 | > 0.5 |
| Trigger rate: bucket 1 (weakest) | 50-80% | calibrated per piece |
| Trigger rate: bucket 5 (strongest) | 15-40% | calibrated per piece |
| No single dimension > X% of triggers | < 50% | < 35% |

### Current State

Deployed. AUC 0.845 on training data (masterclass recordings). Never tested on intermediate students with consumer microphones (the actual target population). The masterclass-to-YouTube domain gap is the biggest unknown.

### Key Files

- `apps/api/src/services/stop.rs` -- Classifier weights and computation

---

## 3. Teaching Moment Selection

### What It Does

Given that STOP triggered on multiple chunks in a session, the system picks the ONE teaching moment that matters most right now. Not the loudest signal, but the most pedagogically useful one -- the blind spot the student doesn't know about, or the breakthrough they should celebrate.

### Inputs / Outputs

- **In:** All scored chunks, student baselines (per-dimension rolling averages), recent observations (last 3 for dedup), score context (optional)
- **Out:** `TeachingMoment { dimension, score, deviation, chunk_index, is_positive, reasoning, bar_range }`
- **Depends on:** STOP classification, student baselines (from memory/D1), score following (for bar ranges)

### How It Works

1. **Filter:** Keep only chunks where STOP triggered
2. **Rank by blind spot:** For each passing chunk, compute per-dimension deviation from student baseline. Rank by largest negative deviation (worst relative to their own norm).
3. **Deduplicate:** Skip candidates whose top dimension matches the last 3 observations
4. **Positive fallback:** If no chunks pass STOP, find the dimension with largest positive deviation from baseline -- celebrate it

The accumulator's `top_moments()` algorithm selects up to 8 moments for synthesis: top-1 per dimension by |deviation|, plus top-1 positive per dimension if different, sorted by chunk_index for narrative coherence.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Selected dimension matches piece style | > 70% | > 85% |
| Positive moments: deviation actually positive (> +0.15) | > 90% | > 95% |
| Framing matches deviation direction | > 85% | > 95% |
| No dimension selected > 2x in one session | > 90% | 100% |

"Matches piece style" means: pedaling flagged for Chopin/Debussy, articulation for Bach, dynamics for Beethoven. This requires an LLM judge with piece knowledge.

### Current State

Implemented with heuristic blind-spot ranking. The dedup window (3) and positive threshold are unvalidated. The biggest risk: the system doesn't yet consider piece style when ranking dimensions -- it treats all dimensions equally regardless of whether pedaling matters for Bach Inventions.

### Key Files

- `apps/api/src/services/teaching_moments.rs` -- Selection algorithm
- `apps/api/src/practice/accumulator.rs` -- SessionAccumulator and top_moments()

---

## 4. Practice Mode Detection

### What It Does

The system understands HOW the student is practicing, not just WHAT they're playing. When they drill bars 20-28 four times, the synthesis says "you drilled bars 20-28 four times and your pedaling cleaned up between attempt 1 and attempt 4." When they do a full run-through, synthesis treats it as a holistic performance. The system never interrupts warm-up with feedback.

### Inputs / Outputs

- **In:** Per-chunk signals: pitch bigrams (consecutive pitch pairs from AMT MIDI), bar ranges (from score following), piece match status, timestamps
- **Out:** `PracticeMode { Warming | Drilling | Running | Regular | Winding }` with `ObservationPolicy { suppress, min_interval_ms, comparative }`
- **Depends on:** AMT transcription (for pitch bigrams), score following (for bar ranges)

### How It Works

4-chunk sliding window. Two repetition signals:
- **Bar overlap:** If same bar range appears in consecutive chunks with > 50% overlap, student is repeating a passage
- **Pitch bigram Dice similarity:** If consecutive chunks share > 60% of their pitch bigram sets, content is repeated

State transitions:
- Warming -> Drilling: 3+ chunks with detected repetition
- Warming -> Running: Piece matched, bars progressing forward
- Running -> Drilling: 30s dwell elapsed + repetition detected
- Drilling -> Running: 30s dwell + no repetition + forward bar progress
- Any -> Winding: 60s silence gap

Drilling passages are tracked with first/final scores for before/after comparison in synthesis.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Drilling detection precision | > 70% | > 90% |
| Drilling detection recall | > 60% | > 85% |
| Run-through detection | > 75% | > 90% |
| Mode transition latency | < 3 chunks (45s) | < 2 chunks (30s) |
| Drilling bar range accuracy | > 50% overlap with actual | > 75% |

### Current State

Implemented with heuristic thresholds (Dice 0.6, bar overlap 0.5, 4-chunk window). These thresholds were set by intuition, not tuned on data. The T5 eval corpus is our first opportunity to validate them on real student recordings. If precision is below 50%, the thresholds need recalibration -- and that's a finding, not a failure of the eval.

### Key Files

- `apps/api/src/practice/practice_mode.rs` -- State machine, repetition detection, observation policy

---

## 5. Session Synthesis

### What It Does

The core deliverable. Student finishes playing. Within 3 seconds, they see a cohesive 3-6 sentence response from their teacher:

> "You spent most of this session on the B section of Fur Elise -- good instinct, that's where the piece gets tricky. Your pedaling cleaned up noticeably between your first and fourth time through bars 20-28; the bass line is much clearer now. The dynamic contrast in the A section could use attention next time -- you're hovering around mezzo-forte throughout, but the score asks for pianissimo at bar 9. Try playing just the right hand melody from bars 9-15 at a true pianissimo before adding the left hand back in."

Everything upstream exists to make this paragraph specific, accurate, and useful.

### Inputs / Outputs

- **In:** `SessionAccumulator` (teaching moments with deviations/bar_ranges, mode transitions with dwell times, drilling records with first/final scores, timeline events), piece context (composer/title/score data), student baselines, student memory (optional)
- **Out:** 3-6 sentence natural language synthesis + optional exercise artifact via tool_use
- **Depends on:** All upstream capabilities

### How It Works

1. `build_synthesis_prompt()` constructs structured JSON from the accumulator: session duration, practice pattern (modes/transitions), top teaching moments (up to 8, selected by `top_moments()`), drilling progress, baselines, piece context
2. Single Anthropic Sonnet 4.6 call with `SESSION_SYNTHESIS_SYSTEM` prompt. No subagent stage -- the structured accumulator IS the analysis.
3. Teacher LLM narrates the data in warm, specific, actionable language. Optionally invokes `create_exercise` tool.
4. Synthesis persisted to D1 messages table. Accumulated moments persisted to observations table.
5. Delivered via WebSocket as `{ type: "synthesis" }` event.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Musical grounding (references specific passage/bars) | > 70% | > 90% |
| Actionability (concrete practice strategy) | > 75% | > 90% |
| Differentiation (different feedback for different students) | substantive difference across skill levels | personalized to individual learning arc |
| Skill calibration (language matches level) | appropriate for bucket | adapted to individual |
| Coherence (reads as one thought, not checklist) | > 85% | > 95% |
| Drilling acknowledgment (when drilling occurred) | > 85% | > 95% |
| Bar reference accuracy (matches DTW output) | > 75% | > 90% |
| Piece context delta (with vs without piece ID) | measured | minimized (good feedback even without piece) |
| Conciseness | 3-8 sentences | 3-6 sentences |

### Current State

Fully implemented: accumulator, synthesis prompt, LLM call, persistence, WebSocket delivery, deferred recovery. Never evaluated. The system prompt includes calibration guidance (R2~0.5, deviation 0.1 is noise, 0.2+ is meaningful) but we don't know if the teacher follows it.

### Key Files

- `apps/api/src/practice/synthesis.rs` -- Prompt builder, LLM call, persistence
- `apps/api/src/practice/accumulator.rs` -- SessionAccumulator struct and accumulation logic
- `apps/api/src/services/prompts.rs` -- SESSION_SYNTHESIS_SYSTEM prompt
- `apps/api/src/practice/session_finalization.rs` -- Synthesis trigger and safety net

---

## 6. Exercise Generation

### What It Does

After identifying that pedaling in bars 20-24 is the key issue, the teacher optionally offers a targeted exercise:

> "Try this: Take just bars 20-24, left hand only with pedal. Lift and re-depress the sustain pedal on each new bass note. Listen for the moment the previous harmony clears before the new one sounds. Then add the right hand back in."

Exercises are grounded in the student's actual piece and passage, target the identified weakness, and give a concrete physical action. They appear inline in the chat as expandable artifact cards.

### Inputs / Outputs

- **In:** Teaching moment (dimension + passage), exercise catalog (curated entries from D1), synthesis context
- **Out:** `create_exercise` tool call with `{ source_passage, target_skill, exercises[] }` or no tool call (text-only synthesis)
- **Depends on:** Teaching moment selection (which dimension/passage), piece context (which piece/bars)

### How It Works

The teacher LLM has access to the `create_exercise` tool via Anthropic native tool_use (`tool_choice: "auto"`). The teacher decides autonomously whether an exercise would help more than verbal guidance.

When invoked:
1. Teacher specifies `source_passage`, `target_skill`, and 1-3 exercise steps
2. System checks if a curated catalog exercise matches the dimension
3. If no catalog match, the teacher's generated exercise is persisted to the exercises table
4. Exercise rendered as `exercise_set` artifact in the chat

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Invocation rate | 20-40% of sessions | adaptive to student preference |
| Dimension targeting (matches top teaching moment) | > 85% | > 95% |
| Passage grounding (references actual bars/piece) | > 60% | > 85% |
| Actionability (pianist can follow instructions) | > 80% | > 95% |
| Catalog utilization (uses curated when available) | > 40% when match exists | adaptive |
| Specificity (piece-specific vs generic scale) | mean > 3.0 / 5.0 | mean > 4.0 / 5.0 |

### Current State

Tool defined in prompts.rs. Catalog has schema but sparse seed data (target: 20-30 curated exercises, current count unverified). Exercise assignment/completion endpoints implemented. Focus mode (multi-exercise guided sequences) deferred to Phase 3. Tool invocation frequency and quality never tested.

### Key Files

- `apps/api/src/services/exercises.rs` -- Catalog, assignment, completion endpoints
- `apps/api/src/services/prompts.rs` -- `exercise_tool_definition()`, teacher prompt with catalog
- `apps/api/src/services/ask.rs` -- `process_exercise_tool_call()`

---

## 7. Score Following & Bar Analysis

### What It Does

When the system says "bars 20-24," those are actually the bars the student was playing. When it says "the crescendo peaked too early," there actually is a crescendo marking in the score at that location. Bar references are grounded in the sheet music, not hallucinated by the LLM.

### Inputs / Outputs

- **In:** AMT MIDI (performance notes), score MIDI (from R2), previous follower state (cross-chunk continuity)
- **Out:** `ChunkAnalysis { bar_range, tier, dimensions: { per-dim musical facts } }` with three tiers of detail
- **Depends on:** Piece identification (must know which piece to load score for), AMT transcription quality

### How It Works

**Tier 1 (full score context):** DTW aligns performance MIDI onsets against score MIDI onsets. Maps each performance note to a bar number. Computes per-bar statistics: velocity mean/std vs reference, onset deviation, pedal duration, note duration ratios. Produces bar-level musical facts per dimension.

**Tier 2 (absolute MIDI, no score):** When piece is unidentified or DTW confidence is low. Analyzes AMT MIDI features directly: velocity contour, inter-onset intervals, pedal density. No bar references, but still provides dimensional analysis.

**Tier 3 (scores only):** When AMT fails entirely. Only MuQ 6-dim scores available. No MIDI analysis. Teaching moments based purely on score deviations from baseline.

Cross-chunk continuity: follower state carries the last bar position forward, so consecutive chunks produce monotonically advancing bar numbers during run-throughs.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Bar alignment accuracy (+/- 2 bars) | > 75% | > 90% |
| Tier degradation correctness | Tier 2 on low-confidence DTW, not wrong bars | verified |
| Cross-chunk continuity (no bar regression in run-through) | > 90% | > 98% |
| Musical marking accuracy in synthesis | > 80% (verifiable against ScoreData) | > 95% |
| Reference profile utilization | used when available | comparative language in synthesis |

### Current State

DTW score follower implemented with cross-chunk state. Bar analysis produces per-dimension facts at Tier 1. Unit-tested against known score/performance pairs. Never validated end-to-end: does a bar reference in the synthesis actually correspond to what the student played? The main risk: DTW misalignment cascades through teaching moment bar ranges into synthesis, producing specific-sounding but wrong feedback.

### Key Files

- `apps/api/src/practice/score_follower.rs` -- DTW alignment, cross-chunk state
- `apps/api/src/practice/analysis.rs` -- Bar-level dimensional analysis, tier logic
- `apps/api/src/practice/score_context.rs` -- Score/reference loading from R2

---

## Capability Dependencies

```
AMT Transcription
  |
  +---> [1] Piece Identification ---> Score Context loads
  |                                        |
  +---> [4] Practice Mode Detection        |
  |       (pitch bigrams, bar ranges)      |
  |                                        v
  +---> [7] Score Following <-------- Score MIDI
  |       (bar alignment, tier analysis)
  |
MuQ Scores
  |
  +---> [2] STOP Classification
  |       |
  |       v
  +---> [3] Teaching Moment Selection <--- Student Baselines
          |                                     (from Memory/D1)
          v
        SessionAccumulator
          |
          +---> [5] Session Synthesis <--- Piece Context, Drilling Records
          |       |
          |       v
          +---> [6] Exercise Generation (via tool_use)
```

Failures cascade downward. If AMT transcription is poor, piece ID fails, score following has no score to follow, and synthesis loses all musical grounding. If STOP never triggers, no teaching moments accumulate, and synthesis has nothing to say. The eval must test each layer independently AND the full cascade.
