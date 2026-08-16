# Capabilities

> Status: 2026-08-04 -- Seven core capabilities defined. Quality targets set for beta and north star. This doc owns targets; for current measured state see `docs/apps/07-evaluation.md`.
>
> **Status update (2026-08-16, score-first redesign shipped -- epics #154/#162, cleanup #164):** The chat surface has been removed as the delivery mechanism -- the web chat UI, the chat-container artifact system, and the API chat routes are deleted. Session Synthesis (#4) and Exercise Generation (#5) below are rewritten for the current delivery objects: marks on the score/timeline, a session verdict, and one carry-forward (post-session review, #159), with exercises launching as "drills" from a mark (#158) instead of a `create_exercise` chat tool call. Chat survives only as future passage-scoped ask threads on marks (#161, unbuilt). MuQ has been removed from the request path; capabilities #1/#3/#6 and the dependency diagram now describe the current pipeline (Transkun AMT -> alignment -> MPM features -> MoonBeam-839M, currently a deterministic mock scorer, #162); see `docs/apps/02-pipeline.md` and `docs/apps/05-ui-system.md` for the full shape.

CrescendAI's teaching pipeline is built from seven capabilities. Each converts raw audio into progressively richer understanding of what the student played and what they should work on. They compose into a single user-facing experience: the student plays, the system listens, and at session end the teacher gives one cohesive, specific, actionable response.

This document defines what each capability does, what "good" looks like quantitatively, and how they connect. It is the quality contract for the product.

---

## 1. Piece Identification

### What It Does

The student opens the app and plays. Within 60 seconds, the system silently identifies the piece from audio alone -- no "what are you playing?" prompt. Score context activates, bar references become available, and the teacher can say "bars 20-24" instead of "that middle section."

### Inputs / Outputs

- **In:** AMT-transcribed MIDI notes from the Transkun `/transcribe` service, N-gram index + rerank features (R2 bucket), 242-piece score library
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

Code complete and merged to main (2026-03-22). N-gram index + rerank features committed to `model/data/fingerprints/`. Pending AMT container deployment to Cloudflare (AMT_SERVICE binding temporarily disabled in wrangler.toml). Current measured state: see docs/apps/07-evaluation.md (this doc owns targets, 07 owns results). The 242-piece library covers ASAP dataset; coverage of common student repertoire (Fur Elise, Clair de Lune, Bach Inventions) is good, but long-tail pieces will miss.

### Key Files

- `apps/api/src/wasm/piece-identify/src/{identify.rs,gate.rs,chroma.rs,text_match.rs}` -- N-gram + rerank + DTW pipeline (Rust/WASM)
- `apps/api/src/services/wasm-bridge.ts` -- TS bridge into the piece-identify WASM module
- `apps/api/src/services/scores.ts` -- Score/reference loading from R2
- `apps/inference/amt/transcription.py` -- Transkun-backed endpoint handler
- `apps/inference/amt/transkun_cli.py` -- shared isolated Transkun runner and MIDI parser

---

## 2. Teaching Moment Selection

### What It Does

Given the chunks where the student fell below baseline (the deviation gate, worst dimension `deviation < 0`), the system picks the ONE teaching moment that matters most right now. Not the loudest signal, but the most pedagogically useful one -- the blind spot the student doesn't know about, or the breakthrough they should celebrate.

### Inputs / Outputs

- **In:** All scored chunks, student baselines (per-dimension rolling averages), recent observations (last 3 for dedup), score context (optional)
- **Out:** `TeachingMoment { dimension, score, deviation, chunk_index, is_positive, reasoning, bar_range }`
- **Depends on:** the deviation gate (worst dimension below baseline), student baselines (from memory/D1), score following (for bar ranges)

### How It Works

1. **Filter:** Keep only chunks whose worst dimension is below the student's baseline (`deviation < 0`)
2. **Rank by blind spot:** For each passing chunk, compute per-dimension deviation from student baseline. Rank by largest negative deviation (worst relative to their own norm).
3. **Deduplicate:** Skip candidates whose top dimension matches the last 3 observations
4. **Positive fallback:** If no chunk is below baseline anywhere, find the dimension with largest positive deviation from baseline -- celebrate it

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

- `apps/api/src/wasm/score-analysis/src/teaching_moments.rs` -- Selection algorithm (Rust/WASM)
- `apps/api/src/services/accumulator.ts` -- SessionAccumulator and topMoments()

---

## 3. Practice Mode Detection

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

- `apps/api/src/services/practice-mode.ts` -- State machine, repetition detection, observation policy

---

## 4. Session Synthesis

### What It Does

The core deliverable. Student stops playing. At session review (`docs/apps/05-ui-system.md`) they see: one teacher verdict sentence (qualitative, no numbers, no dimension scores), the session's marks consolidated on the score or timeline, and one carry-forward -- a single next-session focus. There is no multi-sentence chat paragraph; each mark carries its own MPM-grounded evidence, available on tap, and the verdict is the one place prose ties the session together.

Everything upstream exists to make the verdict, the marks, and the carry-forward specific, accurate, and useful.

### Inputs / Outputs

- **In:** `SessionAccumulator` (teaching moments with deviations/bar_ranges, mode transitions with dwell times, drilling records with first/final scores, timeline events), piece context (composer/title/score data), student baselines, student memory (optional)
- **Out:** one verdict sentence + the session's consolidated marks + one carry-forward (the `synthesizeV6` contract; no chat prose, no artifact cards)
- **Depends on:** All upstream capabilities

### How It Works

1. The accumulator (`apps/api/src/services/accumulator.ts`) builds structured session state: session duration, practice pattern (modes/transitions), top teaching moments (up to 8, selected by `topMoments()`), drilling progress, baselines, piece context.
2. `synthesizeV6()` (`apps/api/src/services/teacher.ts`) makes the teacher LLM call (model ID in wrangler.toml) against this structured state -- still a single call, no subagent stage, the accumulator IS the analysis.
3. The teacher LLM's output is the verdict sentence plus the consolidated mark set plus one carry-forward -- not narrated prose, and no `create_exercise`-style tool invocation (drills are proposed via `prescribe_exercise`, see Exercise Generation below).
4. Persisted to D1 (session/observations tables) via `apps/api/src/services/synthesis.ts`. Delivered to the client as the review-state payload, not a WebSocket `synthesis` chat event.
5. Failure is loud: a synthesis failure renders an explicit retry state, never a blank screen and never a fabricated verdict.

### Quality Targets

| Metric | Beta | North Star |
|--------|------|------------|
| Musical grounding (marks/verdict reference specific passage/bars) | > 70% | > 90% |
| Actionability (carry-forward is a concrete practice strategy) | > 75% | > 90% |
| Differentiation (different feedback for different students) | substantive difference across skill levels | personalized to individual learning arc |
| Skill calibration (language matches level) | appropriate for bucket | adapted to individual |
| Coherence (verdict reads as one thought) | > 85% | > 95% |
| Drilling acknowledgment (when drilling occurred) | > 85% | > 95% |
| Bar reference accuracy (mark anchors match alignment output) | > 75% | > 90% |
| Piece context delta (with vs without piece ID) | measured | minimized (good feedback even without piece) |
| Conciseness | verdict is one sentence | one sentence |

### Current State

The pipeline (accumulator, marks, verdict, carry-forward, D1 persistence) shipped under #157/#158/#162; the pre-pivot chat/WebSocket synthesis paragraph and the `create_exercise` tool are gone. Current measured state: see docs/apps/07-evaluation.md (this doc owns targets, 07 owns results; the judge prompts there still target chat prose and need recalibration against the verdict+marks contract). Calibration guidance (R2~0.5, deviation 0.1 is noise, 0.2+ is meaningful) carries over to the new contract but has not been re-validated against it.

### Key Files

- `apps/api/src/services/teacher.ts` -- `synthesizeV6`, teacher LLM call
- `apps/api/src/services/accumulator.ts` -- SessionAccumulator and accumulation logic
- `apps/api/src/services/synthesis.ts` -- Persistence (messages/observations)
- `apps/api/src/services/prompts.ts` -- Synthesis system prompt, `prescribe_exercise` tool definition
- `apps/api/src/do/session-brain.ts` -- Synthesis trigger (`finalizeSession`) and safety net

---

## 5. Exercise Generation

### What It Does

After identifying that pedaling in bars 20-24 is the key issue, the teacher optionally prescribes a targeted drill. Drills are grounded in the student's actual piece and passage, target the identified weakness, and give a concrete physical action -- either the student's own passage looped at reduced tempo (`own_passage_loop`) or a matched catalog primitive (`corpus_drill`, see `docs/apps/04-exercises.md`). Drills launch from a mark on the score or session timeline ("work on this"), not from a chat message; there is no `exercise_set` chat artifact card.

### Inputs / Outputs

- **In:** Teaching moment / mark (dimension + passage), exercise catalog (curated entries from D1), synthesis context
- **Out:** `prescribe_exercise` tool call emitting an `ExerciseRoutingDecision` (`own_passage_loop | corpus_drill`) or no tool call
- **Depends on:** Teaching moment selection (which dimension/passage), piece context (which piece/bars)

### How It Works

The teacher LLM has access to the `prescribe_exercise` tool via provider tool_use (`tool_choice: "auto"`). The teacher decides autonomously whether a drill would help more than the verdict alone.

When invoked:
1. Teacher emits an `ExerciseRoutingDecision`: `target_dimension`, `bar_range`, `tempo_factor`, and (for `corpus_drill`) a selected primitive
2. `own_passage_loop` needs no catalog lookup -- it loops the student's own passage
3. `corpus_drill` resolves a primitive from the exercise catalog via FILTER -> RANK -> ADAPT (`docs/apps/04-exercises.md`)
4. The decision is persisted to `pending_exercises` and surfaced as a "work on this" entry from the relevant mark, not rendered in a chat thread

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

`ExerciseRoutingDecision` (`own_passage_loop | corpus_drill`) shipped and superseded the free-text `create_exercise` tool (#29); FILTER/ADAPT shipped, RANK built but not wired (see `docs/apps/04-exercises.md` for the full status ladder). Exercise assignment/completion endpoints implemented and tested. Drill launch now happens from a mark rather than inline chat (#158); Focus mode (multi-exercise guided sequences) remains deferred and unbuilt. Tool invocation frequency and quality against the mark-launched contract have not been re-measured since the redesign.

### Key Files

- `apps/api/src/services/exercises.ts` -- Catalog, assignment, completion endpoints
- `apps/api/src/services/corpus-drill.ts` -- `corpus_drill` primitive selection and score-clip assembly
- `apps/api/src/services/pending-exercise.ts` -- `pending_exercises` persistence
- `apps/api/src/services/prompts.ts` -- `prescribe_exercise` tool definition, teacher prompt with catalog
- `apps/api/src/harness/artifacts/exercise-routing.ts` -- `ExerciseRoutingDecision` type and validation

---

## 6. Score Following & Bar Analysis

### What It Does

When the system says "bars 20-24," those are actually the bars the student was playing. When it says "the crescendo peaked too early," there actually is a crescendo marking in the score at that location. Bar references are grounded in the sheet music, not hallucinated by the LLM.

### Inputs / Outputs

- **In:** AMT MIDI (performance notes), score MIDI (from R2), previous follower state (cross-chunk continuity)
- **Out:** `ChunkAnalysis { bar_range, tier, dimensions: { per-dim musical facts } }` with three tiers of detail
- **Depends on:** Piece identification (must know which piece to load score for), AMT transcription quality

### How It Works

**Tier 1 (full score context):** DTW aligns performance MIDI onsets against score MIDI onsets. Maps each performance note to a bar number. Computes per-bar statistics: velocity mean/std vs reference, onset deviation, pedal duration, note duration ratios. Produces bar-level musical facts per dimension.

**Tier 2 (absolute MIDI, no score):** When piece is unidentified or DTW confidence is low. Analyzes AMT MIDI features directly: velocity contour, inter-onset intervals, pedal density. No bar references, but still provides dimensional analysis.

**Tier 3 (scores only):** When AMT fails entirely. Only the 6-dim scoring output is available (Transkun/MPM -> MoonBeam-839M, currently the deterministic mock scorer, #162). No MIDI analysis. Teaching moments based purely on score deviations from baseline.

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

- `apps/api/src/wasm/score-analysis/src/{chroma_dtw.rs,note_align.rs,bar_analysis.rs}` -- DTW alignment, cross-chunk state, bar-level dimensional analysis, tier logic (Rust/WASM)
- `apps/api/src/services/bar-analysis-facts.ts` -- TS-side bar analysis fact building
- `apps/api/src/services/scores.ts` -- Score/reference loading from R2

---

## Capability Dependencies

```
AMT Transcription
  |
  +---> [1] Piece Identification ---> Score Context loads
  |                                        |
  +---> [3] Practice Mode Detection        |
  |       (pitch bigrams, bar ranges)      |
  |                                        v
  +---> [6] Score Following <-------- Score MIDI
  |       (bar alignment, tier analysis)
  |
MPM Features -> 6-dim Scores (mock scorer now; MoonBeam-839M when the head ships, #162)
  |
  +---> [2] Teaching Moment Selection <--- Student Baselines
          |       (deviation-magnitude gate)    (from Memory/D1)
          v
        SessionAccumulator
          |
          +---> [4] Session Synthesis (verdict + marks + carry-forward)
          |       |     <--- Piece Context, Drilling Records
          |       v
          +---> [5] Exercise Generation (prescribe_exercise -> mark "work on this")
```

Failures cascade downward. If AMT transcription is poor, piece ID fails, score following has no score to follow, and synthesis loses all musical grounding. If no chunk falls below baseline, no corrective teaching moments accumulate (a positive moment is surfaced instead). The eval must test each layer independently AND the full cascade.
