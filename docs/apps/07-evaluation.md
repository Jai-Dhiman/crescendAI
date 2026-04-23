# Evaluation Framework

> Status: 2026-03-26 -- Comprehensive eval design for all seven capabilities. Two tiers per capability: ideal eval (with perfect data) and practical eval (with T5 YouTube Skill corpus).

This document defines how to evaluate each of the seven capabilities described in `06-capabilities.md`. For each capability, we describe:

- **Ideal eval:** What we'd measure with unlimited resources and perfect data (expert teacher annotations, multi-session longitudinal data, studio-quality recordings)
- **Practical eval (T5):** What we can measure now with the T5 YouTube Skill corpus (361 recordings, 4 pieces, 5 skill buckets, consumer microphones, intermediate students)

The practical eval is designed to run end-to-end through the synthesis pipeline, not the legacy per-observation path.

---

## Test Corpus: T5 YouTube Skill

**What it is:** 361 recordings of intermediate piano students on YouTube, playing 4 pieces:
- Bach Invention No. 1 in C Major
- Bach Prelude in C Major (WTC Book 1)
- Fur Elise (Beethoven)
- (4th piece TBD from scenario files)

**Why it matters:** These are real students on real upright pianos with consumer microphones -- the exact target population for CrescendAI. They are NOT masterclass performers (MuQ training data) or competition-level pianists (T2 data).

**Labeling:** Each recording has a `skill_level` label (1-5 buckets) assigned during data collection. Bucket 1 = beginner/weak intermediate, bucket 5 = advanced intermediate/early advanced.

**Inference cache:** MuQ scores and Aria-AMT transcriptions are pre-computed and cached per recording. The eval pipeline replays cached data through the API's WebSocket pipeline, avoiding repeated inference costs.

---

## Eval Pipeline Architecture

All seven evals share a common data flow. The eval client sends cached chunks through the **production synthesis path** (not the legacy `is_eval_session` observation path):

```
For each T5 recording:
  Pass A (with piece_query):
    1. POST /api/practice/start -> session_id
    2. WS connect -> send set_piece { piece_query: "Bach Invention 1" }
    3. For each cached chunk: send eval_chunk { predictions, midi_notes, pedal_events }
    4. Wait for chunk_processed acknowledgment
    5. Send end_session
    6. Capture: synthesis text, accumulated state, piece ID result
    7. Close WS

  Pass B (without piece_query):
    1-4. Same, but skip set_piece message
    5-7. Same capture

  Offline analysis:
    - Statistical metrics (no LLM): STOP, piece ID, mode detection, bar analysis
    - Quality metrics (LLM judge): synthesis, exercises, teaching moment selection
    - Delta metrics: compare Pass A vs Pass B outputs
```

**Key change from current eval:** The eval client must route through the production accumulation path. This means either:
- Remove the `is_eval_session` flag (eval_chunk messages go through production accumulation)
- Or add a new message type (e.g., `eval_chunk_synthesis`) that explicitly routes to the production path

The synthesis is triggered by `end_session`, captured via WebSocket `synthesis` event.

### Eval Client Modifications

The `pipeline_client.py` WebSocket client needs:
1. **New capture target:** `synthesis` message type (replacing `observation` list)
2. **Accumulator state capture:** The DO should include accumulator snapshot in synthesis response for offline analysis (teaching moments, mode transitions, drilling records)
3. **Two-pass mode:** Run each recording twice (with/without piece_query)
4. **Piece ID capture:** Already implemented for `piece_identified` messages

### Session Isolation Fix

The current eval creates fresh sessions via `POST /api/practice/start`. Each session maps to a unique Durable Object ID. The identical-STOP-values bug suggests either:
- The inference cache is returning the same data for different recordings (cache key collision)
- The DO is reusing in-memory state across sessions (unlikely given unique IDs, but verify)

**Fix:** Add a cache integrity check before each recording: verify `recording_id` in cache matches the scenario's `video_id`. Log a warning and skip on mismatch.

---

## 0. Signal Ablation (White-Noise Test)

Eval #0 because it answers a load-bearing question that invalidates everything below if it fails: **are MuQ and AMT signals actually load-bearing in synthesis, or is the teacher LLM producing defensible-sounding output from language priors alone?**

### Motivation

From the Mahler wiki's Music AI Systems page: the MuChoMusic benchmark exposed that most audio-language models answered music questions by relying on language priors rather than processing audio. Replacing the audio clip with white noise produced no statistically significant drop in accuracy for most models tested. For CrescendAI this translates to a direct risk: the two-stage pipeline may be producing teaching observations that are internally-consistent narratives about the piece, the student baseline, and the session arc, without the MuQ 6-dim scores or AMT midi_notes actually conditioning the output.

If that is happening, every downstream eval measures the wrong thing and every hour of Phase B/C model work is miscalibrated.

### Design

- **Corpus:** 20 T5 sessions (same scenarios as the other evals).
- **Pass A (real):** Run current synthesis path with real cached signals.
- **Pass B (ablated):** Run synthesis path with signals substituted. Two substitution strategies, each as a separate pass:
  - **B1 (shuffled):** MuQ 6-dim vectors + AMT midi_notes from a different random session in the corpus, preserving marginal distribution but breaking within-session consistency.
  - **B2 (marginal):** MuQ 6-dim vectors sampled IID from the empirical marginal distribution per dimension; AMT midi_notes replaced with a plausible-but-random note stream matching the real tempo.
- **Judges:** Dual-judge pipeline (Gemma-4 + GPT-5.4-mini per Phase 1 plan). Two metrics:
  - Semantic similarity between Pass A and Pass B outputs (if > 0.85, signals are not load-bearing).
  - Judge-detected difference rate ("which output reflects the provided signals more accurately?"). Near-50% means signals are decorative.

### Kill Criteria

- If A vs B1 similarity > 0.85 OR judge difference rate < 60%, pause Model v2 Phase B/C investment until signals are wired into synthesis in a load-bearing way.
- If A vs B2 similarity > 0.90 OR judge difference rate < 55%, same.

### Status

NOT STARTED. Standalone eval -- does not require playbook.yaml wiring or Phase 2 infrastructure. Can run in a day against the existing cached corpus.

---

## Per-Tier Reliability Evals (V5 Pre-Work)

Per-capability evals below measure whole-system behavior. A complementary layer -- added as V5 skills land -- evaluates each **tier** of the three-tier skill catalog (atoms / molecules / compounds) independently. From the Mahler wiki's *Skill Graphs 2*: reliability at every tier is non-trivial; every atom must be solid, every molecule must chain dependably, every compound must stay under its reliability ceiling. Testing the stack as a whole hides which tier failed.

### Atom Reliability

Atoms are near-deterministic. Reliability evals are close to unit tests: given a canonical input signal, does the atom return the expected output within tolerance? Most atoms are computational (velocity curves, onset drift, IOI correlation); a few are retrieval (similar past observation). Autoresearch-style automated eval is a good fit here -- every atom has a precise input/output spec.

### Molecule Reliability

Molecules are the pedagogical moves. Each gets a narrow 5-criterion rubric scored by the Phase 1 dual-judge (Gemma-4 + GPT-5.4-mini). Criteria should be specific and testable, not "quality" or "helpfulness." Rubrics live alongside the molecule file (e.g., `docs/harness/skills/molecules/voicing-diagnosis.md` embeds its rubric).

Rationale: Skill Design wiki argues that composite task evals conflate failure modes and produce diffuse feedback. A molecule with pre/post contracts and a narrow rubric lets judges give discriminating signal. Without this layer, judges rate "was the synthesis good," which degrades into style scoring.

### Compound Reliability

Compounds orchestrate molecules under a hook. Compound evals measure end-to-end scenario outcomes: given a T5 session, did `session-synthesis` select the right molecules, dispatch them correctly, and produce a coherent artifact? This is closest to the current whole-system eval but targeted at one compound at a time.

### Production Review Agent (Middleware)

A runtime-level `after_model` middleware that re-scores compound outputs in production using the Phase 1 judge rubric. From *Multi-Agents: What's Actually Working*: the review agent performs better **without** shared context -- it receives only the synthesis artifact + student baselines + rubric, not the raw signals or accumulator. Forced to reason backward from the output, it catches drift that in-context scoring cannot. Runs on 10% of production synthesis traffic. See `docs/apps/02-pipeline.md` Middleware Hooks section.

### Sequencing

1. Signal Ablation (this doc, eval #0) -- runs independently, no dependencies.
2. Playbook.yaml wiring in `run_eval.py` and `apps/api/src/services/prompts.*`.
3. Atom reliability harness -- runs against the V5 atoms/ directory as each is drafted.
4. Molecule rubrics drafted alongside V5 molecule decomposition.
5. Phase 1 dual-judge locked baseline.
6. Compound reliability evals (end-to-end per-compound scenarios).
7. Production review agent wired as `after_model` middleware.
8. Per-capability evals below.

---

## 1. Piece Identification Eval

### Ideal Eval

**Data:** 1,000+ recordings spanning the full 242-piece library, with expert-verified piece labels. Include 200+ recordings of pieces NOT in the library (negative controls). Include partial performances (starting from middle of piece, playing only one section).

**Metrics:**
- Top-1, top-3, top-5 accuracy stratified by piece familiarity (popular vs obscure)
- Notes-to-identify distribution (median, p90, p99)
- False positive rate at multiple confidence thresholds (0.5, 0.7, 0.8, 0.9)
- Confusion matrix: which pieces get confused with which? (e.g., does "Fur Elise" A section match any other piece?)
- Robustness: accuracy as a function of recording quality (studio vs consumer mic vs phone speaker)
- Partial performance: accuracy when student starts from bar 20 or plays only the B section

### Practical Eval (T5)

**Data:** 361 T5 recordings across 4 pieces. Ground truth: piece identity from YouTube title/filename. Need to map each T5 piece to our `pieces` table ID. Recordings of pieces NOT in our library serve as natural negative controls (if any exist in the T5 set).

**Methodology:**
- Pass B only (no piece_query). Capture `piece_identified` WebSocket message.
- Extract: `piece_id`, `confidence`, `method`, `notes_consumed`
- Match against ground truth piece ID

**Metrics (computed offline, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Top-1 accuracy | `correct / total` where `correct = (identified_piece == ground_truth_piece)` | > 80% |
| Mean notes to identify | `mean(notes_consumed)` across correct identifications | < 300 |
| p90 notes to identify | `percentile(notes_consumed, 90)` | < 500 |
| False positive rate | `false_positives / (false_positives + true_negatives)` at confidence > 0.8 | < 10% |
| Per-piece accuracy | Breakdown by piece (Bach Inv 1 vs Fur Elise vs etc.) | identify weak pieces |
| Confidence calibration | Of identifications at confidence > 0.8, what % are correct? | > 90% |

**Limitations:** Only 4 pieces. Cannot test confusion between similar pieces (e.g., Bach Invention 1 vs Invention 4). Limited negative control set. All recordings are full performances, not partial starts.

---

## 2. STOP Classification Eval

### Ideal Eval

**Data:** 500+ recordings with per-chunk annotations from expert piano teachers: "Would you stop the student here? Which dimension?" Gold-standard STOP labels. Multiple annotators per chunk for inter-rater agreement.

**Metrics:**
- AUC against expert labels
- Per-dimension precision/recall (when STOP triggers on "pedaling," how often does the expert agree?)
- Cross-piece calibration: same skill level, different pieces -- do difficulty differences affect STOP appropriately?
- Temporal patterns: does STOP trigger at musically meaningful boundaries (phrase ends, section transitions)?

### Practical Eval (T5)

**Data:** 361 T5 recordings with skill_level labels (1-5). No per-chunk STOP ground truth.

**Methodology:**
- Extract STOP probabilities from every chunk (from the `chunk_processed` responses or accumulator state)
- No per-chunk correctness (no ground truth), but we can measure distributional properties

**Metrics (computed offline, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Spearman rho (STOP rate vs skill) | Correlation between per-recording trigger rate and skill_level | > 0.3 |
| Cohen's d (adjacent buckets) | Effect size between bucket N and N+1 STOP rate distributions | > 0.5 |
| Trigger rate by bucket | `triggered_chunks / total_chunks` per skill bucket | Bucket 1: 50-80%, Bucket 5: 15-40% |
| Dimension distribution | Frequency of each dimension as top_dimension across all triggers | No dimension > 50% |
| Per-piece STOP rate | Same metric broken down by piece | Pieces with harder passages have higher rates |
| STOP probability distribution | Histogram of probabilities (should be bimodal, not uniform) | Clear separation |

**Advanced (requires LLM judge):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| STOP-synthesis alignment | When STOP triggers, does the eventual synthesis reference that moment? | > 60% |

**Limitations:** No ground truth for individual chunks. We can only measure whether STOP correlates with skill level (a proxy for "there are more things to teach weaker students"). We cannot measure whether it triggers at the RIGHT moments within a recording.

---

## 3. Teaching Moment Selection Eval

### Ideal Eval

**Data:** 200+ sessions where expert piano teachers have annotated: "The most important thing to teach this student about this performance is [X dimension] in [Y passage] because [Z reason]." Multiple experts per session for agreement measurement.

**Metrics:**
- Dimension agreement: does the system pick the same dimension as the expert?
- Passage agreement: does the system flag the same bars?
- Framing agreement: does the system's correction/recognition/encouragement match what the expert would choose?
- Ranking quality: if the expert lists top-3 priorities, is the system's top-1 in that set?

### Practical Eval (T5)

**Data:** 361 T5 recordings. Accumulated teaching moments from SessionAccumulator (captured after full chunk replay).

**Methodology:**
- Extract all accumulated teaching moments from the synthesis pipeline
- Extract `top_moments()` selection (the subset that goes into the synthesis prompt)
- Evaluate selection quality using LLM judge and statistical checks

**Metrics (statistical, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Blind-spot detection | Is selected moment's deviation in bottom 2 of all triggered moments? | > 60% |
| Positive moment validity | For `is_positive=true` moments, is deviation > +0.15? | > 90% |
| Framing-deviation alignment | Correction when deviation < -0.1, recognition when > +0.1 | > 85% |
| Dedup effectiveness | No dimension appears > 2x in top_moments() for one session | > 90% |
| Moment diversity | Number of unique dimensions in top_moments() / total selected | > 0.5 |

**Metrics (LLM judge):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Dimension-piece fit | "Given [piece], is [dimension] among the most important to address?" Judge sees all 6 dimensions + scores. | > 70% |
| Moment ranking quality | "Given these 5 teaching moments, is the top-selected one the most pedagogically important?" Judge ranks them independently. | Top-1 in judge's top-3 > 60% |

**Judge prompt (dimension-piece fit):**
```
You are an experienced piano teacher. A student just performed [piece_name] by [composer].

The system identified these teaching opportunities (dimension: score, deviation from baseline):
[list all accumulated moments with dimension, score, deviation]

The system selected [selected_dimension] as the most important.

Is [selected_dimension] among the top 2 most important dimensions to address for this piece and this student's performance? Consider:
- The musical demands of the piece (e.g., pedaling matters more in Chopin than Bach)
- The magnitude of the deviations (larger negative = bigger issue)
- The student's overall profile (are they weak across the board or strong with one blind spot?)

Answer: YES or NO, with a one-sentence justification.
```

**Limitations:** No expert ground truth for "what should be taught." The LLM judge is a proxy, not a piano teacher. Dimension-piece fit is a heuristic (not all Bach pieces need articulation focus). The judge has no audio context -- it evaluates based on scores and piece identity only.

---

## 4. Practice Mode Detection Eval

### Ideal Eval

**Data:** 100+ full practice sessions (20-60 minutes each) with human-annotated mode labels per chunk: warming, drilling (with bar range), running, regular, winding. Video recordings to verify behavior (student stops, rewinds, repeats).

**Metrics:**
- Per-mode precision and recall
- Mode transition latency (chunks between behavior change and detection)
- Drilling passage bar range accuracy (IoU against annotated range)
- Inter-annotator agreement on mode labels

### Practical Eval (T5)

**Data:** 361 T5 recordings. Most are single performances (not full practice sessions), which limits mode detection testing. However, some YouTube practice videos include drilling sections.

**Methodology:**
- Compute ground truth proxy from AMT MIDI offline:
  - **Drilling proxy:** Consecutive chunks where pitch bigram Dice > 0.6 AND bar overlap > 0.5. Label as "ground truth drilling."
  - **Run-through proxy:** 6+ consecutive chunks with monotonically advancing bar ranges (each chunk's start bar > previous chunk's start bar).
  - **Warming proxy:** First 2-3 chunks of any recording (before stable pattern emerges).
- Compare mode detector output against these proxies

**Metrics (computed offline, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Drilling precision | `true_drilling / detected_drilling` (vs proxy ground truth) | > 70% |
| Drilling recall | `detected_drilling / true_drilling` (vs proxy ground truth) | > 60% |
| Run-through detection | `correct_running / proxy_running` | > 75% |
| Mode transition count | Total transitions per recording (should be low for simple performances) | < 5 for single play-throughs |
| Spurious transitions | Transitions that reverse within 2 chunks (Drilling->Running->Drilling in 30s) | < 10% of all transitions |
| Drilling bar range overlap | IoU between detected drilling passage and proxy passage | > 50% |

**What feeds into synthesis eval:** If mode detection says "drilling detected in bars 20-28" and the synthesis says "you drilled bars 20-28," we can verify the claim. If mode detection is wrong, the synthesis propagates the error.

**Limitations:** T5 recordings are mostly single performances, not full practice sessions. Drilling detection may have very few positive examples in the corpus. The proxy ground truth (computed from the same AMT MIDI the mode detector uses) creates circular validation -- high agreement might just mean both algorithms make the same assumptions. Ideally, drilling ground truth would come from video annotation.

---

## 5. Session Synthesis Eval

### Ideal Eval

**Data:** 200+ sessions with expert piano teacher reviews. Each expert listens to the same recording the system heard, then:
1. Writes their own 3-6 sentence feedback (gold standard)
2. Rates the system's synthesis on each criterion (1-5 scale)
3. Identifies factual errors (wrong bar references, wrong dimension claims)

**Metrics:**
- Per-criterion expert ratings (grounding, actionability, calibration, coherence)
- Factual error rate (bar references that don't match the recording)
- BLEU/ROUGE against expert feedback (directional, not definitive)
- Student preference: A/B test system synthesis vs expert feedback

### Practical Eval (T5)

**Data:** 361 T5 recordings, each producing one synthesis (Pass A: with piece_query) and one synthesis (Pass B: without piece_query).

**Methodology:**
- Run full pipeline: all chunks -> accumulation -> end_session -> capture synthesis + accumulator state
- LLM judge evaluates each synthesis against criteria
- Compare Pass A vs Pass B to quantify piece context impact

**Two-pass design:**
- **Pass A (with piece_query):** Tests synthesis quality with full score context. Isolates synthesis prompt quality from piece ID accuracy.
- **Pass B (without piece_query):** Tests real-world experience. Piece ID may or may not succeed. If it fails, synthesis runs without bar references.
- **Delta:** Pass A score - Pass B score = impact of piece identification on feedback quality.

**Metrics (LLM judge, per-synthesis):**

| Criterion | Judge prompt | Pass/Fail threshold |
|-----------|-------------|-------------------|
| Musical Grounding | "Does this synthesis reference at least one specific passage, bar range, or musical event from the performance?" | Binary: references specific moment = PASS |
| Actionability | "Could the student identify one concrete thing to practice differently based on this feedback? Not vague encouragement ('keep it up') but a specific action ('try lifting the pedal at each bass note change')." | Binary |
| Differentiation | (Pairwise, not per-synthesis -- see below) | |
| Skill Calibration | "The student is at skill level [N] (1=beginner, 5=advanced). Is the language complexity and musical sophistication of this feedback appropriate for their level?" | Binary |
| Coherence | "Does this read as one teacher's cohesive thought about the session, or as a disconnected list of dimension-by-dimension comments?" | Binary |
| Drilling Acknowledgment | "The session included [N] drilling episodes on [passage]. Does the synthesis mention or acknowledge the drilling practice?" (Only evaluated when drilling detected.) | Binary |
| Score Accuracy | "The synthesis references [bar range / passage]. The system's DTW analysis mapped the performance to [actual bar range]. Do these match?" (Cross-referenced against accumulator data.) | Binary |

**Aggregate metrics:**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Grounding pass rate | % of syntheses that pass Musical Grounding | Pass A: > 70%, Pass B: > 50% |
| Actionability pass rate | % that pass Actionability | > 75% |
| Skill calibration pass rate | % that pass Skill Calibration | > 70% |
| Coherence pass rate | % that pass Coherence | > 85% |
| Drilling acknowledgment rate | % that mention drilling when it occurred | > 85% |
| Score accuracy rate | % of bar references that match DTW output | > 75% |
| Piece context delta (grounding) | Pass A grounding - Pass B grounding | measured (expect > 0.2) |
| Piece context delta (actionability) | Pass A actionability - Pass B actionability | measured |
| Mean synthesis length | sentence count | 3-8 |

**Differentiation test (special methodology):**

Select 3 recordings of the same piece at skill levels 1, 3, and 5. Run synthesis on all three. Present all three syntheses to the LLM judge:

```
Three students at different skill levels played the same piece ([piece_name]).
Their syntheses were:

Student A (skill level 1): [synthesis_1]
Student B (skill level 3): [synthesis_3]
Student C (skill level 5): [synthesis_5]

Are these three syntheses substantively different in:
1. What they address (different dimensions/issues)?
2. How they frame feedback (different sophistication)?
3. What they suggest practicing?

Rate differentiation: HIGH (clearly different feedback), MEDIUM (some differences), LOW (could be interchanged).
```

Target: > 70% rated HIGH or MEDIUM.

**Limitations:** LLM judge has no audio context -- it can't verify "your pedaling was muddy" by listening. It evaluates structural quality (does the synthesis reference specific things, is it actionable) not musical correctness (is the pedaling actually muddy). The differentiation test requires matched triplets across skill levels for the same piece, which may limit sample size.

---

## 6. Exercise Generation Eval

### Ideal Eval

**Data:** 100+ synthesized exercises reviewed by piano teachers. Each teacher:
1. Rates whether the exercise targets the identified weakness
2. Rates whether a student could physically follow the instructions
3. Attempts the exercise themselves and rates whether it would actually help
4. Compares curated vs LLM-generated exercise quality

**Metrics:**
- Expert agreement on exercise relevance (dimension targeting)
- Feasibility rating (can a student do this?)
- Pedagogical value (would this actually help?)
- Curated vs generated preference rate

### Practical Eval (T5)

**Data:** Subset of 361 T5 recordings where the teacher LLM invokes `create_exercise` during synthesis.

**Methodology:**
- Run synthesis pipeline with exercise tool enabled
- Capture tool invocations (or lack thereof)
- Parse exercise artifacts for structural analysis
- LLM judge evaluates exercise quality

**Metrics (statistical, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Invocation rate | `sessions_with_exercise / total_sessions` | 20-40% |
| Dimension targeting | `exercise.focus_dimension == top_teaching_moment.dimension` | > 85% |
| Passage grounding | `exercise.source_passage` is non-empty and references specific bars | > 60% |
| Catalog utilization | When a curated exercise matches, is it referenced? | > 40% |
| Exercise count per invocation | Number of exercise steps (target: 1-3) | mean 1.5-2.5 |

**Metrics (LLM judge):**

| Criterion | Judge prompt | Target |
|-----------|-------------|--------|
| Actionability | "Could a pianist at skill level [N] follow these instructions and know exactly what to do?" | > 80% pass |
| Specificity | "Rate 1-5: Is this exercise specific to this piece and passage (5) or a generic drill (1)?" | mean > 3.0 |
| Dimension alignment | "The system identified [dimension] as the key weakness. Does this exercise directly address that dimension?" | > 85% pass |

**Limitations:** If the teacher rarely invokes the tool (< 20% of sessions), the exercise eval has very few samples. We should NOT force tool invocation -- the invocation rate itself is a metric. If the sample is too small, we can run a separate exercise-focused eval that artificially includes exercise tool in the system prompt instruction.

---

## 7. Score Following & Bar Analysis Eval

### Ideal Eval

**Data:** 100+ recordings with expert-annotated bar-level timestamps: "At time 0:15, the student begins bar 12." Ground truth bar maps from manual alignment of performance to score.

**Metrics:**
- Bar alignment accuracy: mean absolute error in bar numbers (system vs expert)
- Onset alignment accuracy: mean absolute error in milliseconds for note onset mapping
- Tier classification accuracy: does the system choose the right analysis tier?
- Cross-chunk consistency: bar numbers never regress during run-throughs

### Practical Eval (T5)

**Data:** 361 T5 recordings (Pass A only -- requires piece_query for score context).

**Methodology:**
- Extract per-chunk `ChunkAnalysis` from accumulator state (bar_range, tier, dimensional facts)
- Validate structural properties (no LLM needed for most metrics)
- Cross-reference bar references in synthesis text against accumulator data

**Metrics (computed offline, no LLM):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Bar range coverage | % of chunks that have a non-null bar_range | > 70% (Tier 1) |
| Tier distribution | % of chunks at Tier 1 / 2 / 3 | Tier 1 > 60% when piece identified |
| Cross-chunk monotonicity | During detected run-throughs: bar_range.start monotonically non-decreasing | > 90% |
| Bar range plausibility | bar_range.end - bar_range.start is 2-16 bars (reasonable for 15s chunk) | > 85% |
| Drilling bar consistency | During detected drilling: bar ranges overlap across repetitions | > 70% |
| Reference profile utilization | When reference data exists, is it included in analysis | > 80% |

**Cross-component validation (LLM-assisted):**

| Metric | How to compute | Target |
|--------|---------------|--------|
| Synthesis-bar accuracy | When synthesis text mentions "bars X-Y", does it match accumulator bar_range for the relevant teaching moment? | > 80% |
| Hallucinated bar references | Does synthesis mention bar numbers that appear nowhere in the accumulator data? | < 5% |

**Limitations:** No ground truth bar alignment. We can only verify internal consistency (DTW output matches what synthesis says) not external correctness (DTW output matches what the student actually played). The plausibility checks (2-16 bars per chunk, monotonic in run-throughs) are structural heuristics, not correctness tests.

---

## Eval Execution Plan

### Prerequisites

1. **Inference cache populated:** MuQ + AMT cached for all 361 T5 recordings
2. **API running locally:** `just dev-light` or `just dev-muq` with wrangler dev
3. **Ground truth mapping:** T5 piece titles mapped to `pieces` table IDs
4. **Cache integrity check:** Verify no recording_id collisions in cache loader

### Execution Order

**Phase 1: Fix and validate plumbing (1-2 days)**
- Fix eval client to route through production synthesis path (not `is_eval_session` observation path)
- Add accumulator state capture to synthesis WebSocket response
- Add cache integrity check (recording_id matches video_id)
- Validate: run 3 recordings, verify different STOP values and different synthesis texts

**Phase 2: Statistical metrics -- no LLM cost (1 day)**
- Run all 361 recordings (Pass A: with piece_query)
- Extract and compute: STOP metrics, piece ID metrics (Pass B subset), mode detection metrics, bar analysis metrics
- Report: per-capability metric tables with confidence intervals

**Phase 3: Quality metrics -- LLM judge (2-3 days)**
- Run Pass B (without piece_query) for all 361 recordings
- Design and validate judge prompts (test on 10 samples, check inter-run agreement)
- Run full judge evaluation: synthesis quality, teaching moment selection, exercise quality
- Compute piece context delta (Pass A vs Pass B)

**Phase 4: Analysis and iteration (ongoing)**
- Identify worst-performing capabilities
- Drill into failure modes (why did synthesis fail actionability? Was it STOP selection? Prompt? Missing context?)
- Iterate on prompts, thresholds, algorithms
- Re-run eval to measure improvement

### Cost Estimate

| Component | Per recording | 361 recordings | 722 recordings (both passes) |
|-----------|-------------|-----------------|------------------------------|
| Synthesis LLM (Anthropic Sonnet) | ~$0.03 | ~$11 | ~$22 |
| Judge LLM (5 criteria x $0.01) | ~$0.05 | ~$18 | ~$36 |
| Piece ID (no LLM, computed) | $0 | $0 | $0 |
| STOP/mode (no LLM, computed) | $0 | $0 | $0 |
| **Total** | ~$0.08 | ~$29 | ~$58 |

Inference costs are zero (cached). The eval is LLM-judge-dominated but still under $60 for the full suite.

### Judge Reliability

LLM judges have variance. To calibrate:
1. Run 20 syntheses through the judge 3 times each
2. Measure inter-run agreement (expect > 85% on binary criteria)
3. If agreement < 80% on any criterion, tighten the judge prompt or switch to a scale (1-5) instead of binary

The memory system eval found sigma ~0.036 across 3 runs -- similar variance expected here.

---

## Appendix: Judge Prompt Templates

### Synthesis: Musical Grounding
```
You are evaluating a piano teaching assistant's session synthesis.

The student played: [piece_name] by [composer]
Student skill level: [skill_level] (1=beginner intermediate, 5=advanced intermediate)

The system's synthesis:
---
[synthesis_text]
---

Does this synthesis reference at least one specific musical passage, bar range,
section, or musical event from the performance? Vague references like "the middle
section" count only if they identify a distinguishable part of the piece. Generic
statements like "your playing" or "the piece" do not count.

Answer PASS or FAIL with a one-sentence justification.
```

### Synthesis: Actionability
```
You are evaluating a piano teaching assistant's session synthesis.

The student played: [piece_name] by [composer]
Student skill level: [skill_level]

The system's synthesis:
---
[synthesis_text]
---

Could the student identify one concrete thing to practice differently based on this
feedback? A concrete action is a physical change the student can make at the piano
(e.g., "lift the pedal at each bass note change", "play the RH melody alone at
pianissimo"). Vague encouragement ("keep working on it", "trust your instincts")
or abstract advice ("focus on dynamics") does NOT count.

Answer PASS or FAIL with a one-sentence justification.
```

### Synthesis: Skill Calibration
```
You are evaluating whether feedback is appropriately calibrated to a student's level.

Student skill level: [skill_level] (1=beginner intermediate, 5=advanced intermediate)
Piece: [piece_name] by [composer]

The system's synthesis:
---
[synthesis_text]
---

Is the language complexity and musical sophistication appropriate for this skill level?
- Level 1-2: Should use accessible language, focus on fundamentals (hand position,
  basic dynamics, pedal on/off), avoid advanced concepts (voicing, rubato, tonal color)
- Level 3: Can reference standard musical concepts, moderate detail
- Level 4-5: Can use advanced terminology, discuss interpretation choices, reference
  performance practice conventions

Answer PASS or FAIL with a one-sentence justification.
```

### Synthesis: Coherence
```
You are evaluating the coherence of a piano teaching session synthesis.

The system's synthesis:
---
[synthesis_text]
---

Does this read as one teacher's cohesive thought about the practice session, or as a
disconnected list of dimension-by-dimension comments? A cohesive synthesis has a
narrative arc (e.g., "you focused on X, here's what improved, here's what to try next").
A disconnected one reads like: "Dynamics: good. Timing: needs work. Pedaling: OK."

Answer PASS or FAIL with a one-sentence justification.
```

### Teaching Moment: Dimension-Piece Fit
```
You are an experienced piano teacher. A student just performed [piece_name] by [composer].

The system detected these teaching opportunities across the session:
[For each accumulated moment: dimension, score (0-1), deviation from baseline, is_positive]

The system selected [selected_dimension] (deviation: [deviation], score: [score]) as
the most important teaching focus.

Is [selected_dimension] among the top 2 most important dimensions to address for this
piece? Consider:
- The musical demands of [piece_name] (e.g., pedaling is critical in Chopin nocturnes,
  articulation in Bach inventions, dynamics in Beethoven sonatas)
- The magnitude of the deviations (larger negative = bigger issue)
- Whether another dimension with a larger deviation would be more impactful

Answer YES or NO with a one-sentence justification.
```

### Exercise: Actionability
```
A piano teaching system generated this exercise for a skill level [skill_level] student
working on [piece_name]:

Exercise:
---
Source passage: [source_passage]
Target skill: [target_skill]
Instructions: [exercise_instructions]
---

Could a pianist at this skill level follow these instructions and know exactly what
to do at the piano? The instructions should specify: what to play (which hand, which
passage), how to play it (specific technique change), and what to listen for.

Answer PASS or FAIL with a one-sentence justification.
```

### Differentiation (Pairwise)
```
Three students at different skill levels played [piece_name] by [composer].
Their session syntheses were:

Student A (skill level 1):
---
[synthesis_1]
---

Student B (skill level 3):
---
[synthesis_3]
---

Student C (skill level 5):
---
[synthesis_5]
---

Are these three syntheses substantively different in:
1. What issues they address (different dimensions or passages)?
2. How they frame feedback (different sophistication or expectations)?
3. What they suggest practicing (different exercises or focus areas)?

Rate: HIGH (clearly different feedback tailored to each level), MEDIUM (some differences
but significant overlap), LOW (could be interchanged without noticing).

Provide your rating and a brief justification.
```
