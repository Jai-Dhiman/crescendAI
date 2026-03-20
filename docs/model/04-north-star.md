# North Star: The Perfect Pipeline

Vision document for the ideal piano performance evaluation system, from recording to actionable feedback. Captures the complete 8-stage pipeline, phased implementation roadmap, and the rationale behind every architectural decision.

> **Status (2026-03-19):** Vision document. **Phase 1 (score infrastructure) COMPLETE.** Phase 0+3 collapsed into "Model v2" (Aria + MuQ + gated fusion). Clean-fold baseline optimized: **A1-Max 79.85% pairwise, R2=0.336** (4-fold mean, optimized weights). E1 audio gate PASSED (>75%). See `03-encoders.md` for encoder details and Aria architecture.

---

## The Core Insight

The current system evaluates performance quality in absolute terms. The perfect system evaluates quality *relative to what the score asks for.* This single shift fixes the dynamics inversion (rho=-0.917 in competition -- the model captures "amount" not "appropriateness"), enables rubato detection, and transforms feedback from "dynamics score 0.35" to "the crescendo in bars 12-16 doesn't reach the forte Chopin marked."

80% of the user-facing improvement comes from giving the LLM better context (bar-aligned musical facts, reference comparisons), not from changing the model itself. The remaining 20% comes from Aria + MuQ gated fusion with score conditioning that evaluates relative to the score at the architecture level.

---

## The Perfect Pipeline (8 Stages)

```
USER PRESSES RECORD
        |
        v
 STAGE 0: CONTEXT PRELOAD
   Score MIDI + reference performances + student history
   Score-conditioned expectations per bar
        |
        v
 STAGE 1: AUDIO CAPTURE + LIVE AMT + SCORE FOLLOWING
   Audio stream -> MuQ (quality scores)
   Audio stream -> AMT (performance MIDI)
   Performance MIDI + score MIDI -> bar positions
   Phrase-aware segmentation (musical units, not fixed 15s)
        |
        v
 STAGE 2: MULTI-MODAL SCORING (Aria + MuQ)
   z_audio  = MuQ + LoRA (audio quality embedding)
   z_perf   = Aria (performance MIDI embedding)
   z_score  = Aria (score MIDI embedding)
   delta    = z_perf - z_score (deviation from written intent)
   Per-dimension gated fusion with learned routing
   Output: 6 relative dimension assessments (quality vs. what was written)
        |
        v
 STAGE 3: TEMPORAL REASONING
   Rubato detection (compensatory return analysis)
   Passage repetition tracking (attempt 1 vs attempt 5)
   Within-session trajectory (warm-up, fatigue, improvement)
        |
        v
 STAGE 4: STRUCTURED MUSICAL ANALYSIS
   Bar-aligned score comparison: "bars 12-16: velocity reaches 70% of reference"
   MIDI-derived features: velocity curves, onset deviations, pedal durations
   Difficulty-aware context: calibrate expectations per passage
        |
        v
 STAGE 5: MULTI-SIGNAL TEACHING MOMENT SELECTION
   Score-conditioned STOP classifier
   Blind spot detection (vs. student baselines)
   Positive moment detection (breakthroughs, improvements)
   Musical priority weighting (Chopin -> pedaling, Bach -> articulation)
   Passage repetition awareness (don't judge a drill like a performance)
   Novelty constraint (don't say the same thing 3 times in a row)
        |
        v
 STAGE 6: SUBAGENT REASONING
   5-step framework: learning arc, delta, musical context, selection, framing
   Input: bar-aligned musical facts (not raw scores)
   Output: structured handoff + narrative reasoning
        |
        v
 STAGE 7: TEACHER VOICE
   Natural 1-3 sentence observation referencing specific bars
   Quote bank for embodied language
   Modality declaration (text, score highlight, keyboard guide, exercise)
        |
        v
 STAGE 8: RICH DELIVERY
   Score highlight (bars 20-24 on rendered score)
   Reference clip (timestamped link to professional recording)
   Practice exercise (specific drill for this weakness)
   Progress chart (sparkline across sessions)
```

---

## The Complete Perception System (8 Capabilities)

The model v2 system targets 8 perceptual capabilities, from immediate inference outputs to downstream reasoning:

| # | Capability | Description | Encoder | Status |
|---|-----------|-------------|---------|--------|
| 1 | **AMT** | Automatic music transcription (audio -> MIDI) | ByteDance piano transcription | DEPLOYED |
| 2 | **Piece identification** | Fuzzy matching against score library | Score following (DTW) | COMPLETE |
| 3 | **Quality assessment** | 6-dimension relative quality scoring | MuQ + Aria (gated fusion) | BASELINE ESTABLISHED (clean folds: 77.5% pairwise, optimized weights found) |
| 4 | **Skill level** | Beginner/intermediate/advanced classification | MuQ + Aria (ordinal training) | NOT STARTED (A1-Max has zero discrimination) |
| 5 | **STOP detection** | "Would a teacher stop here?" | Logistic regression on scores | COMPLETE (needs retrain on clean folds) |
| 6 | **Difficulty estimation** | Per-passage technical difficulty | Score analysis + reference stats | COMPLETE (score infrastructure) |
| 7 | **Temporal reasoning** | Rubato, repetition tracking, trajectory | Score following + onset analysis | NOT STARTED |
| 8 | **Robustness** | Stable scores across audio conditions | AMT validation + calibration | VALIDATED |

---

## Aria + MuQ Gated Fusion (Stage 2 Detail)

### Architecture

```
AUDIO -> [MuQ + LoRA] -> z_audio [512]

PERF MIDI  -> [Aria] -> z_perf [512]
SCORE MIDI -> [Aria] -> z_score [512]
                         delta = z_perf - z_score

                    GATED FUSION (per-dimension)

  For each dimension d:
    gate_d = sigmoid(W_d * [z_audio; z_perf; delta])
    fused_d = gate_d * z_audio + (1 - gate_d) * z_perf
    quality_d = MLP_d(fused_d, delta)

  Output: 6 scores (0-1) relative to score
```

The `delta = z_perf - z_score` vector directly encodes "what's different between what was played and what was written." The quality head learns which deltas are good (rubato, dynamic shading) and which are bad (wrong notes, missed dynamics).

### What Score Conditioning Fixes

| Problem | Current Behavior | With Score Conditioning |
|---------|-----------------|------------------------|
| Dynamics inversion | rho=-0.917 (pianist plays pp = low score) | pp when score says pp = HIGH score |
| Rubato confusion | Any timing deviation = "timing issues" | Deviation + compensatory return = intentional |
| Difficulty blindness | Same standards for easy and hard passages | Calibrated expectations per passage |
| Open-ended pieces | Works for any audio | Requires score MIDI (degrades to absolute for unknown pieces) |

### Training Data: Reference-Anchored

Instead of expensive expert annotation, use professional recordings as implicit quality anchors:

- MAESTRO has 204 pieces with 2+ performers and paired score MIDIs
- Multiple recordings of the same piece = relative quality ordering
- Training triple: (performance_audio, performance_midi, score_midi) with relative ranking label
- Loss: ListMLE ranking (same as A1-Max), conditioned on score
- Data mix: PercePiano as anchor (20%), ordinal competition data (80%)

### Separate-Then-Fuse Protocol

1. **Independent training:** Fine-tune MuQ and Aria separately on PercePiano with clean folds
2. **Quality-aware contrastive pretraining:** Symmetric contrastive training for both encoders (MuQ: NT-Xent on quality pairs; Aria: SimCSE already done, extend with quality pairs)
3. **Error correlation measurement:** Both models on validation sets, per-dimension error correlation. Target: r < 0.5
4. **Gated fusion training:** Freeze encoders, train fusion gates + quality MLPs
5. **End-to-end fine-tuning (optional):** Unfreeze top layers, very low LR (1e-6)

---

## Eval Tier Requirements

### Training Evaluation (E1-E3)

| Tier | Metric | Requirement | Purpose |
|------|--------|-------------|---------|
| **E1** | Pairwise accuracy (clean folds) | > 75% (audio), > 70% (symbolic), > 80% (fused) | Core ranking quality. **Audio: 77.5% PASS (4-fold mean, original weights)** |
| **E2** | Per-dimension error correlation | r < 0.5 between audio and symbolic | Fusion viability gate |
| **E3** | Skill-level discrimination | Cohen's d > 0.8 between adjacent levels | Multi-tier training validation |

### Deployment Validation (E4-E6)

| Tier | Metric | Requirement | Purpose |
|------|--------|-------------|---------|
| **E4** | Competition correlation (Spearman rho) | > 0.6 on held-out competition | External validity |
| **E5** | STOP classifier AUC | > 0.80 on retrained scores | Teaching moment quality |
| **E6** | AMT robustness (pairwise drop) | < 5% across audio conditions | Production reliability |

All evaluation on clean piece-stratified folds. Bootstrap CIs required for all comparisons. See `03-encoders.md` Verification Criteria.

---

## Score Infrastructure (Phase 1 -- COMPLETE)

### Score MIDI Library

Sources (by priority): ASAP (242 pieces, V1), MAESTRO (~300 pieces, needs external score sourcing), IMSLP MIDI collection (~5K), MuseScore corpus (~100K+), Kern Scores (~3K).

Start with ASAP score MIDIs (242 pieces, covers standard classical repertoire). Expand to MAESTRO and external sources after V1 proves useful. Graceful degradation to absolute scoring for unknown pieces. Track "piece not found" events to prioritize additions.

Per-score data model (V1, MIDI-only): bar structure, tempo markings, pedal events (CC64), key/time signatures, per-bar note data (pitch, velocity, onset, duration, track). Richer annotations (explicit dynamics text like pp/ff/cresc., articulation marks, section labels, difficulty annotations per bar) require MusicXML import -- deferred to a future enrichment pass.

Design spec: `docs/superpowers/specs/2026-03-14-score-midi-library-design.md`

### Cloud AMT Service

ByteDance piano transcription (validated: 0% pairwise drop on MAESTRO, 79.9% agreement on mediocre YouTube audio) deployed alongside MuQ on HF endpoint. Single upload, two outputs (scores + MIDI). Total endpoint latency stays under 2s.

### Score Following

Onset-based DTW between AMT output and score MIDI. Produces bar map: `[{chunk_offset, bar, beat}]`. Re-anchors when lost (DTW cost spike > threshold for 3+ consecutive notes: search forward in score for matching onset pattern).

### Bar-Aligned Musical Analysis Engine

Transforms model scores + AMT + score into structured musical facts per passage:

- Dynamics: score marking, performance velocity, reference velocity, analysis sentence
- Timing: score tempo, performance tempo, onset deviation pattern, rubato classification
- Pedaling: score marking, pedal durations, harmony change intervals, bleed analysis
- Articulation: score marking, note duration ratios, overlap analysis
- Difficulty: technical density, note speed, hand span

### Reference Performance Cache

For each piece in the library with MAESTRO recordings: run A1-Max, align to score via DTW, compute per-bar statistics (velocity curves, onset deviation patterns, pedal usage). Cache as `reference_profiles/{piece_id}.json`. When student's performance deviates from ALL references, strong signal. When some professionals do the same, likely valid interpretation.

---

## Temporal Reasoning (Phase 2)

### Rubato Detection

Onset deviation analysis with compensatory return check:

```
deviations[] = performance_onsets - score_onsets

For each phrase:
  first_half_deviation = mean(deviations in first half)
  second_half_deviation = mean(deviations in second half)
  compensatory_ratio = |first + second| / max(|first|, |second|)

  IF ratio < 0.3:  -> INTENTIONAL RUBATO
  ELIF monotonic drift AND |mean| > threshold: -> RUSHING/DRAGGING
  ELSE: -> UNCERTAIN (silence on timing)
```

**Confidence threshold (decided):** Only flag timing issues when: (1) compensatory_ratio > 0.7, AND (2) |mean_deviation| > 100ms, AND (3) pattern persists across 2+ phrases. Otherwise: silence. Better to miss a real issue than destroy trust by calling beautiful rubato a timing problem.

### Passage Repetition Tracking

Detect overlapping bar ranges across recent chunks (>60% overlap = repetition). Track per-attempt scores and compute improvement trends. Enables: "Your pedaling improved from 0.28 to 0.38 across 5 attempts."

### Practice Mode Auto-Detection (with confirmation)

| Signal | Detection | Indicator |
|--------|-----------|-----------|
| One-hand (RH) | >80% AMT notes above C4 | "RH only" |
| One-hand (LH) | >80% AMT notes below C4 | "LH only" |
| Slow practice | Tempo <60% of score marking | "Slow practice" |
| Section drill | Same bars repeated 3x+ | "Drilling bars X-Y" |
| Full run-through | Sequential bars, no repeats | "Run-through" |

System shows subtle indicator ("Looks like RH only"), student can tap to correct.

### Within-Session Trajectory

Track warm-up (first 3-5 min, lower scores expected), peak phase (primary evaluation window), and fatigue (after ~40 min, quality may decline). Suggest breaks when fatigue detected.

---

## Teaching Moment Selection (Upgraded)

Six signals combined:

1. **Score-conditioned STOP:** "Would a teacher stop here given what the score asks?"
2. **Blind spot detection:** Dimension deviated significantly from baseline
3. **Positive moment detection:** Breakthrough, recovery, passage mastery, session best
4. **Musical priority weighting:** Chopin -> pedaling/phrasing 2x; Bach -> articulation/timing 2x
5. **Passage repetition:** Don't judge a drill like a performance; track improvement trajectory
6. **Novelty constraint:** `recency_penalty = count(same dim in last 5 obs) * 0.2`. After 3 pedaling observations, other dimensions get priority.

Positive/corrective ratio target: 25-35% positive. Track per student, adjust if ratio drifts.

---

## Real Audio + Expert Labels (Phase 4)

### Recording Collection

Target: 2K-5K annotated segments across 3+ skill levels, 5+ piano types, 5+ acoustic environments, 100+ pieces.

Sources: university partnerships (~500), user-contributed opt-in (~1000+), YouTube with permission (~500), commissioned (~200).

### Annotation Protocol

3-5 piano teachers rate each dimension 1-5 with score context provided. For ratings <3: annotate bar range, what's wrong, what they'd tell the student. For ratings >4: what's impressive, whether above expected for level.

Active learning after initial 500: prioritize segments where model is most uncertain.

### What Real Audio Unlocks

Pedal resonance subtlety (sympathetic vibration, half-pedaling), piano character (Steinway vs Yamaha vs digital), room acoustics adaptation, formal phone audio validation, skill level calibration (PercePiano is 98% advanced-level).

---

## Phased Implementation Roadmap

### Phase 1: Score Infrastructure (COMPLETE)

Built: score MIDI library (242 pieces), cloud AMT service, score following (DTW), bar-aligned analysis engine (all 6 dims), reference cache script.

Result: Same A1-Max scores, but teacher says "In bars 12-16, the crescendo is timid" instead of "dynamics score 0.35."

### Model v2: Aria + MuQ + Gated Fusion (2-4 months, engineering + ML)

*Collapses previous Phase 0 (A2 multi-tier) and Phase 3 (Symbolic FM) into a single effort.*

Build:

- ~~Retrain A1-Max on clean piece-stratified folds (establish valid baselines)~~ DONE (2026-03-19): 77.5% pairwise (4-fold, original weights). Loss weights optimized via autoresearch: contrastive=0.6, regression=0.8.
- Fine-tune Aria on PercePiano (performance MIDI + score MIDI)
- Quality-aware contrastive pretraining for both encoders
- Per-dimension gated fusion with score conditioning (delta = z_perf - z_score)
- Ordinal-dominated training (80% competition data, 20% PercePiano)
- Skill-level discrimination via multi-tier training data

Result: Score-conditioned dual-encoder scoring. Dynamics inversion fixed. Skill-level separation. Fusion viable because both encoders now have comparable pretraining scale.

Eval gates: E1-E3 (see Eval Tier Requirements above). Error correlation r < 0.5 required before fusion ships. **Phase A validation (2026-03-19): phi=0.043 -- gate passed. Frozen Aria 59.6% pairwise (marginal), frozen MuQ 62.2% (confirmed). Proceed to contrastive pretraining + LoRA fine-tuning.**

### Phase 2: Temporal + Practice Intelligence (2-3 months, engineering)

Build: rubato detection, passage repetition tracking, practice mode auto-detect, within-session trajectory, novelty constraint, positive moment detection.

Result: System becomes a practice partner, not just a judge.

Depends on Phase 1 (score alignment). Ships independently from Model v2.

### Phase 4: Real Audio + Expert Labels (6+ months)

Build: collect expert-annotated real piano recordings, retrain with acoustic diversity, active learning for annotation prioritization.

Result: Model hears real pianos, not synthesized audio.

Depends on Model v2 (architecture to retrain). Cost: ~$50-100K for annotation campaign.

---

## Pipeline Error Handling

| Stage | Failure Mode | Severity | Handling |
|-------|-------------|----------|---------|
| 0 | Piece not in library | Medium | Absolute scoring (current mode), Aria processes perf MIDI only |
| 1 | AMT fails (noisy audio) | High | Skip symbolic path, audio-only |
| 1 | Score following lost | High | Re-anchor on next clear onset match |
| 2 | HF endpoint timeout | High | Retry 2x, degrade to "feedback when online" |
| 2 | Aria inference fails | Medium | Fall back to audio-only scoring |
| 3 | Rubato false positive | High | Confidence threshold + silence |
| 4 | Score parsing error | Medium | Skip score comparison for that passage |
| 5 | No teaching moment | Low | "Sounded good, keep going" |
| 6 | Subagent hallucination | High | Structured input only, validate bar numbers |
| 8 | Reference not found | Low | Text-only observation |

---

## Performance Budget

| Stage | Latency | Cost/Chunk |
|-------|---------|-----------|
| 0: Context preload | <500ms | Negligible |
| 1: AMT | <1s (parallel with MuQ) | ~$0.01 |
| 2: MuQ scoring | <2s | ~$0.02 |
| 2: Aria scoring (perf + score) | <1s (parallel with MuQ) | ~$0.01 |
| 2: Gated fusion | <100ms | Negligible |
| 3-5: Analysis + selection | <500ms total | Negligible |
| 6: Subagent | <500ms | ~$0.001 |
| 7: Teacher | <1.5s | ~$0.01 |
| **Total** | **<5s** | **~$6/session (30 min)** |

Aria inference runs in parallel with MuQ on the same HF endpoint (or a separate lightweight endpoint -- Aria's 650M params are smaller than MuQ). The dual-encoder adds ~$0.01/chunk (~$1/session) over the current single-encoder pipeline.

---

## Key Decisions Log

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Symbolic encoder | Aria (EleutherAI, 650M) | SOTA on 6 benchmarks, 820K MIDI pretraining, Apache 2.0, eliminates 6-12mo custom FM research |
| Score conditioning | Immediate (via Aria delta) | Aria encodes both perf and score MIDI natively; no reason to defer |
| Fusion strategy | Separate-then-fuse with learned gates | Measure error correlation before committing; per-dimension routing |
| Training data mix | 20% PercePiano + 80% ordinal | Expert annotations as anchor, competition data for scale |
| Contrastive pretraining | Symmetric (both MuQ and Aria) | Both encoders get quality-aware embeddings before fusion |
| Data strategy | Reference-anchored (MAESTRO) | No new annotation needed; ranking signal from multiple performers of same piece |
| Temporal focus | Rubato + passage repetition | Highest user value; concrete, testable; skip harder narrative-arc problem |
| Sequencing | Score infrastructure first, then Model v2 | 80% of user-facing improvement from better LLM context |
| Rubato safety | Confidence threshold + silence | Trust preservation; better to miss than to falsely flag expression |
| Practice modes | Auto-detect with confirmation | AMT enables detection; confirmation keeps student in the loop |
| Fold leak response | Retrain everything on clean folds | All prior numbers invalid; no shortcuts |

---

## What Already Exists (Reusable)

| Component | Status | Role in Perfect Pipeline |
|-----------|--------|-------------------------|
| A1-Max (deployed, numbers invalid) | DEPLOYED (needs retrain) | Audio encoder for gated fusion |
| MuQ backbone (160K hrs) | DEPLOYED | Pretrained audio foundation |
| Aria base + embedding | AVAILABLE (HuggingFace) | Symbolic encoder, score encoder |
| STOP classifier (AUC 0.845) | COMPLETE (needs retrain) | Upgraded with score context |
| Two-stage subagent | IMPLEMENTED | Same architecture, richer inputs |
| ByteDance AMT | VALIDATED | Cloud-deployed, production-ready |
| Score following (DTW) | COMPLETE | Bar alignment for score conditioning |
| Bar-aligned analysis (6 dims) | COMPLETE | Structured musical facts |
| Score MIDI library (242 pieces) | COMPLETE | Score conditioning input |
| MAESTRO contrastive (24K segs) | COMPLETE | Reference-anchored training data |
| Composite labels (6 dims) | COMPLETE | Target dimensions unchanged |
| Quote bank (60 quotes) | COMPLETE | Teacher voice grounding |
| Clean piece-stratified folds | VERIFIED | All future training/eval |

## Not In Scope

| Item | Rationale |
|------|-----------|
| Custom symbolic FM (pretrain from scratch) | Aria eliminates the need; 820K MIDI pretraining already done |
| Full hierarchical temporal model (phrase -> section -> piece) | Requires labeled data that doesn't exist; rubato + repetition covers highest value |
| Automatic piece identification (audio fingerprinting) | Separate ML problem, limited ROI |
| On-device inference (Core ML) | Cloud-only is correct for foreseeable future |
| Multi-instrument support | Entire pipeline is piano-specific |
| Video analysis (hand position, posture) | Separate modality, separate research problem |
| Ensemble/duet evaluation | Multi-source separation unsolved for piano |
