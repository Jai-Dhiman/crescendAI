# North Star: The Perfect Pipeline

Vision document for the ideal piano performance evaluation system, from recording to actionable feedback. Captures the complete 8-stage pipeline, phased implementation roadmap, and the rationale behind every architectural decision.

> **Status (2026-03-15):** Vision document. **Phase 1 (score infrastructure) is COMPLETE:** score library (242 pieces), cloud AMT with pedal, score following (DTW), bar-aligned analysis (all 6 dims), reference cache script. Phase 2 (temporal + practice intelligence) is the recommended next focus. See `03-encoders.md` for model status, `00-research-timeline.md` for Phase 1 details.

---

## The Core Insight

The current system evaluates performance quality in absolute terms. The perfect system evaluates quality *relative to what the score asks for.* This single shift fixes the dynamics inversion (rho=-0.917 in competition -- the model captures "amount" not "appropriateness"), enables rubato detection, and transforms feedback from "dynamics score 0.35" to "the crescendo in bars 12-16 doesn't reach the forte Chopin marked."

80% of the user-facing improvement comes from giving the LLM better context (bar-aligned musical facts, reference comparisons), not from changing the model itself. The remaining 20% comes from a score-conditioned scoring model that evaluates relative to the score at the architecture level.

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
 STAGE 2: MULTI-MODAL SCORING
   z_audio (MuQ + LoRA) + z_perf (Symbolic FM) + z_score (Symbolic FM)
   Score-conditioned gated fusion
   Per-dimension routing: audio for timing, symbolic for dynamics/structure
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

## Why Fusion Failed (and When It Becomes Viable)

The ISMIR paper (`paper/ismir_v2/main.tex`) tested audio-symbolic fusion:

| Model | R2 |
|-------|-----|
| Audio only (MuQ + LoRA) | 0.537 |
| **Audio-symbolic fusion (concat)** | **0.524** |
| Symbolic only (baseline) | 0.347 |

Fusion *underperformed* audio-only. Error correlation between modalities: **r = 0.738** -- both fail on the same samples.

### Root Cause: Pretraining Asymmetry

Both modalities derive from identical MIDI source data (PercePiano uses Pianoteq-rendered audio). MuQ was pretrained on 160K hours; the symbolic encoder was trained from scratch on ~24K graphs. The gap reflects pretraining scale, not modality choice.

### When Fusion Becomes Viable

Fusion requires:
- A symbolic foundation model (pretrained on millions of MIDI files) that matches MuQ's representation quality
- Error correlation below ~0.5 (genuinely different failure modes)
- Per-dimension gated fusion that exploits real complementarity: audio for timing/resonance, symbolic for structure/dynamics

---

## Score-Conditioned Scoring Model

### Architecture

```
                   Score MIDI
                      |
                      v
                   [Symbolic Foundation Model]
                      |
                      v
                   z_score [512]
                      |
AUDIO -> [MuQ+LoRA] -> z_audio [512]      |
                      |                     |
PERF MIDI -> [Symb FM] -> z_perf [512]     |
                      |                     |
                      v                     v
            SCORE-CONDITIONED GATED FUSION

            Per-dimension learned gates:
              timing:         audio 0.7 | symb 0.3
              dynamics:       audio 0.3 | symb 0.7
              pedaling:       audio 0.5 | symb 0.5
              articulation:   audio 0.4 | symb 0.6
              phrasing:       audio 0.5 | symb 0.5
              interpretation: cross-attention over all 3

            delta_d = z_perf - z_score  (what's different)
            quality_d = MLP_d(fused_d, delta_d)

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
- A1-Max ensemble scores provide ranking signal (already computed)
- Training triple: (performance_audio, performance_midi, score_midi) with relative ranking label
- Loss: ListMLE ranking (same as A1-Max), conditioned on score

This fixes dynamics inversion naturally: when the score says pp and the best performer plays pp, the model learns pp + score_says_pp = HIGH quality.

---

## Symbolic Foundation Model

### Why

The current S2 GNN (71.3% pairwise) is trained from scratch on ~24K graphs. Pretraining on millions of MIDI performances would develop rich representations of voice leading, harmonic progression, and rhythmic patterns -- complementary to MuQ's audio representations.

### Pretraining Data (~370K+ performances)

| Source | Pieces | Performances |
|--------|--------|-------------|
| MAESTRO v3 | ~300 | 1,276 |
| GiantMIDI-Piano | ~10,800 | 10,800 |
| ATEPP | ~11,000 | 11,697 |
| ASAP | ~222 | 1,066 |
| PianoMIDI | ~100,000+ | ~100,000+ |
| Lakh MIDI (piano) | ~50,000+ | ~50,000+ |
| MuseScore exports | ~200,000+ | ~200,000+ |

~7.4M training segments at 20 segments per performance. Comparable to MuQ's pretraining scale.

### Pretraining Objectives

1. **Masked note prediction (BERT-style):** Mask 15% of notes, predict pitch/velocity/duration/onset. Forces harmonic and voice-leading understanding.
2. **Contrastive same-piece learning:** Same piece, different performer = positive pair. Separates piece identity from performance quality.
3. **Next-bar prediction (autoregressive):** Given bars 1-N, predict bar N+1. Forces understanding of musical form and phrase structure.
4. **Score-performance alignment (optional):** Learn to align performance MIDI to score MIDI. Directly trains for score conditioning.

### Architecture

Transformer encoder (12-24 layers) with:
- Note tokenizer: pitch, velocity (32 bins), onset (64 bins), duration (16 bins), voice, pedal_state
- Relative position encoding (bar-aware)
- Voice-aware attention mask
- Hierarchical pooling: note -> beat -> bar -> phrase -> z_symbolic [512]

Transformer over GNN for the foundation model because self-attention can learn graph-like relationships from data without hardcoding edge types. Same architecture encodes both score MIDI (z_score) and performance MIDI (z_perf).

---

## Score Infrastructure (Phase 1 Components)

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

## Temporal Reasoning

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

## Real Audio + Expert Labels

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

### Phase 1: Score Infrastructure (3-4 months, engineering)

Build: score MIDI library, cloud AMT service, score following, bar-aligned analysis engine, reference performance cache, structured musical facts as subagent input.

Result: Same A1-Max scores, but teacher says "In bars 12-16, the crescendo is timid" instead of "dynamics score 0.35."

Ships independently. Rollback: disable score context, revert to current pipeline.

### Phase 2: Temporal + Practice Intelligence (2-3 months, engineering)

Build: rubato detection, passage repetition tracking, practice mode auto-detect, within-session trajectory, novelty constraint, positive moment detection.

Result: System becomes a practice partner, not just a judge.

Depends on Phase 1 (score alignment). Ships independently.

### Phase 3: Symbolic Foundation Model (6-12 months, research)

Build: pretrain symbolic FM on 370K+ MIDI performances, score-conditioned gated fusion architecture, reference-anchored training on MAESTRO.

Result: Dynamics inversion fixed. True multi-modal scoring relative to score.

Research risk: HIGH (symbolic FM doesn't exist yet). Can start in parallel with Phases 1-2.

### Phase 4: Real Audio + Expert Labels (6+ months)

Build: collect expert-annotated real piano recordings, retrain with acoustic diversity, active learning for annotation prioritization.

Result: Model hears real pianos, not synthesized audio.

Depends on Phase 3 (architecture to retrain). Cost: ~$50-100K for annotation campaign.

---

## Pipeline Error Handling

| Stage | Failure Mode | Severity | Handling |
|-------|-------------|----------|---------|
| 0 | Piece not in library | Medium | Absolute scoring (current mode) |
| 1 | AMT fails (noisy audio) | High | Skip symbolic path, audio-only |
| 1 | Score following lost | High | Re-anchor on next clear onset match |
| 2 | HF endpoint timeout | High | Retry 2x, degrade to "feedback when online" |
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
| 2: Multi-modal scoring | <2s | ~$0.02 (MuQ) + ~$0.01 (Symb FM) |
| 3-5: Analysis + selection | <500ms total | Negligible |
| 6: Subagent | <500ms | ~$0.001 |
| 7: Teacher | <1.5s | ~$0.01 |
| **Total** | **<5s** | **~$5/session (30 min)** |

---

## Key Decisions Log

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Data strategy | Reference-anchored (MAESTRO) | No new annotation needed; ranking signal from multiple performers of same piece |
| Temporal focus | Rubato + passage repetition | Highest user value; concrete, testable; skip harder narrative-arc problem |
| Sequencing | Score infrastructure first | 80% of user-facing improvement from better LLM context, not better model |
| Rubato safety | Confidence threshold + silence | Trust preservation; better to miss than to falsely flag expression |
| Practice modes | Auto-detect with confirmation | AMT enables detection; confirmation keeps student in the loop |
| Symbolic FM arch. | Transformer (not GNN) | Self-attention learns graph relationships from data; same arch for score + perf |

---

## What Already Exists (Reusable)

| Component | Status | Role in Perfect Pipeline |
|-----------|--------|-------------------------|
| A1-Max (80.8% pairwise) | DEPLOYED | Audio encoder, unchanged |
| MuQ backbone (160K hrs) | DEPLOYED | Pretrained audio foundation |
| STOP classifier (AUC 0.845) | COMPLETE | Upgraded with score context |
| Two-stage subagent | IMPLEMENTED | Same architecture, richer inputs |
| ByteDance AMT | VALIDATED | Needs real-time cloud deployment |
| MAESTRO contrastive (24K segs) | COMPLETE | Reference-anchored training data |
| Symbolic pretrain corpus (24K graphs) | COMPLETE | Seed for symbolic FM pretraining |
| Composite labels (6 dims) | COMPLETE | Target dimensions unchanged |
| Quote bank (60 quotes) | COMPLETE | Teacher voice grounding |

## Not In Scope

| Item | Rationale |
|------|-----------|
| Full hierarchical temporal model (phrase -> section -> piece) | Requires labeled data that doesn't exist; rubato + repetition covers highest value |
| Automatic piece identification (audio fingerprinting) | Separate ML problem, limited ROI |
| On-device inference (Core ML) | Cloud-only is correct for foreseeable future |
| Multi-instrument support | Entire pipeline is piano-specific |
| Video analysis (hand position, posture) | Separate modality, separate research problem |
| Ensemble/duet evaluation | Multi-source separation unsolved for piano |
