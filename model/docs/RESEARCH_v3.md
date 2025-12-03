# CrescendAI Research v3: Multi-Modal Piano Performance Evaluation

**Date**: December 2024  
**Status**: Post-Phase 2 Analysis & Revised Research Direction  
**Author**: Jai  

---

## Executive Summary

This document synthesizes findings from Phase 2 experiments with a corrected understanding of available data and a refined research direction. Key conclusions:

1. **Phase 2 achieved r=0.96 correlation**—but this reflects artifact detection, not quality assessment. The synthetic degradations (Gaussian noise, bandpass filtering) were trivially detectable by MERT, explaining why fusion provided 0% improvement over audio-only.

2. **PercePiano provides expert labels but NOT rendered audio**. The dataset includes 1,202 MIDI segments with 12,652 expert annotations across 19 perceptual dimensions—but the Logic Pro renders that annotators listened to are not distributed. This fundamentally shapes viable research approaches.

3. **The research question shifts from "when does fusion help?" to "what information survives transcription?"** For a practical system where users provide audio recordings, the key question is whether transcription + symbolic analysis can match or exceed direct audio analysis.

4. **Two-track approach**: Hackathon demonstrates symbolic-only pipeline; research validates modality contributions using PercePiano MIDI + annotations.

---

## Table of Contents

1. [Background & Problem Statement](#1-background--problem-statement)
2. [Phase 2 Results & Analysis](#2-phase-2-results--analysis)
3. [Data Availability Correction](#3-data-availability-correction)
4. [Revised Research Questions](#4-revised-research-questions)
5. [Theoretical Framework](#5-theoretical-framework)
6. [Experimental Design](#6-experimental-design)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Success Criteria & Decision Points](#8-success-criteria--decision-points)
9. [References & Resources](#9-references--resources)

---

## 1. Background & Problem Statement

### 1.1 Original Vision

Build a multi-modal piano performance evaluation system combining:

- **Audio analysis** (MERT encoder) for acoustic/timbral qualities
- **Symbolic analysis** (MIDI encoder) for structural/technical qualities
- **Fusion** to capture complementary information

Target: Professional-grade assessment across 10-20 dimensions with correlations approaching human inter-rater agreement (r ≈ 0.6-0.7).

### 1.2 The Label Problem (Pre-Phase 2)

The original system achieved only r=0.2. Root cause: synthetic labels derived from MIDI complexity metrics measured **piece difficulty**, not **performance quality**. All MAESTRO performances are virtuoso-level, so there was no quality variance to learn from.

### 1.3 Phase 2 Approach: Controlled Degradation

To create quality variance without expert labels, Phase 2 introduced controlled degradation:

| Tier | Score Range | MIDI Degradation | Audio Degradation |
|------|-------------|------------------|-------------------|
| Pristine | 95-100 | None | None |
| Good | 80-95 | 10ms jitter | 35dB SNR noise |
| Moderate | 65-80 | 30ms jitter, 2% wrong notes | 25dB SNR noise |
| Poor | 50-65 | 60ms jitter, 5% wrong notes | 20dB SNR, bandpass |

This expanded the dataset from 114K to 450K samples.

---

## 2. Phase 2 Results & Analysis

### 2.1 Performance Summary

| Model | Mean Pearson r | Mean MAE | Notes |
|-------|---------------|----------|-------|
| audio_only | 0.9609 | 3.16 | Dominant performance |
| midi_only | 0.6503 | 8.03 | Significantly worse |
| gated | 0.9610 | 3.16 | No improvement over audio |
| concat | 0.9605 | 3.16 | No improvement |
| crossattn | 0.9276 | 4.03 | Worse than audio-only |

### 2.2 Critical Observations

**Observation 1: All dimensions scored identically (r = 0.960-0.962)**

In a well-calibrated system, dimensions should vary:

- Technical dimensions (note accuracy, timing) → easier → higher r
- Interpretive dimensions (expression, musicality) → harder → lower r

Uniform scores indicate the model learned a **single underlying signal**, not distinct quality facets.

**Observation 2: Correlation exceeds human ceiling**

- PercePiano inter-rater ICC(1,1): "poor" for individuals
- Human inter-rater agreement: r ≈ 0.6-0.7
- Our result: r = 0.96

This exceeds what human experts achieve with each other, suggesting **artifact detection**, not quality assessment.

**Observation 3: Fusion provided exactly 0% improvement**

```
Improvement = (gated - audio_only) / audio_only
            = (0.9610 - 0.9609) / 0.9609
            = 0.01%
```

The gate criteria required ≥10% improvement. This indicates MIDI added no information beyond what audio already captured.

**Observation 4: Cross-attention performed worst**

The most sophisticated fusion mechanism underperformed simpler alternatives. This pattern occurs when:

- Modalities aren't semantically aligned
- Attention finds noise rather than signal
- Simpler methods succeed by passing through dominant features unchanged

### 2.3 Root Cause Analysis

**The degradations were trivially detectable in audio:**

| Degradation | Audio Signature |
|-------------|-----------------|
| Gaussian noise | Distinct spectral characteristics |
| Bandpass filtering | Missing frequency content |
| Timing jitter | Irregular onset patterns (but subtle) |
| Wrong notes | Changed harmonic content |

MERT, pretrained on 160K hours of music, easily distinguishes "clean piano" from "noisy/filtered piano." The high correlation reflects this discrimination, not quality assessment.

**MIDI degradations were less salient:**

- Timing jitter: 60ms is noticeable but not catastrophic
- Wrong notes at 5%: Sparse signal
- Velocity compression: Affects dynamics but doesn't create obvious artifacts

This explains the audio-only dominance: audio degradations were easier to detect than MIDI degradations.

### 2.4 What Phase 2 Proved

✅ The model architecture is functional  
✅ Quality variance in labels is learnable  
✅ Degradation tiers are separable  

❌ Synthetic degradations don't correlate with perceptual quality  
❌ Fusion adds no value when signal is redundantly present in one modality  
❌ Missing diagnostic logging prevents definitive root cause analysis  

---

## 3. Data Availability Correction

### 3.1 PercePiano: What's Actually Available

The deep research initially stated "PercePiano annotations are publicly available" implying full usability. This requires correction:

| Component | Available? | Location | Notes |
|-----------|------------|----------|-------|
| MIDI files | ✅ Yes | GitHub | 1,202 segments from MAESTRO/e-Competition |
| Score files | ✅ Yes | GitHub | MusicXML, manually aligned to Henle editions |
| Annotations | ✅ Yes | GitHub | 12,652 ratings, 19 dimensions, 53 experts |
| Rendered audio | ❌ No | Not released | Was Logic Pro + Yamaha Grand Piano VST |

### 3.2 Implications

**For symbolic-only research (Hackathon track):**  
PercePiano is fully usable. MIDI + annotations are sufficient.

**For audio research:**  
Three options, each with tradeoffs:

| Option | Pros | Cons |
|--------|------|------|
| Re-render MIDI | Reproducible, same content | Not identical to annotator experience |
| Match to MAESTRO audio | Real recordings | Different acoustic characteristics |
| Self-annotate audio | Direct audio labels | Time-intensive, smaller scale |

**Recommendation:** Use PercePiano for symbolic validation. For audio experiments, either re-render with consistent synthesizer or map to MAESTRO source audio.

### 3.3 PercePiano Benchmark Results

From the paper (Nature Scientific Reports, 2024):

| Model | R² (Piece Split) | R² (Performer Split) |
|-------|-----------------|---------------------|
| Mean Prediction | ~0 | ~0 |
| Bi-LSTM | 0.185 | 0.236 |
| MidiBERT | 0.313 | — |
| Bi-LSTM + Score Alignment | 0.304 | 0.270 |
| Bi-LSTM + SA + HAN | **0.397** | **0.285** |

Key finding: **Score alignment provides 21.2% absolute improvement** over performance-only features. This validates the importance of reference information for evaluation.

---

## 4. Revised Research Questions

### 4.1 Primary Question (Practical)

> **For a system where users provide audio recordings, does transcription + symbolic analysis outperform direct audio analysis for technical dimensions?**

This is the production-relevant question. Most users will record themselves playing acoustic pianos with phone/computer mics. They won't have direct MIDI.

### 4.2 Secondary Questions (Theoretical)

**Q1: Modality Contribution by Dimension**
> Which of PercePiano's 19 dimensions are best predicted by symbolic features vs. audio features vs. both?

Hypothesis:

- Timing, articulation → Symbolic dominates
- Timbre, tone quality → Audio dominates (but PercePiano used synthesized audio!)
- Dynamics, expression → Both contribute

**Q2: Transcription Error Propagation**
> At what transcription accuracy does symbolic analysis become unreliable?

Current SOTA: ~96% note F1 (studio), ~80% (phone recordings)
Need to establish: Error rate threshold where symbolic analysis degrades below audio baseline

**Q3: Fusion Value with Real Labels**
> With expert labels (not synthetic degradation), does fusion improve over best single modality?

Phase 2 failed because degradation artifacts were redundant. With labels capturing actual perceptual quality, fusion might demonstrate value for dimensions requiring both timing patterns AND timbral information.

### 4.3 Non-Questions (Resolved by Phase 2)

- ~~Can we detect synthetic degradation?~~ Yes, trivially (r=0.96)
- ~~Is cross-attention the best fusion method?~~ No, simpler methods suffice
- ~~Does MERT pretrained features work?~~ Yes, when there's signal to learn

---

## 5. Theoretical Framework

### 5.1 What MIDI Captures Well

| Feature | MIDI Representation | Quality |
|---------|---------------------|---------|
| Note identity | Pitch (0-127) | Exact |
| Timing | Onset/offset timestamps | ~1ms resolution |
| Dynamics | Velocity (0-127) | 128 levels |
| Articulation | Note duration, overlap | Derived |
| Pedal | CC64 events | On/off (half-pedal lost) |

**Research supports symbolic analysis for:**

- Expressive timing (rubato, micro-timing) — Bruno Repp's work
- Dynamic structure (velocity curves, accents)
- Rhythmic accuracy (IOI analysis)
- Articulation patterns (key overlap timing)

### 5.2 What MIDI Cannot Capture

| Feature | Why MIDI Fails |
|---------|----------------|
| Tone quality | No overtone/timbre information |
| Sympathetic resonance | Pedal creates acoustic phenomena |
| Attack characteristics | Velocity ≠ precise hammer dynamics |
| Recording quality | No room acoustics, noise |
| Touch nuance | Key release not captured |

### 5.3 The Velocity-Loudness Problem

From Dannenberg (CMU): MIDI velocity relates approximately to **square root of RMS amplitude**, not logarithmically. The mapping varies **11-89 dB** across synthesizers.

**Implication:** Velocity-based dynamic analysis is valid for relative comparisons (within a performance) but not absolute loudness assessment.

### 5.4 Transcription as Information Bottleneck

```
Audio Recording
     │
     ▼ [Transcription - lossy]
Transcribed MIDI
     │
     ▼ [Analysis]
Quality Scores
```

Transcription errors propagate differently per dimension:

| Dimension | Transcription Robustness |
|-----------|-------------------------|
| Timing | **High** — onset deviation is relative |
| Note accuracy | **Low** — transcription errors = "wrong notes" |
| Velocity | **Medium** — ~80% F1, 1 in 5 may be wrong |
| Articulation | **Low** — offset detection less reliable |

---

## 6. Experimental Design

### 6.1 Experiment 1: PercePiano Symbolic Baseline

**Goal:** Establish baseline performance with expert labels using symbolic-only analysis

**Data:** PercePiano MIDI + annotations (1,202 segments, 19 dimensions)

**Models to compare:**

1. Simple features + linear regression (baseline)
2. Bi-LSTM (PercePiano's baseline)
3. Our MIDI encoder (from Phase 2)
4. MidiBERT pretrained

**Evaluation:**

- Per-dimension R²
- Piece split and performer split
- Compare to PercePiano published results

**Success criterion:** Match or exceed Bi-LSTM baseline (R² ≥ 0.185)

### 6.2 Experiment 2: Transcription Error Tolerance

**Goal:** Determine how transcription errors affect symbolic analysis

**Method:**

1. Train symbolic model on clean PercePiano MIDI
2. Test on artificially degraded MIDI simulating transcription errors:
   - Onset jitter: σ = {10, 20, 30, 50}ms
   - Missed notes: {5%, 10%, 15%, 20%}
   - Velocity noise: σ = {5, 10, 15, 20}
3. Plot performance vs. error rate

**Success criterion:** Identify threshold where R² drops >20% from clean performance

### 6.3 Experiment 3: Real Transcription Pipeline

**Goal:** Validate end-to-end audio → MIDI → scores pipeline

**Method:**

1. Take MAESTRO audio corresponding to PercePiano segments
2. Transcribe using Kong High-Resolution model
3. Run symbolic analysis on transcribed MIDI
4. Compare to analysis on ground-truth MIDI

**Key metrics:**

- Transcription note F1
- Correlation preservation (transcribed vs. ground-truth scores)
- Per-dimension degradation analysis

**Success criterion:** Transcribed analysis achieves >80% of ground-truth analysis R²

### 6.4 Experiment 4: Audio vs. Symbolic (Requires Re-rendering)

**Goal:** Direct comparison of modality contributions

**Method:**

1. Re-render PercePiano MIDI using FluidSynth + piano soundfont
2. Train audio-only model (MERT) on rendered audio + labels
3. Compare to symbolic-only model
4. Identify dimensions where each modality dominates

**Caveat:** Rendered audio ≠ original Logic Pro renders. Results apply to synthesized audio domain.

**Success criterion:** Identify ≥3 dimensions where modalities differ by >0.1 R²

### 6.5 Experiment 5: Targeted Fusion (Conditional on Exp 4)

**Goal:** If modalities show complementary strengths, test targeted fusion

**Method:**

1. Based on Exp 4, categorize dimensions:
   - Symbolic-dominant (fuse unnecessary)
   - Audio-dominant (fuse unnecessary)
   - Complementary (fusion may help)
2. Implement dimension-specific fusion:
   - Symbolic head for timing/articulation
   - Audio head for timbre
   - Gated fusion for dynamics/expression
3. Compare to best single-modality baselines

**Success criterion:** Fusion improves R² by ≥5% on complementary dimensions

---

## 7. Implementation Roadmap

### Phase A: Symbolic Validation (Weeks 1-2)

| Task | Days | Deliverable |
|------|------|-------------|
| Download & process PercePiano | 1 | Clean MIDI + label dataset |
| Implement data loading | 1 | PyTorch DataLoader with splits |
| Train simple baseline | 1 | Linear regression R² |
| Train Bi-LSTM | 2 | Match published baseline |
| Train MIDI encoder | 2 | Compare to Bi-LSTM |
| Analysis & documentation | 1 | Per-dimension results |

**Exit criteria:** R² ≥ 0.185 on piece split

### Phase B: Transcription Robustness (Weeks 3-4)

| Task | Days | Deliverable |
|------|------|-------------|
| Implement degradation pipeline | 2 | Controlled MIDI corruption |
| Run error tolerance experiments | 3 | Error rate vs. performance curves |
| Integrate transcription model | 2 | Kong model pipeline |
| End-to-end validation | 3 | Transcribed vs. ground-truth comparison |
| Analysis & documentation | 2 | Robustness characterization |

**Exit criteria:** Identify transcription quality threshold

### Phase C: Audio Experiments (Weeks 5-6)

| Task | Days | Deliverable |
|------|------|-------------|
| Set up audio rendering | 2 | FluidSynth pipeline |
| Render PercePiano dataset | 1 | Audio files for all segments |
| Train MERT model | 3 | Audio-only baseline |
| Compare modalities | 2 | Per-dimension analysis |
| Document findings | 2 | Modality contribution report |

**Exit criteria:** Characterize audio vs. symbolic by dimension

### Phase D: Fusion (Conditional, Week 7+)

Only proceed if Phase C shows complementary contributions.

| Task | Days | Deliverable |
|------|------|-------------|
| Design targeted fusion | 2 | Architecture specification |
| Implement fusion module | 3 | Dimension-aware fusion |
| Train and evaluate | 4 | Fusion vs. single-modality comparison |
| Final analysis | 3 | Research conclusions |

**Exit criteria:** Fusion provides ≥5% improvement on target dimensions

---

## 8. Success Criteria & Decision Points

### 8.1 Week 2 Decision Point

**Question:** Does symbolic analysis work with expert labels?

| Outcome | Action |
|---------|--------|
| R² ≥ 0.30 | Proceed to transcription experiments |
| R² 0.15-0.30 | Investigate, may need architecture changes |
| R² < 0.15 | Major pivot needed, revisit fundamentals |

### 8.2 Week 4 Decision Point

**Question:** Is transcription + symbolic viable for production?

| Outcome | Action |
|---------|--------|
| Transcribed achieves >80% of clean | Production pipeline validated |
| Transcribed achieves 50-80% of clean | Acceptable with caveats |
| Transcribed achieves <50% of clean | Reconsider audio-only approach |

### 8.3 Week 6 Decision Point

**Question:** Does audio add unique value?

| Outcome | Action |
|---------|--------|
| Audio dominates on ≥3 dimensions | Proceed to fusion experiments |
| Modalities equivalent | Single-modality system sufficient |
| Symbolic dominates all dimensions | Focus on symbolic-only system |

### 8.4 Final Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Technical dimensions R² | ≥ 0.35 | ≥ 0.45 |
| Interpretive dimensions R² | ≥ 0.25 | ≥ 0.35 |
| Overall R² | ≥ 0.30 | ≥ 0.40 |
| Transcription robustness | ≥ 80% | ≥ 90% |
| Fusion improvement (if applicable) | ≥ 5% | ≥ 10% |

---

## 9. References & Resources

### 9.1 Key Papers

**PercePiano** (Park et al., 2024)

- Nature Scientific Reports
- 1,202 segments, 19 dimensions, 53 expert annotators
- Baseline: R² = 0.397 with score alignment + HAN
- GitHub: github.com/JonghoKimSNU/PercePiano

**MERT** (Yizhi et al., 2023)

- ICML 2023
- 330M parameters, 160K hours pretraining
- Dual-teacher: acoustic (RVQ-VAE) + musical (CQT)
- HuggingFace: m-a-p/MERT-v1-330M

**MidiBERT-Piano** (Chou et al., 2021)

- 12-layer transformer, 768-dim
- Pretrained on ~4,166 pieces (MLM)
- GitHub: github.com/wazenmai/MIDI-BERT

**Piano Transcription** (Kong et al., 2020)

- ByteDance high-resolution model
- ~96% note onset F1 on MAESTRO
- GitHub: bytedance/piano_transcription

### 9.2 Datasets

| Dataset | Size | Content | Use Case |
|---------|------|---------|----------|
| PercePiano | 1,202 segments | Expert annotations | Validation |
| MAESTRO | 200 hours | Audio + MIDI pairs | Source data |
| GiantMIDI-Piano | 10,855 files | Transcribed MIDI | Pretraining |

### 9.3 Code Resources

- piano_transcription_inference: `pip install piano_transcription_inference`
- pretty_midi: `pip install pretty_midi`
- MidiTok: github.com/Natooz/MidiTok
- allRank (ranking losses): github.com/allegro/allRank
- coral-pytorch (ordinal regression): github.com/Raschka-research-group/coral-pytorch

### 9.4 Our Codebase

| Component | Path | Description |
|-----------|------|-------------|
| Audio Encoder | `src/models/audio_encoder.py` | MERT-95M wrapper |
| MIDI Encoder | `src/models/midi_encoder.py` | Custom transformer |
| Fusion | `src/models/fusion_*.py` | Gated, concat, cross-attn |
| Labels | `src/labeling_functions.py` | Heuristic labelers |
| Degradation | `src/data/degradation.py` | Synthetic degradation |

---

## Appendix A: PercePiano Dimension Categories

| Level | Category | Dimensions |
|-------|----------|------------|
| Low | Timing | Stable ↔ Unstable |
| Low | Articulation | Short ↔ Long, Soft/Cushioned ↔ Hard/Solid |
| Mid-Low | Pedal | Sparse/Dry ↔ Saturated/Wet, Clean ↔ Blurred |
| Mid-Low | Timbre | Even ↔ Colorful, Shallow ↔ Rich, Bright ↔ Dark, Soft ↔ Loud |
| Mid-High | Dynamic | Sophisticated/Mellow ↔ Raw/Crude, Little Range ↔ Large Range |
| Mid-High | Music Making | Fast-paced ↔ Slow-paced, Flat ↔ Spacious, Disproportioned ↔ Balanced, Pure ↔ Dramatic |
| High | Emotion & Mood | Optimistic ↔ Dark, Low Energy ↔ High Energy, Honest ↔ Imaginative |
| High | Interpretation | Unsatisfactory ↔ Convincing |

---

## Appendix B: Phase 2 Architecture (Reference)

```
Audio Input ──► MERT-95M ──► Projection (768→512) ──┐
                                                     ├──► Fusion ──► BiLSTM ──► MTL Head ──► 8 Scores
MIDI Input ──► MIDIBert ──► Projection (256→512) ──┘
```

**Fusion options tested:**

- GatedFusion: `fused = gate * audio + (1-gate) * midi`
- ConcatFusion: `fused = Linear(concat(audio, midi))`
- CrossAttention: Bidirectional attention between modalities

**Loss function:**

- Huber (base) + Ranking (0.2) + Contrastive (0.1) + CORAL (0.3)
- Bootstrap (β=0.8) + LDS (enabled)

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| IOI | Inter-Onset Interval: time between successive note onsets |
| ICC | Intraclass Correlation Coefficient: measure of rater agreement |
| CQT | Constant-Q Transform: frequency representation with log spacing |
| MLM | Masked Language Modeling: BERT-style pretraining objective |
| HAN | Hierarchical Attention Network: multi-level attention architecture |
| MTL | Multi-Task Learning: shared representations for multiple outputs |
| CORAL | Consistent Rank Logits: ordinal regression method |
| LDS | Label Distribution Smoothing: technique for imbalanced regression |

---

*Document version: 3.0*  
*Last updated: December 2024*  
*Next review: After Week 2 experiments*
