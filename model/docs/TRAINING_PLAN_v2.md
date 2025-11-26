# Piano Performance Evaluation Model - Training Plan v2

**Date**: 2025-01-24
**Status**: Planning Phase
**Authors**: Development Team

---

## Executive Summary

After analyzing baseline results (fusion underperforming single-modal by -19%), we identified that the primary bottleneck is **label quality, not model architecture or data quantity**. This document outlines our path forward: introducing quality variance through controlled degradation, adding comprehensive diagnostics, and validating fusion architecture before scaling.

---

## Current State Analysis

### Baseline Results (114K MAESTRO samples, 6 dimensions)

- Audio-only: r=0.205
- MIDI-only: r=0.199
- Fusion: r=0.166 (WORSE than single-modal)

**Key findings**:
- Only note_accuracy shows moderate signal (r=0.572 MIDI)
- All other dimensions extremely weak (r=0.05-0.39)
- Fusion actively destroys information instead of combining it

### Root Causes Identified

**1. Label Quality Crisis (CRITICAL)**
- Current synthetic labels measure complexity, not quality
- All MAESTRO performances are virtuosos (no quality variance)
- Labels inversely correlate with difficulty: harder pieces get lower scores
- Model learns "simple piece = high quality" which is incorrect

**2. Missing Contrastive Alignment (HIGH)**
- MERT and MIDIBert encoders trained on different objectives
- Representations live in incompatible spaces
- Cross-attention cannot bridge misaligned feature spaces

**3. Fusion Architecture Unknown (HIGH)**
- No diagnostics on attention patterns
- Unknown if fusion uses both modalities or collapses to one
- Sequence alignment may introduce artifacts

**4. Data Quantity (LOW PRIORITY)**
- 114K samples is sufficient if labels were good
- Adding more MAESTRO won't help (same label problem)

---

## Rejected Alternatives

### Why NOT: Improve Heuristic Labeling Functions
Current labeling functions are already sophisticated (spectral features, timing variance, velocity analysis). The problem isn't measurement quality—it's that we're measuring the wrong thing (complexity vs quality). Better heuristics won't fix this fundamental issue.

### Why NOT: Add More MAESTRO Data
More samples of virtuoso performances won't create quality variance. We'd just have 200K samples with the same problem.

### Why NOT: Expert Annotation First
At $75-100K for full annotation, we need to validate that fusion architecture works before investing. Current results suggest architectural issues that expert labels won't fix.

### Why NOT: Hard-Code Modality Routing
The assumption "MIDI for technical, audio for interpretive" seems intuitive but isn't supported by current results. Only note_accuracy is MIDI-strong; dynamics_control actually favors audio. Let the model learn modality weights per dimension rather than hard-coding incorrect assumptions.

---

## Chosen Approach: Controlled Degradation

### Core Strategy

Artificially introduce quality variance by creating degraded versions of MAESTRO performances with known error levels. This provides ground truth quality labels without expert annotation.

### Quality Tiers (4 levels from each MAESTRO segment)

**Pristine (30% of data)**: Original MAESTRO, score 95-100
**Good (30%)**: Light degradation (10ms timing jitter, 35dB noise), score 80-95
**Moderate (25%)**: Moderate degradation (30ms jitter, 2% wrong notes, dynamics compression), score 65-80
**Poor (15%)**: Heavy degradation (60ms jitter, 5% wrong notes, audio filtering), score 50-65

**Result**: 114K → 450K segments with actual quality variance orthogonal to piece difficulty

### Dimension Updates (6→8, PercePiano-Inspired)

**Technical (4)**: note_accuracy, rhythmic_stability, articulation_clarity, pedal_technique
**Timbre/Dynamics (2)**: tone_quality, dynamic_range
**Interpretive (2)**: musical_expression, overall_interpretation

Rationale: PercePiano dataset uses 18 dimensions across technical/expressive/interpretive hierarchy. We balanced coverage against data efficiency (14K samples per dimension).

---

## Implementation Plan

### Phase 1: Clean Resegmentation (Week 1-2)

**Objectives**: Fix corrupted files/paths, generate degradation-based labels

**Deliverables**:
- maestro_with_variance.tar.gz with 450K segments
- Clean annotation files with correct paths
- Documented degradation parameters per quality tier

**Success criteria**: Clean pipeline runs without path correction scripts

### Phase 2: Diagnostic Training (Week 2-3)

**Objectives**: Understand why fusion fails, validate degradation labels

**Experiments** (5 epochs each, parallel):
1. Audio-only baseline
2. MIDI-only baseline
3. Fusion-concatenation (simple)
4. Fusion-cross-attention (current)

**Diagnostic metrics**:
- Attention entropy (high = using both modalities, low = collapsed)
- Attention sparsity (fraction of attention on top 10% of keys)
- Cross-modal alignment (cosine similarity between audio/MIDI features)
- Feature diversity (within-modality similarity, detect mode collapse)

**Success criteria**:
- Models learn quality NOT difficulty (higher scores for pristine vs degraded)
- Fusion beats best single-modal by ≥10%
- Attention diagnostics show both modalities being used

### Phase 3: Contrastive Pre-training (Week 3-4)

**Only execute if Phase 2 succeeds**

**Objectives**: Align MERT and MIDIBert representation spaces

**Approach**: Train projection heads with InfoNCE loss on paired audio-MIDI samples (10-15 epochs, batch size 64-128)

**Success criteria**: Cross-modal alignment score >0.6 (up from current ~0.2)

### Phase 4: Full Training (Week 4-5)

**Only execute if Phase 3 succeeds**

**Configuration**: Best fusion variant from Phase 2, with contrastive-aligned encoders, 20 epochs

**Targets**:
- Technical dimensions: r>0.50
- Interpretive dimensions: r>0.35
- Fusion beats single-modal by ≥15%

### Phase 5: Expert Validation (Week 6)

**Only execute if Phase 4 meets targets**

**Scope**: 200-300 test segments only, $6-10K investment

**Purpose**:
- Validate synthetic labels correlate with expert judgments
- Identify which dimensions transfer to real quality assessment
- Inform full annotation investment decision ($75-100K)

---

## Success Criteria & Go/No-Go Gates

### Phase 2 Gate (Diagnostic Training)
✅ GO: Fusion >10% better than single-modal, learns quality not difficulty
❌ NO-GO: Debug fusion architecture, try simpler fusion, don't proceed to Phase 3

### Phase 3 Gate (Contrastive)
✅ GO: Alignment score >0.6, fusion improvement increases to ≥15%
❌ NO-GO: Investigate alignment failure, don't proceed to Phase 4

### Phase 4 Gate (Full Training)
✅ GO: Technical r>0.50, Interpretive r>0.35
❌ NO-GO: Don't invest in expert annotation, reassess approach

### Phase 5 Gate (Expert Validation)
✅ GO: Synthetic labels r>0.5 with expert labels for 3+ dimensions
❌ NO-GO: Identify which dimensions need full expert annotation, which can use synthetic

---

## Risk Mitigation

**Risk**: Degradation doesn't match real amateur errors
**Mitigation**: Expert validation in Phase 5 tests transfer to real quality

**Risk**: Fusion still fails after contrastive training
**Mitigation**: Phase 2 tests simpler fusion variants; fall back to single-modal specialization

**Risk**: Synthetic labels don't correlate with experts
**Mitigation**: Phase 5 identifies which dimensions work; pivot to hybrid approach

**Risk**: Compute constraints prevent full experiments
**Mitigation**: Phase 2 uses only 5 epochs for fast iteration; scale up only after validation

---

## References

**PercePiano Dataset**: 18-dimension evaluation framework with technical/expressive/interpretive hierarchy (Nature Scientific Reports, 2024)

**Multi-Modal Contrastive Learning**: MGA-CLAP, MA-AVT approaches for audio-language/audio-visual alignment (CVPR 2024)

**Weak Supervision**: Snorkel framework for noisy label aggregation; ranking loss for relative comparisons

**Prior Results**: PercePiano baseline R²=0.397 (piece split), our target: r>0.50 technical, r>0.35 interpretive

---

## Decision Log

**Why 8 dimensions**: Balance between PercePiano's 18 (too many for data) and current 6 (missing interpretive coverage)

**Why degradation over expert labels**: $100K investment requires proof fusion works; degradation provides cheaper validation

**Why not resegment length**: Current 10s windows are standard; only change if diagnostics show context issues

**Why quality tiers**: Creates supervised learning target; more robust than purely contrastive or ranking approaches

**Why Phase 2 before contrastive**: Contrastive training takes 2-3 days GPU; validate fusion architecture first with quick experiments
