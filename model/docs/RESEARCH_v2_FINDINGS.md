# Research Findings v1: Phase 2 Results and Analysis

**Date**: 2024-12-02
**Status**: Post-Experiment Analysis
**Experiment**: Phase 2 Diagnostic Training (TRAINING_PLAN_v2.md)

---

## Background and Context

### Project Goal

Build a multi-modal piano performance evaluation system that combines audio analysis (MERT-95M encoder) with symbolic MIDI analysis (custom MIDIBert-style encoder) to assess performance quality across 8 dimensions.

### The Label Problem

The original system (pre-Phase 2) achieved only r=0.2 correlation. Root cause analysis identified that synthetic labels derived from MAESTRO data measured **piece complexity** rather than **performance quality** - all MAESTRO performances are by virtuosos, so there was no quality variance to learn from.

### Phase 2 Approach: Controlled Degradation

To create quality variance without expert labels, Phase 2 introduced controlled degradation of MAESTRO performances:

| Tier | Probability | Score Range | MIDI Degradation | Audio Degradation |
|------|-------------|-------------|------------------|-------------------|
| Pristine | 30% | 95-100 | None | None |
| Good | 30% | 80-95 | 10ms jitter | 35dB SNR noise |
| Moderate | 25% | 65-80 | 30ms jitter, 2% wrong notes, dynamics compression | 25dB SNR noise |
| Poor | 15% | 50-65 | 60ms jitter, 5% wrong notes, 50% dynamics compression | 20dB SNR noise, bandpass filter |

This expanded the dataset from 114K to 450K samples across 8 evaluation dimensions.

### Phase 2 Success Criteria

From TRAINING_PLAN_v2.md:
- Fusion must beat best single-modal by >= 10%
- Models must learn quality (degradation tier), not piece difficulty
- Attention diagnostics should show both modalities being used

---

## Experiment Results

### Model Performance Summary

| Model | Mean Pearson r | Mean MAE | Type | Best Epoch |
|-------|---------------|----------|------|------------|
| audio_only | 0.9609 | 3.16 | Single-modal | 0 |
| midi_only | 0.6503 | 8.03 | Single-modal | 4 |
| crossattn | 0.9276 | 4.03 | Fusion | 2 |
| gated | 0.9610 | 3.16 | Fusion | 3 |
| concat | 0.9605 | 3.16 | Fusion | 3 |

### Per-Dimension Correlation (Pearson r)

| Dimension | audio_only | midi_only | crossattn | gated | concat |
|-----------|------------|-----------|-----------|-------|--------|
| note_accuracy | 0.960 | 0.654 | 0.928 | 0.960 | 0.960 |
| rhythmic_stability | 0.960 | 0.651 | 0.928 | 0.960 | 0.960 |
| articulation_clarity | 0.961 | 0.652 | 0.929 | 0.962 | 0.961 |
| pedal_technique | 0.961 | 0.647 | 0.927 | 0.961 | 0.961 |
| tone_quality | 0.962 | 0.651 | 0.929 | 0.962 | 0.961 |
| dynamic_range | 0.961 | 0.648 | 0.927 | 0.961 | 0.960 |
| musical_expression | 0.960 | 0.651 | 0.926 | 0.961 | 0.960 |
| overall_interpretation | 0.961 | 0.648 | 0.928 | 0.961 | 0.961 |

### Statistical Significance (Paired t-test: gated vs audio_only)

| Dimension | p-value | Winner | Significant? |
|-----------|---------|--------|--------------|
| note_accuracy | 0.0000 | gated | Yes |
| rhythmic_stability | 0.0000 | audio_only | Yes |
| articulation_clarity | 0.0233 | audio_only | Yes |
| pedal_technique | 0.5013 | Tie | No |
| tone_quality | 0.5929 | Tie | No |
| dynamic_range | 0.0045 | audio_only | Yes |
| musical_expression | 0.1618 | Tie | No |
| overall_interpretation | 0.0025 | audio_only | Yes |

**Result**: audio_only is statistically better on 4/8 dimensions; gated wins on 1/8.

### Fusion Diagnostics

```
CROSSATTN:
  Cross-modal alignment: Not logged
  MSE/Huber loss: 1.7766
  CORAL loss: 0.2669

GATED:
  Cross-modal alignment: Not logged
  Gate mean: Not logged
  MSE/Huber loss: 1.3455
  CORAL loss: 0.2586

CONCAT:
  Cross-modal alignment: Not logged
  MSE/Huber loss: 1.3419
  CORAL loss: 0.2587
```

Key diagnostic metrics (gate values, cross-modal alignment) were not logged during training.

### Modality Contribution Analysis

Audio-only to MIDI-only ratio: 1.48 (0.9609 / 0.6503)

Both encoders produce non-trivial signal independently, but audio dominates.

---

## Analysis and Observations

### Observation 1: Dramatic Improvement from Baseline

Correlation improved from r=0.2 (original synthetic labels) to r=0.96 (degradation-based labels). This confirms:
- The model architecture is functional
- Quality variance in labels is learnable
- The controlled degradation approach creates separable tiers

### Observation 2: Fusion Provides No Improvement Over Audio-Only

```
Improvement = (gated - audio_only) / audio_only
            = (0.9610 - 0.9609) / 0.9609
            = 0.01%
```

Phase 2 gate criteria required >= 10% improvement. The observed 0.01% is within noise.

### Observation 3: MIDI Encoder Underperforms Audio Encoder

Despite pretraining, MIDI-only (r=0.65) significantly underperforms audio-only (r=0.96). Possible explanations:
- Audio degradation artifacts (noise, filtering) are easier to detect than MIDI degradation (timing jitter, wrong notes)
- MERT's 160K-hour pretraining provides stronger features than MIDI encoder's limited pretraining
- The degradation types chosen favor audio detection

### Observation 4: All Dimensions Score Identically

All 8 dimensions achieve r=0.960-0.962. In a well-calibrated system, dimensions should have varying difficulty:
- Technical dimensions (note_accuracy, rhythmic_stability) should be easier
- Interpretive dimensions (musical_expression) should be harder
- Research targets from RESEARCH_v2.md: Technical r>0.50, Interpretive r>0.35

Uniform scores suggest the model predicts a single underlying signal rather than distinct quality dimensions.

### Observation 5: Cross-Attention Performs Worst Among Fusion Methods

```
crossattn: 0.9276
concat:    0.9605
gated:     0.9610
```

The most complex fusion mechanism underperforms simpler alternatives. This pattern often indicates:
- Attention finds noise when modalities aren't semantically aligned
- Simpler methods succeed by passing through dominant features

### Observation 6: Correlation Exceeds Expected Human Ceiling

Reference benchmarks:
- PercePiano (expert labels): R^2 = 0.397 (approximately r = 0.63)
- Human inter-rater agreement: r = 0.6-0.7

The r=0.96 result exceeds what expert human raters achieve with each other, suggesting the model is predicting something other than human-perceived quality.

---

## Identified Issues

### Issue 1: Degradation Artifacts Are Trivially Detectable

The synthetic degradations introduce artifacts that MERT can easily detect:

| Artifact | Detection Method |
|----------|-----------------|
| Gaussian noise | Spectral characteristics differ from music |
| Bandpass filtering | Missing frequency content |
| Timing jitter | Irregular onset patterns |
| Wrong notes | Changed harmonic content |

All artifacts are present in audio, explaining why audio-only matches fusion performance.

### Issue 2: Missing Diagnostic Logging

Critical metrics were not captured:
- `gate_mean` in GatedFusion - cannot verify if both modalities contribute
- `cross_modal_alignment` - cannot verify encoder alignment
- Per-modality attention weights in CrossAttention

Without these, the cause of fusion failure cannot be diagnosed definitively.

### Issue 3: Labels May Not Correlate with Perceived Quality

The degradation approach assumes:
- Timing jitter reduces perceived quality
- Wrong notes reduce perceived quality
- Noise reduces perceived quality

These assumptions are reasonable but unvalidated. Small timing variations may be imperceptible; some noise levels may not affect quality perception.

### Issue 4: Production Architecture Unclear

For fusion to work in production:
- User must provide both audio AND MIDI
- If user has only audio, MIDI must be transcribed (introduces errors)
- If user has only MIDI (digital piano), audio provides no additional technical information

The value proposition of fusion for technical assessment is unclear.

---

## Codebase Reference

### Architecture Components

| Component | File | Description |
|-----------|------|-------------|
| Audio Encoder | `src/models/audio_encoder.py` | MERT-95M, 768-dim output, layers 5-7 selection |
| MIDI Encoder | `src/models/midi_encoder.py` | 6-layer transformer, 256-dim, OctupleMIDI tokens |
| Projection | `src/models/projection.py` | Aligns both to 512-dim shared space |
| Gated Fusion | `src/models/fusion_gated.py` | GMU-style learned gating |
| Concat Fusion | `src/models/fusion_concat.py` | Simple concatenation + projection |
| CrossAttn Fusion | `src/models/fusion_crossattn.py` | Bidirectional cross-attention |
| Aggregation | `src/models/aggregation.py` | BiLSTM + multi-head attention |
| MTL Head | `src/models/mtl_head.py` | 8-dimension output with uncertainty weighting |
| Lightning Module | `src/models/lightning_module.py` | Full training loop |

### Loss Components

| Loss | Weight | Purpose |
|------|--------|---------|
| Huber (base) | 1.0 | Robust regression |
| Ranking | 0.2 | Pairwise ordering |
| Contrastive | 0.1 | Cross-modal alignment |
| CORAL | 0.3 | Ordinal regression |
| Bootstrap | 0.8 beta | Noisy label handling |
| LDS | enabled | Label distribution smoothing |

### Labeling Functions

Existing heuristics in `src/labeling_functions.py`:
- `MIDITimingVariance` - Grid deviation analysis
- `MIDITempoStability` - Tempo coefficient of variation
- `MIDIVelocityRange` - Dynamic range measurement
- `MIDIVelocitySmoothing` - Transition smoothness
- `MIDINoteDurationVariance` - Articulation variety
- `MIDIPedalCoherence` - Pedal change frequency
- `MIDIPedalTiming` - Pedal-note alignment
- `AudioSpectralCentroid` - Tone brightness
- `AudioAttackTransients` - Articulation clarity

### Degradation Pipeline

`src/data/degradation.py` implements:
- `degrade_midi_timing()` - Gaussian jitter on onsets/offsets
- `inject_wrong_notes()` - Random pitch alterations
- `compress_midi_dynamics()` - Velocity range compression
- `degrade_audio_quality()` - Noise injection and filtering

### Checkpoints

```
/tmp/checkpoints/audio_only/audio_only-epoch=00-val_loss=3.0603.ckpt
/tmp/checkpoints/midi_only/midi_only-epoch=04-val_loss=7.1064.ckpt
/tmp/checkpoints/crossattn/crossattn-epoch=02-val_loss=3.3362.ckpt
/tmp/checkpoints/gated/gated-epoch=03-val_loss=2.6034.ckpt
/tmp/checkpoints/concat/concat-epoch=03-val_loss=2.6008.ckpt
/tmp/checkpoints/midi_pretrain/encoder_pretrained.pt
```

---

## Open Questions

### On Label Quality

1. Do the synthetic degradation labels correlate with human-perceived quality? (Unvalidated)
2. What degradation parameters would create artifacts that are perceptible but not trivially detectable?
3. Could pairwise ranking (A vs B comparisons) reduce label burden while providing useful signal?
4. Are there existing datasets with quality labels that could be used for validation?

### On Fusion Architecture

5. What are the actual gate values in GatedFusion? Is it passing audio through unchanged?
6. Why does CrossAttention underperform simpler fusion methods?
7. Would contrastive pretraining (Phase 3) improve cross-modal alignment, or is it solving a different problem?
8. Is there a fusion architecture that would force MIDI contribution?

### On Audio vs Symbolic

9. For technical dimensions (note accuracy, timing), is symbolic analysis strictly superior to audio?
10. What is the production use case - do users have MIDI, audio, or both?
11. Could audio-to-MIDI transcription + symbolic analysis outperform end-to-end audio models?
12. What dimensions genuinely require audio that MIDI cannot capture?

### On Evaluation

13. What is a realistic performance ceiling given label noise and task difficulty?
14. Should different dimensions have different success thresholds?
15. How should fusion improvement be measured when single-modal performance is already high?

---

## Related Documents

- `docs/RESEARCH_v2.md` - Literature review and theoretical foundation
- `docs/TRAINING_PLAN_v2.md` - Phase 2 experimental protocol
- `docs/ARCHITECTURE.md` - System architecture details
- `configs/experiment_full.yaml` - Training configuration used

---

## Summary

Phase 2 achieved high correlation (r=0.96) on synthetic degradation labels, but fusion provided no improvement over audio-only. The result suggests the model detects degradation artifacts rather than learning generalizable quality assessment. Key diagnostic metrics were not logged, preventing definitive root cause analysis. The relationship between synthetic labels and human-perceived quality remains unvalidated.
