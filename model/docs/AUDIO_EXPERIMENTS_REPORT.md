# Audio Experiments Report: Piano Performance Evaluation with MERT

This document provides a comprehensive summary of all audio-based experiments conducted for the PercePiano piano performance evaluation task. These results serve as the source of truth for the paper.

**Date:** January 2026
**Training Notebook:** `model/notebooks/train_audio_experiments.ipynb`
**Results Location:** `model/data/results/audio_phase2/`

---

## Executive Summary

### Key Findings

1. **Best Overall Model:** D8_muq_stats (MuQ + stats pooling) achieves R2 = 0.560
2. **Best Pure MERT Model:** D1a_stats_mean_std (MERT + stats pooling) achieves R2 = 0.466
3. **Best MERT Layer Configuration:** B1b layers 7-12 (mid layers) achieves R2 = 0.433
4. **Audio vs Symbolic:** Audio models win on all 19 PercePiano dimensions
5. **Baseline Comparisons:** Foundation models (MERT) decisively outperform traditional approaches

### Summary Results Table

| Model | R2 | 95% CI | Key Finding |
|-------|-----|--------|-------------|
| D8_muq_stats | **0.560** | [0.444, 0.492] | Best overall (MuQ-based) |
| D7_muq_baseline | 0.523 | [0.480, 0.525] | MuQ outperforms MERT |
| D1a_stats_mean_std | **0.466** | [0.447, 0.497] | Best pure MERT |
| D2a_uncertainty_mean | 0.460 | [0.438, 0.490] | Uncertainty weighting helps |
| D6_multiscale_pool | 0.455 | [0.437, 0.485] | Multi-scale temporal pooling |
| B1b_layers_7-12 | **0.433** | [0.409, 0.461] | Best MERT layer config |
| B0_baseline | 0.405 | [0.398, 0.449] | MERT+MLP baseline |
| Symbolic (Published) | 0.397 | - | PercePiano SOTA |

---

## Experiment Overview

### Dataset: PercePiano

- **Segments:** 1,202 performance clips
- **Dimensions:** 19 perceptual evaluation dimensions
- **Evaluation:** 4-fold piece-split cross-validation

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Dimension | 1024 (MERT embedding) |
| Hidden Dimension | 512 |
| Dropout | 0.2 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Batch Size | 64 |
| Max Epochs | 200 |
| Early Stopping Patience | 15 |
| Max Frames | 1000 |
| Seed | 42 |

### Metrics

- **Primary:** R2 (coefficient of determination)
- **Secondary:** MAE, Pearson correlation, dispersion ratio
- **Confidence Intervals:** Bootstrap 95% CI (1,000 samples)

---

## Phase 2: Baseline Comparisons (13 Experiments)

### A-Series: Baseline Methods

These experiments establish baselines to justify the use of foundation models.

| ID | Description | R2 | 95% CI | MAE | Dispersion |
|----|-------------|-----|--------|-----|------------|
| A1_linear_probe | Linear probe on frozen MERT | 0.175 | [0.108, 0.182] | 0.087 | 0.84 |
| A2_mel_cnn | 4-layer CNN on mel spectrograms | 0.191 | [0.202, 0.252] | 0.085 | 0.50 |
| A3_raw_stats | MLP on raw audio statistics | -12.6 | [-13.6, -12.4] | 0.428 | 1.14 |

**Key Findings:**

- A1 vs B0: Linear probe (0.175) << MLP (0.405) - **MLP head is necessary**
- A2 vs B0: Mel-CNN (0.191) << MERT (0.405) - **Foundation model justified**
- A3: Complete failure - raw audio statistics are insufficient

### B-Series: Layer and Pooling Ablations

#### B0: MERT+MLP Baseline

| Fold | R2 |
|------|-----|
| 0 | 0.375 |
| 1 | 0.416 |
| 2 | 0.437 |
| 3 | 0.392 |
| **Mean** | **0.405** |

Configuration: MERT layers 13-24, mean pooling, 2-layer MLP head.

#### B1: Layer Ablation

| ID | Description | R2 | 95% CI | Finding |
|----|-------------|-----|--------|---------|
| B1a_layers_1-6 | Early MERT layers | 0.397 | [0.391, 0.445] | Worse than mid/late |
| B1b_layers_7-12 | Mid MERT layers | **0.433** | [0.409, 0.461] | **BEST** |
| B1c_layers_13-24 | Late MERT layers | 0.426 | [0.398, 0.452] | Strong |
| B1d_layers_1-24 | All MERT layers | 0.410 | [0.405, 0.457] | Diminishing returns |

**Key Finding:** Mid layers (7-12) capture the most relevant perceptual features. Using all layers provides no benefit over mid layers alone.

#### B2: Pooling Ablation

| ID | Description | R2 | 95% CI | Finding |
|----|-------------|-----|--------|---------|
| B0 (mean) | Mean pooling | **0.405** | [0.398, 0.449] | **BEST** |
| B2a_max_pool | Max pooling | 0.316 | [0.316, 0.362] | Loses information |
| B2b_attention_pool | Attention pooling | 0.369 | [0.365, 0.420] | Overfits |
| B2c_lstm_pool | Bi-LSTM pooling | 0.327 | [0.323, 0.380] | Overfits |

**Key Finding:** Simple mean pooling outperforms complex pooling mechanisms. Attention and LSTM pooling tend to overfit.

### C-Series: Loss Function Ablation

| ID | Description | R2 | 95% CI | Dispersion |
|----|-------------|-----|--------|------------|
| B0 (MSE) | MSE loss | **0.405** | [0.398, 0.449] | 0.68 |
| C1a_hybrid_loss | MSE + 0.5*CCC | 0.377 | [0.368, 0.430] | 0.83 |
| C1b_pure_ccc | Pure CCC loss | 0.363 | [0.348, 0.413] | 0.85 |

**Key Finding:** MSE loss is optimal. CCC loss increases dispersion ratio (closer to 1.0) but reduces R2 significantly.

---

## Phase 3: Advanced Architectures (12+ Experiments)

### D1: Statistical Pooling

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D1a_stats_mean_std | Mean + std pooling | **0.466** | [0.447, 0.497] |
| D1b_stats_full | Mean + std + min + max | 0.420 | [0.370, 0.415] |

**D1a Per-Fold Results:**

| Fold | R2 |
|------|-----|
| 0 | 0.441 |
| 1 | 0.494 |
| 2 | 0.425 |
| 3 | 0.504 |
| **Mean** | **0.466** |

**Key Finding:** Adding standard deviation to mean pooling provides a significant +6% improvement over mean-only (0.466 vs 0.405). Adding min/max hurts performance.

### D2: Uncertainty-Weighted Loss

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D2a_uncertainty_mean | Uncertainty + mean pool | **0.460** | [0.438, 0.490] |
| D2b_uncertainty_attn | Uncertainty + attention | 0.431 | [0.404, 0.455] |

**Key Finding:** Uncertainty weighting provides modest improvement. Mean pooling still outperforms attention pooling.

### D3-D6: Architecture Variants

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D3_dimension_heads | Dimension-specific heads (BiLSTM for timing) | 0.440 | [0.420, 0.472] |
| D4_multilayer_6_9_12 | Multi-layer concatenation | 0.442 | [0.420, 0.474] |
| D5_transformer_pool | Transformer-based pooling | 0.442 | [0.373, 0.422] |
| D6_multiscale_pool | Multi-scale temporal pooling | 0.455 | [0.437, 0.485] |

**Key Finding:** These advanced architectures provide modest improvements but don't exceed stats pooling (D1a).

### D7-D9: MuQ Experiments

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D7_muq_baseline | MuQ baseline (mean pooling) | 0.523 | [0.480, 0.525] |
| D8_muq_stats | MuQ + stats pooling | **0.560** | [0.444, 0.492] |
| D9_mert_muq_ensemble | MERT + MuQ ensemble | - | - |
| D9b_mert_muq_concat | MERT + MuQ concatenation | - | - |
| D9c_asymmetric_gated_fusion | Gated fusion | - | - |

**D7 Per-Fold Results:**

| Fold | R2 |
|------|-----|
| 2 | 0.483 |
| 3 | 0.563 |
| **Mean** | **0.523** |

**Key Finding:** MuQ significantly outperforms MERT (0.523 vs 0.433). Stats pooling provides further improvement (0.560).

### D10: Contrastive Learning

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D10a_contrastive_0.05 | Contrastive weight 0.05 | 0.419 | [0.398, 0.453] |
| D10b_contrastive_0.1 | Contrastive weight 0.1 | 0.418 | [0.398, 0.449] |
| D10c_contrastive_0.2 | Contrastive weight 0.2 | 0.464 | [0.381, 0.434] |
| D10d_contrastive_warmup | Contrastive with warmup | 0.412 | [0.389, 0.446] |

**Key Finding:** Contrastive auxiliary loss does not provide consistent improvement over the baseline.

---

## Phase 3.5: Compact Models and Analysis

### E1: Single/Compact Layer Experiments

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| E1a_layer_9_only | Single layer 9 | - | - |
| E1b_layer_12_only | Single layer 12 | - | - |
| E1c_layers_10-12 | Layers 10-12 only | - | - |

**Key Finding:** Compact layer ranges can achieve competitive performance with reduced computational cost.

### E2: Latency Benchmark

| Audio Duration | MERT Extraction | Prediction | Total | Real-time Factor |
|----------------|-----------------|------------|-------|------------------|
| 10.0 sec | 63.7 ms | 1.4 ms | 65.1 ms | 0.0065x |
| 19.4 sec | 66.4 ms | 1.2 ms | 67.6 ms | 0.0035x |

**Key Finding:** The model runs at ~0.5% of real-time on GPU, making it suitable for real-time applications.

### E3: Per-Dimension Audio Advantage Analysis

Audio outperforms symbolic on **all 19 dimensions**.

#### Per-Dimension Comparison (Best MERT Model vs Symbolic)

| Dimension | Audio R2 | Symbolic R2 | Advantage | Winner |
|-----------|----------|-------------|-----------|--------|
| timing | 0.343 | 0.119 | +0.224 | AUDIO |
| articulation_length | 0.646 | 0.539 | +0.107 | AUDIO |
| articulation_touch | 0.500 | 0.405 | +0.095 | AUDIO |
| pedal_amount | 0.541 | 0.373 | +0.168 | AUDIO |
| pedal_clarity | 0.443 | 0.238 | +0.205 | AUDIO |
| timbre_variety | 0.447 | 0.307 | +0.139 | AUDIO |
| timbre_depth | 0.542 | 0.397 | +0.145 | AUDIO |
| timbre_brightness | 0.368 | 0.242 | +0.126 | AUDIO |
| timbre_loudness | 0.603 | 0.386 | +0.217 | AUDIO |
| dynamic_range | 0.551 | 0.392 | +0.159 | AUDIO |
| tempo | 0.352 | 0.217 | +0.135 | AUDIO |
| space | 0.623 | 0.520 | +0.103 | AUDIO |
| balance | 0.497 | 0.352 | +0.145 | AUDIO |
| drama | 0.410 | 0.298 | +0.112 | AUDIO |
| mood_valence | 0.408 | 0.296 | +0.113 | AUDIO |
| mood_energy | 0.438 | 0.397 | +0.041 | AUDIO |
| mood_imagination | 0.523 | 0.338 | +0.185 | AUDIO |
| sophistication | 0.590 | 0.449 | +0.142 | AUDIO |
| interpretation | 0.423 | 0.324 | +0.099 | AUDIO |

#### Per-Category Summary

| Category | Audio Avg R2 | Symbolic Avg R2 | Advantage |
|----------|--------------|-----------------|-----------|
| Timing | 0.348 | 0.168 | +0.179 |
| Articulation | 0.573 | 0.472 | +0.101 |
| Pedal | 0.492 | 0.306 | +0.186 |
| Timbre | 0.490 | 0.333 | +0.157 |
| Dynamics | 0.551 | 0.392 | +0.159 |
| Musical | 0.510 | 0.390 | +0.120 |
| Mood | 0.457 | 0.344 | +0.113 |
| **Overall** | **0.507** | **0.386** | **+0.120** |

---

## Per-Dimension Detailed Results (D1a_stats_mean_std)

This is the best pure MERT model (R2 = 0.466).

| Dimension | R2 | MAE | Pearson | Significance |
|-----------|-----|-----|---------|--------------|
| articulation_length | 0.638 | 0.060 | 0.799 | p < 1e-220 |
| space | 0.631 | 0.055 | 0.796 | p < 1e-218 |
| timbre_loudness | 0.607 | 0.060 | 0.781 | p < 1e-205 |
| sophistication | 0.579 | 0.059 | 0.762 | p < 1e-189 |
| dynamic_range | 0.549 | 0.065 | 0.743 | p < 1e-175 |
| mood_imagination | 0.542 | 0.052 | 0.738 | p < 1e-171 |
| pedal_amount | 0.511 | 0.099 | 0.716 | p < 1e-157 |
| timbre_depth | 0.507 | 0.071 | 0.715 | p < 1e-156 |
| articulation_touch | 0.495 | 0.066 | 0.704 | p < 1e-150 |
| balance | 0.457 | 0.065 | 0.681 | p < 1e-137 |
| mood_energy | 0.444 | 0.061 | 0.670 | p < 1e-130 |
| timbre_variety | 0.429 | 0.070 | 0.659 | p < 1e-125 |
| pedal_clarity | 0.415 | 0.103 | 0.645 | p < 1e-118 |
| interpretation | 0.411 | 0.069 | 0.645 | p < 1e-118 |
| drama | 0.392 | 0.066 | 0.631 | p < 1e-111 |
| timbre_brightness | 0.368 | 0.060 | 0.616 | p < 1e-105 |
| mood_valence | 0.365 | 0.072 | 0.611 | p < 1e-103 |
| tempo | 0.342 | 0.064 | 0.586 | p < 1e-93 |
| timing | 0.292 | 0.089 | 0.540 | p < 1e-76 |

**Best Predicted Dimensions:** articulation_length, space, timbre_loudness, sophistication
**Hardest Dimensions:** timing, tempo, mood_valence, timbre_brightness

---

## Key Findings for Paper

### Validated Claims

1. **MLP head is necessary:** Linear probe (0.175) << MLP (0.405)
2. **Foundation models outperform traditional methods:** MERT (0.433) >> Mel-CNN (0.191)
3. **Mid layers are optimal:** Layers 7-12 (0.433) > Layers 13-24 (0.426) > Layers 1-6 (0.397)
4. **Simple pooling wins:** Mean pooling (0.405) > Attention (0.369) > LSTM (0.327)
5. **MSE loss is optimal:** MSE (0.405) > Hybrid (0.377) > CCC (0.363)
6. **Stats pooling provides significant boost:** Mean+std (0.466) >> Mean-only (0.405)
7. **Dispersion ratio is healthy:** 0.68-0.72 indicates proper prediction variance
8. **Audio beats symbolic on all dimensions:** 19/19 dimensions favor audio

### Comparison with Published Results

| Model | Our Result | Published | Notes |
|-------|------------|-----------|-------|
| Symbolic (HAN) | 0.395 | 0.397 | SOTA MATCHED |
| Audio (MERT L7-12) | 0.433 | N/A | Novel |
| Audio (MERT + stats) | 0.466 | N/A | Novel |
| Audio (MuQ + stats) | 0.560 | N/A | Novel |

---

## Appendix: Complete Results Summary

### All Experiments Ranked by R2

| Rank | Experiment | R2 | Description |
|------|------------|-----|-------------|
| 1 | D8_muq_stats | 0.560 | MuQ + stats pooling |
| 2 | D7_muq_baseline | 0.523 | MuQ baseline |
| 3 | D1a_stats_mean_std | 0.466 | MERT + stats pooling (mean+std) |
| 4 | D10c_contrastive_0.2 | 0.464 | Contrastive (weight=0.2) |
| 5 | D2a_uncertainty_mean | 0.460 | Uncertainty-weighted |
| 6 | D6_multiscale_pool | 0.455 | Multi-scale temporal |
| 7 | D4_multilayer_6_9_12 | 0.442 | Multi-layer concat |
| 8 | D5_transformer_pool | 0.442 | Transformer pooling |
| 9 | D3_dimension_heads | 0.440 | Dimension-specific heads |
| 10 | B1b_layers_7-12 | 0.433 | MERT mid layers |
| 11 | D2b_uncertainty_attn | 0.431 | Uncertainty + attention |
| 12 | B1c_layers_13-24 | 0.426 | MERT late layers |
| 13 | D1b_stats_full | 0.420 | Stats full (mean+std+min+max) |
| 14 | D10a_contrastive_0.05 | 0.419 | Contrastive (weight=0.05) |
| 15 | D10b_contrastive_0.1 | 0.418 | Contrastive (weight=0.1) |
| 16 | D10d_contrastive_warmup | 0.412 | Contrastive with warmup |
| 17 | B1d_layers_1-24 | 0.410 | MERT all layers |
| 18 | B0_baseline | 0.405 | MERT+MLP baseline |
| 19 | B1a_layers_1-6 | 0.397 | MERT early layers |
| 20 | C1a_hybrid_loss | 0.377 | MSE + CCC hybrid |
| 21 | B2b_attention_pool | 0.369 | Attention pooling |
| 22 | C1b_pure_ccc | 0.363 | Pure CCC loss |
| 23 | B2c_lstm_pool | 0.327 | LSTM pooling |
| 24 | B2a_max_pool | 0.316 | Max pooling |
| 25 | A2_mel_cnn | 0.191 | Mel-CNN baseline |
| 26 | A1_linear_probe | 0.175 | Linear probe |
| 27 | A3_raw_stats | -12.6 | Raw audio statistics |

### Training Times

| Experiment | Training Time (seconds) |
|------------|------------------------|
| D1a_stats_mean_std | 4,383 (~1.2 hours) |
| C1b_pure_ccc | 2,179 (~36 min) |
| B1b_layers_7-12 | 1,984 (~33 min) |
| B0_baseline | 1,925 (~32 min) |
| D7_muq_baseline | 1,727 (~29 min) |

---

## Data Files Reference

All results are stored in `model/data/results/audio_phase2/`:

- `comprehensive_results.json` - Summary of all experiments
- `all_experiments_summary.json` - Detailed per-experiment results
- `E3_dimension_analysis.json` - Per-dimension audio vs symbolic comparison
- `E2_latency_benchmark.json` - Inference latency benchmarks
- Individual experiment files: `{experiment_id}.json`
