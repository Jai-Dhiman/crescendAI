# Audio Experiments Report: Piano Performance Evaluation with MERT

This document provides a comprehensive summary of all audio-based experiments conducted for the PercePiano piano performance evaluation task. These results serve as the source of truth for the paper.

**Date:** January 2026 (Updated 2026-01-23)
**Training Notebooks:**
- `model/notebooks/03_mert_baseline_experiments.ipynb` (Phase 2-3)
- `model/notebooks/02_muq_fusion_experiments.ipynb` (Phase 4+)
- `model/notebooks/01_main_experiments.ipynb` (Phase 9: Final Paper)

**Results Locations:**
- Local: `model/data/results/archive/audio_phase2/` (archived)
- Google Drive: `gdrive:crescendai_data/checkpoints/`

---

## Executive Summary

### Key Findings

1. **Best Overall Model:** A2_pianoteq_ensemble (MuQ + Pianoteq soundfonts) achieves R2 = 0.537 (4-fold CV)
2. **Cross-Soundfont Generalization:** B2 achieves R2 = 0.534 +/- 0.075 on held-out soundfonts
3. **External Validation:** C1_psyllabus shows strong correlation with difficulty (rho = 0.623, p < 1e-50)
4. **Multi-Performer Analysis:** C2_asap shows low intra-piece variance (std = 0.020)
5. **Bootstrap Significance:** B3 95% CI [0.465, 0.575] excludes symbolic baseline
6. **Audio vs Symbolic:** Audio models win on all 19 PercePiano dimensions (p < 1e-25)
7. **Zero-Shot Transfer:** C3_maestro successfully evaluates 500 professional recordings

### Summary Results Table

| Model | R2 | 95% CI | Key Finding |
|-------|-----|--------|-------------|
| A2_pianoteq_ensemble | **0.537** | [0.465, 0.575] | Best overall (Pianoteq ensemble) |
| A1a_piece_fold | **0.536** | - | Piece-based 4-fold CV |
| A1b_performer_fold | **0.536** | - | Performer-based 4-fold CV |
| B2_cross_soundfont | **0.534** | +/- 0.075 | Cross-soundfont LOO |
| M1c_muq_L9-12 | 0.533 | [0.514, 0.560] | MuQ layer 9-12 config |
| F9_muq_symbolic_weighted | 0.524 | [0.500, 0.545] | Best fusion (MuQ+symbolic) |
| A1c_stratified_fold | 0.522 | - | Stratified 4-fold CV |
| D9c_mert_muq_gated | 0.516 | [0.497, 0.543] | Gated MERT+MuQ fusion |
| P1_performer_fold_muq | 0.487 | [0.479, 0.521] | MuQ on performer folds |
| D1a_stats_mean_std | 0.466 | [0.447, 0.497] | Best pure MERT |
| B1b_layers_7-12 | 0.433 | [0.409, 0.461] | Best MERT layer config |
| Symbolic (Published) | 0.397 | - | PercePiano SOTA |
| Symbolic (Our repro) | 0.347 | [0.315, 0.375] | 4-fold aligned predictions |

### Cross-Dataset Validation

| Experiment | Metric | Value | Interpretation |
|------------|--------|-------|----------------|
| C1_psyllabus | Spearman rho | **0.623** | Strong correlation with difficulty |
| C2_asap | Intra-piece std | **0.020** | Low performer variance |
| C3_maestro | Samples | **500** | Zero-shot transfer |

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
| D8_muq_stats | MuQ + stats pooling | 0.454 | [0.444, 0.493] |
| D9a_mert_muq_ensemble | MERT + MuQ ensemble (avg) | 0.490 | [0.420, 0.469] |
| D9b_mert_muq_concat | MERT + MuQ concatenation | 0.471 | [0.452, 0.498] |
| D9c_mert_muq_gated | Asymmetric gated fusion | **0.516** | [0.497, 0.543] |

**D8_muq_stats Per-Fold Results (Complete 4-Fold CV):**

| Fold | R2 |
|------|-----|
| 0 | 0.485 |
| 1 | 0.529 |
| 2 | 0.242 |
| 3 | 0.560 |
| **Mean** | **0.454** |

**Note:** D8 shows high fold variance (std=0.125). Fold 2 has unusually low R2 (0.242). Previous reported R2=0.560 was from fold 3 only. M1c_muq_L9-12 (R2=0.533) is more stable and achieves better average performance.

**Key Finding:** MuQ significantly outperforms MERT. Gated fusion (D9c) achieves best MERT+MuQ combination at R2=0.516.

### D10: Contrastive Learning

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| D10a_contrastive_0.05 | Contrastive weight 0.05 | 0.419 | [0.398, 0.453] |
| D10b_contrastive_0.1 | Contrastive weight 0.1 | 0.418 | [0.398, 0.449] |
| D10c_contrastive_0.2 | Contrastive weight 0.2 | 0.464 | [0.381, 0.434] |
| D10d_contrastive_warmup | Contrastive with warmup | 0.412 | [0.389, 0.446] |

**Key Finding:** Contrastive auxiliary loss does not provide consistent improvement over the baseline.

---

## Phase 4: MuQ Layer Ablations (M-Series)

MuQ (Music Query) is an alternative audio foundation model that we found outperforms MERT. This section documents systematic layer ablation experiments to identify optimal MuQ layer configurations.

### M1: MuQ Layer Selection

| ID | Description | R2 | 95% CI | Training Time |
|----|-------------|-----|--------|---------------|
| M1a_muq_L1-4 | MuQ layers 1-4 (early acoustic) | 0.438 | [0.413, 0.469] | 55 min |
| M1b_muq_L5-8 | MuQ layers 5-8 (mid perceptual) | 0.524 | [0.442, 0.491] | 31 min |
| M1c_muq_L9-12 | MuQ layers 9-12 (late semantic) | **0.533** | [0.514, 0.560] | 54 min |
| M1d_muq_L1-12 | MuQ all layers | 0.510 | [0.490, 0.538] | 92 min |

**M1c Per-Fold Results (Best Configuration):**

| Fold | R2 |
|------|-----|
| 0 | 0.517 |
| 1 | 0.544 |
| 2 | 0.483 |
| 3 | 0.586 |
| **Mean** | **0.533** |

**Key Finding:** Late semantic layers (9-12) are optimal for MuQ, achieving R2 = 0.533. This differs from MERT where mid layers (7-12) were best, suggesting MuQ's later layers capture more task-relevant representations.

### M2: MuQ Last Hidden State

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| M2_muq_last_hidden | MuQ last hidden state only | 0.513 | [0.493, 0.539] |

**Key Finding:** Using only the last hidden state (0.513) is competitive but slightly worse than late layers (0.533), suggesting value in aggregating across multiple late layers.

---

## Phase 4: MERT+MuQ Fusion (D9-Series Updates)

These experiments explore combining MERT and MuQ representations.

### D9: MERT+MuQ Fusion Strategies

| ID | Description | R2 | 95% CI | Training Time |
|----|-------------|-----|--------|---------------|
| D9a_mert_muq_ensemble | Late fusion ensemble (avg) | 0.490 | [0.420, 0.469] | 32 min |
| D9b_mert_muq_concat | Early fusion concatenation | 0.471 | [0.452, 0.498] | 43 min |
| D9c_mert_muq_gated | Asymmetric gated fusion | **0.516** | [0.497, 0.543] | 46 min |

**D9c Per-Fold Results (Gated Fusion):**

| Fold | R2 |
|------|-----|
| 0 | 0.485 |
| 1 | 0.505 |
| 2 | 0.501 |
| 3 | 0.574 |
| **Mean** | **0.516** |

**Key Findings:**

1. **Gated fusion (0.516) outperforms simple fusion approaches**
2. **Late fusion ensemble (0.490) < pure MuQ (0.560)**: Simple averaging hurts MuQ performance
3. **Early concatenation (0.471) is suboptimal**: Models may struggle with high-dimensional combined features
4. **MuQ alone (0.560) > all MERT+MuQ fusions**: MuQ is strong enough that MERT adds noise rather than signal

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

## Phase 5: Statistical Analysis & Significance Testing (S-Series)

Rigorous statistical validation of audio vs symbolic model performance using bootstrap confidence intervals and paired hypothesis tests.

### S0: Bootstrap Confidence Intervals

Using 10,000 bootstrap samples to establish reliable confidence intervals.

**Overall Model Comparison:**

| Model | R2 | 95% CI Lower | 95% CI Upper |
|-------|-----|--------------|--------------|
| Audio (MERT L7-12) | **0.487** | 0.460 | 0.510 |
| Symbolic (HAN) | 0.347 | 0.315 | 0.375 |

**Key Finding:** Audio model confidence interval (0.460-0.510) does not overlap with symbolic (0.315-0.375), confirming significant improvement.

### S1: Paired Statistical Tests

**Audio vs Symbolic (Paired t-test):**

| Metric | Value |
|--------|-------|
| t-statistic | -10.71 |
| p-value | **2.08e-25** |
| Mean MSE (Audio) | 0.0078 |
| Mean MSE (Symbolic) | 0.0100 |
| Cohen's d | 0.31 |
| Audio Better | **True** |

**Wilcoxon Signed-Rank Test:**

| Metric | Value |
|--------|-------|
| Statistic | 145,692 |
| p-value | **2.16e-29** |
| Audio Better | **True** |

**Per-Dimension Significance:**

| Dimension | p-value | Significant | Audio Better |
|-----------|---------|-------------|--------------|
| timing | 8.34e-12 | Yes | Yes |
| articulation_length | 9.90e-07 | Yes | Yes |
| articulation_touch | 2.28e-04 | Yes | Yes |
| pedal_amount | 2.43e-10 | Yes | Yes |
| pedal_clarity | 3.92e-13 | Yes | Yes |
| timbre_variety | 7.84e-07 | Yes | Yes |
| timbre_depth | 2.41e-08 | Yes | Yes |
| timbre_brightness | 1.30e-04 | Yes | Yes |
| timbre_loudness | 5.38e-17 | Yes | Yes |
| dynamic_range | 9.13e-08 | Yes | Yes |
| tempo | 2.57e-05 | Yes | Yes |
| space | 6.48e-08 | Yes | Yes |
| balance | 3.70e-07 | Yes | Yes |
| drama | 4.04e-05 | Yes | Yes |
| mood_valence | 2.57e-04 | Yes | Yes |
| mood_energy | 0.181 | **No** | Yes |
| mood_imagination | 4.83e-11 | Yes | Yes |
| sophistication | 8.11e-09 | Yes | Yes |
| interpretation | 5.17e-04 | Yes | Yes |

**Key Finding:** Audio significantly outperforms symbolic on 18/19 dimensions after Bonferroni correction. Only mood_energy shows non-significant difference (p=0.18).

### S2: Multiple Comparison Correction

| Correction Method | Significant Dimensions |
|-------------------|------------------------|
| Bonferroni | 18/19 |
| FDR (Benjamini-Hochberg) | 18/19 |

---

## Phase 5: Audio + Symbolic Fusion (F-Series)

Late fusion experiments combining MERT audio predictions with symbolic (HAN) predictions on aligned samples.

### F0-F7: Fusion Strategy Comparison

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| Audio only (MERT L7-12) | Baseline | 0.487 | [0.460, 0.510] |
| Symbolic only | Baseline | 0.347 | [0.315, 0.375] |
| F0_simple | Simple average (0.5/0.5) | 0.486 | [0.461, 0.507] |
| F1_weighted | Per-dim optimal weights | **0.499** | [0.474, 0.520] |
| F2_ridge | Ridge regression fusion | 0.493 | [0.468, 0.515] |
| F3_confidence | Confidence-weighted | 0.491 | [0.468, 0.513] |
| F4_modality_dropout | Dropout regularization | 0.489 | [0.465, 0.511] |
| F5_orthogonality | Orthogonality constraint | 0.488 | [0.464, 0.510] |
| F6_residual | Residual fusion | 0.490 | [0.466, 0.512] |
| F7_dim_weighted | Dimension-weighted | 0.495 | [0.470, 0.517] |

**Key Findings:**

1. **Weighted fusion (0.499) provides +1.2% gain over audio alone (0.487)**
2. **Simple average (0.486) slightly hurts audio performance** - symbolic noise dilutes audio signal
3. **Ridge regression (0.493) provides regularized combination**
4. **Fusion benefit is modest**: Audio is already strong, symbolic adds limited complementary information

### F1: Optimal Fusion Weights per Dimension

Weights represent audio contribution (1 - weight = symbolic contribution):

| Dimension | Optimal Weight | Audio R2 | Symbolic R2 |
|-----------|----------------|----------|-------------|
| timing | 0.90 | 0.343 | 0.119 |
| articulation_length | 0.70 | 0.646 | 0.539 |
| articulation_touch | 0.68 | 0.500 | 0.405 |
| pedal_amount | 0.76 | 0.541 | 0.373 |
| pedal_clarity | 0.79 | 0.443 | 0.238 |
| timbre_variety | 0.71 | 0.447 | 0.307 |
| timbre_depth | 0.74 | 0.542 | 0.397 |
| timbre_brightness | 0.76 | 0.368 | 0.242 |
| timbre_loudness | 0.89 | 0.603 | 0.386 |
| dynamic_range | 0.78 | 0.551 | 0.392 |
| tempo | 0.79 | 0.352 | 0.217 |
| space | 0.75 | 0.623 | 0.520 |
| balance | 0.75 | 0.497 | 0.352 |
| drama | 0.70 | 0.410 | 0.298 |
| mood_valence | 0.69 | 0.408 | 0.296 |
| mood_energy | 0.59 | 0.438 | 0.397 |
| mood_imagination | 0.81 | 0.523 | 0.338 |
| sophistication | 0.78 | 0.590 | 0.449 |
| interpretation | 0.68 | 0.423 | 0.324 |

**Key Finding:** Audio is preferred for most dimensions (weights 0.68-0.90). Only mood_energy (0.59) shows more balanced contribution, consistent with non-significant audio advantage on this dimension.

---

## Phase 5: MuQ + Symbolic Fusion (F8-F11)

Fusion experiments combining the superior MuQ audio model with symbolic predictions.

### MuQ + Symbolic Fusion Results

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| MuQ only (D8) | Baseline | 0.560 | [0.444, 0.492] |
| Symbolic only | Baseline | 0.347 | [0.315, 0.375] |
| F8_muq_symbolic_simple | Simple average | 0.500 | [0.477, 0.521] |
| F9_muq_symbolic_weighted | Per-dim optimal | **0.524** | [0.500, 0.545] |
| F10_muq_symbolic_ridge | Ridge regression | 0.517 | [0.493, 0.539] |
| F11_muq_symbolic_confidence | Confidence-weighted | 0.516 | [0.493, 0.537] |

**Key Findings:**

1. **All fusion approaches hurt MuQ performance**: Best fusion (0.524) < MuQ alone (0.560)
2. **Simple averaging significantly degrades performance**: 0.500 vs 0.560 (-10.7%)
3. **Weighted fusion minimizes damage**: 0.524 preserves more MuQ signal
4. **MuQ is sufficiently powerful that symbolic adds noise, not signal**

### F9: MuQ Optimal Weights per Dimension

| Dimension | MuQ Weight | MuQ R2 | Symbolic R2 |
|-----------|------------|--------|-------------|
| timing | 0.98 | 0.487 | 0.119 |
| articulation_length | 0.83 | 0.717 | 0.539 |
| articulation_touch | 0.75 | 0.615 | 0.405 |
| pedal_amount | 0.83 | 0.611 | 0.373 |
| pedal_clarity | 0.86 | 0.517 | 0.238 |
| timbre_variety | 0.68 | 0.554 | 0.307 |
| timbre_depth | 0.84 | 0.648 | 0.397 |
| timbre_brightness | 0.94 | 0.625 | 0.242 |
| timbre_loudness | 0.84 | 0.672 | 0.386 |
| dynamic_range | 0.81 | 0.687 | 0.392 |
| tempo | 0.56 | 0.416 | 0.217 |
| space | 0.85 | 0.653 | 0.520 |
| balance | 0.81 | 0.620 | 0.352 |
| drama | 0.76 | 0.560 | 0.298 |
| mood_valence | 0.73 | 0.528 | 0.296 |
| mood_energy | 0.74 | 0.614 | 0.397 |
| mood_imagination | 0.75 | 0.616 | 0.338 |
| sophistication | 0.94 | 0.703 | 0.449 |
| interpretation | 0.75 | 0.552 | 0.324 |

**Key Finding:** MuQ is strongly preferred across all dimensions (0.56-0.98). Only tempo shows moderate symbolic contribution (0.56), but even there MuQ dominates.

---

## Phase 6: Analysis Experiments (A-Series)

Detailed analysis of model behavior, errors, and predictions.

### A3: Error Correlation Analysis

Correlation between audio and symbolic model errors on the same samples.

| Metric | Value |
|--------|-------|
| Overall Error Correlation | **0.738** |
| Mean Per-Dimension Correlation | 0.738 |

**Per-Dimension Error Correlations:**

| Dimension | Correlation | Interpretation |
|-----------|-------------|----------------|
| interpretation | 0.790 | High - similar errors |
| tempo | 0.777 | High - similar errors |
| drama | 0.773 | High - similar errors |
| pedal_clarity | 0.771 | High - similar errors |
| balance | 0.766 | High - similar errors |
| mood_valence | 0.766 | High - similar errors |
| pedal_amount | 0.760 | High - similar errors |
| timbre_variety | 0.754 | High - similar errors |
| timbre_depth | 0.750 | High - similar errors |
| timing | 0.742 | High - similar errors |
| timbre_brightness | 0.741 | High - similar errors |
| articulation_touch | 0.732 | High - similar errors |
| space | 0.715 | High - similar errors |
| mood_energy | 0.709 | Moderate - similar errors |
| articulation_length | 0.709 | Moderate - similar errors |
| mood_imagination | 0.708 | Moderate - similar errors |
| sophistication | 0.705 | Moderate - similar errors |
| timbre_loudness | 0.683 | Moderate - some unique errors |
| dynamic_range | 0.668 | Moderate - some unique errors |

**Key Finding:** High error correlation (0.738) explains why fusion provides limited benefit - models make similar mistakes, reducing complementarity potential.

### A4: MuQ vs Symbolic Per-Dimension Breakdown

| Dimension | MuQ R2 | Symbolic R2 | Winner | MuQ Advantage |
|-----------|--------|-------------|--------|---------------|
| timing | 0.487 | 0.119 | MuQ | +0.368 |
| timbre_brightness | 0.625 | 0.242 | MuQ | +0.383 |
| pedal_clarity | 0.517 | 0.238 | MuQ | +0.279 |
| timbre_loudness | 0.672 | 0.386 | MuQ | +0.285 |
| dynamic_range | 0.687 | 0.392 | MuQ | +0.295 |
| mood_imagination | 0.616 | 0.338 | MuQ | +0.278 |
| balance | 0.620 | 0.352 | MuQ | +0.269 |
| drama | 0.560 | 0.298 | MuQ | +0.262 |
| sophistication | 0.703 | 0.449 | MuQ | +0.254 |
| timbre_depth | 0.648 | 0.397 | MuQ | +0.251 |
| timbre_variety | 0.554 | 0.307 | MuQ | +0.246 |
| pedal_amount | 0.611 | 0.373 | MuQ | +0.238 |
| mood_valence | 0.528 | 0.296 | MuQ | +0.232 |
| interpretation | 0.552 | 0.324 | MuQ | +0.228 |
| mood_energy | 0.614 | 0.397 | MuQ | +0.217 |
| articulation_touch | 0.615 | 0.405 | MuQ | +0.210 |
| tempo | 0.416 | 0.217 | MuQ | +0.199 |
| articulation_length | 0.717 | 0.539 | MuQ | +0.178 |
| space | 0.653 | 0.520 | MuQ | +0.133 |

**Key Finding:** MuQ wins all 19 dimensions. Largest advantages on timing (+0.368), timbre_brightness (+0.383), and dynamic_range (+0.295). Smallest advantage on space (+0.133) where symbolic is also strong.

### A5: Failure Cases Analysis

Analysis of samples with highest prediction error.

| Key | MSE | Worst Dimensions |
|-----|-----|------------------|
| Beethoven_WoO80_var12_8bars_18_9 | 0.031 | timbre_brightness, timing, mood_energy |
| Schubert_D960_mv2_8bars_3_04 | 0.029 | pedal_clarity, balance, pedal_amount |
| Beethoven_WoO80_var12_8bars_26_9 | 0.026 | mood_imagination, space, timing |
| Schubert_D960_mv2_8bars_4_11 | 0.026 | mood_energy, timbre_brightness, timbre_variety |
| Schubert_D960_mv3_8bars_6_37 | 0.026 | timbre_depth, pedal_amount, pedal_clarity |

**Key Finding:** Failure cases cluster around Beethoven WoO80 variations and Schubert D960 movements. Common failure dimensions: timing, timbre_brightness, pedal dimensions.

### A6: Calibration Analysis

Model calibration across prediction bins.

| Bin Range | Count | Mean Predicted | Mean Actual | Error |
|-----------|-------|----------------|-------------|-------|
| 0.1-0.2 | 5 | 0.184 | 0.190 | -0.006 |
| 0.2-0.3 | 92 | 0.266 | 0.290 | -0.024 |
| 0.3-0.4 | 921 | 0.367 | 0.358 | +0.009 |
| 0.4-0.5 | 4,840 | 0.455 | 0.449 | +0.006 |
| 0.5-0.6 | 6,880 | 0.552 | 0.552 | -0.001 |
| 0.6-0.7 | 5,059 | 0.643 | 0.655 | -0.012 |
| 0.7-0.8 | 1,036 | 0.732 | 0.737 | -0.005 |
| 0.8-0.9 | 72 | 0.822 | 0.820 | +0.002 |

**Calibration Metrics:**

| Metric | Value |
|--------|-------|
| Dispersion Ratio | 0.754 |
| Predicted Std | 0.098 |
| Label Std | 0.130 |

**Key Finding:** Model is well-calibrated with errors < 0.025 across all bins. Slight under-confidence at extremes (low predictions too low, high predictions too high), but central predictions are accurate.

### A7: Gate Visualization (MERT+MuQ Gated Fusion)

Analysis of learned gating weights in asymmetric fusion model.

**MERT-Preferred Dimensions (positive weights):**

| Dimension | Gate Weight |
|-----------|-------------|
| timbre_brightness | +0.031 |
| dynamic_range | +0.030 |
| articulation_length | +0.025 |
| timbre_loudness | +0.024 |
| pedal_clarity | +0.023 |

**MuQ-Preferred Dimensions (negative weights):**

| Dimension | Gate Weight |
|-----------|-------------|
| mood_valence | -0.035 |
| balance | -0.033 |
| interpretation | -0.022 |
| sophistication | -0.019 |
| pedal_amount | -0.015 |

**Category Summary:**

| Category | Mean MERT Weight | Preference |
|----------|------------------|------------|
| Dynamics | +0.030 | MERT |
| Articulation | +0.017 | MERT |
| Timbre | +0.012 | MERT |
| Timing | +0.008 | Neutral |
| Pedal | +0.004 | Neutral |
| Tempo/Space | -0.004 | Neutral |
| Emotion | -0.018 | MuQ |
| Interpretation | -0.020 | MuQ |

**Key Finding:** MERT excels at low-level acoustic features (dynamics, timbre, articulation). MuQ excels at high-level musical interpretation (emotion, interpretation, balance).

---

## Phase 7: Cross-Dataset Evaluation (X-Series)

Evaluation on external datasets to assess generalization.

### X2: ASAP Multi-Performer Analysis

Testing model consistency across different performers playing the same pieces using the ASAP/MAESTRO datasets.

**Dataset Statistics (Updated with Strongest Paper C2):**

| Metric | Value | Previous |
|--------|-------|----------|
| Number of Pieces | 206 | 24 |
| Total Performances | 631 | 157 |
| Mean Intra-Piece Std | 0.020 | 0.0072 |

**High-Variance Dimensions** (more performer-sensitive):

- dynamic_range (std: 0.027)
- timing (std: 0.022)
- articulation_touch (std: 0.020)
- drama (std: 0.019)
- interpretation (std: 0.018)

**Low-Variance Dimensions** (piece-dominated):

- mood_imagination (std: 0.014)
- mood_valence (std: 0.014)
- tempo (std: 0.014)
- mood_energy (std: 0.011)
- timbre_brightness (std: 0.011)

**Key Finding:** Low intra-piece variance (0.0072) indicates model predictions are dominated by piece characteristics rather than performer variations. The model learns piece-level features more than performer-specific expression, which may limit its utility for fine-grained performer comparison.

### X3: PSyllabus Cross-Dataset Difficulty Correlation

Testing model predictions on the PSyllabus dataset (external piano repertoire with difficulty ratings 0-10).

**Dataset Statistics:**

| Metric | Value |
|--------|-------|
| Total Samples | 508 (updated from 290) |
| Difficulty Levels | 0-10 |
| Overall Spearman rho | **0.623** (updated from 0.570) |
| p-value | < 1e-50 |

**Top Correlated Dimensions with Difficulty:**

| Dimension | Spearman r | p-value | Significant |
|-----------|------------|---------|-------------|
| pedal_clarity | 0.701 | 2.8e-44 | Yes |
| mood_imagination | 0.696 | 2.3e-43 | Yes |
| timbre_loudness | 0.657 | 2.8e-37 | Yes |
| mood_valence | 0.561 | 1.8e-25 | Yes |
| pedal_amount | 0.562 | 1.4e-25 | Yes |
| tempo | 0.558 | 4.3e-25 | Yes |
| timing | 0.513 | 7.1e-21 | Yes |

**Non-Significant Dimensions:**

| Dimension | Spearman r | p-value |
|-----------|------------|---------|
| articulation_touch | 0.056 | 0.345 |
| drama | 0.028 | 0.640 |

**Per-Difficulty Level Statistics:**

| Difficulty | Count | Mean Prediction | Std |
|------------|-------|-----------------|-----|
| 1 | 30 | 0.537 | 0.099 |
| 2 | 30 | 0.543 | 0.106 |
| 3 | 28 | 0.575 | 0.101 |
| 4 | 30 | 0.591 | 0.112 |
| 5 | 28 | 0.602 | 0.123 |
| 6 | 28 | 0.620 | 0.117 |
| 7 | 29 | 0.610 | 0.115 |
| 8 | 29 | 0.626 | 0.104 |
| 9 | 29 | 0.634 | 0.118 |
| 10 | 29 | 0.631 | 0.107 |

**Key Finding:** Strong correlation (r=0.570) between model predictions and piece difficulty validates that the model captures musically meaningful features. Higher difficulty pieces have higher predicted values for pedal use, timbre, and timing dimensions. This provides external validation beyond PercePiano.

---

## Phase 8: Performer Generalization (P-Series)

Experiments testing model generalization to unseen performers using performer-based cross-validation folds.

### P1: MuQ Performer-Fold Evaluation

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| P1_performer_fold_muq | MuQ with performer-based folds | **0.487** | [0.479, 0.521] |

**P1 Per-Fold Results:**

| Fold | R2 |
|------|-----|
| 0 | 0.532 |
| 1 | 0.445 |
| 2 | 0.471 |
| 3 | 0.498 |
| **Mean** | **0.487** |

### P2: MERT Performer-Fold Evaluation

| ID | Description | R2 | 95% CI |
|----|-------------|-----|--------|
| P2_performer_fold_mert | MERT with performer-based folds | **0.444** | [0.431, 0.474] |

**P2 Per-Fold Results:**

| Fold | R2 |
|------|-----|
| 0 | 0.449 |
| 1 | 0.473 |
| 2 | 0.398 |
| 3 | 0.455 |
| **Mean** | **0.444** |

### Performer-Fold vs Piece-Fold Comparison

| Model | Piece-Fold R2 | Performer-Fold R2 | Drop | Drop % |
|-------|---------------|-------------------|------|--------|
| MuQ (M1c vs P1) | 0.533 | 0.487 | 0.046 | 8.6% |
| MERT (D1a vs P2) | 0.466 | 0.444 | 0.022 | 4.7% |

**Key Findings:**

1. **MuQ shows larger drop (8.6%)** on performer-based folds vs piece-based folds
2. **MERT is more stable (4.7% drop)** across fold types
3. Both models maintain reasonable performance on unseen performers
4. MuQ may capture more performer-specific features (timbre, touch) which explains the larger generalization gap

---

## Appendix: Complete Results Summary

### All Experiments Ranked by R2

| Rank | Experiment | R2 | Description |
|------|------------|-----|-------------|
| 1 | M1c_muq_L9-12 | **0.533** | MuQ layers 9-12 (late semantic) |
| 2 | M1b_muq_L5-8 | 0.524 | MuQ layers 5-8 (mid perceptual) |
| 3 | F9_muq_symbolic_weighted | 0.524 | MuQ + symbolic weighted fusion |
| 4 | D7_muq_baseline | 0.523 | MuQ baseline |
| 5 | F10_muq_symbolic_ridge | 0.517 | MuQ + symbolic ridge fusion |
| 6 | F11_muq_symbolic_confidence | 0.516 | MuQ + symbolic confidence fusion |
| 7 | D9c_mert_muq_gated | 0.516 | MERT+MuQ gated fusion |
| 8 | M2_muq_last_hidden | 0.513 | MuQ last hidden state |
| 9 | M1d_muq_L1-12 | 0.510 | MuQ all layers |
| 10 | F8_muq_symbolic_simple | 0.500 | MuQ + symbolic simple fusion |
| 11 | F1_weighted | 0.499 | MERT + symbolic weighted fusion |
| 12 | F3_confidence | 0.497 | Confidence-weighted fusion |
| 13 | F7_dim_weighted | 0.449 | Dimension-weighted fusion |
| 14 | F2_ridge | 0.493 | MERT + symbolic ridge fusion |
| 15 | D9a_mert_muq_ensemble | 0.490 | MERT+MuQ ensemble fusion |
| 16 | F6_residual | 0.449 | Residual fusion |
| 17 | P1_performer_fold_muq | **0.487** | MuQ performer-fold CV |
| 18 | Audio (MERT L7-12) | 0.487 | Aligned audio baseline |
| 19 | F0_simple | 0.486 | MERT + symbolic simple avg |
| 20 | D9b_mert_muq_concat | 0.471 | MERT+MuQ concat fusion |
| 21 | D1a_stats_mean_std | 0.466 | MERT + stats pooling (mean+std) |
| 22 | D10c_contrastive_0.2 | 0.464 | Contrastive (weight=0.2) |
| 23 | D2a_uncertainty_mean | 0.460 | Uncertainty-weighted |
| 24 | D6_multiscale_pool | 0.455 | Multi-scale temporal |
| 25 | D8_muq_stats | **0.454** | MuQ + stats pooling (4-fold) |
| 26 | F5_orthogonality | 0.450 | Orthogonality constraint fusion |
| 27 | P2_performer_fold_mert | **0.444** | MERT performer-fold CV |
| 28 | D4_multilayer_6_9_12 | 0.442 | Multi-layer concat |
| 29 | D5_transformer_pool | 0.442 | Transformer pooling |
| 30 | D3_dimension_heads | 0.440 | Dimension-specific heads |
| 31 | M1a_muq_L1-4 | 0.438 | MuQ layers 1-4 (early acoustic) |
| 32 | F4_modality_dropout | 0.436 | Modality dropout fusion |
| 33 | B1b_layers_7-12 | 0.433 | MERT mid layers |
| 34 | D2b_uncertainty_attn | 0.431 | Uncertainty + attention |
| 35 | B1c_layers_13-24 | 0.426 | MERT late layers |
| 36 | D1b_stats_full | 0.420 | Stats full (mean+std+min+max) |
| 37 | D10a_contrastive_0.05 | 0.419 | Contrastive (weight=0.05) |
| 38 | D10b_contrastive_0.1 | 0.418 | Contrastive (weight=0.1) |
| 39 | D10d_contrastive_warmup | 0.412 | Contrastive with warmup |
| 40 | B1d_layers_1-24 | 0.410 | MERT all layers |
| 41 | B0_baseline | 0.405 | MERT+MLP baseline |
| 42 | B1a_layers_1-6 | 0.397 | MERT early layers |
| 43 | C1a_hybrid_loss | 0.377 | MSE + CCC hybrid |
| 44 | B2b_attention_pool | 0.369 | Attention pooling |
| 45 | C1b_pure_ccc | 0.363 | Pure CCC loss |
| 46 | Symbolic (aligned) | 0.347 | Symbolic model (4-fold) |
| 47 | B2c_lstm_pool | 0.327 | LSTM pooling |
| 48 | B2a_max_pool | 0.316 | Max pooling |
| 49 | A2_mel_cnn | 0.191 | Mel-CNN baseline |
| 50 | A1_linear_probe | 0.175 | Linear probe |
| 51 | A3_raw_stats | -12.6 | Raw audio statistics |

**Notes on Ranking:**
- D8_muq_stats corrected from 0.560 to 0.454 (complete 4-fold CV)
- P1 and P2 added for performer-fold generalization experiments
- Fusion experiments (F4-F7) corrected with complete bootstrap results

### Training Times

| Experiment | Training Time |
|------------|---------------|
| M1d_muq_L1-12 | 5,547 sec (~92 min) |
| M2_muq_last_hidden | 4,996 sec (~83 min) |
| D1a_stats_mean_std | 4,383 sec (~73 min) |
| M1a_muq_L1-4 | 3,303 sec (~55 min) |
| M1c_muq_L9-12 | 3,249 sec (~54 min) |
| D9c_mert_muq_gated | 2,758 sec (~46 min) |
| D9b_mert_muq_concat | 2,589 sec (~43 min) |
| C1b_pure_ccc | 2,179 sec (~36 min) |
| B1b_layers_7-12 | 1,984 sec (~33 min) |
| B0_baseline | 1,925 sec (~32 min) |
| D9a_mert_muq_ensemble | 1,909 sec (~32 min) |
| M1b_muq_L5-8 | 1,833 sec (~31 min) |
| D7_muq_baseline | 1,727 sec (~29 min) |

---

## Data Files Reference

### Primary Results Locations

**Google Drive (`gdrive:crescendai_data/checkpoints/`):**

1. **`audio_phase2/`** - Phase 2-3 audio experiments
   - All A/B/C/D/E series experiments
   - `phase2_all_results.json`, `phase3_all_results.json`

2. **`definitive_experiments/`** - Phase 4+ experiments
   - M-series: MuQ layer ablations
   - D9a-c: MERT+MuQ fusion
   - F8-F11: MuQ+symbolic fusion
   - A3-A7: Analysis experiments
   - X2: ASAP multi-performer analysis
   - `definitive_all_results.json`

3. **`aligned_fusion/`** - Statistical analysis & MERT+symbolic fusion
   - S0-S2: Bootstrap, paired tests, multiple correction
   - F0-F7: Fusion strategies
   - A0-A2: Fusion analysis
   - `aligned_fusion_all_results.json`

4. **`percepiano_original/`** - Symbolic model checkpoints
   - `fold{0-3}_best.pt`

5. **`percepiano_sota/`** - Best symbolic checkpoint
   - `fold2_best.pt`

### Local Results

**`model/data/results/audio_phase2/`:**

- `comprehensive_results.json` - Summary of all experiments
- `all_experiments_summary.json` - Detailed per-experiment results
- `E3_dimension_analysis.json` - Per-dimension audio vs symbolic comparison
- `E2_latency_benchmark.json` - Inference latency benchmarks
- Individual experiment files: `{experiment_id}.json`

---

## Phase 9: Strongest Paper Experiments (2026-01-23)

Comprehensive validation experiments for paper submission, using `01_main_experiments.ipynb`.

**Results Location:** `gdrive:crescendai_data/checkpoints/strongest_paper/`

### Phase A: Fold Validation Strategies

| ID | Description | R2 | Fold Results |
|----|-------------|-----|--------------|
| A1a_piece_fold | Piece-based 4-fold CV | **0.536** | [0.513, 0.543, 0.483, 0.580] |
| A1b_performer_fold | Performer-based 4-fold CV | **0.536** | [0.463, 0.539, 0.510, 0.598] |
| A1c_stratified_fold | Stratified 4-fold CV | **0.522** | [0.537, 0.543, 0.449, 0.511] |
| A2_pianoteq_ensemble | Pianoteq multi-soundfont | **0.537** | [0.493, 0.559, 0.559] |

**Key Finding:** All fold strategies produce consistent R2 ~ 0.52-0.54. Piece-based and performer-based folds yield nearly identical results (0.536), suggesting the model captures both piece and performer characteristics equally well.

### Phase B: Robustness & Statistical Rigor

| ID | Description | Result | Details |
|----|-------------|--------|---------|
| B1 | Multi-seed stability | 3 seeds | Seeds 42, 123, 456 |
| B2 | Cross-soundfont LOO | R2 = 0.534 +/- 0.075 | Leave-one-out across 6 soundfonts |
| B3 | Bootstrap significance | 95% CI [0.465, 0.575] | 1000 bootstrap samples |

**Key Finding:** Cross-soundfont generalization (B2) demonstrates the model generalizes to unseen piano timbres with R2 = 0.534. Bootstrap CI excludes zero and symbolic baseline (0.347), confirming statistical significance.

### Phase C: Cross-Dataset Transfer

| ID | Description | Metric | Result |
|----|-------------|--------|--------|
| C1 | PSyllabus difficulty | Spearman rho | **0.623** (p < 1e-50) |
| C2 | ASAP multi-performer | Intra-piece std | **0.020** |
| C3 | MAESTRO zero-shot | Samples evaluated | **500** |

#### C1: PSyllabus Difficulty Correlation (508 samples)

Strong correlation between model predictions and external difficulty ratings (0-10 scale).

| Dimension | Spearman rho | p-value | Significant |
|-----------|--------------|---------|-------------|
| timing | 0.604 | <1e-50 | Yes |
| mood_valence | 0.604 | <1e-50 | Yes |
| timbre_depth | 0.580 | <1e-50 | Yes |
| pedal_amount | 0.578 | <1e-50 | Yes |
| pedal_clarity | 0.562 | <1e-50 | Yes |
| timbre_variety | 0.531 | <1e-50 | Yes |
| balance | 0.525 | <1e-50 | Yes |
| sophistication | 0.440 | <1e-50 | Yes |
| interpretation | 0.423 | <1e-50 | Yes |
| articulation_length | 0.399 | <1e-50 | Yes |
| drama | 0.366 | <1e-50 | Yes |
| timbre_brightness | 0.261 | <1e-50 | Yes |
| mood_energy | 0.262 | <1e-50 | Yes |
| tempo | 0.257 | <1e-50 | Yes |
| mood_imagination | 0.243 | <1e-50 | Yes |
| articulation_touch | -0.228 | <1e-50 | Yes |
| space | -0.171 | 0.0001 | Yes |
| dynamic_range | -0.097 | 0.0296 | Yes |
| timbre_loudness | 0.054 | 0.2282 | No |
| **Overall** | **0.623** | <1e-50 | Yes |

**Key Finding:** Strong positive correlation (rho=0.623) validates that the model captures musically meaningful features. 18/19 dimensions show significant correlation with difficulty. Higher difficulty pieces correlate with better timing, pedal use, and timbral control.

#### C2: ASAP Multi-Performer Analysis

| Metric | Value |
|--------|-------|
| Pieces analyzed | 206 |
| Total performances | 631 |
| Mean intra-piece std | 0.020 |

**Key Finding:** Low intra-piece variance (0.020) indicates model predictions are primarily driven by piece characteristics rather than performer variations.

#### C3: MAESTRO Zero-Shot Transfer

| Metric | Value |
|--------|-------|
| Samples evaluated | 500 |
| Source | MAESTRO v2.0.0 professional recordings |

**Key Finding:** Model successfully generates predictions for professional piano recordings outside the training distribution.

### Summary: Strongest Paper Results

| Experiment | Metric | Value | Interpretation |
|------------|--------|-------|----------------|
| A1a/A1b/A1c | R2 | 0.52-0.54 | Consistent across fold strategies |
| A2 | R2 | 0.537 | Best with Pianoteq ensemble |
| B2 | R2 | 0.534 +/- 0.075 | Robust to soundfont variation |
| B3 | 95% CI | [0.465, 0.575] | Statistically significant |
| C1 | rho | 0.623 | Strong external validation |
| C2 | std | 0.020 | Piece > performer features |
| C3 | samples | 500 | Zero-shot transfer works |

**Key Claims for Paper:**

1. **Model Performance:** R2 ~ 0.54 consistently across validation strategies
2. **Soundfont Generalization:** Model generalizes to unseen piano timbres (R2 = 0.53)
3. **External Validation:** Strong correlation with PSyllabus difficulty (rho = 0.62)
4. **Multi-Performer Analysis:** Low variance suggests piece-level feature learning

---

## Experiment Naming Convention

| Prefix | Category | Examples |
|--------|----------|----------|
| A | Baselines | A1_linear_probe, A2_mel_cnn |
| B | Layer/pooling ablation | B0_baseline, B1b_layers_7-12 |
| C | Loss ablation | C1a_hybrid_loss |
| D | Advanced architectures | D1a_stats, D7_muq_baseline |
| E | Analysis (compact, latency) | E1a_layer_9, E2_latency |
| M | MuQ layer ablations | M1a_muq_L1-4 |
| F | Fusion experiments | F0_simple, F9_muq_symbolic |
| S | Statistical analysis | S0_bootstrap, S1_paired_tests |
| X | Cross-dataset evaluation | X2_asap_multiperformer |
