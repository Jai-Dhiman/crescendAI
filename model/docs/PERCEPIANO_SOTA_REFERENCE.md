# PercePiano SOTA Reference

**Last Updated**: 2024-12-23
**Purpose**: Source of truth for reproducing PercePiano SOTA results (R2 = 0.397)

---

## Paper Citation

```
Park, J., Kim, J., Park, J.M., Choi, A., Li, W.S., Park, J., & Hwang, S.W. (2024).
Piano performance evaluation dataset with multilevel perceptual features.
Scientific Reports, 14, 23002.
DOI: 10.1038/s41598-024-73810-0
```

**Resources**:

- Paper: <https://www.nature.com/articles/s41598-024-73810-0>
- PMC Full Text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11450231/>
- GitHub: <https://github.com/JonghoKimSNU/PercePiano>
- Dataset: <https://doi.org/10.5281/zenodo.13269613>

---

## Published Results (Table 3)

### Piece-Split (Primary Benchmark)

| Model | R2 | MSE |
|-------|-----|-----|
| Mean Prediction | 0.00 | 2.28 |
| Bi-LSTM | 0.185 | 10.8E-03 |
| MidiBERT | 0.313 | 9.24E-03 |
| **Bi-LSTM + SA + HAN (SOTA)** | **0.397** | **8.11E-03** |

### Performer-Split

| Model | R2 | MSE |
|-------|-----|-----|
| Bi-LSTM | 0.236 | 9.09E-03 |
| MidiBERT | 0.212 | 9.30E-03 |
| **Bi-LSTM + SA + HAN (SOTA)** | **0.285** | **8.70E-03** |

### Ablation Study (Piece-Split)

| Configuration | R2 | Delta |
|---------------|-----|-------|
| Bi-LSTM (baseline) | 0.185 | - |
| + Score Alignment (SA) | 0.304 | +0.119 |
| + Hierarchical Attention (HAN) | 0.397 | +0.093 |

**Key Insight**: Score alignment contributes +11.9% R2, HAN contributes +9.3% R2.

---

## Model Architecture

### Overview

```
Input (78-dim VirtuosoNet features)
    |
    v
[Note FC] Linear(78 -> hidden)
    |
    v
[Note LSTM] BiLSTM(hidden, layers=2) --> note_out [B, T, hidden*2]
    |                                         |
    v                                         v
[Voice LSTM] BiLSTM(hidden, layers=2) --> voice_out [B, T, hidden*2]
    |                                         |
    +------ cat(note_out, voice_out) ---------+
                        |
                        v
            hidden_out [B, T, hidden*4]
                        |
                        v
[Beat Attention] --> [Beat LSTM] BiLSTM --> beat_out [B, T_beat, hidden*2]
                                                |
                                                v
[Measure Attention] --> [Measure LSTM] BiLSTM --> measure_out [B, T_meas, hidden*2]
                        |
                        v
        span_beat_to_note + span_measure_to_note
                        |
                        v
    total_note_cat = cat(hidden_out, beat_spanned, measure_spanned)
                    [B, T, hidden*8]  (2048 for hidden=256)
                        |
                        v
[Performance Contractor] Linear(hidden*8 -> hidden*2)  [B, T, 512]
                        |
                        v
[Final Attention] ContextAttention(hidden*2, heads=8)  [B, 512]
                        |
                        v
[Prediction Head] Dropout -> Linear -> GELU -> Dropout -> Linear
                        |
                        v
                  [Sigmoid]
                        |
                        v
                 predictions [B, 19]
```

### Critical Architecture Details

1. **Voice LSTM runs in PARALLEL with Note LSTM** (not sequential)
   - Both take the same 256-dim projected input
   - Outputs are concatenated: `hidden_out = cat(note_out, voice_out)`
   - Reference: `encoder_score.py:496-516`

2. **Performance Contractor is CRITICAL**
   - Reduces 2048-dim total_note_cat to 512-dim
   - Without this, model has gradient flow issues
   - Reference: `model_m2pf.py` VirtuosoNetMultiLevel class

3. **Sigmoid on output** (confirmed from code)
   - `train_m2pf.py:135`: `loss = loss_calculator(sigmoid(logitss[-1]), targets)`
   - Targets are normalized to [0, 1], sigmoid constrains predictions to same range

---

## Hyperparameters

### From Published Paper

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden sizes | {64, 128, 256} | Hyperparameter search |
| Num layers | {1, 2} | Hyperparameter search |
| Batch size | 8 | Fixed |
| Learning rate | {5e-5, 2.5e-5} | Hyperparameter search |
| Loss function | MSE | Fixed |
| Cross-validation | 4-fold | Piece-split and performer-split |

### From GitHub Repository (SOTA Config)

**Config file**: `ymls/shared/label19/han_measnote_nomask_bigger256.yml`

| Parameter | Value |
|-----------|-------|
| input_size | 78 |
| note_size | 256 |
| voice_size | 256 |
| beat_size | 256 |
| measure_size | 256 |
| note_layer | 2 |
| voice_layer | 2 |
| beat_layer | 2 |
| measure_layer | 1 |
| num_attention_heads | 8 |
| dropout | 0.2 |
| final_fc_size | 128 |
| final_fc_layer | 1 |
| encoder | HAN |
| meas_note | true |
| use_voice_net | true |
| sequence_iteration | 3 |

### Training Script Parameters

**Script**: `2_run_comp_multilevel_total.sh`

| Parameter | Value |
|-----------|-------|
| learning_rate | 2.5e-5 |
| batch_size | 8 |
| weight_decay | 1e-5 |
| grad_clip | 2.0 |
| lr_scheduler | StepLR(step_size=3000, gamma=0.98) |
| optimizer | Adam |
| augment_train | False |

---

## Input Features

### VirtuosoNet 78-Dimension Feature Vector

Extracted via pyScoreParser from aligned MusicXML + MIDI pairs.

| Index | Feature | Dim | Normalized |
|-------|---------|-----|------------|
| 0 | midi_pitch | 1 | z-score |
| 1 | duration | 1 | z-score |
| 2 | beat_importance | 1 | z-score |
| 3 | measure_length | 1 | z-score |
| 4 | qpm_primo | 1 | z-score |
| 5 | following_rest | 1 | z-score |
| 6 | distance_from_abs_dynamic | 1 | z-score |
| 7 | distance_from_recent_tempo | 1 | z-score |
| 8 | beat_position | 1 | no |
| 9 | xml_position | 1 | no |
| 10 | grace_order | 1 | no |
| 11 | preceded_by_grace_note | 1 | no |
| 12 | followed_by_fermata_rest | 1 | no |
| 13-25 | pitch (octave + 12-class one-hot) | 13 | octave normalized |
| 26-30 | tempo marking | 5 | no |
| 31-34 | dynamic marking | 4 | no |
| 35-43 | time_sig_vec | 9 | no |
| 44-49 | slur_beam_vec | 6 | no |
| 50-66 | composer_vec | 17 | no |
| 67-75 | notation | 9 | no |
| 76-77 | tempo_primo | 2 | no |

**Total**: 78 dimensions

### Additional Unnormalized Features (for augmentation)

| Index | Feature |
|-------|---------|
| 78 | midi_pitch_unnorm |
| 79 | duration_unnorm |
| 80 | beat_importance_unnorm |
| 81 | measure_length_unnorm |
| 82 | following_rest_unnorm |

**Note**: SOTA config uses 78 features. The 5 unnorm features are for key augmentation only.

---

## Target Labels

### 19 Perceptual Dimensions

| Group | Dimension |
|-------|-----------|
| Timing | timing |
| Articulation | articulation_length, articulation_touch |
| Pedal | pedal_amount, pedal_clarity |
| Timbre | timbre_variety, timbre_depth, timbre_brightness, timbre_loudness |
| Dynamics | dynamic_range |
| Tempo | tempo |
| Space | space |
| Balance | balance |
| Interpretation | drama, mood_valence, mood_energy, mood_imagination, sophistication, interpretation |

### Label Processing

1. Multiple annotators (5-12) per segment
2. "I don't know" responses removed before aggregation
3. Values averaged across annotators
4. **Normalized to [0, 1] range** (critical for sigmoid output)

---

## Data Split

### Piece-Split (Primary)

- Same piece cannot appear in both train and validation
- Tests generalization to new compositions
- **Use this for primary benchmarking**

### Performer-Split

- Same performer cannot appear in both train and validation
- Tests generalization to new playing styles
- Generally lower R2 scores

### Dataset Size

- Total: 1,202 segments
- Total annotations: 12,652
- Segment lengths: 4, 8, or 16 bars

---

## Known Implementation Details

### From train_m2pf.py

1. **Loss computation** (line 135):

   ```python
   loss = loss_calculator(sigmoid(logitss[-1]), targets)
   ```

2. **Gradient clipping** (line 173):

   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
   ```

3. **R2 computation** (line 222):

   ```python
   from sklearn.metrics import r2_score
   r2 = r2_score(targets, predictions)  # uniform_average by default
   ```

4. **Early stopping**: Based on validation loss improvement

### From encoder_score.py

1. **Sequence iteration** (line 33):

   ```python
   for i in range(self.num_sequence_iteration):
       # Refine hierarchy representations
   ```

   Default: 3 iterations (from config)

2. **Voice processing** (lines 496-549):
   - Runs in parallel with note LSTM
   - Uses same input embeddings (not note LSTM output)
   - Scatters results back via batched matrix multiply

---

## Debugging Checklist

When R2 is below expected:

### Check 1: Prediction Statistics

```python
print(f"Pred mean: {preds.mean():.4f}, std: {preds.std():.4f}")
print(f"Pred min: {preds.min():.4f}, max: {preds.max():.4f}")
# Expected: mean ~0.5, std > 0.1, range covers [0.1, 0.9]
```

### Check 2: Collapsed Dimensions

```python
for i, dim in enumerate(dimensions):
    dim_std = preds[:, i].std()
    if dim_std < 0.05:
        print(f"COLLAPSED: {dim} has std={dim_std:.4f}")
```

### Check 3: Target Distribution

```python
print(f"Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
# Ensure targets are in [0, 1] and not already scaled differently
```

### Check 4: Feature Statistics

```python
print(f"Feature mean: {features.mean():.4f}, std: {features.std():.4f}")
# Z-scored features should have mean ~0, std ~1
```

### Check 5: Gradient Flow

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

---

## Our Implementation Status

**Last Updated**: 2025-12-25

### Issue History

#### 2024-12-24: Prediction Collapse & Gradient Explosion

**Symptoms observed**:

- Gradient norms 800-2500+ (before clipping)
- Prediction std ~0.03-0.05 vs target std ~0.15
- R2 stuck negative, slowly approaching 0 but never crossing
- Key layers affected: `note_fc.weight` (16-800), `prediction_head.1.weight` (108-900)

**Experiments run**:

| LR | Result |
|----|--------|
| 2.5e-5 (SOTA) | R2: -0.23 to -0.05 over 20 epochs, pred_std ~0.04 |
| 1e-5 | R2: -0.15 to ~0.00 over 25 epochs, pred_std ~0.05 |
| 5e-6 | R2: -0.17 to -0.05 over 18 epochs, pred_std ~0.035 |

**Root cause analysis**:

- Feature normalization verified CORRECT (only 8 scalar features z-scored, rest categorical)
- Gradient clipping at 2.0 verified CONFIGURED
- Issue: 78->256 projection (`note_fc`) without normalization causes gradient instability
- Original PercePiano uses PyTorch defaults but may have different seed/conditions

#### 2025-12-25: Stability Fixes Applied

**Attempt 1: LayerNorm after note_fc + Xavier init** - FAILED
- Gradients got WORSE (1586 -> 9159 by step 1000)
- LayerNorm after Linear doesn't help because gradient explosion happens in backprop through Linear

**Attempt 2: Input LayerNorm + Aggressive Clipping** (`percepiano_replica.py:114-115`)

```python
self.input_norm = nn.LayerNorm(input_size)  # Normalize 78-dim input
self.note_fc = nn.Linear(input_size, note_size)

# In forward():
x_normed = self.input_norm(x)
x_embedded = self.note_fc(x_normed)
```

Also reduced gradient clipping from 2.0 to 1.0 (`kfold_trainer.py:352`)

**Rationale**: The 78-dim input features have varied scales (8 z-scored + 70 categorical/embeddings).
LayerNorm on INPUT normalizes all features to unit variance before projection, preventing gradient explosion.

**Status**: Testing

### Verified Matching SOTA

| Component | Status | Notes |
|-----------|--------|-------|
| Model architecture | MATCH | HAN, performance_contractor, final_attention, prediction head |
| Hidden size | MATCH | 256 for all levels |
| Layer counts | MATCH | note=2, voice=2, beat=2, measure=1 |
| Attention heads | MATCH | 8 |
| Dropout | MATCH | 0.2 |
| Learning rate | MATCH | 2.5e-5 |
| Batch size | MATCH | 8 |
| Weight decay | MATCH | 1e-5 |
| Optimizer | MATCH | Adam (not AdamW) |
| LR Scheduler | MATCH | StepLR(step_size=3000, gamma=0.98, interval=step) |
| Gradient clipping | MATCH | 2.0 |
| Loss function | MATCH | MSE after sigmoid |
| R2 computation | MATCH | sklearn r2_score with uniform_average |

### Ruled Out Issues

1. **Feature normalization** - CORRECT: 8 features z-scored, 70 categorical/embeddings unchanged
2. **Measure-aligned slicing** - Samples 51-413 notes, max_notes=1024 sufficient
3. **Sequence iteration** - HanEncoder doesn't use it (only ISGN encoders do)
4. **Fold assignment** - Fixed with greedy bucket balancing
5. **Labels** - Correct range (0.24-0.90 within [0,1])

### Active Diagnostics

1. **Prediction collapse detection** (`percepiano_replica.py`)
   - Logs pred mean, std, range after each validation epoch
   - Warns if `pred_std < 0.05`

2. **Gradient monitoring** (`kfold_trainer.py:76-107`)
   - `GradientMonitorCallback` logs gradient norms every 100 steps
   - Tracks `note_fc`, `prediction_head` layers
   - Warns on explosion (>100) or vanishing (<1e-6)

---

## File Locations

### Our Implementation

- Model: `src/percepiano/models/percepiano_replica.py`
- HAN encoder: `src/percepiano/models/han_encoder.py`
- Hierarchy utils: `src/percepiano/models/hierarchy_utils.py`
- Feature extractor: `src/percepiano/data/virtuosonet_feature_extractor.py`
- Dataset: `src/percepiano/data/percepiano_vnet_dataset.py`
- Trainer: `src/percepiano/training/kfold_trainer.py`
- Training notebook: `notebooks/train_percepiano_replica.ipynb`

### Original PercePiano (in data/raw)

- Encoder: `data/raw/PercePiano/virtuoso/virtuoso/encoder_score.py`
- Training: `data/raw/PercePiano/virtuoso/virtuoso/train_m2pf.py`
- Dataset: `data/raw/PercePiano/virtuoso/virtuoso/dataset.py`
- Config: `data/raw/PercePiano/virtuoso/ymls/shared/label19/han_measnote_nomask_bigger256.yml`
- Feature extraction: `data/raw/PercePiano/virtuoso/virtuoso/pyScoreParser/data_for_training.py`
