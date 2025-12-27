# PercePiano SOTA Reference

**Purpose**: Source of truth for PercePiano architecture and hyperparameters (R2 = 0.397)

For our experiment history and debugging journey, see `EXPERIMENT_LOG.md`.

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
[Note FC] Linear(78 -> 256)
    |
    v
[Note LSTM] BiLSTM(256, layers=2) --> note_out [B, T, 512]
    |                                       |
    v                                       v
[Voice LSTM] BiLSTM(256, layers=2) --> voice_out [B, T, 512]
    |                                       |
    +------ cat(note_out, voice_out) -------+
                        |
                        v
            hidden_out [B, T, 1024]
                        |
                        v
[Beat Attention] --> [Beat LSTM] BiLSTM --> beat_out [B, T_beat, 512]
                                                |
                                                v
[Measure Attention] --> [Measure LSTM] BiLSTM --> measure_out [B, T_meas, 512]
                        |
                        v
        span_beat_to_note + span_measure_to_note
                        |
                        v
    total_note_cat = cat(hidden_out, beat_spanned, measure_spanned)
                    [B, T, 2048]
                        |
                        v
[Performance Contractor] Linear(2048 -> 512)
                        |
                        v
[Final Attention] ContextAttention(512, heads=8)  [B, 512]
                        |
                        v
[Prediction Head] Dropout -> Linear(512, 512) -> GELU -> Dropout -> Linear(512, 19)
                        |
                        v
                  [Sigmoid]
                        |
                        v
                 predictions [B, 19]
```

**IMPORTANT CORRECTION (2025-12-25)**: The prediction head uses 512->512->19, NOT 512->128->19.
The `final_fc_size: 128` in the config file is for the **decoder** (not used in classification),
not the `out_fc` classifier head. See `model_m2pf.py:118-124` for actual implementation.

### Critical Architecture Details

1. **Voice LSTM runs in PARALLEL with Note LSTM** (not sequential)
   - Both take the same 256-dim projected input
   - Outputs are concatenated: `hidden_out = cat(note_out, voice_out)`
   - Reference: `encoder_score.py:496-516`

2. **Performance Contractor is CRITICAL**
   - Reduces 2048-dim total_note_cat to 512-dim
   - Without this, model has gradient flow issues
   - Reference: `model_m2pf.py` VirtuosoNetMultiLevel class

3. **ContextAttention structure**
   - `Linear(size, size)` + tanh activation
   - Learnable context vectors per head (8 vectors of 64-dim each)
   - Softmax over sequence dimension
   - Reference: `virtuoso/module.py:348-383`

4. **Sigmoid on output** (confirmed from code)
   - `train_m2pf.py:135`: `loss = loss_calculator(sigmoid(logitss[-1]), targets)`
   - Targets are normalized to [0, 1], sigmoid constrains predictions to same range

5. **MixEmbedder is SEPARATE from HAN** (discovered 2025-12-25)
   - Input embedding (`MixEmbedder`: Linear 78->256) is a separate module
   - HAN's `note_fc` is **commented out** in original (`encoder_score.py:509`)
   - Flow: `x -> MixEmbedder -> HAN encoder` (not `x -> HAN.note_fc -> HAN encoder`)
   - Reference: `model_m2pf.py:67-68`, `encoder_score.py:509`

6. **NO LayerNorm anywhere** (discovered 2025-12-25)
   - Original has NO normalization layers in the entire model
   - We incorrectly added LayerNorm in Round 1 to fix gradient issues
   - Original relies on proper data preprocessing for magnitude control

---

## Data Pipeline (CRITICAL - discovered 2025-12-25)

The original uses a fundamentally different data pipeline than our implementation:

### Slicing Strategy

Each performance is sliced into **multiple overlapping segments**:

```python
# From dataset.py:67-77
def update_slice_info(self):
    self.slice_info = []
    for i, data in enumerate(self.data):
        slice_indices = make_slicing_indexes_by_measure(
            len(data['input']),
            data['note_location']['measure'],
            self.slice_len  # default: 5000 notes
        )
        for idx in slice_indices:
            self.slice_info.append((i, idx))  # Each slice is a training sample
```

**Impact**: Creates 3-5x more training samples from the same data via overlapping slices.

### Variable-Length Batching (PackedSequence)

Original uses `pack_sequence` for variable-length sequences:

```python
# From dataset.py:223 (FeatureCollate)
batch_x = pack_sequence([sample[0] for sample in batch], enforce_sorted=False)
```

**NOT** `pad_sequence` with fixed-length tensors and attention masks.

### Our Implementation (WRONG)

| Aspect | Original SOTA | Our Implementation |
|--------|---------------|-------------------|
| Sampling | Multiple overlapping slices per performance | One sample per performance |
| Slice size | ~5000 notes with random overlap | Entire performance (truncate to max_notes) |
| Training samples | 1000s (slices across all performances) | ~500-600 (just the pickle files) |
| Batching | `pack_sequence()` - PackedSequence | `pad_sequence()` - Fixed-length tensors |

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| len_slice | 5000 | parser.py:123 |
| len_valid_slice | 5000 | parser.py:131 |

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
| final_fc_size | 128 | **NOTE: This is for decoder, NOT classifier** |
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
| precision | FP32 |

### Prediction Head (CORRECTED 2025-12-25)

The actual classifier head from `model_m2pf.py:118-124`:

```python
self.out_fc = nn.Sequential(
    nn.Dropout(net_param.drop_out),
    nn.Linear(net_param.encoder.size * 2, net_param.encoder.size * 2),  # 512 -> 512
    nn.GELU(),
    nn.Dropout(net_param.drop_out),
    nn.Linear(net_param.encoder.size * 2, net_param.num_label),  # 512 -> 19
)
```

**Architecture**: `512 -> 512 -> 19` (NOT 512 -> 128 -> 19)

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

## File Locations

### Our Implementation

- Model: `src/percepiano/models/percepiano_replica.py`
- Attention: `src/percepiano/models/context_attention.py`
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
