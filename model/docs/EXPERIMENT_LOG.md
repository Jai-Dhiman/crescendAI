# PercePiano Replica Experiment Log

**Last Updated**: 2025-12-27 (Round 12)
**Purpose**: Track our debugging journey to reproduce PercePiano SOTA (R2 = 0.397)

For architecture details and hyperparameters, see `PERCEPIANO_SOTA_REFERENCE.md`.

---

## Current Status

**Round 12** - BI-LSTM BASELINE ARCHITECTURE MISMATCH (CRITICAL FINDING)

| Change | Description |
|--------|-------------|
| Data investigation | **COMPLETE** - Confirmed we have full PercePiano dataset (1,179 of 1,202 segments) |
| Architecture comparison | **COMPLETE** - Our HAN matches original 99.8% |
| Bi-LSTM baseline | **CRITICAL MISMATCH** - Our ablation uses wrong architecture |
| Checkpoint saving | **FIXED** - ModelCheckpoint filename template corrected |

**Key Discovery (Round 12):**
Our "ablation" test comparing Full HAN vs Bi-LSTM baseline uses the WRONG baseline
architecture. The original PercePiano Bi-LSTM baseline (`VirtuosoNetSingle`) is a
**7-layer single LSTM**, NOT our "zeroed hierarchy" approach.

**Critical Architecture Mismatch:**

| Aspect | Original VirtuosoNetSingle | Our "Ablated" Model |
|--------|---------------------------|---------------------|
| LSTM structure | Single 7-layer LSTM | 2-layer note + 2-layer voice (separate) |
| Voice processing | None | Full voice LSTM |
| Total backbone params | ~2M | ~4M (higher capacity) |
| Input to contractor | 512-dim (dense) | 2048-dim (1024 zeros) |
| Architecture | Clean baseline | Broken HAN with zeros |

**Why This Matters:**
- Original paper: Bi-LSTM baseline R2 = 0.187, HAN R2 = 0.397 (gain = +0.21)
- Our "ablation" measures HAN vs broken-HAN, NOT HAN vs true baseline
- The +0.13 hierarchy gain we measured is comparing wrong things
- The overall low R2 (-0.1 to +0.15) may be due to architectural issues

**Data Investigation Results:**
- PercePiano has 1,202 labeled segments total
- We have 1,179 preprocessed (98.1% match - small loss from filtering/alignment)
- 22 unique pieces, 49 unique performers
- Data heavily imbalanced: Schubert D960 mvmt 3 = 40% of all data

**Next Steps:**
1. Implement proper `VirtuosoNetSingle` baseline matching original exactly
2. Update ablation callback to use correct baseline for comparison
3. Re-run ablation to get true hierarchy gain measurement
4. If true baseline R2 ~0.19 and our HAN still low, investigate HAN issues

---

## Verified Configuration (Matching SOTA - Round 8)

| Component | Status | Notes |
|-----------|--------|-------|
| Model architecture | MATCH | HAN, performance_contractor, final_attention, prediction head |
| Hidden size | MATCH | 256 for all levels |
| Layer counts | MATCH | note=2, voice=2, beat=2, measure=1 |
| Attention heads | MATCH | 8 |
| Dropout | MATCH | 0.2 |
| **Learning rate** | **FIXED** | 2.5e-5 (was incorrectly 5e-5 in Round 4) |
| Batch size | MATCH | 8 |
| Weight decay | MATCH | 1e-5 |
| Optimizer | MATCH | Adam (not AdamW) |
| LR Scheduler | MATCH | StepLR(step_size=3000, gamma=0.98, interval=step) |
| Gradient clipping | MATCH | 2.0 |
| Loss function | MATCH | MSE after sigmoid |
| R2 computation | MATCH | sklearn r2_score with uniform_average |
| Precision | MATCH | FP32 (fixed in Round 1) |
| Final Attention | MATCH | ContextAttention with context vectors (fixed in Round 3) |
| **Prediction head** | **FIXED** | 512->512->19 (was incorrectly 512->128->19 in Round 5) |
| **LayerNorm** | **FIXED** | REMOVED (was incorrectly added in Round 1) |
| Initialization | MATCH | PyTorch defaults (kaiming_uniform) |
| **Slice sampling** | **FIXED** | 3-5 overlapping slices/sample, regenerated each epoch (Round 8) |
| **max_notes/slice_len** | **FIXED** | 5000 (was 1024 before Round 8) |

---

## Issue History

### 2024-12-24: Initial Problem - Prediction Collapse & Gradient Explosion

**Symptoms observed**:
- Gradient norms 800-2500+ (before clipping)
- Prediction std ~0.03-0.05 vs target std ~0.15
- R2 stuck negative, slowly approaching 0 but never crossing
- Key layers affected: `note_fc.weight` (16-800), `prediction_head.1.weight` (108-900)

**Initial experiments**:

| LR | Result |
|----|--------|
| 2.5e-5 (SOTA) | R2: -0.23 to -0.05 over 20 epochs, pred_std ~0.04 |
| 1e-5 | R2: -0.15 to ~0.00 over 25 epochs, pred_std ~0.05 |
| 5e-6 | R2: -0.17 to -0.05 over 18 epochs, pred_std ~0.035 |

---

### 2025-12-25: Round 1 - Architecture & Normalization Fixes

**Diagnostic run with activation logging** revealed root causes:

```
Input features: mean=0.01, std=0.17 (OK)
HAN total_note_cat: mean=-0.0002, std=0.0139 (100x TOO SMALL!)
Contracted: mean=0.0001, std=0.0149 (TOO SMALL)
Aggregated: mean=0.0003, std=0.0211 (TOO SMALL)
Logits: mean=-0.004, std=0.03 (50x TOO SMALL - should be 1-3)
Predictions: mean=0.499, std=0.008 (all ~0.5, sigmoid of ~0)

Gradient by category:
  han_encoder:            norm=70,  max=16
  performance_contractor: norm=187, max=152
  final_attention:        norm=0.1, max=0.1 (NEGLIGIBLE!)
  prediction_head:        norm=622, max=561 (DOING ALL THE WORK)
```

**Root Causes Identified**:

1. **FP16 Mixed Precision**: Original uses FP32, we defaulted to FP16-mixed
2. **ContextAttention Architecture Wrong**: Used `Linear(size, size)` but original uses `Linear(size, num_head)`
3. **Magnitude Collapse**: Near-uniform attention weights averaged all positions
4. **No Normalization**: Values collapsed to near-zero

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Precision FP32 | `kfold_trainer.py:351-353` | Changed from `"16-mixed"` to `"32"` |
| ContextAttention | `context_attention.py` | Fixed to use `Linear(size, num_head)` |
| HAN output norm | `percepiano_replica.py:761` | Added `LayerNorm(2048)` |
| Pre-prediction norm | `percepiano_replica.py:775` | Added `LayerNorm(512)` |
| Prediction head init | `percepiano_replica.py:787-790` | Xavier init gain=0.1 |

**Results**: Gradient explosion fixed, but prediction collapse persisted (logits std=0.05)

---

### 2025-12-25: Round 2 - Prediction Head Initialization

**Root Cause**: Xavier gain=0.1 produced tiny logits (std=0.05)

**Fix Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Prediction head init | `percepiano_replica.py:791-796` | Xavier gain=2.0, bias uniform [-1, 1] |

**Results**:

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Logits std | 0.8-1.5 | 1.317 | PASS |
| Predictions std | 0.10-0.15 | 0.250 | PASS |
| Gradient balance | 0.3-0.5x | 0.4x | PASS |
| R2 after 7 epochs | >+0.05 | -0.13 | FAIL |

Activation collapse fixed but R2 still negative.

---

### 2025-12-25: Round 3 - Final Attention Architecture

**Root Cause**: `FinalContextAttention` was too simplified - missing context vectors

Original PercePiano uses:
- `Linear(512, 512)` + tanh activation
- Learnable context vectors per head (8 vectors of 64-dim each)

Our simplified version had:
- `Linear(512, 8)` only
- No context vectors

**Fix Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Final attention | `percepiano_replica.py:775-777` | Changed to `ContextAttention` |

**Results**:

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Context vectors | Present | Present | PASS |
| Predictions std | 0.10-0.15 | 0.242 | HIGH (2.3x target) |
| R2 after 12 epochs | >+0.05 | -0.136 | FAIL |

Context vectors present but predictions overshooting.

---

### 2025-12-25: Round 4 - Learning Rate & Initialization Tuning

**Root Cause**: Prediction head initialization too aggressive (gain=2.0, bias [-1,1])

Predictions std=0.242 was 2.3x higher than target std=0.106.

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Learning rate | `notebook cell-12` | Increased from 2.5e-5 to 5e-5 |
| Xavier gain | `percepiano_replica.py:798` | Reduced from 2.0 to 1.0 |
| Bias range | `percepiano_replica.py:799` | Reduced from [-1,1] to [-0.1,0.1] |

**Results** (23 epochs, interrupted):

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Learning rate | 5e-5 | 5e-5 | PASS |
| Logits std | 0.5-1.0 | 0.541 | PASS |
| Predictions std | 0.10-0.15 | 0.128 | PASS |
| Pred/target std ratio | 0.8-1.5x | 0.98x | PASS |
| R2 after 5 epochs | > 0 | -0.081 | FAIL |
| R2 after 14 epochs | > 0.10 | -0.039 | FAIL |
| R2 after 23 epochs | > 0.15 | -0.039 | FAIL |

All activation health metrics passed, but R2 progression far too slow.

---

### 2025-12-25: Round 5 - Prediction Head Size Fix (INCORRECT)

**Hypothesis**: Prediction head using wrong hidden size (512 vs 128)

The SOTA config specifies `final_fc_size: 128`, so we changed to 512->128->19.

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Prediction head size | `percepiano_replica.py:790-793` | Changed to `final_hidden=128` |
| Remove custom init | `percepiano_replica.py:795-796` | Use PyTorch defaults |

**Actual Results** (28 epochs, interrupted):

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Prediction head | 512->128, 128->19 | Correct | PASS |
| Logits std | 0.5-1.5 | 0.203 | **FAIL** |
| Predictions std | 0.10-0.15 | 0.050 | **FAIL** |
| Pred/target std ratio | 0.8-1.5x | 0.41x | **FAIL** |
| Collapsed dimensions | 0/19 | 5/19 | **FAIL** |
| R2 after 5 epochs | > 0 | +0.017 | PASS |
| R2 after 10 epochs | > 0.05 | -0.040 | **FAIL** |
| R2 after 20 epochs | > 0.15 | -0.032 | **FAIL** |
| Best R2 | 0.30+ | +0.017 (epoch 8) | **FAIL** |

**Conclusion**: Round 5 did not help. R2 plateaued around 0 and activation metrics regressed.

---

### 2025-12-25: Round 6 - Systematic Original Code Analysis

**Approach**: Stop guessing, systematically compare our implementation with original code.

**Key Discoveries from Original Code Analysis**:

1. **Prediction head is 512->512->19, NOT 512->128->19** (CRITICAL)
   - We misread the config: `final_fc_size: 128` is for the DECODER, not the classifier
   - Actual code (`model_m2pf.py:118-124`):
     ```python
     nn.Linear(net_param.encoder.size * 2, net_param.encoder.size * 2),  # 512 -> 512
     nn.Linear(net_param.encoder.size * 2, net_param.num_label),  # 512 -> 19
     ```

2. **Original has NO LayerNorm anywhere** (CRITICAL)
   - We added LayerNorm in Round 1 to fix gradient issues
   - Original code has zero normalization layers
   - LayerNorm may be preventing the model from learning

3. **MixEmbedder is separate from HAN**
   - Input embedding (78->256) is done by `MixEmbedder` before HAN
   - HAN's `note_fc` is commented out in original (`encoder_score.py:509`)

4. **Learning rate is 2.5e-5** (not 5e-5)
   - We incorrectly increased to 5e-5 in Round 4
   - Original uses 2.5e-5 (`2_run_comp_multilevel_total.sh`)

5. **Data is already pre-segmented**
   - Our pickle files are individual 4/8/16-bar segments (1202 total)
   - No additional slicing needed - that's for full performances

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Remove LayerNorm | `percepiano_replica.py:761-764` | Removed `han_output_norm` |
| Remove LayerNorm | `percepiano_replica.py:782` | Removed `pre_prediction_norm` |
| Prediction head | `percepiano_replica.py:784-790` | Changed to 512->512->19 |
| Learning rate | `notebook cell-12` | Reverted to 2.5e-5 |

**Expected Results (Round 6)**:

| Metric | Round 5 | Expected Round 6 |
|--------|---------|------------------|
| Prediction head | 512->128->19 | 512->512->19 |
| LayerNorm | Present | Removed |
| Learning rate | 5e-5 | 2.5e-5 |
| R2 after 5 epochs | +0.017 | > 0 |
| R2 after 20 epochs | -0.032 | > 0.15 |
| R2 at convergence | +0.017 | +0.30-0.40 (SOTA) |

---

### 2025-12-25: Round 7 - Data Processing Fix (section_tempo + PackedSequence)

**Problem**: Round 6 showed all context_vectors had exactly zero gradients, preventing attention
mechanism from learning. Root cause: differences in data processing pipeline vs original PercePiano.

**Key Differences Found**:

| Aspect | Original | Ours (before) | Ours (after) |
|--------|----------|---------------|--------------|
| Feature count | 79 base + 5 unnorm = 84 | 78 base + 5 unnorm = 83 | 79 base + 5 unnorm = 84 |
| section_tempo | Present at index 5 | Missing | Restored at index 5 |
| Batching | pack_sequence (PackedSequence) | pad_sequence (fixed 1024) | pack_sequence (PackedSequence) |
| NORM_FEAT_KEYS | 9 features | 8 features | 9 features |

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Add section_tempo | `virtuosonet_feature_extractor.py` | Restored section_tempo at index 5 |
| Update dimensions | `virtuosonet_feature_extractor.py` | BASE_FEATURE_DIM=79, TOTAL_FEATURE_DIM=84 |
| PackedSequence collate | `percepiano_vnet_dataset.py` | Added `percepiano_pack_collate()` function |
| Handle PackedSequence | `percepiano_replica.py` | Model unpacks PackedSequence in forward() |
| Reprocess data | `data/percepiano_vnet_84dim/` | New 84-dim train/val/test splits |
| Upload to GDrive | `gdrive:crescendai_data/percepiano_vnet_84dim` | 955 train, 27 val, 197 test |

**Data Location**:
- Local: `data/percepiano_vnet_84dim/` (84-dim with section_tempo)
- GDrive: `gdrive:crescendai_data/percepiano_vnet_84dim`
- Old 83-dim: `gdrive:crescendai_data/percepiano_vnet_split` (deprecated)

**Expected Results (Round 7)**:

| Metric | Round 6 | Expected Round 7 |
|--------|---------|------------------|
| context_vectors gradient | 0.000000 | > 0 |
| Feature count | 83 (78 base) | 84 (79 base) |
| Batching | padded (1024) | PackedSequence |
| R2 after 5 epochs | -0.037 | > 0 |
| R2 at convergence | ~ 0 | +0.30-0.40 (SOTA) |

**Actual Results (Round 7)**:

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Feature count | 84 (79 base) | 84 (79 base) | PASS |
| Batching | PackedSequence | PackedSequence | PASS |
| R2 | > 0.30 | 0.0017 | FAIL |
| Predictions std | 0.10-0.15 | 0.007 | FAIL (collapse) |

**Conclusion**: Data pipeline fixed but severe prediction collapse (std=0.007). Model
not learning despite correct features and batching.

---

### 2025-12-26: Round 8 - SLICE SAMPLING (Critical Missing Piece)

**Problem**: Round 7 achieved R2=0.0017 with prediction collapse (std=0.007).
After thorough investigation comparing our implementation against original PercePiano
codebase, identified one critical issue.

**Root Cause Analysis**:

| Aspect | Original PercePiano | Our Implementation | Impact |
|--------|---------------------|-------------------|--------|
| Samples/performance | 3-5 overlapping slices | 1 sample | 3-5x less training data |
| Training samples | ~600-1000 slices | ~200 samples | Critical for learning |
| Slice regeneration | Each epoch | None | No variation |

**Evidence from Original Code**:
- `dataset.py:67-77`: `update_slice_info()` called each epoch
- `data_process.py:53-103`: `make_slicing_indexes_by_measure()` creates overlapping slices
- `train_m2pf.py:288`: Slice regeneration at epoch start

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| Slice function | `percepiano_vnet_dataset.py:45-128` | Ported `make_slicing_indexes_by_measure()` |
| Preload data | `percepiano_vnet_dataset.py:839-851` | Added `_preload_samples()` method |
| Slice info | `percepiano_vnet_dataset.py:853-879` | Added `update_slice_info()` method |
| Dataset changes | `percepiano_vnet_dataset.py:928-1008` | Modified `__len__()` and `__getitem__()` |
| Epoch callback | `kfold_trainer.py:187-220` | Added `SliceRegenerationCallback` |
| DataModule | `percepiano_vnet_dataset.py:1058-1134` | Added `slice_len` parameter |
| Trainer | `kfold_trainer.py:556-571` | Pass `slice_len` to DataModule |
| Config | `notebook cell-12` | `max_notes=5000`, `slice_len=5000` |

**How Slice Sampling Works**:

1. Each performance is divided into 3-5 overlapping slices of ~5000 notes
2. Slices are aligned to measure boundaries (no mid-measure cuts)
3. At each epoch start, `update_slice_info()` regenerates slices with different random boundaries
4. This provides data augmentation through varied slice boundaries

**Expected Results (Round 8)**:

| Metric | Round 7 | Expected Round 8 |
|--------|---------|------------------|
| Training samples | ~200 | ~600-1000 slices |
| Predictions std | 0.007 | 0.10-0.15 |
| R2 after 10 epochs | 0.0017 | > 0.10 |
| R2 at convergence | 0.0017 | +0.30-0.40 (SOTA) |

**Actual Results (Round 8)**:

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Training samples | ~600-1000 | ~600 slices | PASS |
| Predictions std | 0.10-0.15 | 0.10-0.12 | PASS |
| R2 after 10 epochs | > 0.10 | ~0.10 | PASS |
| CV R2 average | 0.30-0.40 | 0.1496 | FAIL |
| Test R2 | 0.30-0.40 | 0.0398 | FAIL |

**Conclusion**: Slice sampling fixed prediction collapse. Model now learns (R2 ~0.15)
but matches Bi-LSTM baseline only. Hierarchical components not contributing.

---

### 2025-12-27: Round 12 - Bi-LSTM Baseline Architecture Mismatch (CRITICAL)

**Problem**: Rounds 9-11 showed hierarchy gain of +0.13 R2, but overall model R2 stuck at
-0.1 to +0.15 instead of target 0.397. Deep investigation of original PercePiano source
code revealed critical architectural mismatch in our Bi-LSTM baseline.

**Investigation Approach**:
1. Researched all PercePiano publications (Nature Scientific Reports, ISMIR, GitHub, Zenodo)
2. Compared our preprocessed data against original dataset counts
3. Deep-dived into original source code (`model_m2pf.py`, `encoder_score.py`, etc.)
4. Analyzed `VirtuosoNetSingle` (Bi-LSTM baseline) vs our ablation implementation

**Data Investigation Results**:

| Metric | Original PercePiano | Our Data | Status |
|--------|---------------------|----------|--------|
| Labeled segments | 1,202 | 1,179 | MATCH (98.1%) |
| Unique pieces | 22 | 22 | MATCH |
| Unique performers | 49 | 49 | MATCH |
| Perceptual dimensions | 19 | 19 | MATCH |

**Data Distribution (Highly Imbalanced)**:
```
Schubert_D960_mv3_8bars: 481 segments (40%)
Schubert_D960_mv2_8bars: 221 segments (18%)
Schubert_D935_no.3_4bars: 117 segments (10%)
16 Beethoven variations: ~14 segments each (19%)
Other pieces: remaining (13%)
```

**Critical Finding: Bi-LSTM Baseline Architecture Mismatch**

Original `VirtuosoNetSingle` (from `model_m2pf.py:56-85`):
```python
class VirtuosoNetSingle(nn.Module):
    def __init__(self, net_param, data_stats):
        # SINGLE LSTM combining ALL hierarchy layers
        self.lstm = nn.LSTM(
            net_param.note.size,  # 256
            net_param.note.size,  # 256
            net_param.note.layer + net_param.voice.layer +
            net_param.beat.layer + net_param.measure.layer,  # 2+2+2+1 = 7 LAYERS
            batch_first=True, bidirectional=True, dropout=net_param.drop_out
        )
        self.note_contractor = nn.Linear(512, 512)
        self.note_attention = ContextAttention(512, 8)
        self.out_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 19),
        )

    def forward(self, x):
        x_embedded = self.note_embedder(x)  # 79 -> 256
        score_embedding, _ = self.lstm(x_embedded)  # 7-layer LSTM
        score_embedding, _ = pad_packed_sequence(score_embedding, True)
        note = self.note_contractor(score_embedding)  # 512 -> 512
        note_output = self.note_attention(note)  # Attention aggregation
        outputs = self.out_fc(note_output)
        return outputs
```

Our Ablation (from `diagnostics.py:720-773`):
```python
def _forward_ablated(self, pl_module, batch, note_locations):
    # Uses separate LSTMs
    note_out, _ = han.note_lstm(x_packed)  # 2-layer LSTM
    voice_out = han._run_voice_processing(...)  # 2-layer voice LSTM
    hidden_out = torch.cat([note_out, voice_out], dim=-1)  # 1024

    # Zero out hierarchy (but still process through full pipeline!)
    beat_spanned = torch.zeros(...)  # 512 zeros
    measure_spanned = torch.zeros(...)  # 512 zeros
    total_note_cat = torch.cat([hidden_out, beat_spanned, measure_spanned], dim=-1)  # 2048

    # Process 2048-dim through contractor (half zeros!)
    contracted = pl_module.performance_contractor(total_note_cat)  # 2048 -> 512
```

**Architecture Comparison Table**:

| Aspect | Original VirtuosoNetSingle | Our "Ablated" Model |
|--------|---------------------------|---------------------|
| LSTM architecture | Single 7-layer bidirectional | 2x separate 2-layer bidirectional |
| Voice processing | None | Full voice LSTM (2 layers) |
| Total LSTM layers | 7 | 4 (split into note + voice) |
| Backbone params | ~2M | ~4M (double capacity) |
| Input to contractor | 512 (LSTM output) | 2048 (1024 real + 1024 zeros) |
| Contractor behavior | 512 -> 512 (normal) | 2048 -> 512 (half zeros corrupt weights) |

**Impact of Mismatch**:

1. **Wrong comparison**: Our ablation compares HAN vs "broken HAN with zeros", not HAN vs true baseline
2. **Higher baseline capacity**: 4-layer split backbone may have different learning dynamics than 7-layer single
3. **Corrupted contractor**: Feeding 50% zeros to linear layer corrupts weight learning
4. **Hierarchy gain invalid**: The +0.13 R2 gain we measured is NOT the true hierarchy contribution

**Fixes Applied (Round 12)**:

| Fix | File | Description |
|-----|------|-------------|
| Checkpoint filename | `kfold_trainer.py:504-512` | Changed to `best-epoch={epoch:02d}-r2={val/mean_r2:.4f}` |
| auto_insert_metric_name | `kfold_trainer.py:511` | Added `auto_insert_metric_name=False` |
| get_trained_model() | `kfold_trainer.py:438-451` | Added method to retrieve in-memory trained model |
| Model storage | `kfold_trainer.py:673-674` | Store trained model after training completes |

**Next Steps (Implementation Required)**:

1. **Implement `VirtuosoNetSingle`**: Create proper 7-layer baseline matching original exactly
2. **Update ablation callback**: Use `VirtuosoNetSingle` for true baseline comparison
3. **Re-run ablation**: Measure true hierarchy gain (expected: +0.21 R2)
4. **Diagnostic check**: If baseline R2 ~0.19 and HAN still low, investigate HAN-specific issues

---

### 2025-12-26: Round 9 - Diagnostic Infrastructure + Hierarchy Investigation

**Problem**: Round 8 achieved R2 ~0.15 (CV average), matching Bi-LSTM baseline.
The hierarchical components (beat/measure attention) should add +0.21 R2 (per paper),
but appear to be contributing nothing.

**Hypothesis**: Either:
1. Beat/measure indices have issues preventing proper aggregation
2. Attention is collapsing or not learning
3. `span_beat_to_note_num` clamping corrupts first beat representation

**Diagnostic Infrastructure Added**:

| Component | File | Description |
|-----------|------|-------------|
| `DiagnosticCallback` | `diagnostics.py:166-562` | Captures activation variances, attention entropy |
| `HierarchyAblationCallback` | `diagnostics.py:564-721` | Compares full model vs Bi-LSTM only |
| `analyze_indices()` | `diagnostics.py:111-205` | Validates beat/measure index health |
| `run_full_diagnostics()` | `diagnostics.py:724-808` | Standalone diagnostic function |

**Issues Found & Fixed**:

| Issue | Root Cause | Fix |
|-------|------------|-----|
| Beat range [0, 26] | Including padding zeros in analysis | Only analyze valid positions (> 0) |
| 657 negative values | Zero-shifting applied to padding | Zero-shift only valid positions |
| Ablation device error | PackedSequence not moved to GPU | Explicitly handle PackedSequence device transfer |

**Fixes Applied**:

| Fix | File | Description |
|-----|------|-------------|
| analyze_indices | `diagnostics.py:111-205` | Only analyze valid (non-padding) positions |
| Ablation callback | `diagnostics.py:605-623` | Move PackedSequence components to device |
| run_full_diagnostics | `diagnostics.py:760-781` | Same PackedSequence fix |
| Re-enable ablation | `kfold_trainer.py:540-552` | Uncommented ablation callback |

**Initial Diagnostic Output** (before fixes):

```
[1] INDEX ANALYSIS:
  Beat range: [0, 26]
  Measure range: [0, 10]
  [WARNING] Negative zero-shifted values: 657

[2] ACTIVATION VARIANCES (std):
  note_out (LSTM)               0.0297 [LOW - ISSUE]
  beat_nodes                    0.0540 [OK]
  beat_spanned                  0.0540 [OK]
  measure_nodes                 0.0663 [OK]
  measure_spanned               0.0663 [OK]

[4] HIERARCHY CONTRIBUTION (fraction of total variance):
  hidden_out (Bi-LSTM):  47.6%
  beat_spanned:          26.2%
  measure_spanned:       26.2%
```

**Expected Diagnostic Output** (after fixes):

```
[1] INDEX ANALYSIS:
  Beat range: [1, 26]  (no padding zeros)
  Measure range: [1, 10]
  Zero-shifted values: OK (no negatives)

[ABLATION] Full R2: 0.15
[ABLATION] Ablated (Bi-LSTM only) R2: ???
[ABLATION] Hierarchy gain: +??? (expected ~+0.21)
```

**Remaining Investigation**:

If ablation shows hierarchy gain < 0.05 R2:
1. Check `span_beat_to_note_num` in `hierarchy_utils.py` - clamping may corrupt first beat
2. Verify `make_higher_node` boundary detection is correct
3. Compare attention weights with original implementation

---

## Ruled Out Issues

1. **Feature normalization** - CORRECT: 9 features z-scored (includes section_tempo), 70 categorical/embeddings unchanged
2. **Measure-aligned slicing** - Samples 51-413 notes, max_notes=1024 sufficient
3. **Sequence iteration** - HanEncoder doesn't use it (only ISGN encoders do)
4. **Fold assignment** - Fixed with greedy bucket balancing
5. **Labels** - Correct range (0.24-0.90 within [0,1])

---

## Deviations from Original (Round 8: SLICE SAMPLING FIXED)

As of Round 8, we now match the original PercePiano exactly:
- Architecture: No LayerNorm, prediction head 512->512->19, LR 2.5e-5
- Data: 84-dim features (79 base + 5 unnorm), includes section_tempo at index 5
- Batching: PackedSequence (not padded to fixed 1024)
- **Slice sampling**: 3-5 overlapping slices per performance, regenerated each epoch

---

## Active Diagnostics

### 1. Activation Check (`kfold_trainer.py:ActivationDiagnosticCallback`)

Runs on first batch only. Reports:
- Model parameter count
- Prediction head architecture verification
- Logits std, prediction std, health status
- Pred/target std ratio
- Per-dimension collapse check
- Context vectors presence

### 2. Gradient Monitor (`kfold_trainer.py:GradientMonitorCallback`)

Logs every 100 steps (verbose first 5). Reports:
- Gradient norms by layer category
- Balance ratio (prediction_head vs encoder)
- Context vector gradient norms
- Flags: imbalance (>10x), explosion, vanishing

### 3. TensorBoard Logging (`percepiano_replica.py`)

- val/pred_mean and val/pred_std each epoch
- Per-dimension R2 values

### 4. Comprehensive Diagnostics (`diagnostics.py:DiagnosticCallback`) - Round 9

Runs every 5 epochs during validation. Reports:
- Index analysis: beat/measure ranges, zero-shift validation
- Activation variances at each hierarchy level
- Attention entropy (collapsed vs uniform vs healthy)
- Hierarchy contribution analysis (% variance from each component)
- Gradient flow through all components

### 5. Hierarchy Ablation (`diagnostics.py:HierarchyAblationCallback`) - Round 9

Runs every 10 epochs. Reports:
- Full model R2
- Bi-LSTM only R2 (hierarchy zeroed out)
- Hierarchy gain (full - ablated)
- Warning if hierarchy contributing < 0.01 R2

---

## Expected Diagnostic Output (Round 5)

```
============================================================
  ACTIVATION CHECK - Batch 0
============================================================
  Model parameters: ~X,XXX,XXX (reduced from Round 4)
  Prediction head: 512->128, 128->19
    [OK] Prediction head architecture correct (Round 5)
  Learning rate: 5.00e-05
  Targets:     mean=0.55, std=0.106
  Predictions: mean=0.50, std=0.12, range=[0.1, 0.9]
  Logits:      mean=0.0, std=0.6

  Health Check:
    [OK] Logits std=0.60 in good range
    [OK] Prediction std=0.12 in good range
    [OK] Pred/target std ratio=1.13x (target: 0.8-1.5x)
    [OK] All 19 dimensions have healthy variance
    [OK] Context vectors present (Round 3 fix active)
============================================================

  [GRAD] Step 0: total=0.5-2.0
    han=0.3, contractor=0.8, attn=0.03, head=0.4
    context_vectors=0.002 [OK: learning]
    Balance (head/encoder): 0.4x [OK]
```

---

## If Problems Persist After Round 5

Potential next fixes (in priority order):

1. **Try PyTorch default initialization completely** - Remove all custom init
2. **Try gain=0.5** - Even more conservative than gain=1.0
3. **Check data preprocessing** - Compare feature distributions with original
4. **Increase training epochs** - May just need more time
5. **Learning rate warmup** - Gradual increase to 5e-5

---

## File Locations

### Our Implementation
- Model: `src/percepiano/models/percepiano_replica.py`
- Hierarchy utils: `src/percepiano/models/hierarchy_utils.py`
- Attention: `src/percepiano/models/context_attention.py`
- Trainer: `src/percepiano/training/kfold_trainer.py`
- Diagnostics: `src/percepiano/training/diagnostics.py` (Round 9)
- Dataset: `src/percepiano/data/percepiano_vnet_dataset.py`
- Notebook: `notebooks/train_percepiano_replica.ipynb`

### Original PercePiano (in data/raw)
- Encoder: `data/raw/PercePiano/virtuoso/virtuoso/encoder_score.py`
- Model utils: `data/raw/PercePiano/virtuoso/virtuoso/model_utils.py`
- Utils: `data/raw/PercePiano/virtuoso/virtuoso/utils.py`
- Training: `data/raw/PercePiano/virtuoso/virtuoso/train_m2pf.py`
- Config: `data/raw/PercePiano/virtuoso/ymls/shared/label19/han_measnote_nomask_bigger256.yml`
