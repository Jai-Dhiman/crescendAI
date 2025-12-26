# PercePiano Replica Experiment Log

**Last Updated**: 2025-12-25 (Round 6)
**Purpose**: Track our debugging journey to reproduce PercePiano SOTA (R2 = 0.397)

For architecture details and hyperparameters, see `PERCEPIANO_SOTA_REFERENCE.md`.

---

## Current Status

**Round 6** - Major architecture corrections based on original code analysis

| Change | Description |
|--------|-------------|
| LayerNorm | **REMOVED** - original has none |
| Prediction head | **FIXED** to 512->512->19 (was 512->128->19) |
| Learning rate | **REVERTED** to 2.5e-5 (was 5e-5) |

**Key Discovery (Round 6):**
The config's `final_fc_size: 128` is for the DECODER (not used), NOT the classifier.
The actual classifier uses `encoder.size * 2 = 512`. See `model_m2pf.py:118-124`.

**Expected after Round 6:**
- R2 crosses 0 by epoch 5
- R2 > 0.15 by epoch 20
- R2 reaches 0.30-0.40 at convergence (SOTA range)

---

## Verified Configuration (Matching SOTA - Round 6)

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

## Ruled Out Issues

1. **Feature normalization** - CORRECT: 8 features z-scored, 70 categorical/embeddings unchanged
2. **Measure-aligned slicing** - Samples 51-413 notes, max_notes=1024 sufficient
3. **Sequence iteration** - HanEncoder doesn't use it (only ISGN encoders do)
4. **Fold assignment** - Fixed with greedy bucket balancing
5. **Labels** - Correct range (0.24-0.90 within [0,1])

---

## Deviations from Original (Round 6: ALL REMOVED)

As of Round 6, we now match the original architecture exactly:
- No LayerNorm anywhere (removed)
- Prediction head: 512->512->19 (corrected)
- Learning rate: 2.5e-5 (reverted)

---

## Active Diagnostics

### 1. Activation Check (`kfold_trainer.py:ActivationDiagnosticCallback`)

Runs on first batch only. Reports:
- Model parameter count
- Prediction head architecture verification (Round 5)
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
- Attention: `src/percepiano/models/context_attention.py`
- Trainer: `src/percepiano/training/kfold_trainer.py`
- Notebook: `notebooks/train_percepiano_replica.ipynb`

### Original PercePiano (in data/raw)
- Encoder: `data/raw/PercePiano/virtuoso/virtuoso/encoder_score.py`
- Training: `data/raw/PercePiano/virtuoso/virtuoso/train_m2pf.py`
- Config: `data/raw/PercePiano/virtuoso/ymls/shared/label19/han_measnote_nomask_bigger256.yml`
