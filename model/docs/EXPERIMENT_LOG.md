# PercePiano Replica Experiment Log

**Last Updated**: 2025-12-29 (Round 15 - Attention Fixes)
**Purpose**: Track debugging and reproduce PercePiano SOTA (R2 = 0.397)
**Reference**: See `PERCEPIANO_SOTA_REFERENCE.md` for architecture details.

---

## Current Status: Round 15 (Attention Fixes)

### Round 15 Changes Applied

Based on investigation of near-uniform attention (entropy=0.95), five fixes were implemented:

| Fix | Description | File | Impact |
|-----|-------------|------|--------|
| 1 | Apply temperature in hierarchy softmax | `hierarchy_utils.py:get_softmax_by_boundary()` | Sharpens attention in make_higher_node |
| 2 | Xavier init for attention_net, wider context_vector | `context_attention.py:ContextAttention.__init__()` | Better gradient flow at init |
| 3 | Separate LR for attention (10x multiplier) | `percepiano_replica.py:configure_optimizers()` | Faster attention learning |
| 4 | Entropy regularization | `percepiano_replica.py:compute_loss()` | Penalizes uniform attention > 0.6 |
| 5 | Boundary diagnostic logging | `hierarchy_utils.py:find_boundaries_batch()` | Logs ==1 vs >0 boundary differences |

### Root Cause Analysis

**Problem**: Beat attention entropy = 0.9508 (near-uniform) at initialization and not improving during training.

**Findings**:
1. Temperature=0.5 was set in ContextAttention but **never applied** in the hierarchy path
   - `make_higher_node()` calls `get_attention()` which returns raw similarity
   - `get_softmax_by_boundary()` applied raw softmax without temperature
2. Default PyTorch Linear initialization produces too-small weights (std ~0.04 for size=512)
   - `tanh(small) ≈ small` → small similarity values → uniform softmax
3. Attention parameters have same LR as encoder, but need to learn faster

### New Hyperparameters

```python
CONFIG = {
    ...
    'attention_lr_multiplier': 10.0,  # Higher LR for attention params
    'entropy_weight': 0.01,           # Penalty for uniform attention
    'entropy_target': 0.6,            # Target entropy (0=focused, 1=uniform)
}
```

### Expected Outcome (Round 15)

With fixes applied, expected improvements:
- Beat attention entropy: 0.95 → < 0.8 (more focused)
- Beat hierarchy gain: +0.041 → +0.10-0.15
- Total R2: +0.22 → +0.30-0.35

---

## Previous Status: Round 14 (Phase 2 Incremental Build)

### Phase 2 Results (Fold 2 - Longest Pieces)

| Model | Val R2 | Best Epoch | Hierarchy Gain | Expected Gain |
|-------|--------|------------|----------------|---------------|
| Baseline (7-layer BiLSTM) | **+0.1765** | 5 | - | ~0.19 |
| Baseline + Beat | **+0.2175** | 26 | **+0.041** | +0.10 to +0.15 |
| Baseline + Beat + Measure | (pending) | - | - | +0.05 to +0.10 |

**Key Finding**: Beat hierarchy provides +0.041 R2 gain, which is positive but below the expected +0.10-0.15. Total expected hierarchy gain is ~+0.21.

### Critical Issues Identified

1. **Beat Attention Near-Uniform**: Entropy = 0.9508 (1.0 = perfectly uniform)
   - Attention is NOT learning to focus on specific beats
   - All beats treated equally, adding noise rather than signal

2. **Prediction Collapse at Init**: pred_std = 0.008-0.009 (target: 0.10-0.15)
   - Logits std = 0.031-0.034 (target: 0.5-1.5)
   - 19/19 dimensions collapsed at initialization
   - Improves during training but remains suboptimal

3. **High Validation Variance**: R2 jumps wildly between epochs
   - Epoch 8: +0.2108, Epoch 10: +0.0522, Epoch 22: +0.1852, Epoch 26: +0.2175
   - Suggests sensitivity to slice sampling or batch composition

4. **Diagnostic Callback Bug** (FIXED): `DiagnosticCallback._run_diagnostic_forward()` only supports full HAN model, shows all zeros for incremental models

5. **Checkpoint Loading Bug** (FIXED): `kfold_trainer.py:771-793` was loading wrong model class for `MODEL_TYPE_BASELINE_BEAT`

### Previous Status: Round 13

| Model | Our R2 | Paper R2 | Status |
|-------|--------|----------|--------|
| Bi-LSTM Baseline | **+0.1931** (Fold 2) | 0.185 | MATCHED |
| HAN (Hierarchical) | **+0.2908** (Fold 2) | 0.397 | PARTIAL |
| Hierarchy Gain | +0.0977 | +0.212 | UNDER-PERFORMING |

### Key Finding: Piece-Length Dependency

The hierarchy only works on longer pieces:

| Fold | Beat Range | Measure Range | Measure Contribution | Val R2 |
|------|------------|---------------|---------------------|--------|
| 0 | 1-26 | 1-10 | 12-15% | +0.1868 |
| 1 | 1-8 | 1-5 | **1-2%** (collapsed) | +0.0965 |
| 2 | 1-54 | 1-18 | 16-21% (healthy) | **+0.2908** |
| 3 | (incomplete) | - | 13% | -0.1587 |

**Root Cause**: Short pieces (Fold 1: only 5 measures) provide insufficient hierarchical structure for beat/measure LSTMs to learn meaningful patterns.

### Persistent Issues

1. **Prediction Collapse**: pred_std = 0.06-0.08 vs target_std = 0.12-0.15
2. **Near-Uniform Attention**: Entropy 0.95-1.0 (should be < 0.8)
3. **Generalization Failure**: CV R2 = +0.19, Test R2 = -0.45

---

## Verified Configuration (Matching SOTA)

| Component | Value | Verified |
|-----------|-------|----------|
| Hidden size | 256 (all levels) | Yes |
| Layer counts | note=2, voice=2, beat=2, measure=1 | Yes |
| Attention heads | 8 | Yes |
| Dropout | 0.2 | Yes |
| Learning rate | 2.5e-5 | Yes |
| Batch size | 8 | Yes |
| Weight decay | 1e-5 | Yes |
| Optimizer | Adam (not AdamW) | Yes |
| LR Scheduler | StepLR(step_size=3000, gamma=0.98) | Yes |
| Gradient clipping | 2.0 | Yes |
| Loss function | MSE after sigmoid | Yes |
| Precision | FP32 | Yes |
| Prediction head | 512->512->19 | Yes |
| LayerNorm | None (removed) | Yes |
| Slice sampling | 3-5 overlapping slices, regenerated each epoch | Yes |
| max_notes/slice_len | 5000 | Yes |
| Features | 79-dim (includes section_tempo) | Yes |
| Batching | PackedSequence | Yes |
| LSTM init | Orthogonal + forget gate bias=1.0 | Yes |
| Attention temperature | 0.5 | Yes |

---

## Round 13 Detailed Results

### Per-Fold Performance

| Fold | Epochs | Best Epoch | Val R2 | Test R2 | Notes |
|------|--------|------------|--------|---------|-------|
| 0 | 58 | 36 | +0.1868 | -0.6093 | Inverted train/val split |
| 1 | 46 | 24 | +0.0965 | -0.4750 | Measure hierarchy collapsed (1-2%) |
| 2 | 29 | 7 | **+0.2908** | -0.3406 | Best fold, longest pieces |
| 3 | 1 | - | -0.1587 | - | Interrupted |

### Cross-Fold Dimension Analysis (Mean R2)

| Dimension | Mean R2 | Std | Status |
|-----------|---------|-----|--------|
| space | +0.4026 | 0.047 | Strong |
| timbre_brightness | +0.3764 | 0.139 | Strong |
| mood_energy | +0.3655 | 0.254 | Strong |
| timbre_loudness | +0.3595 | 0.106 | Strong |
| mood_imagination | +0.3021 | 0.095 | Strong |
| articulation_touch | +0.2641 | 0.116 | OK |
| sophistication | +0.2564 | 0.163 | OK |
| dynamic_range | +0.2365 | 0.053 | OK |
| articulation_length | +0.2253 | 0.141 | OK |
| balance | +0.1772 | 0.104 | OK |
| timbre_depth | +0.1529 | 0.078 | OK |
| timbre_variety | +0.1300 | 0.056 | OK |
| interpretation | +0.1208 | 0.035 | OK |
| drama | +0.0872 | 0.031 | Weak |
| timing | +0.0748 | 0.054 | Weak |
| tempo | +0.0648 | 0.052 | Weak |
| pedal_amount | +0.0450 | 0.136 | Weak |
| mood_valence | +0.0008 | 0.196 | Weak |
| pedal_clarity | -0.0061 | 0.081 | Failed |

**Summary**: 18/19 positive R2, 9/19 strong (>= 0.2)

### Ablation Analysis (Hierarchy Gain)

| Fold | Full HAN R2 | Zeroed Hierarchy R2 | Gain |
|------|-------------|---------------------|------|
| 0 | +0.19 | -0.07 | **+0.26** |
| 1 | +0.10 | +0.01 | +0.09 (weak) |
| 2 | +0.29 | +0.12 | **+0.17** |

**Observation**: Hierarchy gain varies 3x depending on piece length.

---

## Baseline Checkpoint Analysis

The best baseline checkpoint (`best-epoch=14-r2=0.2998.ckpt`) from Fold 2:

| Parameter | Value |
|-----------|-------|
| Model | PercePianoBiLSTMBaseline (7-layer BiLSTM) |
| R2 | 0.2998 |
| Epoch | 14 |
| Global Step | 1,410 |
| Parameters | 11,333,139 |
| LSTM | 7-layer bidirectional |
| Hidden | 256 |
| Attention | ContextAttention(512, 8 heads, temp=0.5) |

---

## Next Steps: Incremental Build Approach

### Phase 1: Verify Baseline - COMPLETE
- [x] Train `PercePianoBiLSTMBaseline` on Fold 2
- [x] Result: R2 = +0.1765 (matches expected ~0.19)

### Phase 2: Incremental Hierarchy - IN PROGRESS
- [x] Baseline (7-layer BiLSTM): R2 = +0.1765
- [x] Baseline + Beat: R2 = +0.2175 (gain: +0.041, expected: +0.10-0.15)
- [ ] Baseline + Beat + Measure: (pending)

### Phase 2 Investigation Priorities
The beat hierarchy gain (+0.041) is below expected (+0.10-0.15). Key issues to investigate:

1. **Beat Attention Not Learning** (HIGH PRIORITY)
   - Entropy = 0.9508 at init (1.0 = uniform)
   - Need to check if entropy decreases during training
   - Consider: lower temperature, higher LR for attention, different init

2. **Prediction Collapse at Init** (MEDIUM PRIORITY)
   - Logits std = 0.031 (target: 0.5-1.5)
   - Model starts predicting mean, has to "unlearn" this
   - Consider: different head initialization, output scaling

3. **High Validation Variance** (MEDIUM PRIORITY)
   - R2 jumps from +0.21 to +0.05 between epochs
   - Slice regeneration may cause instability
   - Consider: fixed slices, larger patience, lower LR

4. **DiagnosticCallback Bug** (LOW PRIORITY - cosmetic)
   - `_run_diagnostic_forward()` only supports full HAN
   - Shows all zeros for incremental models
   - Need to add support for `PercePianoBaselinePlusBeat` architecture

### Phase 3: Full Hierarchy
- [ ] Complete Beat+Measure training
- [ ] Compare total hierarchy gain to paper (+0.21)
- [ ] If still underperforming, investigate attention mechanisms

---

## Debugging History Summary

| Round | Date | Focus | Outcome |
|-------|------|-------|---------|
| 1-2 | 12-24 | Gradient explosion, prediction collapse | Fixed with FP32, LayerNorm |
| 3-4 | 12-25 | Context attention architecture | Fixed context vectors |
| 5 | 12-25 | Prediction head size (512->128) | Wrong - reverted |
| 6 | 12-25 | Remove LayerNorm, fix LR | Matched original config |
| 7 | 12-25 | Add section_tempo, PackedSequence | Fixed data pipeline |
| 8 | 12-26 | Slice sampling | Fixed - prediction collapse resolved |
| 9 | 12-26 | Diagnostic infrastructure | Added ablation callbacks |
| 10-12 | 12-27 | Baseline architecture mismatch | Created proper VirtuosoNetSingle |
| 13 | 12-28 | HAN LSTM initialization | Applied orthogonal init, temp=0.5 |
| 14 | 12-29 | Phase 2 incremental hierarchy | Beat gain +0.041 (below expected +0.10-0.15) |
| 15 | 12-29 | Attention fixes | Temperature in softmax, Xavier init, separate LR, entropy reg |

### Key Fixes Applied

1. **FP32 Precision** - Original uses FP32, not mixed precision
2. **No LayerNorm** - Original has zero normalization layers
3. **Prediction Head 512->512->19** - Config's `final_fc_size=128` is for decoder, not classifier
4. **section_tempo Feature** - Restored at index 5 (was missing)
5. **PackedSequence Batching** - Original uses pack_sequence, not pad_sequence
6. **Slice Sampling** - 3-5 overlapping slices per performance, regenerated each epoch
7. **Orthogonal LSTM Init** - Prevents signal collapse in deep LSTMs
8. **Attention Temperature 0.5** - Sharpens attention, prevents uniform weights
9. **Temperature in Hierarchy Softmax** (Round 15) - Temperature now applied in `get_softmax_by_boundary()`
10. **Xavier Attention Init** (Round 15) - Better gradient flow than default kaiming
11. **Separate LR for Attention** (Round 15) - 10x multiplier for faster attention learning
12. **Entropy Regularization** (Round 15) - Penalizes uniform attention > 0.6 entropy

---

## Ruled Out Issues

1. **Feature normalization** - Correct: 9 features z-scored, 70 categorical unchanged
2. **Sequence iteration** - HanEncoder doesn't use it (only ISGN encoders do)
3. **Fold assignment** - Fixed with greedy bucket balancing
4. **Labels** - Correct range (0.24-0.90 within [0,1])
5. **Dataset completeness** - 1,179 of 1,202 segments (98.1%)

---

## Data Distribution

**Highly imbalanced** - Schubert D960 mvmt 3 = 40% of all data:

| Piece | Segments | Percentage |
|-------|----------|------------|
| Schubert_D960_mv3_8bars | 481 | 40% |
| Schubert_D960_mv2_8bars | 221 | 18% |
| Schubert_D935_no.3_4bars | 117 | 10% |
| 16 Beethoven variations | ~14 each | 19% |
| Other pieces | - | 13% |

---

## File Locations

### Our Implementation

| Component | Path |
|-----------|------|
| Model | `src/percepiano/models/percepiano_replica.py` |
| Hierarchy utils | `src/percepiano/models/hierarchy_utils.py` |
| Attention | `src/percepiano/models/context_attention.py` |
| Trainer | `src/percepiano/training/kfold_trainer.py` |
| Diagnostics | `src/percepiano/training/diagnostics.py` |
| Dataset | `src/percepiano/data/percepiano_vnet_dataset.py` |
| Notebook | `notebooks/train_percepiano_replica.ipynb` |

### Original PercePiano (in data/raw)

| Component | Path |
|-----------|------|
| Encoder | `data/raw/PercePiano/virtuoso/virtuoso/encoder_score.py` |
| Model | `data/raw/PercePiano/virtuoso/virtuoso/model_m2pf.py` |
| Training | `data/raw/PercePiano/virtuoso/virtuoso/train_m2pf.py` |
| Config | `data/raw/PercePiano/virtuoso/ymls/shared/label19/han_measnote_nomask_bigger256.yml` |
