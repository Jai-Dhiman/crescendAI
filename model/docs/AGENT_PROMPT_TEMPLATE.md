# Task: Evaluate PercePiano Replica Training Results and Iterate Toward SOTA

## Objective

We are replicating the PercePiano SOTA model (R2 = 0.397) as a baseline before building our custom piano performance evaluation system. After multiple debugging rounds, we need to evaluate the latest training results and iterate until we match the published performance.

## Reference Documents

Read these documents first for full context:

1. **`docs/PERCEPIANO_SOTA_REFERENCE.md`** - Architecture, hyperparameters, and debugging checklist
2. **`docs/EXPERIMENT_LOG.md`** - Our complete debugging history (Rounds 1-5)

## Problem History (Summary)

| Round | Issue | Fix | Result |
|-------|-------|-----|--------|
| 1 | Gradient explosion, prediction collapse | FP32, LayerNorm, ContextAttention fix | Gradients fixed, predictions still collapsed |
| 2 | Logits std too small (0.05) | Xavier gain=2.0, bias [-1,1] | Activations fixed, R2 still negative |
| 3 | Missing context vectors in final attention | Changed to ContextAttention | Context vectors learning, pred std 2.3x too high |
| 4 | Prediction std overshoot (0.242 vs 0.106) | LR=5e-5, gain=1.0, bias [-0.1,0.1] | All activation metrics pass, but R2 stuck at -0.04 |
| 5 | **Prediction head size wrong (512 vs 128)** | Use `final_hidden=128`, PyTorch default init | **PENDING - awaiting results** |

## Round 5 Fix (Current)

The SOTA config specifies `final_fc_size: 128`, but our implementation was using 512. This added ~200k extra parameters, causing slow convergence.

**Before**: `Linear(512, 512) -> GELU -> Linear(512, 19)`
**After**: `Linear(512, 128) -> GELU -> Linear(128, 19)`

## Expected Diagnostic Output (Round 5)

```
============================================================
  ACTIVATION CHECK - Batch 0
============================================================
  Model parameters: ~X,XXX,XXX (should be reduced from Round 4)
  Prediction head: 512->128, 128->19
    [OK] Prediction head architecture correct (Round 5)
  Learning rate: 5.00e-05
  Targets:     mean=0.55, std=0.106
  Predictions: mean=0.50, std=0.12, range=[0.1, 0.9]
  Logits:      mean=0.0, std=0.6

  Health Check:
    [OK] Logits std in good range (0.5-1.5)
    [OK] Prediction std in good range (0.10-0.15)
    [OK] Pred/target std ratio (0.8-1.5x)
    [OK] All 19 dimensions have healthy variance
    [OK] Context vectors present
============================================================
```

## Evaluation Criteria

### 1. Architecture Verification (Check first)

- Model parameters: Should be ~200k less than Round 4
- Prediction head: Should show `512->128, 128->19`
- `[OK] Prediction head architecture correct (Round 5)`

### 2. Activation Health

| Metric | Target Range | If Wrong |
|--------|--------------|----------|
| Logits std | 0.5-1.5 | Init issue |
| Prediction std | 0.10-0.15 | Init or architecture issue |
| Pred/target std ratio | 0.8-1.5x | Init too aggressive/conservative |
| Collapsed dimensions | 0/19 | Gradient flow issue |
| Context vectors | Present | Final attention wrong |

### 3. Gradient Balance

| Metric | Expected | If Wrong |
|--------|----------|----------|
| Total norm | 0.5-2.0 (stable) | LR or init issue |
| Balance ratio | 0.3-1.0x | Gradient imbalance |
| Context vectors | Learning (>1e-6) | Attention stuck |

### 4. R2 Progression (Critical)

| Epoch | Expected R2 | If Below |
|-------|-------------|----------|
| 5 | > 0 | Model not learning |
| 10 | > 0.05 | Slow convergence |
| 20 | > 0.15 | May need more epochs or different approach |
| Convergence | 0.30-0.40 | SOTA range achieved |

## Your Tasks

1. **Read the reference docs** to understand full context
2. **Analyze the training logs** against the criteria above
3. **Verify Round 5 fix is active**:
   - Is prediction head showing `512->128, 128->19`?
   - Are model parameters reduced?
4. **Assess activation health**: All metrics in expected ranges?
5. **Track R2 progression**: Is it on trajectory for SOTA?
6. **Identify remaining issues** if R2 not progressing as expected
7. **Propose next fixes** if needed, prioritized by likely impact
8. **Implement fixes** if requested

## Baselines to Beat

| Model | R2 | Status |
|-------|-----|--------|
| Mean prediction | 0.00 | Minimum viable |
| Bi-LSTM | 0.185 | Basic baseline |
| MidiBERT | 0.313 | Strong baseline |
| **HAN SOTA** | **0.397** | **Target** |

## If R2 Still Not Improving After Round 5

Potential next fixes (in priority order):

1. **Remove all custom initialization** - Use pure PyTorch defaults everywhere
2. **Try smaller learning rate** - 2.5e-5 instead of 5e-5
3. **Check data preprocessing** - Compare feature distributions with original
4. **Add learning rate warmup** - Gradual increase over first few epochs
5. **Increase training epochs** - May just need more time with correct architecture
6. **Compare with original code** - Run original PercePiano on same data

## Key Files

- Model: `model/src/percepiano/models/percepiano_replica.py`
- Attention: `model/src/percepiano/models/context_attention.py`
- Trainer: `model/src/percepiano/training/kfold_trainer.py`
- Notebook: `model/notebooks/train_percepiano_replica.ipynb`
- **SOTA Reference**: `model/docs/PERCEPIANO_SOTA_REFERENCE.md`
- **Experiment Log**: `model/docs/EXPERIMENT_LOG.md`

## Constraints

- Use `uv` for Python packages, `bun` for JavaScript
- No emojis in code/docs
- Explicit exception handling (no silent fallbacks)
- Don't create backup files

---

## Training Logs to Evaluate

[PASTE TRAINING LOGS HERE]
