## Loss Weight Autoresearch Changelog

### Setup
- **Goal:** Maximize pairwise accuracy on fold 0 validation
- **Guard:** R2 > 0.0 (must not collapse)
- **Baseline:** A1-Max defaults (listmle=1.5, contrastive=0.3, regression=0.3, invariance=0.1, mixup=0.2)
- **Baseline metrics:** pairwise=0.7644, R2=-0.055

### Iteration 1 -- REVERT (-0.012)
**Hypothesis:** Contrastive loss is near-zero during training; removing it should be neutral
**Change:** lambda_contrastive 0.3 -> 0.0
**Result:** pairwise 0.7644 -> 0.7529. R2 worsened.
**Why it failed:** Even at near-zero loss value, the contrastive gradient provides regularization that shapes the encoder's representation space. Removing it degrades both metrics.

### Iteration 2 -- REVERT (-0.008)
**Hypothesis:** Stronger ListMLE ranking signal should improve pairwise ordering
**Change:** lambda_listmle 1.5 -> 2.5
**Result:** pairwise 0.7644 -> 0.7564. R2 collapsed to -0.548.
**Why it failed:** Excessive ranking weight dominated the total loss, preventing auxiliary objectives (contrastive, regression) from contributing useful gradient signal. The shared encoder over-specialized for ranking at the expense of representation quality.

### Iteration 3 -- KEEP (+0.000 pairwise, +0.250 R2)
**Hypothesis:** Stronger CCC regression should improve absolute score calibration
**Change:** lambda_regression 0.3 -> 0.6
**Result:** pairwise 0.7644 -> 0.7647 (held), R2 -0.055 -> 0.195 (massive improvement)
**Why it worked:** The shared encoder can serve both ranking and regression at these weight ratios. Doubling regression weight forced the regression head to calibrate without stealing gradient from the ranking objective. CCC loss dropped from 0.27 to 0.08.

### Iteration 4 -- KEEP (+0.007 pairwise, +0.009 R2)
**Hypothesis:** Since removing contrastive hurt (iter 1), increasing it should help
**Change:** lambda_contrastive 0.3 -> 0.6
**Result:** pairwise 0.7647 -> 0.7715 (+0.007), R2 0.195 -> 0.204 (+0.009)
**Why it worked:** Stronger contrastive regularization improved the encoder's representation space, benefiting both ranking and regression downstream. Contrastive loss converges near-zero regardless of weight, but the gradient direction matters.

### Iteration 5 -- REVERT (-0.018)
**Hypothesis:** Mixup may add noise rather than regularization
**Change:** mixup_alpha 0.2 -> 0.0
**Result:** pairwise 0.7715 -> 0.7538. R2 also dropped.
**Why it failed:** With only ~900 training samples (PercePiano fold 0 train), mixup provides essential data augmentation. Removing it causes overfitting and hurts generalization.

### Iteration 6 -- REVERT (-0.006 pairwise, +0.058 R2)
**Hypothesis:** More regression should further improve R2
**Change:** lambda_regression 0.6 -> 1.0
**Result:** pairwise 0.7715 -> 0.7651 (-0.006), R2 0.204 -> 0.262 (+0.058)
**Why it partially failed:** Reveals a Pareto frontier between pairwise and R2. At regression=1.0, the model shifts toward absolute score prediction at ranking's expense.

### Iteration 7 -- KEEP (+0.000 pairwise, +0.036 R2)
**Hypothesis:** Regression=0.8 is the sweet spot between 0.6 and 1.0
**Change:** lambda_regression 0.6 -> 0.8
**Result:** pairwise 0.7715 (unchanged), R2 0.204 -> 0.240 (+0.036)
**Why it worked:** This weight exactly threads the Pareto frontier. Maximum R2 gain without any pairwise sacrifice.

### Iteration 8 -- REVERT (-0.002)
**Hypothesis:** Lower ListMLE with stronger auxiliaries might help
**Change:** lambda_listmle 1.5 -> 1.0
**Result:** pairwise 0.7715 -> 0.7694 (-0.002), R2 0.240 -> 0.269 (+0.029)
**Why it failed:** Same Pareto pattern. Reducing ranking weight improves R2 but costs pairwise.

### Summary

**Best weights found:**
| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| lambda_listmle | 1.5 | 1.5 | unchanged |
| lambda_contrastive | 0.3 | 0.6 | 2x |
| lambda_regression | 0.3 | 0.8 | 2.7x |
| lambda_invariance | 0.1 | 0.1 | unchanged |
| mixup_alpha | 0.2 | 0.2 | unchanged |

**Metric improvement:**
- Pairwise: 0.7644 -> 0.7715 (+0.0071, +0.9%)
- R2: -0.055 -> 0.240 (+0.295, from broken to meaningful)

**Key findings:**
1. The original A1-Max weights drastically underweighted regression (CCC), producing a model that ranked well but gave meaningless absolute scores
2. Contrastive loss provides crucial regularization even when its loss value is near zero
3. Mixup is essential for this dataset size (~900 training samples)
4. ListMLE at 1.5 is optimal -- both higher and lower hurt pairwise
5. There's a clear Pareto frontier between pairwise and R2; regression=0.8 is the optimal balance point
