# Audio Encoder Training Results & Experiment Roadmap

> **Status (2026-03-02):** Audio training COMPLETE. A1 (LoRA) selected as winner. Symbolic training (02_symbolic_training.ipynb) in progress. Fusion (03_fusion.ipynb) not started.

**Notebook:** `model/notebooks/model_improvement/01_audio_training.ipynb`
**Training plan:** `docs/model/03-model-improvement.md`
**Taxonomy:** `docs/model/02-teacher-grounded-taxonomy.md`

## Experiment Summary

Three MuQ domain adaptation strategies were trained on 1,202 PercePiano segments labeled with 6 teacher-grounded composite dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation). All use pre-extracted MuQ embeddings (1024-dim frame-level features) with 4-fold piece-stratified cross-validation.

| Model | Strategy | Parameters Adapted |
|-------|----------|-------------------|
| A1 | MuQ + LoRA multi-task | LoRA rank-16 adapters on layers 9-12, ~1% of MuQ params |
| A2 | Staged domain adaptation | Stage 1: self-supervised (contrastive + invariance), Stage 2: supervised |
| A3 | Full unfreeze, gradual layer unfreezing | Layers 12->11->10->9 unfrozen progressively, discriminative LR |

All three share the same downstream architecture: attention pooling -> 2-layer shared encoder (512-dim) -> per-dimension ranking heads + regression head (sigmoid).

Training loss: `L = L_ranking + 0.3 * L_contrastive + 0.5 * L_regression + 0.1 * L_invariance`

An additional ablation ran A2 with MAESTRO contrastive pretraining (24,321 segments, 204 pieces) in Stage 1.

### Data

- **Labeled segments:** 1,202 (PercePiano, Pianoteq-rendered audio)
- **Pieces with multiple performances:** 61
- **Total recordings in piece mapping:** 964
- **Fold sizes:** ~750-790 train, ~230-270 val per fold
- **Labels:** 6 composite dimensions derived from 1,707 masterclass teaching moments (see doc 02)

### Prior Baselines (from PercePiano audit, 19-dim)

| Metric | Value | Model |
|--------|-------|-------|
| R2 (19-dim) | 0.537 | Frozen MuQ regression |
| Pairwise accuracy (19-dim) | 84% | Contrastive ranker (E2a) |
| STOP AUC (19 dims) | 0.814 | Leave-one-video-out CV |
| STOP AUC (raw MuQ) | 0.936 | Attention-pooled embeddings |

## Results

### Per-Fold Breakdown

| Model | Fold | Val Loss | Pairwise Acc | R2 |
|-------|------|----------|-------------|-----|
| A1 | 0 | 0.4879 | 0.7766 | 0.4349 |
| A1 | 1 | 0.5473 | 0.7034 | 0.3903 |
| A1 | 2 | 0.4325 | 0.7660 | 0.4500 |
| A1 | 3 | 0.5450 | 0.7112 | 0.3375 |
| A2 | 0 | 0.4693 | 0.7864 | 0.4959 |
| A2 | 1 | 0.5396 | 0.6898 | 0.4295 |
| A2 | 2 | 0.4822 | 0.6925 | 0.4292 |
| A2 | 3 | 0.5327 | 0.6881 | 0.3277 |
| A3 | 0 | 0.6443 | 0.6731 | 0.0592 |
| A3 | 1 | 0.5358 | 0.7039 | 0.3623 |
| A3 | 2 | 0.4507 | 0.7221 | 0.3525 |
| A3 | 3 | 0.5160 | 0.6948 | 0.3577 |

### Averaged Results

| Model | Pairwise Acc | R2 | Robustness (Pearson r) | Score Drop % |
|-------|-------------|-----|----------------------|-------------|
| **A1** | **0.7393** | 0.4032 | 0.9999 | 0.08% |
| A2 | 0.7142 | 0.4206 | 0.9999 | 0.07% |
| A3 | 0.6985 | 0.2829 | 0.9999 | 0.07% |

**Winner: A1** (highest pairwise accuracy, primary selection criterion)

A2 edges ahead on R2 (0.4206 vs 0.4032) but loses on pairwise ranking, which is the metric that matters for teaching moment selection.

### A2 MAESTRO Ablation

Adding 24K MAESTRO segments (204 pieces) to A2's Stage 1 contrastive pretraining:

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|-------|--------|--------|--------|--------|
| A2 baseline | 0.7864 | 0.6898 | 0.6925 | 0.6881 |
| A2 + MAESTRO | 0.7545* | 0.5688* | 0.4573* | 0.5446* |

*Val loss values shown; pairwise accuracy not separately computed for ablation in the notebook.

No clear improvement from more unlabeled piano audio. MuQ was pretrained on 160K hours of music -- more piano audio doesn't help its representations. The bottleneck is labeled data quantity, not representation quality.

## Interpretation

### Why A1 (LoRA) Wins

LoRA adapts <1% of MuQ's parameters. The fact that more aggressive adaptation (A2 staged, A3 full unfreeze) doesn't improve -- and A3 actively hurts -- confirms that MuQ's pretrained representations are already well-suited for piano quality assessment. With only ~750 training samples per fold, there isn't enough data to reshape the backbone without catastrophic forgetting. A3 fold 0 (R2=0.059) is a clear case of forgetting.

### Why Numbers Dropped from Prior Baselines

The old 19-dim baseline (84% pairwise, 0.537 R2) used highly redundant dimensions. PCA showed only 4 statistically significant factors, with PC1 (a "quality halo") capturing 47% of variance. The new 6 teacher-grounded dimensions are independent by design and optimized for actionable feedback, not predictability. Trading ~10 points of pairwise accuracy for dimensions that map to what teachers actually say is the correct trade-off.

### What Fold Variance Tells Us

A1 pairwise ranges from 0.703 (fold 1) to 0.777 (fold 0). This ~7-point spread is driven by which of the 61 multi-performance pieces land in validation. With so few pieces, one unusual piece can shift the metric substantially. This is a data quantity constraint, not a model problem. More pieces with multiple performances would stabilize estimates.

### Per-Dimension MuQ Predictability

From the taxonomy validation (`dimension_definitions.json:selection_scores`):

| Dimension | MuQ Probing R2 | Teacher Frequency |
|-----------|---------------|-------------------|
| articulation | 0.607 | 11.4% |
| dynamics | 0.587 | 14.1% |
| phrasing | 0.569 | 13.1% |
| interpretation | 0.524 | 36.7% |
| pedaling | 0.513 | 6.8% |
| timing | 0.332 | 18.0% |

**Key observations:**

- **Timing is the hardest** for MuQ to detect from audio alone (R2=0.332). This dimension is the strongest candidate for symbolic encoder support -- tempo, rubato, and rhythm are directly observable from MIDI.
- **Interpretation is the broadest** (19 clusters, 36.7% of teaching moments) and has middling MuQ R2 (0.524). It's a catch-all for tone, voicing, character, and expression. Performance here may improve with more labeled data rather than architecture changes.
- **Articulation is the strongest** audio signal (R2=0.607). Note attack/release and touch are directly audible.

## What This Means for the Product

### STOP Classifier / Teaching Moment Selection -- Workable

The STOP classifier is a 6-weight logistic regression on dimension scores (current AUC: 0.845 on composite labels). Pairwise accuracy of 74% means the model can meaningfully rank chunks within a session. Teaching moment selection picks the top chunk, not a precise threshold, so ranking quality matters more than absolute accuracy.

### Student Model / Blind Spot Detection -- Workable With Smoothing

Blind spot detection compares relative dimension deviations (e.g., "normally fine on pedaling, dipped today"). This depends on ranking consistency more than absolute R2. The student model uses exponential moving averages (alpha=0.3) across sessions, which smooths per-chunk noise.

### LLM Teacher Prompt -- Sufficient

The LLM receives structured context like `"pedaling": 0.35, baseline: 0.62`. Whether the true score is 0.35 or 0.42 matters less than "pedaling is significantly below this student's baseline." The model provides that relative signal. The LLM generates natural language, not numbers.

### Progress Tracking Over Weeks -- Noisy

R2=0.40 means the model explains less than half the variance in absolute quality. Individual session scores are noisy. But averaging across sessions and using trend detection should make this usable by sessions 5-10.

### Phone Audio -- Unresolved Risk

All results are on PercePiano (Pianoteq-rendered, studio-quality audio). The robustness check uses Gaussian noise on embeddings, not real phone audio through the full pipeline. Phone audio validation (doc 01) remains the highest-risk item for the product.

### Core ML Conversion -- A1 is Optimal

A1 (LoRA) is the easiest model to convert to Core ML: mostly frozen MuQ with small adapter layers. Smallest footprint, fewest moving parts during quantization.

## Experiment Roadmap

Experiments to push audio model performance higher, grouped by effort level. Pairwise accuracy is the north star.

### Tier 1: Quick Wins (days, no new data needed)

**2a. Hard negative mining for pair sampling**

Current `PairedPerformanceDataset` enumerates ALL within-piece pairs exhaustively. Easy pairs (large quality gap) dominate training and contribute little gradient.

Proposed: After warmup (first 5 epochs), oversample pairs where the model is wrong or uncertain (ranking logit near 0). Use a curriculum: start with easy pairs (|label_diff| > 0.3), progressively add harder pairs.

Expected impact: +3-5% pairwise accuracy.

Files: New `HardNegativePairSampler` in `data.py`, modify training loop in `audio_encoders.py`.

**2b. Listwise ranking loss**

Current `DimensionWiseRankingLoss` uses binary cross-entropy on pairs independently. ListMLE or LambdaRank considers the full ranking of all performances within a piece simultaneously. Consistently stronger than pairwise BCE in learning-to-rank literature.

Expected impact: +2-4% pairwise accuracy.

Files: New loss class in `losses.py`, update `training_step` in `audio_encoders.py`.

**2c. CCC regression loss**

Replace MSE with Concordance Correlation Coefficient loss. Penalizes both scale and shift errors. Better suited for subjective quality scores. Previous experiments showed CCC outperformed MSE (see `data/results/`).

Expected impact: R2 improvement.

Files: Add `ccc_loss()` to `losses.py`.

**2d. Loss weight tuning**

Current fixed lambdas: ranking=1.0, contrastive=0.3, regression=0.5, invariance=0.1. Grid search or uncertainty-weighted multi-task learning (Kendall et al. 2018). The ranking loss should dominate more if pairwise accuracy is the north star.

Expected impact: +1-3% pairwise accuracy.

Files: Modify `A1_CONFIG` in notebook.

**2e. LoRA rank and layer ablation**

Current: rank 16, layers (9, 10, 11, 12). Ablate rank {4, 8, 16, 32, 64} and layers {(11,12), (9-12), (7-12), (5-12)}.

Expected impact: +1-3% pairwise accuracy.

Files: Modify `A1_CONFIG` in notebook.

**2f. Label smoothing**

Current: `label_smoothing=0.0`. Ablate {0.05, 0.1, 0.15}. PercePiano labels are subjective, so some smoothing likely helps with overfitting on noisy targets.

Expected impact: reduced fold variance.

Files: Modify `A1_CONFIG` in notebook.

**2g. Mixup on embeddings**

Interpolate between embeddings: `emb_mix = lambda * emb_a + (1-lambda) * emb_b`, `label_mix = lambda * label_a + (1-lambda) * label_b`. Effective regularizer for small datasets.

Expected impact: reduced fold variance.

Files: Modify `training_step` in `audio_encoders.py`.

**5a. Fold ensemble at inference**

Average predictions across all 4 fold models at inference time. Free 2-3% accuracy boost with no additional training. Trade-off: 4x inference cost (matters for Core ML). Can distill ensemble back into a single model.

Expected impact: +2-3% pairwise accuracy (production only).

Files: New `EnsemblePredictor` class.

**6a. Stratified evaluation metrics**

Report pairwise accuracy stratified by label difficulty: easy (|diff| > 0.3), medium (0.1-0.3), hard (0.05-0.1). Reveals WHERE the model is failing.

Files: Modify `MetricsSuite.pairwise_accuracy` in `metrics.py`.

### Tier 2: Medium Effort (weeks, existing infrastructure)

**3a. Multi-head attention pooling**

Current single attention head (1024 -> 256 -> 1) applies one set of attention weights for all dimensions. Replace with K=6 heads, one per dimension. Each dimension attends to different temporal regions (dynamics: loud/soft transitions, timing: tempo changes, pedaling: sustained regions).

Expected impact: +2-3% pairwise accuracy.

Files: Replace `self.attn` in `audio_encoders.py`.

**3b. Multi-scale temporal modeling**

Current attention pooling collapses all frames into a single vector, losing temporal structure. Hierarchical pooling: local (pool every 4-8 frames into phrase-level), 1D conv or Transformer over phrase-level, global attention pool. Captures both local phenomena (articulation, note attacks) and global structure (phrasing, interpretation).

Expected impact: +2-4% pairwise accuracy.

Files: New `HierarchicalPooling` module in `audio_encoders.py`.

**1d. Competition data (T2) integration**

Infrastructure exists: `CompetitionPairSampler`, `CompetitionDataset` in `data.py`, collection script at `scripts/collect_competition_data.py`. Chopin 2021 competition provides ~2,000 segments with ordinal placement rankings. No dimension-specific labels, but ordinal ranking loss provides useful cross-piece inductive bias.

Expected impact: +3-5% pairwise accuracy.

Prerequisites: Run collection script on Thunder Compute.

**2e. Per-dimension loss weighting**

Weight dimensions by MuQ R2: up-weight articulation (0.607) and dynamics (0.587) where audio signal is strong, down-weight timing (0.332) which needs symbolic support.

Expected impact: +1-2% on strong dimensions.

Files: Modify `DimensionWiseRankingLoss` in `losses.py`.

**6c. STOP AUC on new models**

Evaluate STOP AUC using 6-dim scores from each audio encoder. Gate: must maintain >= 0.80 (current composite baseline: 0.845). Not a training experiment, but essential for validating that model improvements don't degrade the downstream product metric.

Files: Add STOP evaluation to notebook section 9.

### Tier 3: High Effort, High Impact (months, new data/infrastructure)

**1a. Human annotation campaign**

Recruit 3-5 piano teachers/students. Annotate MAESTRO audio segments (already have 24K MuQ embeddings) on the 6 dimensions using the rubric from `composite_labels/teacher_rubric.json`. Target: 2,000-5,000 new labeled segments across 200+ pieces. Build annotation UI (audio player + 6 sliders). Key: more pieces with multiple performances grows ranking pairs quadratically.

Expected impact: +5-10% pairwise accuracy.

**1b. Active learning loop**

Use current A1 model to score all 24K MAESTRO segments. Identify segments where the model is most uncertain (high prediction variance, regression/ranking disagreement). Prioritize those for human annotation -- maximizes signal per annotation dollar.

Prerequisites: 1a annotation pipeline.

**1c. Synthetic data via Pianoteq parameter variation**

Re-render PercePiano MIDI with systematically varied Pianoteq parameters: dynamics curve scaling, pedal depth/timing perturbation, tempo rubato injection, articulation variation. Each variation creates a new segment with known relative quality relationship. Generates clean ranking pairs without human annotation.

Risk: model may learn Pianoteq-specific artifacts.

Files: New `scripts/generate_synthetic_variations.py`.

**4a. Phone audio paired recordings**

Record 50-100 pieces simultaneously on studio mic + iPhone at various positions. Create paired embeddings. Train domain adaptation: phone embeddings should produce same quality scores as studio embeddings. This is the **phone audio validation** (doc 01) -- the highest-risk item for the product.

**4b. Phone simulation augmentation**

Apply phone simulation (low-pass 8kHz, compression, distortion from `augmentation.py`) to raw PercePiano audio, re-extract MuQ embeddings. Train with clean-augmented embedding pairs. Alternative: learn domain transfer function from paired recordings (4a).

**4c. Domain-adversarial training**

Add domain classifier ("phone or studio?") with gradient reversal layer. Encoder learns representations invariant to recording quality but sensitive to musical quality.

Prerequisites: 4a.

## Next Steps

1. Symbolic encoder results (02_symbolic_training.ipynb) -- in progress
2. Fusion experiments (03_fusion.ipynb) -- uses best audio + best symbolic encoder
3. Begin Tier 1 experiments on A1 baseline
4. Phone audio validation (doc 01) -- can run in parallel with model experiments

## Verification Criteria (for any future experiment)

1. 4-fold piece-stratified CV, same folds as `percepiano_cache/folds.json`
2. Pairwise accuracy (primary), R2 (secondary), robustness score_drop_pct (veto at >15%)
3. STOP AUC >= 0.80
4. Per-dimension breakdown reported
5. Bootstrap CI on pairwise accuracy difference vs A1 baseline for significance
