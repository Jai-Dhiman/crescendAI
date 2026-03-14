# Encoder Training Results & Experiment Roadmap

> **Status (2026-03-13):** Audio training COMPLETE (A1 LoRA winner). A1-Max COMPLETE (best: 80.8% ensemble pairwise, +6.8pp vs A1). Symbolic training COMPLETE (S2 GNN winner). Layer 1 validation COMPLETE (all gates pass). Fusion (03_fusion.ipynb) next after S2-max.

**Notebooks:** `model/notebooks/model_improvement/01_audio_training.ipynb`, `02_symbolic_training.ipynb`
**Training plan:** `docs/model/03-model-improvement.md`
**Taxonomy:** `docs/model/02-teacher-grounded-taxonomy.md`

## Audio Encoder Experiments

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

### A1-Max: Stacked Tier 1 Improvements

A1-Max extends A1 with ListMLE ranking loss, CCC regression loss, embedding mixup, hard negative mining with curriculum, and configurable loss weights. Swept 18 configs (3 LoRA ranks x 2 layer ranges x 3 label smoothing values) x 4 folds = 72 runs.

Loss: `L = L_ranking + 1.5 * L_ListMLE + 0.3 * L_contrastive + 0.3 * L_CCC + 0.1 * L_invariance`

#### A1-Max Top 5 Configs (by mean pairwise accuracy)

| Config | LoRA Rank | Layers | Label Smooth | Pairwise | R2 |
|--------|-----------|--------|-------------|----------|-----|
| **A1max_r32_L7-12_ls0.1** | **32** | **7-12** | **0.1** | **0.7872** | **0.1553** |
| A1max_r8_L7-12_ls0.1 | 8 | 7-12 | 0.1 | 0.7866 | 0.1514 |
| A1max_r32_L7-12_ls0.05 | 32 | 7-12 | 0.05 | 0.7861 | 0.0974 |
| A1max_r8_L9-12_ls0.0 | 8 | 9-12 | 0.0 | 0.7859 | 0.1616 |
| A1max_r16_L9-12_ls0.1 | 16 | 9-12 | 0.1 | 0.7852 | 0.1393 |

All 18 configs beat the A1 baseline (73.9%). The winning config uses wider LoRA adaptation (layers 7-12 vs 9-12) and label smoothing (0.1).

#### A1-Max 4-Fold Ensemble (Deployed)

| Metric | Value | vs A1 Baseline |
|--------|-------|---------------|
| **Pairwise accuracy** | **80.77%** | +6.84pp |
| **R2** | **0.5021** | +0.0989 |
| **Robustness (score drop)** | **0.08%** | Same |
| **Robustness (Pearson r)** | **1.0000** | Same |

Ensembling the 4 fold models boosts pairwise from 78.7% (single fold mean) to 80.8% (+2.1pp). All deployment gates pass: pairwise > 73.9%, score drop < 15%.

#### A1-Max Interpretation

**ListMLE is the biggest contributor.** The ranking-dominant loss weighting (lambda_listmle=1.5) explicitly optimizes the Plackett-Luce ranking likelihood per piece, aligning the loss directly with the pairwise accuracy metric.

**Wider LoRA (layers 7-12) helps.** The top 3 configs all use layers 7-12 instead of 9-12. With ListMLE and CCC providing better learning signals, more adapted parameters can be utilized without overfitting.

**R2 dropped per-fold but recovered in ensemble.** Individual fold R2 (~0.15) is lower than A1 (~0.40) because the ranking-dominant loss de-emphasizes regression. However, the 4-fold ensemble R2 (0.50) recovers to baseline levels, suggesting the regression heads learn complementary patterns across folds.

## Audio Interpretation

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

## Symbolic Encoder Experiments

Four symbolic encoder architectures were trained on MIDI data. All pretrained on a 14,821-sequence corpus (ASAP + MAESTRO + ATEPP + PercePiano), then finetuned on 1,202 PercePiano segments with the same 6-dim composite labels and 4-fold CV as the audio experiments.

| Model | Strategy | Input Representation |
|-------|----------|---------------------|
| S1 | Transformer on REMI tokens | REMI tokenization (~500 vocab), masked token prediction pretrain |
| S2 | GNN on homogeneous score graph | Notes as nodes (pitch, velocity, onset, duration, pedal, voice), GATConv layers |
| S2H | Heterogeneous GNN on score graph | 4 edge types (onset, during, follow, silence), per-type link prediction pretrain |
| S3 | CNN + Transformer on continuous features | MIDI -> continuous curves (pitch, velocity, density, pedal, IOI), multi-scale 1D-CNN |

All share the same downstream pipeline: attention pooling -> z_symbolic [512] -> per-dimension ranking + regression heads.

### Symbolic Per-Fold Breakdown

| Model | Fold | Pairwise Acc | R2 |
|-------|------|-------------|-----|
| S1 | 0 | 0.7324 | 0.4353 |
| S1 | 1 | 0.6608 | 0.2310 |
| S1 | 2 | 0.6840 | 0.2964 |
| S1 | 3 | 0.6589 | 0.3498 |
| S2 | 0 | 0.7504 | 0.3575 |
| S2 | 1 | 0.6880 | 0.2467 |
| S2 | 2 | 0.6998 | 0.2877 |
| S2 | 3 | 0.7150 | 0.3693 |
| S2H | 0 | 0.7205 | 0.3810 |
| S2H | 1 | 0.7041 | 0.3517 |
| S2H | 2 | 0.6845 | 0.3260 |
| S2H | 3 | 0.6971 | 0.3793 |
| S3 | 0 | 0.7469 | 0.4332 |
| S3 | 1 | 0.6823 | 0.3760 |
| S3 | 2 | 0.6895 | 0.3413 |
| S3 | 3 | 0.6816 | 0.3379 |

### Symbolic Averaged Results

| Model | Pairwise Acc | R2 | Robustness (Pearson r) | Score Drop % | Alignment |
|-------|-------------|-----|----------------------|-------------|-----------|
| S1 | 0.6840 | 0.3282 | 0.9807 | 1.74% | 0.6413 |
| **S2** | **0.7133** | 0.3153 | - | - | - |
| S2H | 0.7016 | **0.3595** | - | - | - |
| S3 | 0.7001 | 0.3721 | 0.9993 | 0.21% | - |

**Winner: S2** (highest pairwise accuracy)

S3 is a close second on pairwise (0.7001 vs 0.7133) and leads on R2 (0.3721). S2H has the best R2 among GNNs (0.3595).

### Symbolic Interpretation

**S2 (GNN) wins on ranking because graph structure directly encodes musical relationships.** Note adjacency, harmonic intervals, and voice membership create an inductive bias for comparing performances: the GNN learns which structural patterns correlate with quality differences. The homogeneous graph generalizes better than S2H's richer 4-edge-type representation, likely because the extra edge types overfit with limited data.

**S3 (CNN + Transformer) has the best R2 (0.3721) and best robustness (0.9993 Pearson, 0.21% drop).** Its architecture is most analogous to MuQ (both CNN-based), and continuous feature curves preserve fine-grained timing and velocity information that tokenization (S1) or graph discretization (S2/S2H) may lose. The near-perfect robustness suggests the CNN's multi-scale kernels (3, 7, 15) learn stable, noise-resistant patterns.

**S1 (Transformer on REMI tokens) is the weakest ranker (0.6840) but still robust (0.9807).** REMI tokenization quantizes velocity into 32 bins and time into discrete shifts, losing the continuous nuance that S3 preserves. However, S1's fold 0 (0.7324) is competitive -- the weakness is concentrated in folds 1 and 3, suggesting sensitivity to which pieces land in validation.

**All symbolic encoders substantially beat the prior symbolic baseline** (R2=0.347 from hand-crafted MIDI features). S3's R2=0.3721 represents a 7% relative improvement, and the pairwise accuracy gains (0.68-0.71 vs no prior pairwise baseline for symbolic) are significant.

**Symbolic vs audio gap is real but narrower than expected.** The best symbolic pairwise (S2: 0.7133) trails audio (A1: 0.7393) by only 2.6 percentage points. This is encouraging for fusion -- the symbolic models capture enough complementary signal to potentially help the audio encoder, especially on timing (MuQ's weakest dimension at R2=0.332).

### Cross-Modality Comparison

| Rank | Model | Modality | Pairwise | R2 | Notes |
|------|-------|----------|----------|-----|-------|
| 1 | **A1-Max (ensemble)** | **Audio** | **0.8077** | **0.5021** | **Deployed** |
| 2 | A1-Max (single fold mean) | Audio | 0.7872 | 0.1553 | Best config |
| 3 | A1 (LoRA) | Audio | 0.7393 | 0.4032 | Previous best |
| 4 | A2 (Staged) | Audio | 0.7142 | 0.4206 | |
| 5 | S2 (GNN) | Symbolic | 0.7133 | 0.3153 | |
| 6 | S2H (Hetero GNN) | Symbolic | 0.7016 | 0.3595 | |
| 7 | S3 (CNN+Trans) | Symbolic | 0.7001 | 0.3721 | |
| 8 | A3 (Full Unfreeze) | Audio | 0.6985 | 0.2829 | |
| 9 | S1 (Transformer) | Symbolic | 0.6840 | 0.3282 | |

S2 and S2H outperform A3 (full unfreeze), which suffered catastrophic forgetting. The best symbolic encoders are competitive with audio when the audio model isn't well-adapted.

## What This Means for the Product

### STOP Classifier / Teaching Moment Selection -- Workable

The STOP classifier is a 6-weight logistic regression on dimension scores (current AUC: 0.845 on composite labels). With A1-Max ensemble pairwise accuracy of 80.8%, the model can reliably rank chunks within a session. Teaching moment selection picks the top chunk, not a precise threshold, so ranking quality matters more than absolute accuracy.

### Student Model / Blind Spot Detection -- Workable With Smoothing

Blind spot detection compares relative dimension deviations (e.g., "normally fine on pedaling, dipped today"). This depends on ranking consistency more than absolute R2. The student model uses exponential moving averages (alpha=0.3) across sessions, which smooths per-chunk noise.

### LLM Teacher Prompt -- Sufficient

The LLM receives structured context like `"pedaling": 0.35, baseline: 0.62`. Whether the true score is 0.35 or 0.42 matters less than "pedaling is significantly below this student's baseline." The model provides that relative signal. The LLM generates natural language, not numbers.

### Progress Tracking Over Weeks -- Noisy

R2=0.40 means the model explains less than half the variance in absolute quality. Individual session scores are noisy. But averaging across sessions and using trend detection should make this usable by sessions 5-10.

### Phone Audio -- Unresolved Risk

All results are on PercePiano (Pianoteq-rendered, studio-quality audio). The robustness check uses Gaussian noise on embeddings, not real phone audio through the full pipeline. Phone audio validation (doc 01) remains the highest-risk item for the product.

### Symbolic Input in Production -- AMT Bridge Risk

All symbolic encoder results use ground-truth MIDI from PercePiano. In production, MIDI input comes from automatic music transcription (AMT) of phone-recorded audio. AMT introduces velocity estimation errors, onset/offset jitter, missed/hallucinated notes, and pedal detection failures. The symbolic advantage on dynamics (S2: 0.77 vs A1: 0.70) and articulation (S2: 0.70 vs A1: 0.66) may not survive the AMT bridge. Layer 1 of the roadmap includes an AMT degradation test to quantify this risk before investing in fusion.

### MIDI as LLM Context -- Alternative to Model Fusion

Rather than fusing audio and symbolic at the model level, structured MIDI comparison (velocity curves vs score markings, onset deviations, note accuracy) can be fed directly to the teacher subagent alongside A1 scores. This approach is more robust to AMT noise (rule-based comparison degrades gracefully), fully interpretable, and buildable in days. It sidesteps the fusion problem entirely by letting the LLM reason about complementary signals.

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

See `docs/model/00-research-timeline.md` for the full training roadmap (Waves 1-3).

### Per-Dimension Complementarity

| Dimension | A1 (Audio) | S2 (GNN) | Winner |
|---|---|---|---|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

Audio wins timing decisively. Symbolic wins dynamics, articulation, and interpretation. AMT degradation test (0% drop) confirms the symbolic advantage survives transcription. This validates gated per-dimension fusion (F3).

### Summary

- **Wave 1 (current):** A1-max quick wins (IN PROGRESS), S2-max pretraining expansion + multi-source finetuning, F3 fusion, Core ML conversion
- **Wave 2 (months):** Phone audio paired recordings, diverse skill-level data, expert annotation campaign (2K+ segments), retrain on expanded data
- **Wave 3 (months-year):** Score alignment pipeline, score-conditioned model `quality = f(z_performance, z_score)`, bar-aligned LLM context

## Verification Criteria (for any future experiment)

1. 4-fold piece-stratified CV, same folds as `percepiano_cache/folds.json`
2. Pairwise accuracy (primary), R2 (secondary), robustness score_drop_pct (veto at >15%)
3. STOP AUC >= 0.80
4. Per-dimension breakdown reported
5. Bootstrap CI on pairwise accuracy difference vs A1 baseline for significance

## Layer 1 Validation Results (2026-03-11)

*Status: ALL 4 EXPERIMENTS COMPLETE*

Code: `model/src/model_improvement/layer1_validation.py`, `midi_comparison.py`, `feedback_assessment.py`.
Notebook: `model/notebooks/model_improvement/04_layer1_validation.ipynb`.

### Experiment 1: Competition Correlation -- PASS

A1 scores on 2,293 Chopin 2021 competition segments (11 performers) correlate with expert placement.

| Aggregation | rho | p-value | Gate |
|-------------|-----|---------|------|
| mean | +0.704 | 0.016 | PASS |
| min | +0.654 | 0.029 | PASS |
| median | +0.248 | 0.463 | INVESTIGATE |

Per-dimension (mean aggregation):

| Dimension | rho | p-value |
|-----------|-----|---------|
| dynamics | -0.917 | 0.0001 |
| timing | -0.590 | 0.056 |
| pedaling | +0.887 | 0.0003 |
| articulation | +0.292 | 0.383 |
| phrasing | +0.803 | 0.003 |
| interpretation | +0.169 | 0.620 |

**Key finding:** Pedaling and phrasing are the strongest predictors of competition placement. Dynamics has a strong *negative* correlation -- competition finalists show more controlled (not louder) dynamics. This suggests the dynamics dimension captures "amount" rather than "appropriateness," which may need score-conditioning (Layer 4) to become fully useful.

### Experiment 2: AMT Degradation -- PASS

S2 pairwise accuracy with ByteDance piano transcription vs ground-truth MIDI on 50 MAESTRO recordings (4 contrastive pieces, 107 pairs).

| Source | Overall Pairwise | Gate |
|--------|-----------------|------|
| Ground truth | 100.0% | -- |
| ByteDance | 100.0% (+0.0%) | PASS |

Per-dimension drops: dynamics -0.9%, timing +2.8%, pedaling +0.0%, articulation +0.9%, phrasing +3.7%, interpretation +1.9%. All well within the <10% gate.

**Note:** YourMT3+ was not available (not installed). Only one AMT system tested. Small evaluation set (4 contrastive pieces). Result is unambiguously positive but should be validated on a larger set if fusion is pursued.

### Experiment 3: Dynamic Range -- DIAGNOSTIC COMPLETE

A1 score distributions across skill levels (629 intermediate YouTube recordings, 1,202 advanced PercePiano, 2,293 professional Chopin 2021).

| Group | Mean Score | Std | N |
|-------|-----------|-----|---|
| Intermediate | 0.565 | 0.062 | 629 |
| Advanced (PercePiano) | 0.552 | 0.110 | 1,202 |
| Professional (Chopin 2021) | 0.595 | 0.064 | 2,293 |

| Comparison | Cohen's d |
|-----------|-----------|
| Intermediate vs Professional | 0.47 |
| Advanced vs Professional | 0.47 |
| Advanced vs Intermediate | 0.15 |

**Key finding:** The model separates skill levels at the group level (d=0.47) but individual discrimination is noisy. Advanced (PercePiano) scores are slightly *lower* than intermediate, likely reflecting dataset differences (Pianoteq-rendered vs real YouTube audio) rather than a model flaw. Usable for within-student progress tracking (comparing to own baseline) but not absolute skill-level classification.

### Experiment 4: MIDI-as-Context Feedback -- SKIP

LLM judge compared teacher observations generated with A1 scores only (Condition A) vs A1 scores + structured MIDI comparison (Condition B) on 20 Schubert D960 mv3 segments.

| Condition | Wins | Rate |
|-----------|------|------|
| A (scores only) | 11 | 55% |
| B (MIDI context) | 9 | 45% |

Decision: **SKIP** (below 55% BORDERLINE threshold).

Judge confidence: 10 high, 10 medium. Among high-confidence judgments, B won only 3/10. Among medium-confidence judgments, B won 6/10.

**Key finding:** Adding raw MIDI comparison data (velocity MAE, onset deviations, note F1) does not reliably improve teacher observation quality. The LLM already generates specific, actionable feedback from dimension scores alone. When MIDI data helps, it's through concrete note counts ("48 missed notes"), but the judge often found this added false precision rather than musical insight. Score comparison may become valuable with score-aligned context (bar numbers, passage references) rather than raw statistics.

### Decision Gate Summary

| Experiment | Gate | Result | Implication |
|-----------|------|--------|-------------|
| Competition correlation | rho > 0.3 | PASS (0.704) | A1 quality signal is real, proceed to Layer 2 |
| AMT degradation | drop < 10% | PASS (0.0%) | Symbolic path viable for fusion |
| Dynamic range | diagnostic | d=0.47 | Usable for within-student tracking |
| MIDI-as-context | B wins > 65% | SKIP (45%) | Do not build score comparison pipeline now |

**What this unlocks:**
- Layer 2 quick wins on A1 are validated (model signal is real)
- Fusion (A1 + S2) remains viable (AMT doesn't degrade S2)
- Core ML conversion of A1 is unblocked
- Score comparison pipeline is deprioritized -- revisit with bar-aligned context in Layer 4

### YouTube AMT Validation (2026-03-13)

**Goal:** Test whether ByteDance AMT produces usable MIDI from mediocre YouTube audio and whether S2 pairwise ranking still works. The MAESTRO AMT test (Experiment 2) used studio-quality audio -- this tests the hardest case: phone recordings, home pianos, digital keyboards, varying room acoustics.

**Method:** 51 YouTube recordings downloaded (56 approved, 5 failed due to yt-dlp issues). Audio -> MuQ embeddings -> A1 scores (reference). Audio -> ByteDance AMT -> MIDI -> `parsed_midi_to_graph()` -> S2 scores. Compare A1 vs S2 pairwise agreement at the recording level.

**Gate:** A1-vs-S2 agreement > 60% overall (random = 50%).

Code: `model/scripts/validate_youtube_amt.py`

| Metric | Value |
|--------|-------|
| Recordings | 50 (1 skipped: >10K notes) |
| Pairs evaluated | 1,225 |
| Overall agreement | **79.9%** (4,067/5,093) |

Per-dimension agreement:

| Dimension | Agreement | Pairs |
|-----------|-----------|-------|
| dynamics | 82.9% | 944 |
| timing | 76.7% | 772 |
| pedaling | 78.5% | 1,048 |
| articulation | 72.4% | 692 |
| phrasing | 82.2% | 736 |
| interpretation | 84.7% | 901 |

**Gate: PASS (79.9% >> 60%)**

**Key findings:**
- ByteDance AMT transcribed all 51 recordings successfully (0 failures, note counts 526-14,627 per file). AMT is robust to mediocre audio quality.
- All 6 dimensions exceed the 60% gate individually. Articulation is weakest (72.4%) -- likely because AMT velocity estimation is noisiest, and articulation depends on precise attack/release detection.
- Interpretation and dynamics are strongest (84.7%, 82.9%) -- these depend more on overall musical structure than note-level precision, which AMT preserves well.
- This validates the full production path: phone audio -> ByteDance AMT -> MIDI graph -> S2 scoring. The symbolic encoder provides meaningful signal even from non-studio recordings.

**Comparison with MAESTRO AMT test:** The MAESTRO test showed 0% pairwise accuracy drop (studio audio, ground-truth MIDI available for comparison). YouTube test shows ~80% A1-vs-S2 agreement. These aren't directly comparable (different metric: accuracy-vs-ground-truth vs cross-encoder agreement), but both confirm AMT viability.

**Decision:** Proceed with A1+S2 fusion. The symbolic path survives real-world audio conditions across all dimensions.
