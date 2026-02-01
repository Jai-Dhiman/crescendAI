# Phase 1 Research: Disentangling Piece vs Performer in Piano Evaluation

> Living reference document for CrescendAI's next research phase.
> Last updated: 2026-01-31

---

## 1. Problem Statement

### The Core Limitation (from Paper 1)

Our current MuQ-based model achieves R^2 = 0.537 on PercePiano, but the **multi-performer analysis reveals a critical limitation**:

> "Low multi-performer variance (intra-piece std = 0.020) suggests the model captures piece characteristics more strongly than performer-specific expression."
> -- Section 5.3, Dhiman 2026

**Dimensions with highest performer sensitivity** (still quite low):

- dynamic_range: std = 0.027
- timing: std = 0.022
- articulation_touch: std = 0.020

**Why this happens**: When training with MSE loss on absolute scores, the model learns piece-level priors (e.g., "Chopin Ballade = high sophistication") rather than distinguishing performer quality within a piece. The strong correlation between piece difficulty and evaluation dimensions (PSyllabus rho = 0.623) provides an easy shortcut.

### Research Goal

Train a model that can **rank performances of the same piece** by quality on each evaluation dimension, disentangling:

- **Piece factors**: difficulty, style, composer, era
- **Performer factors**: timing precision, dynamic control, articulation quality, expressiveness

### Success Metric

**Pairwise ranking accuracy on held-out pieces**: Given two performances (A, B) of the same piece, can the model correctly identify which is better on dimension D?

---

## 2. Theoretical Foundation

### 2.1 Why Disentanglement is Hard

The challenge is that piece and performer factors are **entangled in the audio signal**:

| Factor | Appears in Audio as... |
|--------|------------------------|
| Piece difficulty | Note density, tempo, harmonic complexity |
| Performer timing | Micro-deviations from mechanical timing |
| Piece dynamics | Written dynamic markings (fff, pp) |
| Performer dynamics | How the written dynamics are realized |

**Key insight from disentanglement literature**: To separate factors, we need either:

1. **Supervision** - labels for each factor separately
2. **Architectural constraints** - force different parts of the network to capture different factors
3. **Contrastive objectives** - define what should be similar/different

Sources:

- [Disentangling Score Content and Performance Style](https://arxiv.org/html/2509.23878) - defines style as "expressive realization" vs content as "structural characteristics"
- [Guided VAE for Disentanglement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Guided_Variational_Autoencoder_for_Disentanglement_Learning_CVPR_2020_paper.pdf)
- [Rethinking Style and Content Disentanglement in VAEs](https://openreview.net/forum?id=B1rQtwJDG)

### 2.2 Learning to Rank vs Regression

Our current approach: **Pointwise regression** (predict absolute score for each sample)

- Limitation: Model can achieve low loss by predicting piece-level means

Better approach for this task: **Pairwise ranking** (predict relative ordering)

- Forces model to distinguish between performances of same piece
- More aligned with evaluation task (ranking performers)

| Approach | Input | Output | Loss |
|----------|-------|--------|------|
| Pointwise | Single performance | Absolute score | MSE |
| Pairwise | Pair (A, B) same piece | P(A > B) | BCE/Margin |
| Listwise | All performances of piece | Full ranking | ListMLE/NDCG |

Sources:

- [Pairwise Learning to Rank by Neural Networks Revisited](https://link.springer.com/article/10.1007/s10994-024-06644-6) - shows pairwise approaches can learn transitive relations
- [Learning to Rank Overview](https://en.wikipedia.org/wiki/Learning_to_rank)
- [TensorFlow Ranking](https://www.tensorflow.org/ranking) - supports all three paradigms

### 2.3 Contrastive Learning for Same-Content Comparison

Key papers on comparing different versions of same content:

**CLEWS (2025)** - Musical version matching with contrastive learning
> "We propose a method to learn from weakly annotated segments, together with a contrastive loss variant that outperforms alternatives... based on pairwise segment distance reductions."

Source: [CLEWS - Supervised Contrastive Learning for Musical Version Matching](https://proceedings.mlr.press/v267/serra25a.html)

**Triplet Loss vs Contrastive Loss**:
> "Triplet loss preserves greater variance within and across classes, supporting finer-grained distinctions... contrastive loss tends to compact intra-class embeddings, which may obscure subtle semantic differences."

Source: [Triplet Loss Overview](https://www.v7labs.com/blog/triplet-loss)

This is exactly what we need - we want to **preserve variance** between performances of the same piece, not collapse them.

---

## 3. Architecture Design: Three Approaches

### Overview

```
                    +------------------+
                    |   MuQ Encoder    |
                    |  (frozen L9-12)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Temporal Pooling |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v-------+ +----v----+ +-------v--------+
     | Approach A:    | | Approach| | Approach C:    |
     | Contrastive    | | B:      | | Disentangled   |
     | Ranking        | | Siamese | | Encoder        |
     +----------------+ +---------+ +----------------+
```

---

### Approach A: Contrastive Pairwise Ranking

**Core idea**: Use contrastive learning where positives/negatives are defined by piece identity, forcing the model to learn piece-invariant performer representations.

#### Architecture

```
Input: Performance audio x
       |
       v
+------------------+
|  MuQ L9-12       |  (frozen, 4096-dim frame embeddings)
+--------+---------+
         |
+--------v---------+
| Mean Pool        |  (sequence -> single vector)
+--------+---------+
         |
+--------v---------+
| Projection Head  |  z = MLP(pooled) -> 256-dim
+--------+---------+
         |
    +----+----+
    |         |
+---v---+ +---v---+
|Piece  | |Rank   |
|Head   | |Head   |
+-------+ +-------+
```

#### Loss Function

**Combined objective**:

```
L_total = L_ranking + lambda_piece * L_piece_contrastive
```

**L_ranking** (pairwise margin loss):
Given performances (A, B) of same piece, with ground truth A > B on dimension d:

```
L_ranking = max(0, margin - (score_d(A) - score_d(B)))
```

**L_piece_contrastive** (InfoNCE with piece-based positives):

```
For anchor a (performance of piece P):
  - Positive: another performance of P (different performer)
  - Negatives: performances of other pieces

L_piece = -log(exp(sim(z_a, z_pos)/tau) / sum(exp(sim(z_a, z_neg)/tau)))
```

This forces the projection head to learn **piece-invariant** features - what's shared across performers playing the same piece.

#### Training Data Strategy

From ASAP/MAESTRO:

- 206 pieces with multiple performances (631 recordings)
- Sample triplets: (anchor, positive_same_piece, negative_different_piece)
- For ranking: need proxy labels for "which is better" (see Section 5)

#### Hyperparameters (from literature)

| Parameter | Recommended | Source |
|-----------|-------------|--------|
| Temperature tau | 0.07 - 0.1 | [InfoNCE Best Practices](https://lilianweng.github.io/posts/2021-05-31-contrastive/) |
| Projection dim | 128-256 | [SimCLR](https://arxiv.org/abs/2002.05709) |
| Batch size | As large as possible (512+) | [SimCLR](https://arxiv.org/abs/2002.05709) |
| Margin (ranking) | 0.1 - 0.5 | Task-dependent |

Sources:

- [Understanding InfoNCE Loss in PyTorch](https://www.codegenes.net/blog/infonce-loss-pytorch/)
- [Temperature-Free Loss for Contrastive Learning](https://arxiv.org/html/2501.17683v1)

---

### Approach B: Siamese Dimension-Specific Ranking

**Core idea**: Explicitly model the comparison task with a Siamese architecture that takes two performances and outputs dimension-specific rankings.

#### Architecture

```
Performance A          Performance B
     |                      |
     v                      v
+----------+          +----------+
| MuQ      |          | MuQ      |  (shared weights)
+----+-----+          +-----+----+
     |                      |
+----v-----+          +-----v----+
| Pool     |          | Pool     |  (shared weights)
+----+-----+          +-----+----+
     |                      |
     +----------+-----------+
                |
         +------v------+
         | Comparison  |
         | Module      |  (concatenate, subtract, or bilinear)
         +------+------+
                |
         +------v------+
         | Dimension   |
         | Heads (x19) |  -> P(A > B) for each dimension
         +-------------+
```

#### Comparison Module Options

1. **Concatenation**: `[z_A; z_B; z_A - z_B; z_A * z_B]`
2. **Bilinear**: `z_A^T W z_B` for each dimension
3. **Cross-attention**: Let representations attend to each other

Source: [Siamese Networks - The Tale of Two Manifolds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Roy_Siamese_Networks_The_Tale_of_Two_Manifolds_ICCV_2019_paper.pdf)

#### Loss Function

**Binary cross-entropy for ranking**:

```
L = -sum_d [y_d * log(p_d) + (1 - y_d) * log(1 - p_d)]

where:
  y_d = 1 if A is better than B on dimension d
  p_d = sigmoid(comparison_output_d)
```

**With margin for confident predictions**:

```
L_margin = max(0, margin - (s_A - s_B)) if A > B
         = max(0, margin - (s_B - s_A)) if B > A
```

Sources:

- [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- [DirectRanker](https://arxiv.org/abs/1909.02768)

#### Training Strategy

**Pair sampling**:

1. For each piece with N performances, generate N*(N-1)/2 pairs
2. Balance pairs where A > B vs B > A
3. Hard negative mining: focus on pairs with small quality differences

**Data augmentation**:

- Time-shift (small random offset)
- Slight tempo variation (via time-stretching)
- Soundfont variation (use multiple Pianoteq presets)

---

### Approach C: Disentangled Dual-Encoder

**Core idea**: Explicitly separate piece-level and performer-level representations using adversarial training.

#### Architecture

```
Input: Performance audio x
              |
              v
      +-------+-------+
      |  MuQ Encoder  |
      +-------+-------+
              |
       +------+------+
       |             |
+------v-----+ +-----v------+
| Piece      | | Style      |
| Encoder    | | Encoder    |
| (captures  | | (captures  |
| what piece)| | how played)|
+------+-----+ +-----+------+
       |             |
       v             v
    z_piece       z_style
       |             |
       |      +------+------+
       |      |             |
       |  +---v---+   +-----v-----+
       |  | Dim   |   | Piece     |
       |  | Heads |   | Classifier|
       |  +-------+   | (adversary|
       |              +-----------+
       |                    ^
       +-------> GRL -------+
```

**GRL = Gradient Reversal Layer**: During backprop, reverses gradients to make style encoder adversarial to piece classification.

#### Loss Function

```
L_total = L_regression + lambda_adv * L_adversarial + lambda_style * L_style_contrastive

L_regression: MSE on dimension predictions from z_style
L_adversarial: Cross-entropy for piece classification (with gradient reversal)
L_style_contrastive: Contrastive loss to spread out style embeddings
```

**Gradient Reversal Layer**:

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

Sources:

- [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/volume17/15-239/15-239.pdf)
- [PyTorch Gradient Reversal Implementation](https://github.com/tadeephuy/GradientReversal)
- [Domain Agnostic Learning with Disentangled Representations](http://proceedings.mlr.press/v97/peng19b/peng19b.pdf)

#### Disentanglement Evaluation

How to verify piece and style are actually disentangled:

1. **Piece classification from z_style**: Should be at chance level
2. **Style variance within piece**: z_style should vary across performers of same piece
3. **Piece clustering from z_piece**: Should cluster by piece, not performer

Sources:

- [Towards Principled Disentanglement for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Principled_Disentanglement_for_Domain_Generalization_CVPR_2022_paper.pdf)

---

## 4. Data Strategy

### 4.1 Available Datasets

| Dataset | Size | Multi-performer | Annotations |
|---------|------|-----------------|-------------|
| PercePiano | 1,202 segments | 6-13 per piece | 19 dimensions |
| ASAP | 1,067 performances | Yes (206 pieces) | Score alignment |
| MAESTRO | 200+ hours | Yes (linked to ASAP) | None (audio only) |
| PSyllabus | 508 pieces | No | Difficulty only |

### 4.2 Label Strategy for Ranking

**Challenge**: We don't have explicit "A is better than B" labels for same-piece comparisons.

**Proxy label options**:

1. **Competition rankings** (MAESTRO source)
   - MAESTRO data comes from International Piano-e-Competition
   - Performers have competition placements
   - Limitation: Only a few discrete ranks

2. **Synthesized rankings from PercePiano**
   - If piece P has performances with scores [0.7, 0.5, 0.8]
   - Derive pairwise rankings: 0.8 > 0.7 > 0.5
   - Limitation: Assumes within-piece score differences reflect performer quality

3. **Tempo/dynamic deviation as proxy**
   - More expressive = more deviation from mechanical
   - Limitation: Not always true (precision also matters)

4. **Expert annotation** (future)
   - Have teachers rank pairs of performances
   - Most reliable but expensive

Sources:

- [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) - competition context
- [ASAP Dataset](https://github.com/fosfrancesco/asap-dataset) - multi-performer structure

### 4.3 Handling Annotation Noise

PercePiano uses crowdsourced annotations with IRT aggregation, but noise remains.

**Strategies**:

1. **Confidence weighting**: Weight samples by annotator agreement
2. **Label smoothing**: Don't train on hard 0/1 rankings
3. **Margin in loss**: Only penalize if model is confident + wrong

Sources:

- [Learning from Crowdsourced Noisy Labels](https://arxiv.org/abs/2407.06902)
- [Label Correction of Crowdsourced Noisy Annotations](https://proceedings.neurips.cc/paper_files/paper/2023/file/015a8c69bedcb0a7d2ed2e1678f34399-Paper-Conference.pdf)

---

## 5. Evaluation Framework

### 5.1 Primary Metric: Pairwise Ranking Accuracy

For held-out pieces with multiple performances:

```
Accuracy_d = (# correct pairwise predictions on dim d) / (total pairs)
```

**Baseline**: Random = 50%

**Target**: Significantly above random, ideally >70%

### 5.2 Secondary Metrics

| Metric | Measures | How |
|--------|----------|-----|
| Kendall's Tau | Rank correlation | Compare predicted vs true ranking |
| NDCG | Ranking quality | Normalized discounted cumulative gain |
| Intra-piece std | Performer sensitivity | Variance of predictions within piece |
| Piece classification accuracy from style | Disentanglement | Should be low (~chance) |

### 5.3 Ablation Experiments

1. **Approach comparison**: A vs B vs C (and combinations)
2. **Loss function ablation**: Ranking vs contrastive vs combined
3. **Temperature sensitivity**: tau in [0.05, 0.1, 0.2, 0.5]
4. **Projection dimension**: [64, 128, 256, 512]
5. **Frozen vs fine-tuned MuQ**: Does fine-tuning help?

---

## 6. Implementation Plan

### Phase 1a: Data Preparation (Week 1-2)

- [ ] Build ASAP-MAESTRO linked dataset loader
- [ ] Implement pair sampling with piece-stratification
- [ ] Create proxy ranking labels from competition data
- [ ] Extract and cache MuQ embeddings for all performances

### Phase 1b: Baseline Reproduction (Week 2-3)

- [ ] Reproduce current model's multi-performer analysis
- [ ] Establish baseline pairwise ranking accuracy (using regression scores)
- [ ] Profile where current model fails on same-piece comparisons

### Phase 1c: Approach A Implementation (Week 3-5)

- [ ] Implement contrastive pairwise ranking architecture
- [ ] InfoNCE loss with piece-based positives
- [ ] Margin-based ranking loss
- [ ] Train and evaluate

### Phase 1d: Approach B Implementation (Week 5-7)

- [ ] Implement Siamese comparison architecture
- [ ] Dimension-specific ranking heads
- [ ] Pair sampling and training loop
- [ ] Train and evaluate

### Phase 1e: Approach C Implementation (Week 7-9)

- [ ] Implement dual-encoder architecture
- [ ] Gradient reversal layer for adversarial training
- [ ] Disentanglement evaluation metrics
- [ ] Train and evaluate

### Phase 1f: Analysis & Paper (Week 9-12)

- [ ] Compare all approaches
- [ ] Ablation studies
- [ ] Error analysis: where does each approach fail?
- [ ] Write paper: "Disentangling Piece and Performer in Piano Evaluation"

---

## 7. Open Research Questions

### Fundamental Questions

1. **Is piece-performer disentanglement even possible?**
   - Some dimensions may be inherently piece-dependent (e.g., "drama" depends on the music)
   - Need to identify which dimensions can be disentangled

2. **What temporal resolution is needed?**
   - Current: whole-segment predictions
   - Better: phrase-level or measure-level?
   - ASAP provides score alignment for this

3. **How to handle interpretation differences?**
   - A "different" interpretation isn't necessarily "worse"
   - Valid artistic variation vs technical error

### Technical Questions

1. **Optimal way to combine approaches?**
   - A + B: Contrastive pretraining, then Siamese fine-tuning?
   - B + C: Siamese with disentangled encoders?

2. **Self-supervised pretraining on ASAP?**
   - Pretrain on multi-performer data without labels
   - Then fine-tune on PercePiano

3. **Can we leverage score information?**
   - ASAP provides aligned scores
   - Could condition on score difficulty/structure

Sources:

- [Computational Models of Expressive Music Performance](https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00025/full)
- [Semi-supervised Contrastive Learning of Musical Representations](https://arxiv.org/html/2407.13840v1)

---

## 8. Key References

### Disentanglement & Style-Content Separation

1. [Disentangling Score Content and Performance Style](https://arxiv.org/html/2509.23878) - Joint rendering and transcription with disentangled latents
2. [Domain Agnostic Learning with Disentangled Representations (DADA)](http://proceedings.mlr.press/v97/peng19b/peng19b.pdf) - Adversarial disentanglement
3. [MusicVAE](https://magenta.tensorflow.org/music-vae) - Latent space disentanglement for music

### Contrastive Learning for Music

1. [CLEWS - Musical Version Matching](https://proceedings.mlr.press/v267/serra25a.html) - Contrastive learning for same-piece matching
2. [CLMR - Contrastive Learning of Musical Representations](https://arxiv.org/pdf/2103.09410) - SimCLR for music
3. [Semi-supervised Contrastive Learning for Music](https://arxiv.org/html/2407.13840v1) - Combining supervised and self-supervised

### Learning to Rank

1. [Pairwise Learning to Rank Revisited](https://link.springer.com/article/10.1007/s10994-024-06644-6) - Theory of neural ranking
2. [From Audio Encoders to Piano Judges](https://arxiv.org/html/2407.04518v1) - Ranking formulation for piano assessment

### Siamese Networks & Metric Learning

1. [Siamese Networks for Audio Similarity](https://towardsdatascience.com/calculating-audio-song-similarity-using-siamese-neural-networks-62730e8f3e3d/) - Song similarity with Siamese
2. [Triplet Loss Deep Dive](https://www.v7labs.com/blog/triplet-loss) - When to use triplet vs contrastive

### Domain Adaptation & Adversarial Training

1. [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/volume17/15-239/15-239.pdf) - Original DANN paper
2. [Gradient Reversal Layer PyTorch](https://github.com/tadeephuy/GradientReversal) - Implementation

### Expressive Performance Modeling

1. [Reconstructing Human Expressiveness with Transformers](https://arxiv.org/html/2306.06040) - Transformer for expression
2. [ScorePerformer (ISMIR 2023)](https://archives.ismir.net/ismir2023/paper/000069.pdf) - Expressive rendering
3. [Pianist Transformer](https://arxiv.org/pdf/2512.02652) - Latest in expressive piano

### Datasets

1. [ASAP Dataset](https://github.com/fosfrancesco/asap-dataset) - 1,067 performances, 236 scores
2. [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) - 200+ hours, competition data
3. [PercePiano](https://www.nature.com/articles/s41598-024-73810-0) - 19-dimension evaluation

### Multi-Task & Multi-Dimensional Learning

1. [M3BERT - Multi-faceted Music Transformer](https://www.nature.com/articles/s41598-023-36714-z) - Multi-task music learning
2. [Multi-dimensional Music Similarity](https://arxiv.org/abs/2111.01710) - Disentangled similarity dimensions

### Handling Noisy Labels

1. [Learning from Crowdsourced Noisy Labels](https://arxiv.org/abs/2407.06902) - Signal processing perspective
2. [Label Correction for Crowdsourced Annotations](https://proceedings.neurips.cc/paper_files/paper/2023/file/015a8c69bedcb0a7d2ed2e1678f34399-Paper-Conference.pdf) - Instance-dependent noise

---

## 9. Experimental Log

### Experiment Template

```
## Experiment: [NAME]
Date: YYYY-MM-DD
Approach: A / B / C
Hypothesis: [What we expect to learn]

### Configuration
- Model: [architecture details]
- Loss: [loss function]
- Data: [dataset and splits]
- Hyperparameters: [key settings]

### Results
| Metric | Value | Baseline |
|--------|-------|----------|
| Pairwise Accuracy (avg) | | 50% |
| Kendall's Tau | | |
| Intra-piece std | | 0.020 |

### Observations
[What we learned]

### Next Steps
[What to try based on results]
```

---

*This document will be updated as experiments progress.*
