# Encoder Status

Living document tracking audio and symbolic encoder architectures, training results, and next experiments.

> **Status (2026-03-19):** A1-Max DEPLOYED. Clean-fold optimized: **79.85% pairwise, R2=0.336** (4-fold mean). Loss weight autoresearch + best-checkpoint fix recovered nearly all leaked-fold performance (was 80.8% leaked, now 79.9% clean). Aria (EleutherAI) selected as primary symbolic encoder. Fusion now viable (phi=0.043).

**Notebooks:** `model/notebooks/model_improvement/01_audio_training.ipynb`, `02_symbolic_training.ipynb`
**Code:** `model/src/model_improvement/`
**Taxonomy:** `docs/model/02-teacher-grounded-taxonomy.md`

---

## IMPORTANT: Fold Leak Invalidation

**All pairwise accuracy and R2 numbers reported below from prior experiments are INVALID.** The original fold splits leaked pieces across train/validation boundaries. Clean piece-stratified folds have been regenerated and verified in `model/data/labels/percepiano/folds.json`. All models must be retrained and re-evaluated on clean folds before any number can be cited.

Numbers are retained below for relative comparison only (all experiments used the same leaked folds, so relative ordering may hold). Absolute values should not be reported externally.

---

## Audio Encoder

### Architecture: A1-Max (MuQ + LoRA)

Frozen MuQ backbone (pretrained on 160K hours of music) with LoRA rank-32 adapters on layers 7-12 (<1% of MuQ params). Multi-task training on PercePiano (1,202 segments, 6 teacher-grounded dimensions, 4-fold piece-stratified CV).

```
MuQ embeddings [93, 1024]
  -> LoRA adapters (rank-32, layers 7-12)
  -> Attention pooling -> [1024]
  -> Shared encoder (2-layer MLP + LayerNorm) -> z_audio [512]
  -> Per-dimension ranking heads (6 dims)
  -> Regression head (6 dims, sigmoid output)
```

**Loss (original):** `L = L_ranking + 1.5 * L_ListMLE + 0.3 * L_contrastive + 0.3 * L_CCC + 0.1 * L_invariance`

**Loss (optimized, 2026-03-19):** `L = L_ranking + 1.5 * L_ListMLE + 0.6 * L_contrastive + 0.8 * L_CCC + 0.1 * L_invariance`

Autoresearch (8 iterations) found that the original weights drastically underweighted regression -- producing a model that ranked well but gave meaningless absolute scores (R2 < 0). Doubling contrastive and nearly tripling regression weight recovers R2 to +0.24 without sacrificing pairwise accuracy. Key findings: contrastive provides crucial regularization even when near-converged; mixup is essential for PercePiano's small sample size (~900 training samples per fold).

Key improvements over A1 baseline: ListMLE ranking loss (Plackett-Luce likelihood), CCC regression loss, embedding mixup, hard negative mining with curriculum, wider LoRA adaptation (layers 7-12 vs 9-12), label smoothing (0.1).

### Deployed Configuration

- **Endpoint:** HuggingFace inference endpoint (cloud-only)
- **Model:** A1-Max 4-fold ensemble (average predictions across all 4 fold models)
- **Input:** 15-second audio chunks at 24kHz mono
- **Output:** 6 dimension scores (0-1 range)
- **Calibration:** MAESTRO calibration stats in `model/data/maestro_cache/calibration_stats.json`

### Results: Clean Piece-Stratified Folds (2026-03-19)

#### Ablation Sweep (original loss weights)

| Config | Pairwise (4-fold mean) | R2 (4-fold mean) |
|--------|----------------------|-----------------|
| **full_a1max_repro** | **77.50%** | **0.119** |
| bce_listmle_ccc | 75.43% | -1.087 |
| bce_ranking_only | 73.02% | -0.258 |
| bce_plus_listmle | 69.73% | -11.138 |
| frozen_probe | 53.42% | 0.461 |

The ~3.3pp drop from leaked 80.8% to clean 77.5% confirms the fold leak was real and inflated results. Ranking losses (ListMLE) are essential -- removing them drops pairwise by 4-8pp. Frozen probe (regression-only, no LoRA) gets the best absolute R2 (0.46) but terrible ranking (53.4%).

#### Optimized Loss Weights (4-fold validated)

Loss weight autoresearch (8 iterations, single-fold search, then 4-fold validation): found that contrastive=0.6, regression=0.8 (vs original 0.3, 0.3) dramatically improves both metrics.

| Config | Pairwise (4-fold mean) | R2 (4-fold mean) |
|--------|----------------------|-----------------|
| Original weights | 77.50% | 0.119 |
| **Optimized weights** | **79.85%** | **0.336** |

Per-fold optimized results: 76.7%, 78.9%, 81.2%, 82.5% pairwise. R2 per fold: 0.21, 0.24, 0.42, 0.48.

The +2.35pp pairwise gain comes from two fixes: (1) optimized loss weights (contrastive 2x, regression 2.7x), and (2) evaluating the best checkpoint instead of the last epoch. R2 nearly tripled, meaning the model now produces meaningful absolute scores. Full autoresearch log: `model/data/results/loss_weight_autoresearch.tsv`, `model/data/results/loss_weight_changelog.md`.

### Results (INVALID -- fold leak, retained for relative comparison only)

#### A1-Max 4-Fold Ensemble (Deployed)

| Metric | Value (INVALID) | vs A1 Baseline |
|--------|-----------------|---------------|
| **Pairwise accuracy** | **80.77%** | +6.84pp |
| **R2** | **0.5021** | +0.0989 |
| **Robustness (score drop)** | **0.08%** | Same |
| **Robustness (Pearson r)** | **1.0000** | Same |

#### A1-Max Top 5 Configs (from 18-config sweep)

| Config | LoRA Rank | Layers | Label Smooth | Pairwise (INVALID) | R2 (INVALID) |
|--------|-----------|--------|-------------|---------------------|---------------|
| **r32_L7-12_ls0.1** | **32** | **7-12** | **0.1** | **0.7872** | **0.1553** |
| r8_L7-12_ls0.1 | 8 | 7-12 | 0.1 | 0.7866 | 0.1514 |
| r32_L7-12_ls0.05 | 32 | 7-12 | 0.05 | 0.7861 | 0.0974 |
| r8_L9-12_ls0.0 | 8 | 9-12 | 0.0 | 0.7859 | 0.1616 |
| r16_L9-12_ls0.1 | 16 | 9-12 | 0.1 | 0.7852 | 0.1393 |

#### All Audio Experiments (averaged across 4 folds, INVALID)

| Model | Strategy | Pairwise Acc (INVALID) | R2 (INVALID) |
|-------|----------|------------------------|--------------|
| **A1-Max (ensemble)** | **LoRA rank-32 + ListMLE/CCC/mixup** | **80.8%** | **0.50** |
| A1 | MuQ + LoRA rank-16 | 73.9% | 0.40 |
| A2 | Staged domain adaptation | 71.4% | 0.42 |
| A3 | Full unfreeze, gradual | 69.9% | 0.28 |

**A2 MAESTRO ablation:** Adding 24K MAESTRO segments to Stage 1 contrastive pretraining showed no improvement. MuQ was pretrained on 160K hours -- more piano audio doesn't help.

### Audio Interpretation

**Why LoRA wins:** More aggressive adaptation (A2, A3) doesn't improve and A3 actively hurts (fold 0 R2=0.059 = catastrophic forgetting). MuQ's pretrained representations are already well-suited. With ~750 training samples per fold, there isn't enough data to reshape the backbone.

**ListMLE is the biggest A1-Max contributor.** Ranking-dominant loss (lambda=1.5) explicitly optimizes Plackett-Luce ranking likelihood, aligning loss directly with pairwise accuracy.

**R2 trade-off in A1-Max:** Individual fold R2 (~0.15) drops below A1 (~0.40) because ranking-dominant loss de-emphasizes regression. But 4-fold ensemble R2 (0.50) recovers -- regression heads learn complementary patterns across folds.

**Fold variance:** A1 pairwise ranges 70.3-77.7% (~7pp spread). Driven by which of 61 multi-performance pieces land in validation. Data quantity constraint, not model problem.

### Per-Dimension MuQ Predictability

| Dimension | MuQ Probing R2 | Teacher Frequency |
|-----------|---------------|-------------------|
| articulation | 0.607 | 11.4% |
| dynamics | 0.587 | 14.1% |
| phrasing | 0.569 | 13.1% |
| interpretation | 0.524 | 36.7% |
| pedaling | 0.513 | 6.8% |
| timing | 0.332 | 18.0% |

Timing is hardest for audio (R2=0.332) -- strongest candidate for symbolic support. Articulation is strongest (0.607) -- note attack/release directly audible.

### Aria vs MuQ: Frozen Linear Probe Comparison (2026-03-19)

Linear probe on frozen embeddings, 4-fold piece-stratified CV (clean folds). These are the first VALID numbers on clean folds.

| Dimension | Aria-Embedding (512d) | Aria-Base (1536d) | MuQ mean-pooled (1024d) |
|-----------|----------------------|-------------------|------------------------|
| dynamics | 65.8% | 62.5% | **72.4%** |
| timing | 55.8% | 58.0% | **67.5%** |
| pedaling | 58.6% | 60.7% | **66.6%** |
| articulation | 54.2% | 54.3% | **54.7%** |
| phrasing | 57.9% | 54.8% | **60.9%** |
| interpretation | 61.2% | 62.3% | **63.9%** |
| **Overall** | **59.6%** | **59.6%** | **62.2%** |

Error correlation (phi): **0.043** -- near-zero. Models make completely independent mistakes, making fusion highly viable.

MuQ dominates all dimensions from frozen embeddings. This is expected: MuQ was pretrained on 160K hours of audio for music understanding tasks, while Aria was pretrained on MIDI for generation/identity tasks (not quality). The key finding is that Aria has quality signal (significantly above 50% chance) despite never being trained for quality, and its errors are independent from MuQ's.

### MuQ Continued Pretraining Plan: Quality-Aware Contrastive

Before fine-tuning on PercePiano, apply symmetric contrastive pretraining to MuQ so that its embeddings become quality-aware (not just content-aware). This mirrors what Aria's SimCSE contrastive stage does for symbolic.

**Approach:**
- NT-Xent contrastive loss on PercePiano pairs with known quality ordering
- Positive pairs: same piece, different performer (quality-varying)
- Negative pairs: different pieces
- Curriculum: easy negatives first (different composers), then hard (same composer, different piece)
- Training: 20-30 epochs on top of frozen MuQ, adapting only LoRA layers + pooling head
- Goal: reduce error correlation with Aria by making audio embeddings explicitly quality-sensitive before fusion

This is symmetric with Aria's contrastive pretraining -- both encoders get quality-aware contrastive training before fine-tuning.

### Audio Next Experiments

| Experiment | Effort | Expected Impact |
|-----------|--------|-----------------|
| Retrain A1-Max on clean folds | High | Establish valid baselines |
| Quality-aware contrastive pretraining | Medium | Better fusion compatibility |
| Multi-head attention pooling (6 heads, one per dim) | Medium | +2-3% pairwise |
| Multi-scale temporal modeling (hierarchical pooling) | Medium | +2-4% pairwise |
| Competition data (T2) integration | Medium | +3-5% pairwise |
| Per-dimension loss weighting (by MuQ R2) | Low | +1-2% on strong dims |

---

## Symbolic Encoder: Aria (PRIMARY)

### Why Aria

Aria (EleutherAI, 2025) is a 650M-parameter LLaMA-architecture model pretrained on 820K piano MIDI performances (~60K hours). It achieves SOTA on 6 MIR benchmarks. Apache 2.0 license. This replaces ALL custom symbolic encoders (S2 GNN, S2H, S3, S1) and eliminates the need to build a custom symbolic foundation model (previously Phase 3, 6-12 month research effort with HIGH risk).

Aria IS the symbolic foundation model. The pretraining asymmetry that caused fusion failure (MuQ pretrained on 160K hours vs S2 trained from scratch on 24K graphs) no longer exists -- Aria's 820K MIDI pretraining matches MuQ's representation scale.

**Weights:**
- Base (autoregressive): `loubb/aria-medium-base` on HuggingFace
- Embedding (contrastive): `loubb/aria-medium-embedding` on HuggingFace

### Architecture

- **Base architecture:** LLaMA 3.2
- **Parameters:** 650M
- **Layers:** 16
- **Hidden dimension:** 1536
- **Attention heads:** 24
- **Max sequence length:** 8192 tokens (base model), 2048 tokens (embedding model, ~680 notes)
- **Embedding output:** 512-dim from EOS token position

### Tokenization: AbsTokenizer

Aria uses AbsTokenizer, encoding MIDI into 3 tokens per note:

1. **instrument + pitch + velocity** (combined token)
2. **onset_ms** (absolute onset time in milliseconds)
3. **duration_ms** (note duration in milliseconds)

Temporal structure: 5000ms segments marked by `<T>` tokens. 10ms temporal resolution. This absolute tokenization preserves fine-grained timing information that relative tokenizations (like REMI) lose.

### Pretraining

- **Data:** 820K piano MIDI performances (~60K hours)
- **Objective:** Autoregressive next-token prediction
- **Training:** 75 epochs, 9 days on 8xH100
- **Contrastive stage:** SimCSE with NT-Xent loss, tau=0.1, 25 epochs on top of base model

### Benchmark Results (all SOTA)

| Benchmark | Task | Accuracy |
|-----------|------|----------|
| Genre classification | Genre | 92.4% |
| Form analysis | Form | 82.5% |
| Composer identification | Composer | 90.5% |
| Pianist8 | Performer ID | 91.6% |
| Period classification | Period | 84.7% |
| VG-MIDI | Emotion | 63.6% |

These benchmarks demonstrate that Aria has learned rich representations of musical style, structure, and expression -- exactly the capabilities needed for quality assessment.

### Fine-Tuning Strategy for CrescendAI

**Direct fine-tuning on PercePiano:**
- Learning rate: 1e-5
- Dropout: linearly increasing 0.0 to 0.2 across layers (layer 0 = no dropout, layer 15 = 0.2)
- Training: 10 epochs
- Loss: same ranking-dominant loss as A1-Max (ListMLE + contrastive + CCC)

**LoRA option:** Standard LLaMA architecture means full HuggingFace PEFT compatibility. LoRA fine-tuning is viable if full fine-tuning overfits on PercePiano's limited data (~1,202 segments). Start with LoRA rank-32 as baseline, compare with full fine-tuning.

### Score Conditioning (from Day One)

Aria encodes BOTH performance MIDI and score MIDI using the same model:

```
Performance MIDI -> [Aria] -> z_perf [512]
Score MIDI      -> [Aria] -> z_score [512]

delta = z_perf - z_score   (what's different between played and written)
```

The delta vector directly encodes performance deviations from the score. The quality head learns which deltas are good (rubato, dynamic shading) and which are bad (missed dynamics, wrong notes).

This is immediate -- not deferred to a future phase. Aria's architecture naturally handles both score and performance MIDI. No special architecture changes needed.

### AMT Validation (prior results, methodology still valid)

The symbolic encoder requires MIDI input, which in production comes from automatic music transcription (AMT) of audio.

**MAESTRO AMT test (studio audio):** ByteDance piano transcription vs ground-truth MIDI on 50 recordings, 107 pairs. **0% pairwise accuracy drop.** Per-dimension drops all < 4%.

**YouTube AMT test (mediocre audio):** 50 recordings from phone/home setups, 1,225 pairs. **79.9% A1-vs-S2 agreement** (gate: >60%).

| Dimension | Agreement |
|-----------|-----------|
| interpretation | 84.7% |
| dynamics | 82.9% |
| phrasing | 82.2% |
| pedaling | 78.5% |
| timing | 76.7% |
| articulation | 72.4% |

Articulation weakest (AMT velocity estimation noisiest). Interpretation/dynamics strongest (depend on overall structure, not note-level precision).

**Conclusion:** AMT is production-viable. Symbolic path survives real-world audio conditions across all dimensions. These results apply to any symbolic encoder including Aria -- the AMT quality bottleneck is upstream of the encoder choice.

---

## Gated Fusion Architecture

### Why Fusion Is Now Viable

The ISMIR paper tested audio-symbolic fusion and found it *underperformed* audio-only (R2 0.524 vs 0.537). Error correlation between modalities was r=0.738 -- both failed on the same samples.

**Root cause: pretraining scale asymmetry.** MuQ was pretrained on 160K hours; S2 was trained from scratch on ~24K graphs. The symbolic encoder was too weak to contribute novel signal.

**Aria changes this.** With 650M parameters pretrained on 820K MIDI performances (60K hours), Aria matches MuQ's representation scale and quality. The pretraining gap that killed fusion no longer exists. Error correlation should drop significantly because:

1. Aria's representations are genuinely different from MuQ's (MIDI tokens vs audio spectrograms)
2. Both encoders now have comparable pretraining depth
3. Score conditioning gives Aria information MuQ cannot access (what was the composer's intention)

### Architecture: Separate-Then-Fuse

Train MuQ and Aria independently first, measure error correlation, then fuse with per-dimension learned gates.

```
AUDIO -> [MuQ + LoRA] -> z_audio [512]

PERF MIDI  -> [Aria] -> z_perf [512]
SCORE MIDI -> [Aria] -> z_score [512]
                         delta = z_perf - z_score

                    GATED FUSION (per-dimension)

  For each dimension d:
    gate_d = sigmoid(W_d * [z_audio; z_perf; delta])   -- learned gate [0,1]
    fused_d = gate_d * z_audio + (1 - gate_d) * z_perf
    quality_d = MLP_d(fused_d, delta)

  Output: 6 scores (0-1) relative to score
```

### Per-Dimension Complementarity Expectations

Based on MuQ probing R2 and prior cross-modality analysis:

| Dimension | Expected Routing | Rationale |
|-----------|-----------------|-----------|
| timing | Audio-dominant (~0.7 audio) | MuQ captures micro-timing, rubato feel directly from audio waveform |
| dynamics | Symbolic-dominant (~0.7 symbolic) | Score delta (what was written vs played) resolves dynamics inversion |
| pedaling | Balanced (~0.5/0.5) | Audio hears resonance/blur; symbolic sees pedal CC64 events + harmonic context |
| articulation | Slight symbolic (~0.6 symbolic) | Symbolic captures note duration ratios precisely; audio captures attack quality |
| phrasing | Balanced (~0.5/0.5) | Phrase structure from symbolic; breath/shaping from audio |
| interpretation | Cross-attention over all 3 | Holistic dimension requiring audio feel + symbolic structure + score intent |

These are hypotheses. The learned gates will discover the actual optimal routing from data.

### Training Protocol

1. **Phase A -- Frozen linear probe (COMPLETE 2026-03-19):** Validated Aria captures quality signal (59.6% pairwise from frozen embeddings). Error correlation phi=0.043 (near-zero) confirms fusion viability. See "Aria vs MuQ: Frozen Linear Probe Comparison" above.
2. **Phase B -- Contrastive pretraining:** Quality-aware contrastive training for both MuQ and Aria on T2 competition + T5 YouTube Skill data. Teaches quality ordering before fine-tuning.
3. **Phase C -- Independent fine-tuning:** LoRA fine-tune MuQ and Aria separately on all tiers. Establish independent baselines on clean folds.
4. **Phase D -- Gated fusion training:** Freeze both encoders, train only fusion gates + quality MLPs. PercePiano as anchor (20% of training), ordinal competition data (80%).
5. **Phase E -- End-to-end fine-tuning (optional):** Unfreeze top layers of both encoders with very low LR (1e-6) for joint optimization.

### Training Data Mix

- **PercePiano (20%):** Anchor dataset with expert annotations. 6-dimensional continuous labels.
- **Ordinal competition data (80%):** Competition placements provide ranking signal across pieces and performers. Much larger scale. ListMLE ranking loss.

This 20/80 split ensures the model learns robust ranking from abundant ordinal data while anchoring to expert-grounded dimensions from PercePiano.

---

## Cross-Modality Comparison (ALL NUMBERS INVALID -- fold leak)

Retained for relative comparison only. All models used the same leaked folds.

| Rank | Model | Modality | Pairwise (INVALID) | R2 (INVALID) |
|------|-------|----------|---------------------|--------------|
| 1 | **A1-Max (ensemble)** | **Audio** | **80.8%** | **0.50** |
| 2 | A1-Max (single fold mean) | Audio | 78.7% | 0.16 |
| 3 | A1 (LoRA) | Audio | 73.9% | 0.40 |
| 4 | A2 (Staged) | Audio | 71.4% | 0.42 |
| 5 | S2 (GNN) | Symbolic | 71.3% | 0.32 |
| 6 | S2H (Hetero GNN) | Symbolic | 70.2% | 0.36 |
| 7 | S3 (CNN+Trans) | Symbolic | 70.0% | 0.37 |
| 8 | A3 (Full Unfreeze) | Audio | 69.9% | 0.28 |
| 9 | S1 (Transformer) | Symbolic | 68.4% | 0.33 |

---

## LEGACY: Custom Symbolic Encoders (Superseded by Aria)

Retained for reference and ISMIR paper context. These architectures are no longer in active development.

### S2 (GNN on Score Graph)

Notes as nodes with features (pitch, velocity, onset, duration, pedal, voice). Edges encode temporal adjacency, harmonic intervals, and voice membership. GATConv message-passing layers. Pretrained on 14,821-sequence corpus (ASAP + MAESTRO + ATEPP + PercePiano) via link prediction, then finetuned on PercePiano.

```
MIDI -> Score graph (notes as nodes, structural edges)
  -> GATConv layers (message passing)
  -> Attention pooling -> z_symbolic [512]
  -> Per-dimension ranking heads (6 dims)
  -> Regression head (6 dims)
```

#### Results (INVALID -- fold leak)

| Model | Strategy | Pairwise Acc (INVALID) | R2 (INVALID) |
|-------|----------|------------------------|--------------|
| **S2** | **GNN on homogeneous score graph** | **71.3%** | **0.32** |
| S2H | Heterogeneous GNN (4 edge types) | 70.2% | 0.36 |
| S3 | CNN + Transformer on continuous features | 70.0% | 0.37 |
| S1 | Transformer on REMI tokens | 68.4% | 0.33 |

### Why These Were Replaced

The fundamental problem was pretraining scale. All custom symbolic encoders were trained from scratch or with limited pretraining (~24K sequences). This created an asymmetry with MuQ (160K hours of pretraining) that made fusion impossible -- both modalities failed on the same samples (error correlation r=0.738). Aria solves this with 820K MIDI pretraining on a proven LLaMA architecture, achieving SOTA across all benchmarks.

---

## What This Means for the Product

### STOP Classifier / Teaching Moment Selection -- Workable

STOP classifier is a 6-weight logistic regression on dimension scores (AUC: 0.845). With A1-Max ranking chunks within a session, teaching moment selection picks the top chunk -- ranking quality matters more than absolute accuracy. Both run in the cloud worker after HF scores return.

### Student Model / Blind Spot Detection -- Workable With Smoothing

Blind spot detection compares relative dimension deviations. Depends on ranking consistency more than absolute R2. Student model uses exponential moving averages (alpha=0.3) across sessions, smoothing per-chunk noise.

### LLM Teacher Prompt -- Sufficient

The LLM receives structured context like `"pedaling": 0.35, baseline: 0.62`. Whether true score is 0.35 or 0.42 matters less than "pedaling is significantly below baseline." Model provides relative signal.

### Progress Tracking -- Noisy but Usable

R2~0.50 means model explains roughly half the variance. Individual sessions noisy, but averaging across sessions and trend detection makes this usable by sessions 5-10.

### Score Conditioning -- Immediate with Aria

Aria encodes both performance MIDI and score MIDI natively. Score conditioning is available from day one of Aria integration, not deferred to a future phase. This fixes the dynamics inversion (rho=-0.917) because quality becomes relative: pp when score says pp = HIGH quality, pp when score says ff = LOW quality.

Training data: reference-anchored on MAESTRO (ranking signal from multiple performers of the same piece, no new annotation needed).

---

## Layer 1 Validation Results (2026-03-11)

Code: `model/src/model_improvement/layer1_validation.py`, `midi_comparison.py`, `feedback_assessment.py`
Notebook: `model/notebooks/model_improvement/04_layer1_validation.ipynb`

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

Pedaling and phrasing are strongest predictors. Dynamics is inverted -- captures "amount" not "appropriateness." Score conditioning via Aria delta will fix this.

### Experiment 2: AMT Degradation -- PASS

ByteDance piano transcription vs ground-truth MIDI on 50 MAESTRO recordings (107 pairs): **0.0% pairwise accuracy drop.** All per-dimension drops < 4%.

YouTube follow-up (50 mediocre recordings, 1,225 pairs): **79.9% A1-vs-S2 agreement.** All dimensions > 72%.

### Experiment 3: Dynamic Range -- DIAGNOSTIC

| Comparison | Cohen's d |
|-----------|-----------|
| Intermediate vs Professional | 0.47 |
| Advanced vs Professional | 0.47 |
| Advanced vs Intermediate | 0.15 |

Separates skill levels at group level. Usable for within-student tracking, not absolute classification.

### Experiment 4: MIDI-as-Context -- SKIP (raw stats), REVISIT (bar-aligned facts)

LLM judge: A wins 55%, B wins 45% (below 55% BORDERLINE threshold). Raw MIDI stats add "false precision." But bar-aligned passage-specific context is a fundamentally different input. "Velocity MAE = 15" is noise. "Crescendo in bars 12-16 only reaches mf, score asks for ff" is teacher-language.

Phase 1 of the pipeline roadmap (see `04-north-star.md`) builds the correct version: a bar-aligned musical analysis engine that produces structured facts per passage, combining AMT output with score comparison and reference performance statistics. This is the highest-leverage improvement in the entire roadmap.

---

## Verification Criteria (for any future experiment)

1. 4-fold piece-stratified CV, same folds as `model/data/labels/percepiano/folds.json` (CLEAN folds, post-leak fix)
2. Pairwise accuracy (primary), R2 (secondary), robustness score_drop_pct (veto at >15%)
3. STOP AUC >= 0.80
4. Per-dimension breakdown reported
5. Bootstrap CI on pairwise accuracy difference vs A1 baseline
6. Error correlation between audio and symbolic encoders (target: r < 0.5 for fusion viability)
