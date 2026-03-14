# Encoder Status

Living document tracking audio and symbolic encoder architectures, training results, and next experiments.

> **Status (2026-03-14):** A1-Max DEPLOYED (80.8% ensemble pairwise, cloud HF endpoint). S2 COMPLETE (71.3% pairwise). Layer 1 validation COMPLETE. YouTube AMT validation COMPLETE (79.9% agreement). All inference cloud-only.

**Notebooks:** `model/notebooks/model_improvement/01_audio_training.ipynb`, `02_symbolic_training.ipynb`
**Code:** `model/src/model_improvement/`
**Taxonomy:** `docs/model/02-teacher-grounded-taxonomy.md`

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

**Loss:** `L = L_ranking + 1.5 * L_ListMLE + 0.3 * L_contrastive + 0.3 * L_CCC + 0.1 * L_invariance`

Key improvements over A1 baseline: ListMLE ranking loss (Plackett-Luce likelihood), CCC regression loss, embedding mixup, hard negative mining with curriculum, wider LoRA adaptation (layers 7-12 vs 9-12), label smoothing (0.1).

### Deployed Configuration

- **Endpoint:** HuggingFace inference endpoint (cloud-only)
- **Model:** A1-Max 4-fold ensemble (average predictions across all 4 fold models)
- **Input:** 15-second audio chunks at 24kHz mono
- **Output:** 6 dimension scores (0-1 range)
- **Calibration:** MAESTRO calibration stats in `model/data/maestro_cache/calibration_stats.json`

### Results

#### A1-Max 4-Fold Ensemble (Deployed)

| Metric | Value | vs A1 Baseline |
|--------|-------|---------------|
| **Pairwise accuracy** | **80.77%** | +6.84pp |
| **R2** | **0.5021** | +0.0989 |
| **Robustness (score drop)** | **0.08%** | Same |
| **Robustness (Pearson r)** | **1.0000** | Same |

#### A1-Max Top 5 Configs (from 18-config sweep)

| Config | LoRA Rank | Layers | Label Smooth | Pairwise | R2 |
|--------|-----------|--------|-------------|----------|-----|
| **r32_L7-12_ls0.1** | **32** | **7-12** | **0.1** | **0.7872** | **0.1553** |
| r8_L7-12_ls0.1 | 8 | 7-12 | 0.1 | 0.7866 | 0.1514 |
| r32_L7-12_ls0.05 | 32 | 7-12 | 0.05 | 0.7861 | 0.0974 |
| r8_L9-12_ls0.0 | 8 | 9-12 | 0.0 | 0.7859 | 0.1616 |
| r16_L9-12_ls0.1 | 16 | 9-12 | 0.1 | 0.7852 | 0.1393 |

#### All Audio Experiments (averaged across 4 folds)

| Model | Strategy | Pairwise Acc | R2 |
|-------|----------|-------------|-----|
| **A1-Max (ensemble)** | **LoRA rank-32 + ListMLE/CCC/mixup** | **80.8%** | **0.50** |
| A1 | MuQ + LoRA rank-16 | 73.9% | 0.40 |
| A2 | Staged domain adaptation | 71.4% | 0.42 |
| A3 | Full unfreeze, gradual | 69.9% | 0.28 |

**A2 MAESTRO ablation:** Adding 24K MAESTRO segments to Stage 1 contrastive pretraining showed no improvement. MuQ was pretrained on 160K hours -- more piano audio doesn't help.

#### Per-Fold Breakdown (A1)

| Fold | Pairwise Acc | R2 |
|------|-------------|-----|
| 0 | 0.7766 | 0.4349 |
| 1 | 0.7034 | 0.3903 |
| 2 | 0.7660 | 0.4500 |
| 3 | 0.7112 | 0.3375 |

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

### Audio Next Experiments

| Experiment | Effort | Expected Impact |
|-----------|--------|-----------------|
| Multi-head attention pooling (6 heads, one per dim) | Medium | +2-3% pairwise |
| Multi-scale temporal modeling (hierarchical pooling) | Medium | +2-4% pairwise |
| Competition data (T2) integration | Medium | +3-5% pairwise |
| Per-dimension loss weighting (by MuQ R2) | Low | +1-2% on strong dims |

---

## Symbolic Encoder

### Architecture: S2 (GNN on Score Graph)

Notes as nodes with features (pitch, velocity, onset, duration, pedal, voice). Edges encode temporal adjacency, harmonic intervals, and voice membership. GATConv message-passing layers. Pretrained on 14,821-sequence corpus (ASAP + MAESTRO + ATEPP + PercePiano) via link prediction, then finetuned on PercePiano.

```
MIDI -> Score graph (notes as nodes, structural edges)
  -> GATConv layers (message passing)
  -> Attention pooling -> z_symbolic [512]
  -> Per-dimension ranking heads (6 dims)
  -> Regression head (6 dims)
```

### Results

#### All Symbolic Experiments (averaged across 4 folds)

| Model | Strategy | Pairwise Acc | R2 |
|-------|----------|-------------|-----|
| **S2** | **GNN on homogeneous score graph** | **71.3%** | **0.32** |
| S2H | Heterogeneous GNN (4 edge types) | 70.2% | 0.36 |
| S3 | CNN + Transformer on continuous features | 70.0% | 0.37 |
| S1 | Transformer on REMI tokens | 68.4% | 0.33 |

#### Per-Fold Breakdown (S2)

| Fold | Pairwise Acc | R2 |
|------|-------------|-----|
| 0 | 0.7504 | 0.3575 |
| 1 | 0.6880 | 0.2467 |
| 2 | 0.6998 | 0.2877 |
| 3 | 0.7150 | 0.3693 |

### Symbolic Interpretation

**S2 (GNN) wins on ranking** because graph structure directly encodes musical relationships. Homogeneous graph generalizes better than S2H's richer 4-edge-type representation -- extra edge types overfit with limited data.

**S3 leads on R2 (0.3721) and robustness (0.9993 Pearson).** Architecture most analogous to MuQ (both CNN-based). Continuous features preserve fine-grained timing/velocity that tokenization (S1) or graph discretization (S2) may lose.

**Symbolic vs audio gap is narrow:** S2 (71.3%) trails A1 (73.9%) by only 2.6pp. The ISMIR paper showed this gap is due to pretraining scale, not modality choice.

### AMT Validation

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

**Conclusion:** AMT is production-viable. Symbolic path survives real-world audio conditions across all dimensions.

### Symbolic Next Steps: S2-Max

Pretrain on expanded 24,220 graph corpus (up from 14,821), then finetune on PercePiano + MAESTRO contrastive + Competition ordinal (3 data sources, up from 1).

Architecture exploration:
- Edge-type embedding in GATConv attention (S2H's information without its overfitting)
- Multi-scale graph pooling

---

## Cross-Modality Comparison

| Rank | Model | Modality | Pairwise | R2 |
|------|-------|----------|----------|-----|
| 1 | **A1-Max (ensemble)** | **Audio** | **80.8%** | **0.50** |
| 2 | A1-Max (single fold mean) | Audio | 78.7% | 0.16 |
| 3 | A1 (LoRA) | Audio | 73.9% | 0.40 |
| 4 | A2 (Staged) | Audio | 71.4% | 0.42 |
| 5 | S2 (GNN) | Symbolic | 71.3% | 0.32 |
| 6 | S2H (Hetero GNN) | Symbolic | 70.2% | 0.36 |
| 7 | S3 (CNN+Trans) | Symbolic | 70.0% | 0.37 |
| 8 | A3 (Full Unfreeze) | Audio | 69.9% | 0.28 |
| 9 | S1 (Transformer) | Symbolic | 68.4% | 0.33 |

### Per-Dimension Complementarity

| Dimension | A1 (Audio) | S2 (GNN) | Winner |
|---|---|---|---|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

---

## What This Means for the Product

### STOP Classifier / Teaching Moment Selection -- Workable

STOP classifier is a 6-weight logistic regression on dimension scores (AUC: 0.845). With A1-Max at 80.8% pairwise, the model reliably ranks chunks within a session. Teaching moment selection picks the top chunk -- ranking quality matters more than absolute accuracy. Both run in the cloud worker after HF scores return.

### Student Model / Blind Spot Detection -- Workable With Smoothing

Blind spot detection compares relative dimension deviations. Depends on ranking consistency more than absolute R2. Student model uses exponential moving averages (alpha=0.3) across sessions, smoothing per-chunk noise.

### LLM Teacher Prompt -- Sufficient

The LLM receives structured context like `"pedaling": 0.35, baseline: 0.62`. Whether true score is 0.35 or 0.42 matters less than "pedaling is significantly below baseline." Model provides relative signal.

### Progress Tracking -- Noisy but Usable

R2=0.50 means model explains half the variance. Individual sessions noisy, but averaging across sessions and trend detection makes this usable by sessions 5-10.

### MIDI as LLM Context -- Alternative to Model Fusion

Rather than fusing at the model level, structured MIDI comparison (velocity curves, onset deviations) can be fed to the teacher subagent alongside A1 scores. More robust to AMT noise, fully interpretable, and sidesteps the fusion problem by letting the LLM reason about complementary signals.

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

Pedaling and phrasing are strongest predictors. Dynamics is inverted -- captures "amount" not "appropriateness."

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

### Experiment 4: MIDI-as-Context -- SKIP

LLM judge: A wins 55%, B wins 45% (below 55% BORDERLINE threshold). Raw MIDI stats add "false precision." Bar-aligned passage-specific context may help (Wave 3), but raw statistics do not.

---

## Verification Criteria (for any future experiment)

1. 4-fold piece-stratified CV, same folds as `percepiano_cache/folds.json`
2. Pairwise accuracy (primary), R2 (secondary), robustness score_drop_pct (veto at >15%)
3. STOP AUC >= 0.80
4. Per-dimension breakdown reported
5. Bootstrap CI on pairwise accuracy difference vs A1 baseline
