# Model Improvement v2 Design: Teacher-Grounded Dual-Encoder Experiments

## Goal

Train the strongest possible piano performance evaluation model using:

1. Teacher-grounded feedback dimensions (from taxonomy work) instead of raw PercePiano 19-dim labels
2. Four data tiers (labeled, competition, contrastive, invariance) instead of PercePiano alone
3. Staged elimination to maximize experiment coverage while minimizing wasted GPU time
4. Three consolidated notebooks: audio, symbolic, fusion

This plan starts after both prerequisites are complete:
- Teacher-grounded taxonomy validated (all 5 gates pass)
- Repo cleanup and data collection done (T2-T4 available)

## Prerequisites: What This Plan Receives

From the taxonomy work:
- N teacher-grounded dimensions (expected 5-8) with 2-level hierarchy
- `composite_labels/taxonomy.json`: dimension definitions, PercePiano mapping weights
- `composite_labels/labels.json`: composite labels for all 1,202 PercePiano segments
- `composite_labels/quote_bank.json`: teacher quotes per dimension
- STOP prediction AUC baseline >= 0.80

From data collection:
- T1: `percepiano_cache/` -- MuQ embeddings + composite labels
- T2: `competition_cache/chopin2021/` -- MuQ embeddings + ordinal placements
- T3: `maestro_cache/muq_embeddings/` -- cross-performer MuQ embeddings
- T4: `youtube_piano_cache/` (optional) -- clean + augmented embedding pairs

From existing infrastructure:
- `pretrain_cache/` -- tokenized MIDI, score graphs, continuous features (14K+ corpus)
- All model code in `src/model_improvement/` (audio_encoders, symbolic_encoders, data, etc.)

## Current Baseline (from PercePiano audit)

| Metric | Value | Model |
|--------|-------|-------|
| R2 (19-dim) | 0.537 | Frozen MuQ regression |
| Pairwise accuracy (19-dim) | 84% | Contrastive ranker (E2a) |
| Symbolic-only R2 | 0.347 | Hand-crafted MIDI features |
| Audio-symbolic fusion R2 | 0.524 | Concatenation (worse than audio alone) |
| STOP AUC (all 19 dims) | 0.814 | Leave-one-video-out CV |
| STOP AUC (raw MuQ) | 0.936 | Attention-pooled embeddings |

## Architecture Overview

### Training Objective

Multi-task loss across four data tiers:

```
L_total = L_regression + lambda_rank * L_ranking + lambda_contrastive * L_contrastive + lambda_invariance * L_invariance

L_regression     = MSE on composite labels (T1, N dims)
L_ranking        = DimensionWiseRankingLoss on same-piece pairs (T1) + ordinal placement pairs (T2)
L_contrastive    = Piece-based InfoNCE on cross-performer pairs (T3)
L_invariance     = MSE between clean and augmented embeddings (T4)
```

### Audio Encoder Experiments

All operate on pre-extracted MuQ embeddings (1024-dim frame-level features).

**A1: MuQ + LoRA Multi-Task**
- LoRA adapters (rank 16-64) on self-attention layers of MuQ layers 9-12
- 99%+ of MuQ parameters stay frozen
- Multi-task training on all available labels (T1+T2+T3)
- Cheapest, fastest iteration
- Good baseline for what MuQ can do with minimal adaptation

**A2: Staged Domain Adaptation**
- Stage 1 (self-supervised): contrastive + invariance on T3+T4. No labels needed.
  - Cross-performer contrastive: same piece, different performers -> positive pairs
  - Augmentation invariance: same recording + {noise, room IR, phone sim} -> same embedding
  - LoRA adapters on MuQ
- Stage 2 (supervised): multi-task on T1+T2+T3. Aligns representations to teacher dimensions.
- Most principled approach -- separates representation learning from label fitting.

**A3: Full Unfreeze with Gradual Unfreezing**
- Unfreeze MuQ layers progressively: 12 -> 11 -> 10 -> 9
- Discriminative learning rates (deeper = smaller)
- Highest parameter count, highest risk of catastrophic forgetting, highest ceiling

**Shared pipeline:**
```
MuQ embeddings [T, 1024]
  -> Attention pooling -> [1024]
  -> Shared encoder (2-layer MLP + LayerNorm) -> z_audio [512]
  -> Per-dimension ranking heads (N dims)
  -> Regression head (N dims, sigmoid output)
```

### Symbolic Encoder Experiments

All pretrain on the 14K+ MIDI corpus (MAESTRO + ATEPP + ASAP + PercePiano), then finetune on PercePiano composite labels.

**S1: Transformer on REMI Tokens**
- REMI tokenization: note-on, note-off, velocity (32 bins), time-shift, pedal (on/off/partial), bar, tempo
- Vocabulary: ~500 tokens
- Architecture: 6-12 layer Transformer, 512-dim, 8 heads (~25M parameters)
- Pretraining: masked token prediction (15% masking, BERT-style) + cross-performer contrastive
- Most standard approach, strong baseline

**S2: GNN on Homogeneous Score Graph**
- Notes as nodes (pitch, velocity, onset, duration, pedal, voice)
- Edges: temporal adjacency, harmonic interval, voice membership
- Message-passing encoder (GATConv layers)
- Pretraining: link prediction + node attribute prediction (masked velocity/timing)
- Structurally expressive for counterpoint and harmony

**S2H: Heterogeneous GNN on Score Graph**
- 4 edge types: onset (simultaneous), during (overlapping), follow (sequential), silence (gap)
- Richer structural representation than S2
- Pretraining: per-edge-type link prediction

**S3: Continuous MIDI Encoder**
- MIDI -> continuous feature curves (pitch, velocity, density, pedal, IOI)
- 1D-CNN (multi-scale: 3, 7, 15) + Transformer encoder
- wav2vec-style contrastive pretraining
- Most analogous to MuQ's architecture

**Shared pipeline:**
```
MIDI input
  -> Tokenizer/encoder (per experiment)
  -> Token/frame embeddings [T, 512]
  -> Attention pooling -> z_symbolic [512]
  -> Per-dimension ranking heads (N dims)
  -> Regression head (N dims)
```

### Fusion Experiments

Input: best audio encoder + best symbolic encoder (from staged elimination).

**F1: Cross-Attention Fusion**
- z_audio attends to z_symbolic and vice versa
- Breaks the correlated-error pattern that killed concatenation fusion (r=0.738 in audit)

**F2: Concatenation Baseline**
- [z_audio; z_symbolic] -> MLP
- Must beat audio-only or fusion isn't working (current: 0.524 < 0.537)

**F3: Gated Per-Dimension Fusion**
- Learned gate per dimension: some dimensions benefit more from audio, others from symbolic
- gate_d = sigmoid(W_d * [z_audio; z_symbolic])
- z_fused_d = gate_d * z_audio + (1 - gate_d) * z_symbolic
- Interpretable: the gate values tell us which modality each dimension relies on

**Score-conditioned quality (the key unlock):**
```
quality = f(z_performance_audio, z_performance_midi, z_score_midi)
```
The model knows what the score asks for AND how the performance sounds. This enables feedback like "the dynamics are correct for this passage" vs "the dynamics don't match what Chopin wrote."

**Downstream heads (shared across fusion experiments):**
- Quality heads: N-dimension regression (composite labels)
- Ranking heads: N-dimension pairwise ranking (E2a-style)
- Difficulty head: auxiliary regression (PSyllabus, 508 pieces, current Spearman rho 0.623)

**Fusion training:**
- Freeze both encoders initially
- Train fusion module + downstream heads on T1+T2+T3
- If frozen fusion plateaus, optionally unfreeze encoders with very low LR (1e-6)

## Audio Augmentation Suite

Applied on-the-fly during training via AudioAugmentor (already implemented in `src/model_improvement/augmentation.py`):

| Augmentation | Source | Probability | Purpose |
|---|---|---|---|
| Room impulse response | MIT IR Survey, EchoThief (~500 IRs) | 0.3 | Reverb/acoustics robustness |
| Additive noise | ESC-50 (2,000 clips), SNR 10-30dB | 0.3 | Background noise |
| Phone mic simulation | Low-pass 8kHz, compression, distortion | 0.2 | Phone recording bridge |
| Pitch shift | +/- 50 cents | 0.1 | Tuning variation |
| EQ variation | Random 3-band parametric EQ | 0.2 | Timbral variation across pianos |

## Staged Elimination Protocol

### Round 1: Quick Screening

**Purpose:** Identify clear losers before committing full GPU time.

**Config:**
- 50 pretrain epochs (symbolic only)
- 50 finetune epochs (all)
- Fold 0 only (no cross-validation)
- T1 + T2 data (no T3/T4 -- faster iteration)
- Batch size 16

**Evaluate:** Pairwise accuracy (primary) + R2 (secondary) on fold 0 validation set.

**Elimination rule:** Within each track (audio or symbolic), drop any experiment with pairwise accuracy > 10 percentage points below the best. If all are within 10%, keep all.

**Expected cost:** ~2-4 GPU-hours total for all 7 experiments.

### Round 2: Full Training

**Purpose:** Properly train surviving experiments with full data and cross-validation.

**Config:**
- Full pretrain schedule (symbolic: 50 epochs on 14K+ corpus)
- 200 finetune epochs with early stopping (patience 20)
- 4-fold piece-stratified CV (no piece leakage)
- All available data tiers (T1+T2+T3, add T4 if collected)
- Batch size 16

**Evaluate:** Full MetricsSuite on each fold's validation set. Average across folds.

**Select winners:** Best audio encoder + best symbolic encoder by pairwise accuracy. Tiebreak: R2. Veto: robustness drop > 15%.

**Expected cost:** ~8-20 GPU-hours total depending on survivors.

### Round 3: Fusion

**Purpose:** Combine best encoders and evaluate end-to-end.

**Config:** Same as Round 2 but training only the fusion module (encoders frozen initially).

**Evaluate:** Full MetricsSuite + STOP prediction + competition ranking correlation.

## Notebook Structure

### 01_audio.ipynb

```
1. Setup (Thunder Compute, rclone sync)
2. Load composite labels + T1-T4 data
3. Round 1: Quick screen A1/A2/A3 (fold 0, 50 epochs)
4. Elimination: compare pairwise accuracy, drop losers
5. Round 2: Full training on survivors (4-fold CV, 200 epochs)
6. Audio comparison table + per-dimension breakdown
7. Select best audio encoder
8. Upload checkpoints
```

### 02_symbolic.ipynb

```
1. Setup (Thunder Compute, rclone sync)
2. Load composite labels + pretrain cache
3. Pretrain all symbolic encoders on 14K+ corpus
4. Round 1: Quick screen S1/S2/S2H/S3 (fold 0, 50 finetune epochs)
5. Elimination: compare pairwise accuracy, drop losers
6. Round 2: Full training on survivors (4-fold CV, 200 epochs)
7. Symbolic comparison table + per-dimension breakdown
8. Select best symbolic encoder
9. Upload checkpoints
```

### 03_fusion.ipynb

```
1. Setup (Thunder Compute, rclone sync)
2. Load best audio + symbolic encoder checkpoints
3. Extract embeddings on PercePiano
4. Train F1/F2/F3 (4-fold CV)
5. Score-conditioned quality experiment
6. Fusion comparison table
7. Robustness validation (augmented test set)
8. STOP prediction with fused model
9. Competition ranking correlation
10. Final model selection
11. Upload final checkpoints
```

## Code Changes Required

### Label System

Update `src/model_improvement/` modules to load composite labels:

```python
# Before
labels = json.load(open('percepiano_cache/labels.json'))  # 19-dim
num_labels = 19

# After
taxonomy = json.load(open('composite_labels/taxonomy.json'))
labels = json.load(open('composite_labels/labels.json'))  # N-dim composite
num_labels = len(taxonomy['dimensions'])
```

All model configs change `num_labels=19` to `num_labels=N`.

### Data Pipeline

Add T2/T3/T4 to the training data loaders:

- `CompetitionDataset`: already scaffolded in `data.py`, needs to actually load competition embeddings and generate ordinal pairs
- `CrossPerformerDataset`: new class for MAESTRO same-piece contrastive pairs
- `AugmentedEmbeddingDataset`: already exists, needs to use T4 clean+augmented pairs instead of just augmenting T1 on-the-fly

### Multi-Task Collation

Update `multi_task_collate_fn` to handle mixed batches from different data tiers. Each sample has a `tier` field indicating which loss terms apply.

### Metrics

Update `MetricsSuite` to work with N dimensions instead of hardcoded 19. Add competition ranking correlation metric.

## Evaluation Strategy

### Core Metrics (all experiments)

| Metric | Dataset | Current Best |
|--------|---------|-------------|
| R2 (N-dim) | PercePiano composite, 4-fold CV | 0.537 (19-dim MuQ regression) |
| Pairwise accuracy (N-dim) | PercePiano same-piece pairs, 4-fold CV | 84% (19-dim E2a contrastive) |
| Per-dimension breakdown | PercePiano composite | varies by dimension |
| Piece-split vs performer-split R2 | PercePiano | 0.536 both (well-disentangled) |
| Cross-soundfont R2 | PercePiano leave-one-out | 0.534 |
| Difficulty correlation (Spearman rho) | PSyllabus (508 pieces) | 0.623 |
| STOP prediction AUC | Masterclass LOVO CV | 0.814 (19-dim), 0.936 (raw MuQ) |

### Robustness Metrics (audio experiments)

| Metric | Target |
|--------|--------|
| Augmented pairwise accuracy | Drop < 10% vs clean |
| Cross-condition consistency (Pearson r, clean vs augmented) | > 0.9 |
| Real phone recording pilot | Qualitative sanity check |

### Symbolic-Specific Metrics

| Metric | Dataset | Current Baseline |
|--------|---------|-----------------|
| Score alignment accuracy (measure-level) | ASAP | ~30% within 30ms |
| Alignment onset error | ASAP | ~18s (MuQ failed), target < 1s |
| Symbolic-only R2 | PercePiano MIDI path | 0.347 (hand-crafted features) |

### Competition Validation

| Metric | Target |
|--------|--------|
| Spearman rho (model ranking vs placement) | > 0.3 |
| Per-round ranking accuracy | Better than random |

### Fusion-Specific Metrics

| Metric | Target |
|--------|--------|
| Fused R2 | > best single encoder (current fusion: 0.524, must beat 0.537 audio-only) |
| Fused pairwise accuracy | > best single encoder |
| Score-conditioned quality | Qualitative: distinguishes wrong notes from expressive deviations |

### Comparison Table Format

Each notebook produces a comparison table tracking cost alongside performance:

```
Experiment | R2    | Pairwise | Difficulty rho | Robustness | GPU-hours
-----------+-------+----------+----------------+------------+----------
A1 (LoRA)  | 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
A2 (Staged)| 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
A3 (Full)  | 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
```

### Winner Selection

Primary: highest pairwise accuracy on composite labels.
Tiebreak: R2.
Veto: if robustness drop > 15%, experiment disqualified.
Bonus: STOP AUC improvement over composite-label baseline.

## Training Infrastructure

### Thunder Compute

All training notebooks run on Thunder Compute GPU instances (A100 80GB).

**Setup pattern** (per notebook):
1. Clone repo, install deps with `uv sync`
2. Sync data from GDrive via rclone
3. Train
4. Upload checkpoints to GDrive after each fold

**Persistence:** rclone to Google Drive after every fold completes. Thunder instances are ephemeral.

### Cross-Validation

4-fold piece-stratified split. Same folds as existing `percepiano_cache/folds.json`. No piece appears in both train and validation for the same fold.

### Training Hyperparameters

- Optimizer: AdamW
- Scheduler: cosine annealing with warmup (5% of steps)
- Precision: bf16-mixed on A100
- Cross-validation: 4-fold piece-stratified (no piece leakage)
- Early stopping: patience 20, monitor val pairwise accuracy
- Checkpointing: save top-3 by validation metric per experiment
- Gradient clipping: 1.0

### Reproducibility

- CUDA deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`)
- Seed 42 for all random operations
- bf16-mixed precision

## Source Module Structure

Existing modules in `model/src/model_improvement/` (already implemented):

```
model/src/model_improvement/
  __init__.py
  audio_encoders.py       -- A1 MuQLoRAModel, A2 MuQStagedModel, A3 MuQFullUnfreezeModel
  symbolic_encoders.py    -- S1 TransformerSymbolicEncoder, S2 GNNSymbolicEncoder,
                             S2H GNNHeteroSymbolicEncoder, S3 ContinuousSymbolicEncoder
  fusion.py               -- F1/F2/F3 fusion modules, FusedPerformanceModel
  losses.py               -- DimensionWiseRankingLoss, piece_based_infonce_loss
  data.py                 -- PairedPerformanceDataset, CompetitionDataset,
                             AugmentedEmbeddingDataset, MIDIPretrainingDataset,
                             ScoreGraphPretrainingDataset, ContinuousPretrainDataset,
                             HeteroPretrainDataset, multi_task_collate_fn, etc.
  tokenizer.py            -- PianoTokenizer (REMI via miditok), extract_continuous_features
  graph.py                -- midi_to_graph, midi_to_hetero_graph, assign_voices
  metrics.py              -- MetricsSuite (pairwise_accuracy, regression_r2,
                             difficulty_correlation, robustness)
  augmentation.py         -- AudioAugmentor (room IR, noise, phone sim, pitch shift, EQ)
  lora.py                 -- apply_lora_to_muq, count_trainable_params
  training.py             -- train_model, upload_checkpoint
  datasets.py             -- load_maestro_midi_files, load_atepp_midi_files,
                             load_asap_midi_files, load_percepiano_midi_files
  preprocessing.py        -- preprocess_tokens, preprocess_graphs, preprocess_continuous_features
  competition.py          -- load_competition_metadata
```

New modules needed:
- `taxonomy.py` -- composite label loading, PercePiano bridge (produced by taxonomy work)
- Update `data.py` -- add CrossPerformerDataset for T3, update collation for tier-mixed batches

## What This Design Does NOT Cover

- Teacher-grounded taxonomy derivation (separate plan, prerequisite)
- Data collection pipeline (separate plan, prerequisite)
- Production serving or inference optimization
- Real-time streaming inference
- Web app integration
- Masterclass priority signal model (depends on this backbone, future work)
