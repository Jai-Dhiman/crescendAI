# Model Improvement Design: Dual-Encoder Architecture with Domain Adaptation

## Goal

Build the strongest possible backbone model for piano performance evaluation by:

1. Fine-tuning MuQ for piano-specific quality assessment (audio track)
2. Building a symbolic foundation model for piano MIDI (symbolic track)
3. Fusing both encoders for score-conditioned quality assessment and alignment
4. Training on every available data source -- labeled, weakly labeled, and unlabeled at scale

The fused model answers the north star question: "How well is the student playing what the score asks for?" by combining audio understanding (how it sounds) with symbolic understanding (what the score asks).

## Current Baseline

- Paper model: Frozen MuQ (layers 9-12) -> mean pooling -> 2-layer MLP -> 19 PercePiano dimensions. R^2 = 0.537.
- Contrastive ranker (E2a): Attention pooling -> shared encoder -> piece-based InfoNCE + per-dimension ranking heads. 84% pairwise accuracy.
- Symbolic baseline: Hand-crafted MIDI features -> MLP. R^2 = 0.347.
- Audio-symbolic fusion: Concatenation. R^2 = 0.524 (worse than audio alone).
- Score alignment via MuQ: Failed. ~18s mean onset error. MuQ layers 9-12 encode semantic, not temporal content.

## Data Strategy

Four tiers, each providing a different training signal:

### T1: Labeled (PercePiano)

- 1,202 segments with 19-dimension perceptual scores from crowdworkers
- Primary supervised signal for absolute quality prediction
- Already available

### T2: Weakly Labeled (Competition Recordings)

- Piano competitions: International Chopin Competition, Cliburn, Leeds, Van Cliburn archives
- 500-2,000 recordings (estimated, best-effort collection)
- Signal: ordinal ranking from placements (1st > 2nd > semifinal > eliminated)
- Schema: {recording_id, competition, round, placement, piece, performer, audio_path}
- Risk: sourcing difficulty. Some competitions publish freely (Chopin Competition on YouTube), others are behind paywalls. Treat as best-effort, not a blocker. T3 provides ranking signal without explicit labels.

### T3: Paired Unlabeled (Multi-Performer MIDI+Audio)

- MAESTRO v3: 1,276 performances, 200+ hours (already using)
- ASAP: 1,067 performances of 236 scores (already using)
- ATEPP: ~11,000 performances with aligned MIDI (new -- largest multi-performer dataset)
- Signal: same-piece different-performer pairs for contrastive learning
- Piece grouping enables self-supervised ranking signal

### T4: Unlabeled at Scale

- MIDI: GiantMIDI-Piano (~10,000 transcribed recordings)
- Audio: YouTube piano channels (professional recitals, conservatory uploads), target 5,000-10,000 recordings
- Signal: self-supervised pretraining objectives only
- Processing: segment into 10-30s clips, extract and cache embeddings

## Architecture

### Experimental Matrix

Six independent experiments across two tracks, followed by fusion of the best from each track.

### Audio Encoder Experiments

**A1: MuQ + LoRA Fine-tuning**

- LoRA adapters (rank 16-64) on self-attention layers of MuQ layers 9-12
- 99%+ of MuQ parameters stay frozen
- Multi-task training on all available labels (T1+T2+T3)
- Cheapest, fastest iteration

**A2: MuQ Staged Domain Adaptation**

- Stage 1: Self-supervised on T3+T4 data (no labels needed)
  - Cross-performer contrastive: same piece, different performers -> positive pairs
  - Augmentation invariance: same recording + {noise, room IR, phone sim} -> should produce same embedding
  - LoRA adapters on MuQ
- Stage 2: Multi-task supervised fine-tuning on T1+T2+T3
- Most principled approach

**A3: MuQ Full Unfreeze**

- Gradually unfreeze MuQ layers: 12 -> 11 -> 10 -> 9
- Discriminative learning rates (deeper layers get smaller LR)
- Highest parameter count, highest risk of catastrophic forgetting, highest ceiling

**All audio experiments share:**

```
Raw audio (24kHz)
  -> MuQ backbone (layers 9-12, adapted per experiment)
  -> Frame embeddings [T, 4096]
  -> Learned attention pooling -> [4096]
  -> Projection head -> z_audio [512]
```

### Symbolic Encoder Experiments

**S1: Transformer on MIDI Tokens (default)**

- REMI tokenization: note-on, note-off, velocity, time-shift, pedal, bar, tempo
- Vocabulary: ~500 tokens
- Architecture: 6-12 layer Transformer, 512-dim, 8 heads (~25M parameters)
- Pretraining: masked token prediction (15% masking, BERT-style) + cross-performer contrastive
- Pretrain on GiantMIDI + MAESTRO + ASAP + ATEPP

**S2: GNN on Score Graph**

- Notes as nodes, edges for temporal adjacency, harmonic intervals, voice membership
- Message-passing encoder
- Pretraining: link prediction + node attribute prediction (masked velocity/timing)
- Structurally expressive for counterpoint, harmonic progressions

**S3: Continuous MIDI Encoder**

- MIDI -> continuous feature curves (pitch, velocity, pedal depth over time)
- 1D-CNN + Transformer architecture
- wav2vec-style contrastive pretraining: quantize features, predict masked frames
- Most analogous to MuQ's architecture, may make fusion more natural

**All symbolic experiments share:**

```
MIDI performance
  -> Tokenizer/encoder (per experiment)
  -> Token/frame embeddings [T, 512]
  -> Attention pooling -> z_symbolic [512]
```

**Two-phase training for all symbolic experiments:**

1. Self-supervised pretraining on large MIDI corpus (GiantMIDI + MAESTRO + ASAP + ATEPP)
2. Supervised fine-tuning on PercePiano (symbolic path) + MAESTRO pairwise ranking

### Fusion Experiments

Input: best audio encoder (winner of A1/A2/A3) + best symbolic encoder (winner of S1/S2/S3).

**F1: Cross-attention fusion**

- z_audio attends to z_symbolic and vice versa
- Breaks correlated-error pattern that killed concatenation fusion (error correlation r=0.738)

**F2: Concatenation (baseline)**

- [z_audio; z_symbolic] -> MLP
- Must beat current 0.524 or fusion isn't working

**F3: Gated fusion**

- Learned per-dimension weighting: some dimensions may benefit more from audio, others from symbolic
- gate_d = sigmoid(W_d * [z_audio; z_symbolic])
- z_fused_d = gate_d *z_audio + (1 - gate_d)* z_symbolic

**Fusion training:**

- Freeze both encoders initially, train only fusion module + downstream heads on T1+T2+T3
- If frozen fusion plateaus, optionally unfreeze encoders with very low LR (1e-6) for end-to-end tuning

**Downstream heads (shared across fusion experiments):**

- Quality heads: 19-dimension regression (PercePiano)
- Ranking heads: 19-dimension pairwise ranking (E2a-style)
- Difficulty head: auxiliary regression (PSyllabus)

**Score-conditioned quality (the big unlock):**
With both encoders, quality becomes: f(z_performance_audio, z_performance_midi, z_score_midi).
The model knows what the score asks for AND how the performance sounds.

## Training Strategy

### Multi-Task Objective (supervised phases)

```
L_total = L_regression + lambda_rank * L_ranking + lambda_contrastive * L_contrastive + lambda_augment * L_invariance

L_regression   = MSE on PercePiano 19-dim scores (T1)
L_ranking      = DimensionWiseRankingLoss on same-piece pairs (T2+T3)
L_contrastive  = Piece-based InfoNCE (T3)
L_invariance   = MSE between embeddings of clean vs augmented audio (T4)
```

### Audio Augmentation Suite

Applied on-the-fly during training via AudioAugmentor:

| Augmentation | Source | Probability | Purpose |
|---|---|---|---|
| Room impulse response | MIT IR Survey, EchoThief (~500 IRs) | 0.3 | Reverb/acoustics robustness |
| Additive noise | ESC-50 (2,000 clips), SNR 10-30dB | 0.3 | Background noise |
| Phone mic simulation | Low-pass 8kHz, compression, distortion | 0.2 | Phone recording bridge |
| Pitch shift | +/- 50 cents | 0.1 | Tuning variation |
| EQ variation | Random 3-band parametric EQ | 0.2 | Timbral variation across pianos |

### MIDI Tokenization

REMI-style tokenizer for symbolic experiments:

- Position tokens (beat subdivisions)
- Pitch tokens (0-127)
- Velocity tokens (quantized to 32 bins)
- Duration tokens (quantized)
- Pedal tokens (on/off/partial)
- Bar tokens (structural markers)
- Tempo tokens (quantized BPM)
- Vocabulary size: ~500 tokens

### Training Infrastructure

- Optimizer: AdamW
- Scheduler: cosine annealing with warmup (5% of steps)
- Precision: bf16-mixed on A100
- Cross-validation: 4-fold piece-stratified (no piece leakage)
- Early stopping: patience 15, monitor val pairwise accuracy or val R^2
- Checkpointing: save top-3 by validation metric per experiment
- Gradient clipping: 1.0

## Evaluation Strategy

### Core Metrics (all experiments)

| Metric | Dataset | Current Best |
|---|---|---|
| R^2 (19-dim) | PercePiano 4-fold CV | 0.537 (MuQ regression) |
| Pairwise accuracy (19-dim) | PercePiano same-piece pairs | 84% (E2a contrastive) |
| Per-dimension breakdown | PercePiano | varies by dimension |
| Piece-split vs performer-split R^2 | PercePiano | 0.536 both (well-disentangled) |
| Cross-soundfont R^2 | PercePiano leave-one-out | 0.534 |
| Difficulty correlation (Spearman rho) | PSyllabus (508 pieces) | 0.623 |

### Robustness Metrics (audio experiments)

| Metric | Target |
|---|---|
| Augmented pairwise accuracy | Drop < 10% vs clean |
| Cross-condition consistency (Pearson r, clean vs augmented) | > 0.9 |
| Real phone recording pilot | Qualitative sanity check |

### Symbolic-Specific Metrics

| Metric | Dataset | Current (MuQ failed) |
|---|---|---|
| Score alignment accuracy (measure-level) | ASAP | ~30% within 30ms |
| Alignment onset error | ASAP | ~18s (MuQ), target < 1s |
| Symbolic-only R^2 | PercePiano MIDI path | 0.347 (hand-crafted baseline) |

### Fusion Metrics

| Metric | Target |
|---|---|
| Fused R^2 | > 0.537 (must beat audio alone; current fusion is 0.524) |
| Fused pairwise accuracy | > best single encoder |
| Score-conditioned quality | Qualitative: distinguishes wrong notes from expressive deviation |

### Winner Selection

Primary: highest pairwise accuracy.
Tiebreaker: R^2.
Veto: if robustness drop > 15%, experiment disqualified.

### Comparison Tables

Each comparison notebook (07, 08) produces:

```
Experiment | R^2   | Pairwise | Difficulty rho | Robustness | GPU-hours
-----------+-------+----------+----------------+------------+----------
A1 (LoRA)  | 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
A2 (Staged)| 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
A3 (Full)  | 0.xxx | xx.x%    | 0.xxx          | xx.x%      | X
```

## Notebook Structure

```
notebooks/model_improvement/
  01_audio_A1_lora_finetune.ipynb
  02_audio_A2_staged_adaptation.ipynb
  03_audio_A3_full_unfreeze.ipynb
  04_symbolic_S1_transformer.ipynb
  05_symbolic_S2_gnn.ipynb
  06_symbolic_S3_continuous.ipynb
  07_audio_comparison.ipynb
  08_symbolic_comparison.ipynb
  09_fusion_experiments.ipynb
  10_robustness_validation.ipynb
```

Each notebook imports from src/ modules. Notebooks orchestrate and visualize; logic lives in importable modules.

## Source Module Structure

```
model/src/
  model_improvement/
    __init__.py
    audio_encoders.py       -- A1/A2/A3 model definitions, LoRA integration
    symbolic_encoders.py    -- S1/S2/S3 model definitions, tokenizer
    fusion.py               -- F1/F2/F3 fusion modules
    losses.py               -- multi-task loss, augmentation invariance loss
    data.py                 -- T1-T4 dataset classes, augmentation pipeline
    tokenizer.py            -- REMI tokenizer for MIDI
    evaluation.py           -- shared eval suite, comparison tables
    augmentation.py         -- AudioAugmentor class
```

## Caching

```
model/data/
  percepiano_cache/           # existing
  maestro_cache/              # existing
  competition_cache/          # new: audio + metadata
  atepp_cache/                # new: MIDI + audio pairs
  giant_midi_cache/           # new: tokenized MIDI sequences
  youtube_piano_cache/        # new: audio segments
  augmentation_assets/
    room_irs/                 # MIT IR Survey + EchoThief
    noise/                    # ESC-50
  embeddings/
    muq/                      # per-experiment MuQ embeddings
    symbolic/                 # per-experiment symbolic embeddings
```

## What This Design Does NOT Cover

- Data collection automation (scraping, downloading) -- separate pipeline work
- Production serving / inference optimization
- Real-time streaming inference
- Masterclass priority signal model (separate design, depends on this backbone)
- Web app integration
