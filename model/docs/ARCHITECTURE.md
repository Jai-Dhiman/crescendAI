# Piano Performance Evaluation - Architecture Design

## Overview

Multi-modal hierarchical architecture for automated piano performance assessment using pre-trained music encoders with uncertainty-weighted multi-task learning.

**MVP Scope**: Cost-optimized proof-of-concept with 200-300 personal labels demonstrating viability before scaling to professional annotation.

## System Architecture

```
Input: Audio (WAV) + MIDI Score
       ↓
┌──────────────────────────────────────────────┐
│  Multi-Modal Feature Extraction              │
│                                              │
│  Audio Branch          MIDI Branch          │
│  ┌─────────────┐      ┌─────────────┐      │
│  │   CQT       │      │  OctupleMIDI│      │
│  │ Spectrogram │      │  Encoding   │      │
│  └──────┬──────┘      └──────┬──────┘      │
│         │                    │             │
│  ┌──────▼──────┐      ┌──────▼──────┐      │
│  │  MERT-95M   │      │ MIDIBert-   │      │
│  │  Encoder    │◄────►│   Piano     │      │
│  │ (pre-train) │      │             │      │
│  └──────┬──────┘      └──────┬──────┘      │
│         │                    │             │
│         └─────────┬──────────┘             │
│                   │                        │
└───────────────────┼────────────────────────┘
                    ↓
         ┌──────────────────┐
         │ Cross-Attention  │
         │     Fusion       │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │   Hierarchical   │
         │   Aggregation    │
         │  (BiLSTM layers) │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │  Multi-Task MTL  │
         │  (Uncertainty-   │
         │   weighted)      │
         └────────┬─────────┘
                  ↓
    ┌─────────────────────────────┐
    │   8-10 Dimension Outputs    │
    │  (scores + uncertainties)   │
    └─────────────────────────────┘
```

## Component Specifications

### 1. Audio Processing Pipeline

#### Input Preprocessing

- **Format**: 44.1kHz stereo WAV, 16-bit PCM
- **Segmentation**: 10-second windows for training, 20-30s for inference
- **Normalization**: Peak normalize to -3dB, mono mixdown

#### Constant-Q Transform (CQT)

- **Library**: nnAudio (GPU-accelerated) or librosa
- **Parameters**:
  - Bins per octave: 24 (captures piano harmonics)
  - Frequency range: C1 (32.7 Hz) to C8 (4186 Hz) - 7 octaves
  - Hop length: 512 samples (11.6ms @ 44.1kHz)
  - Window: Hann window
- **Output shape**: `[batch, 168, time_frames]` where 168 = 24 bins × 7 octaves
- **Rationale**: CQT's logarithmic spacing matches musical pitch structure better than mel-spectrograms designed for speech

#### Data Augmentation (Training Only)

Applied with probability p per augmentation:

- **Pitch shift**: ±2 semitones (p=0.3)
- **Time stretch**: 0.85-1.15× speed (p=0.3)
- **Gaussian noise**: SNR 25-40dB (p=0.2)
- **Room acoustics**: Impulse response convolution, 5-10 different spaces (p=0.4)
- **Gain variation**: ±6dB (p=0.3)
- **MP3 compression**: 128-320kbps encode-decode (p=0.2)

**Constraint**: Max 3 simultaneous augmentations, no extreme combinations

### 2. Symbolic Processing Pipeline

#### MIDI Preprocessing

- **Library**: mido, pretty_midi
- **Alignment**: DTW-based score-to-performance alignment
- **Segmentation**: Match audio segment timestamps
- **Normalization**: Note velocities to 0-1 range

#### OctupleMIDI Encoding

Eight-dimensional representation per event:

1. **Note type**: note-on, note-off, time-shift, etc.
2. **Beat**: Metrical position
3. **Position**: Position within beat
4. **Pitch**: MIDI note number (21-108 for piano)
5. **Duration**: Note length in ticks
6. **Velocity**: Attack velocity (0-127)
7. **Instrument**: Piano (always 0 for this application)
8. **Bar**: Measure number

**Encoding**: Tokenized and embedded via MIDIBert vocabulary

### 3. Audio Encoder: MERT-95M

#### Model Specification

- **Source**: HuggingFace `m-a-p/MERT-v1-95M`
- **Parameters**: 95M (vs 330M full version)
- **Pre-training**: 160K hours music with dual-teacher approach
  - Acoustic teacher: RVQ-VAE tokenizer
  - Musical teacher: Constant-Q Transform (pitch-aware)
- **Architecture**: Transformer encoder, 12 layers, 768 hidden dim

#### Fine-tuning Strategy

- **Stage 2 (pseudo-labels)**: Fine-tune all layers, lr=1e-5, 10-20 epochs
- **Stage 3 (expert labels)**: Fine-tune all layers, lr=5e-6 to 2e-5, 50 epochs
- **Gradient checkpointing**: Enabled to reduce memory (critical for Colab)

#### Output

- **Shape**: `[batch, time_frames, 768]`
- **Usage**: Frame-level embeddings for hierarchical aggregation

### 4. Symbolic Encoder: MIDIBert-Piano

#### Model Specification

- **Source**: HuggingFace `microsoft/MIDIBert-Piano` (if available) or custom implementation
- **Parameters**: ~12M
- **Pre-training**: Self-supervised on MIDI corpora
- **Architecture**: BERT-style transformer, 6 layers, 256 hidden dim

#### Fine-tuning Strategy

- **Stage 2-3**: Fine-tune all layers alongside audio encoder
- **Learning rate**: 1e-4 (higher than MERT due to smaller pre-training corpus)

#### Output

- **Shape**: `[batch, midi_events, 256]`
- **Usage**: Event-level embeddings for cross-attention fusion

### 5. Cross-Attention Fusion

#### Architecture

Bidirectional cross-attention between modalities:

```python
# Audio attends to MIDI (what notes are written?)
audio_fused = CrossAttention(
    query=audio_features,   # [B, T_audio, 768]
    key=midi_features,      # [B, T_midi, 256]
    value=midi_features
)

# MIDI attends to Audio (how are notes performed?)
midi_fused = CrossAttention(
    query=midi_features,    # [B, T_midi, 256]
    key=audio_features,     # [B, T_audio, 768]
    value=audio_features
)

# Concatenate for joint representation
fused = concat([audio_fused, midi_fused], dim=-1)  # [B, T, 1024]
```

#### Parameters

- **Attention heads**: 8
- **Dropout**: 0.1
- **Layer norm**: Applied before and after attention
- **Positional encoding**: Relative attention (time offsets matter)

#### Fallback Mode

If MIDI unavailable:

- Use audio features only: `fused = audio_features`
- Expected performance degradation: 15-20% on score-dependent dimensions

### 6. Hierarchical Temporal Aggregation

Piano pieces span 2-10 minutes, far exceeding transformer context windows. Three-level hierarchy:

#### Level 1: Frame-level (CQT frames)

- **Granularity**: ~11.6ms per frame
- **Source**: Direct MERT output
- **Use**: Low-level acoustic features (tone, dynamics)

#### Level 2: Note/Phrase-level (segments)

- **Granularity**: 10-second training segments, 20-30s inference windows
- **Architecture**: Bidirectional LSTM
  - 2 layers
  - 256 hidden units per direction (512 total)
  - Dropout 0.2 between layers
- **Aggregation**: Multi-head attention over LSTM outputs (4 heads)
- **Output**: `[batch, 512]` phrase embedding

#### Level 3: Piece-level (full performance)

- **Inference only**: Not used in MVP (segments treated independently)
- **Future**: Music Transformer with relative positional attention
- **Purpose**: Global coherence, long-range structure

### 7. Multi-Task Learning Head

#### Uncertainty-Weighted Loss (Kendall & Gal)

Automatically balances multiple tasks without manual tuning:

```
L_total = Σ_i [ (1 / 2σ_i²) * L_i + log(σ_i) ]

where:
- L_i: Individual task loss (MSE for regression)
- σ_i: Learned uncertainty parameter (per dimension)
- First term: Task loss weighted by inverse uncertainty
- Second term: Regularization preventing σ → ∞
```

**Benefits**:

- No manual loss weight tuning
- Tasks with higher inherent noise (interpretive) automatically downweighted
- Provides per-dimension uncertainty estimates

#### Task Head Architecture

For each of 8-10 dimensions:

```python
shared_features → [512]
    ↓
Linear(512 → 256)
    ↓
ReLU + Dropout(0.1)
    ↓
Linear(256 → 128)
    ↓
ReLU + Dropout(0.1)
    ↓
Linear(128 → 1)
    ↓
Sigmoid × 100  # Scale to 0-100
```

**Shared vs Separate**:

- **Shared**: First feature extractor (linear 512→256) shared across all tasks
- **Separate**: Dimension-specific heads (256→128→1) for each task

Total parameters in MTL head: ~8-10M

### 8. Output Layer

#### Per-Dimension Outputs

For each of 8-10 dimensions:

1. **Score**: 0-100 continuous value
2. **Aleatoric uncertainty**: `σ_i` from uncertainty-weighted loss
3. **Attention weights**: Frame-level importance (for visualization)

#### Dimension Definitions

**Technical (6 dimensions)**:

1. **Note accuracy**: Correctness of pitches (% correct notes)
2. **Rhythmic precision**: Timing accuracy (deviation from score timing)
3. **Dynamics control**: Volume variation and control (velocity consistency)
4. **Articulation quality**: Clarity of note attacks and releases
5. **Pedaling technique**: Appropriate sustain pedal usage
6. **Tone quality**: Timbre, color, register balance

**Interpretive (4 dimensions)**:
7. **Phrasing**: Musical sentence structure and breathing
8. **Musicality**: Overall musical expression and communication
9. **Overall quality**: Holistic assessment
10. **Expressiveness**: Emotional depth and nuance

#### Aggregate Metrics

Weighted average (optional):

```
Overall = 0.35 × (Note + Rhythm) / 2
        + 0.25 × (Dynamics + Tone) / 2
        + 0.25 × (Musicality + Expressiveness) / 2
        + 0.15 × Phrasing
```

Weights can be configured based on pedagogical priorities.

## Training Pipeline

### Stage 1: Initialization (No Training)

- Load MERT-95M pre-trained weights from HuggingFace
- Load MIDIBert-Piano pre-trained weights
- Initialize fusion layers, LSTM, and MTL heads randomly
- **Time**: <1 hour (download + initialization)
- **Compute**: Local (M4 Mac)

### Stage 2: Pseudo-Label Pre-training

#### Data

- **Source**: MAESTRO subset (50-100 pieces, ~10-15 hours)
- **Segments**: 10-second windows, 50% overlap → ~5,000-7,500 segments
- **Labels**: Heuristic pseudo-labels for 6 technical dimensions
  - Note accuracy: MIDI transcription comparison
  - Rhythm: Onset timing deviation from score
  - Dynamics: Velocity range and consistency
  - Articulation: Attack sharpness (spectral flux)
  - Pedaling: CC64 coherence (from performance MIDI)
  - Tone: Spectral centroid, register balance

#### Training Configuration

```yaml
optimizer: AdamW
learning_rate:
  backbone: 1e-5
  heads: 1e-4
weight_decay: 0.01
scheduler: cosine
warmup_steps: 500
max_epochs: 20
batch_size: 8
gradient_accumulation: 4  # Effective batch 32
mixed_precision: fp16
gradient_checkpointing: true
```

#### Compute

- **Hardware**: Colab Pro T4 (16GB) or V100 (16GB)
- **Time**: ~12 GPU hours

#### Expected Outcome

- Adapt MERT to evaluation task
- Reduce expert labeling requirements
- Checkpoint: `checkpoints/pseudo_pretrain_best.ckpt`

### Stage 3: Expert Label Fine-tuning

#### Data

- **Source**: Personal annotations (200-300 segments @ 20-30s each)
- **Labels**: 8-10 dimensions with 0-100 ratings
- **Splits**: 70/15/15 train/val/test (stratified by difficulty, piece)

#### Training Configuration

```yaml
optimizer: AdamW
learning_rate:
  backbone: 5e-6 to 2e-5 (start lower, decay)
  heads: 5e-4
weight_decay: 0.01
scheduler: cosine with warmup
warmup_steps: 500
max_epochs: 50 (early stopping patience=10)
batch_size: 4-8
gradient_accumulation: 4-8  # Effective batch 32
mixed_precision: fp16
gradient_checkpointing: true
label_smoothing: 0.1
dropout: 0.1-0.2
gradient_clip: 1.0
```

#### Regularization

- **Dropout**: 0.1 in fusion layers, 0.2 in LSTM, 0.1 in heads
- **Label smoothing**: 0.1 (targets shifted 10% toward mean)
- **Weight decay**: 0.01
- **Data augmentation**: All augmentations listed above
- **Early stopping**: Patience=10 on validation loss

#### Compute

- **Hardware**: Colab Pro V100 (16-32GB)
- **Time**: ~8 GPU hours
- **Cost**: $0 (Colab Pro)

#### Expected Outcome

- Technical dimensions: r=0.50-0.65 with expert ratings
- Interpretive dimensions: r=0.35-0.50
- Checkpoint: `checkpoints/expert_finetune_best.ckpt`

## Evaluation Metrics

### Correlation Metrics

- **Pearson r**: Linear correlation with expert ratings
- **Spearman ρ**: Rank-order correlation (robust to outliers)
- **Kendall τ**: Alternative rank correlation

**Targets**:

- Technical: r>0.50, ρ>0.50
- Interpretive: r>0.35, ρ>0.40

### Error Metrics

- **MAE**: Mean Absolute Error on 0-100 scale
- **RMSE**: Root Mean Squared Error
- **Range Accuracy**: % predictions within [min_rating-α, max_rating+α]

**Targets**:

- MAE <15 points
- RA(α=10) >75%

### Calibration Metrics

- **ECE**: Expected Calibration Error (binned confidence vs accuracy)
- **Reliability diagram**: Visual calibration assessment
- **Sharpness**: Spread of uncertainty estimates

**Targets**:

- ECE <0.15 (MVP, <0.10 for full system)

### Baseline Comparisons

1. **Random**: Uniform random 0-100
2. **Mean baseline**: Always predict training set mean per dimension
3. **Linear regression**: Simple audio features → ratings
4. **Ablations**:
   - Audio-only (no MIDI)
   - No pre-training (random init)
   - No uncertainty weighting (equal task weights)

**Target**: ≥30% improvement over linear baseline

## Inference Pipeline

### Input

- Audio file (WAV, MP3, FLAC)
- MIDI score (optional but recommended)

### Processing Steps

1. **Preprocess**:
   - Resample to 44.1kHz, convert to mono
   - Segment into 20-30s windows with 50% overlap
   - Compute CQT spectrograms
   - Parse and align MIDI if available

2. **Forward pass**:
   - Encode audio with MERT-95M
   - Encode MIDI with MIDIBert (if available)
   - Cross-attention fusion
   - Hierarchical aggregation
   - Multi-task prediction

3. **Aggregate**:
   - Average predictions across overlapping windows (weighted by center)
   - Pool attention maps across windows
   - Compute final uncertainty estimates

4. **Output**:
   - Per-dimension scores (0-100)
   - Per-dimension uncertainties (standard deviations)
   - Attention heatmap (time-frequency visualization)
   - Temporal attention curve (which bars/phrases are problematic)

### Performance

- **Latency**: 2-5 seconds for 5-minute piece (single V100)
- **Throughput**: 12-30 pieces per minute (batch processing)
- **Memory**: 4-6GB GPU RAM for single inference

## Scaling Roadmap (Post-MVP)

After validating MVP performance, scale to professional-grade system:

### Model Architecture

- **MERT-330M**: 3.5× larger backbone (95M → 330M parameters)
- **Music Transformer**: Replace BiLSTM with relative positional attention
- **Ensemble specialists**: 3-5 variants per interpretive dimension
- **Temperature scaling**: Post-ensemble calibration

### Data

- **2,500 expert labels**: 5-8 raters per segment
- **Active learning**: Iterative selection of high-uncertainty segments
- **PercePiano**: External validation on existing benchmark

### Training

- **Compute**: 1,500 GPU-hours on A100 (vs 20h on T4/V100)
- **Cost**: $2,500-3,500
- **Performance**: r=0.68-0.75 (professional-grade)

### Deployment

- **Cloud API**: FastAPI serving batch predictions
- **Real-time feedback**: Distilled 50-100M student model
- **Mobile**: Quantized INT8 model for edge inference

## Key Design Decisions

### Why MERT-95M (not 330M)?

- **Memory**: 95M fits Colab T4 16GB, 330M requires A100 40GB
- **Cost**: Free Colab Pro vs $4/hour A100
- **Performance**: 85-90% of 330M performance for MVP validation
- **Scaling**: Easy upgrade path (same architecture)

### Why BiLSTM (not Music Transformer)?

- **Simplicity**: Fewer hyperparameters, easier debugging
- **Memory**: 2× less GPU RAM than self-attention
- **Performance**: Adequate for segment-level aggregation
- **Scaling**: Can replace with Music Transformer later

### Why No Ensemble?

- **Cost**: 3-5× more training time (limited budget)
- **Complexity**: Harder to debug and iterate
- **Benefit**: Ensemble mostly helps interpretive dimensions (5-10% gain)
- **MVP Focus**: Validate architecture first, then add ensemble

### Why 10-Second Segments?

- **Memory**: Longer segments exceed Colab GPU RAM
- **Context**: Sufficient for note/phrase-level evaluation
- **Overlap**: 50% overlap at inference provides context
- **Scaling**: Full-piece aggregation added later

### Why 8-10 Dimensions (not 12)?

- **Data**: Limited labels (200-300 segments) can't support 12 tasks
- **Coverage**: 8-10 covers core technical + interpretive aspects
- **Uncertainty**: Fewer tasks → more data per task → better uncertainty estimates
- **Scaling**: Easy to add 2 more dimensions with more data

## Risk Mitigation

### Insufficient Labels (200-300 segments)

- **Pseudo-labeling**: Bootstraps from 5,000+ MAESTRO segments
- **Pre-training**: MERT-95M already knows music, needs task adaptation
- **Augmentation**: 7× data multiplier
- **Multi-task**: Shared representations across correlated dimensions
- **Fallback**: Start with 3-5 technical dimensions if 8-10 underperforms

### Overfitting

- **Regularization**: Dropout, weight decay, label smoothing, gradient clipping
- **Early stopping**: Patience=10 on validation loss
- **Augmentation**: Aggressive audio augmentation
- **Cross-validation**: Monitor pianist-out performance

### Colab Session Limits

- **Checkpointing**: Save every epoch + best model
- **Resume training**: Load checkpoint and continue
- **Efficient training**: Mixed precision, gradient checkpointing
- **Batch size**: Gradient accumulation for larger effective batch

### Poor Calibration

- **Uncertainty weighting**: Learns per-task uncertainties
- **Label smoothing**: Prevents overconfidence
- **Temperature scaling**: Post-hoc calibration (requires validation set)
- **Acceptance**: ECE<0.15 acceptable for MVP (vs <0.10 for production)

## References

- **MERT Paper**: "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training" (ICML 2023)
- **MIDIBert**: "MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding" (2021)
- **Kendall & Gal**: "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
- **PercePiano**: "Predicting Perceptual Ratings of Piano Performances" (2021)
- **Full research**: See `RESEARCH.md`
