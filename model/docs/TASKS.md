# Implementation Tasks - Piano Performance Evaluation MVP

## High-Level Phases

```
Phase 1: Environment & Data Infrastructure (Week 1)
Phase 2: Model Architecture Implementation (Week 2)
Phase 3: Data Labeling & Pseudo-labels (Week 3-4)
Phase 4: Training Pipeline & Experiments (Week 5-6)
Phase 5: Evaluation & Documentation (Week 7)
```

**Total Estimated Time**: 6-8 weeks (part-time)
**Budget**: <$100 (Colab Pro only)

---

## Phase 1: Environment & Data Infrastructure

### 1.1 Project Setup

- [x] Create `pyproject.toml` with dependencies:
  - PyTorch 2.x
  - PyTorch Lightning 2.x
  - transformers (HuggingFace)
  - librosa or nnAudio
  - mido, pretty_midi
  - numpy, pandas, scipy
  - matplotlib, seaborn (visualization)
  - pytest (testing)
  - wandb (optional, experiment tracking)
- [x] Run `uv sync` and verify installation
- [x] Create `.gitignore` (exclude data/, checkpoints/, .venv/)

### 1.2 MAESTRO Data Download

**Time**: 2-3 hours (mostly waiting for downloads)

- [x] Create `scripts/download_maestro.py`:
  - Download MAESTRO v3.0.0 from official source
  - Support `--subset N` argument (download N pieces instead of full 1,276)
  - Verify downloads (checksums if available)
  - Organize: `data/maestro/year/audio/` and `data/maestro/year/midi/`
- [ ] Run download for 50-100 pieces (~10-15 hours audio):

  ```bash
  python scripts/download_maestro.py --subset 50 --output data/maestro
  ```

- [ ] Verify downloaded files (check audio playback, MIDI parsing)
- [ ] Create metadata CSV: `piece_id, audio_path, midi_path, duration, composer, year`

### 1.3 Audio Preprocessing Pipeline

**Time**: 3-4 hours

- [x] Create `src/data/audio_processing.py`:
  - [x] `load_audio(path)`: Load WAV/MP3, resample to 44.1kHz, convert to mono
  - [x] `compute_cqt(audio, sr=44100)`: CQT spectrogram
    - 24 bins per octave
    - 7 octaves (C1 to C8)
    - Hop length 512
    - Return shape: `[168, time_frames]`
  - [x] `segment_audio(audio, segment_len=10, overlap=0.5)`: Split into 10s segments
  - [x] `normalize_audio(audio, target_db=-3)`: Peak normalization
- [ ] Test on 5-10 MAESTRO files, verify CQT shapes
- [ ] Benchmark: GPU vs CPU for CQT (nnAudio vs librosa)

### 1.4 MIDI Preprocessing Pipeline

**Time**: 3-4 hours

- [x] Create `src/data/midi_processing.py`:
  - [x] `load_midi(path)`: Parse MIDI with mido/pretty_midi
  - [x] `align_midi_to_audio(midi, audio_duration)`: Tempo alignment
  - [x] `extract_midi_features(midi)`: Basic features for pseudo-labels
    - Note onset times
    - Note velocities
    - Pedal events (CC64)
  - [x] `encode_octuple_midi(midi)`: OctupleMIDI tokenization
    - 8-tuple per event: (type, beat, position, pitch, duration, velocity, instrument, bar)
    - Return token sequence
  - [x] `segment_midi(midi, timestamps)`: Match audio segments
- [ ] Test on 5-10 MAESTRO MIDI files
- [ ] Verify alignment quality (visual inspection)

### 1.5 Data Augmentation

**Time**: 2-3 hours

- [x] Create `src/data/augmentation.py`:
  - [x] `pitch_shift(audio, semitones)`: ±2 semitones
  - [x] `time_stretch(audio, rate)`: 0.85-1.15× speed
  - [x] `add_noise(audio, snr_db)`: Gaussian noise, SNR 25-40dB
  - [x] `apply_room_acoustics(audio, impulse_response)`: Convolve with IR
  - [x] `compress_audio(audio, bitrate)`: MP3 encode-decode, 128-320kbps
  - [x] `gain_variation(audio, db_range)`: ±6dB
  - [x] `augment_pipeline(audio, config)`: Apply random subset (max 3)
- [ ] Download or generate impulse responses (5-10 room types)
- [ ] Test augmentations (listen to examples)
- [ ] Write unit tests: `tests/test_augmentation.py`

---

## Phase 2: Model Architecture Implementation

### 2.1 MERT Audio Encoder

**Time**: 2-3 hours

- [x] Create `src/models/audio_encoder.py`:
  - [x] `MERTEncoder` class:
    - Load pre-trained `m-a-p/MERT-v1-95M` from HuggingFace
    - Forward: `cqt [B, 168, T] → features [B, T', 768]`
    - Support gradient checkpointing (for memory)
    - Support frozen vs fine-tunable modes
- [ ] Test loading on CPU (M4 Mac)
- [ ] Test forward pass with dummy CQT input
- [ ] Verify output shapes
- [ ] Write unit test: `tests/test_audio_encoder.py`

### 2.2 MIDI Encoder

**Time**: 2-3 hours

- [ ] Create `src/models/midi_encoder.py`:
  - [ ] `MIDIBertEncoder` class:
    - Load pre-trained MIDIBert (or implement lightweight version)
    - Vocabulary for OctupleMIDI tokens
    - Forward: `midi_tokens [B, L] → features [B, L, 256]`
    - Positional embeddings
- [ ] Test loading and forward pass
- [ ] Verify output shapes
- [ ] Write unit test: `tests/test_midi_encoder.py`

### 2.3 Cross-Attention Fusion

**Time**: 3-4 hours

- [ ] Create `src/models/fusion.py`:
  - [ ] `CrossAttentionFusion` class:
    - Bidirectional cross-attention:
      - Audio queries attend to MIDI keys/values
      - MIDI queries attend to audio keys/values
    - 8 attention heads
    - Relative positional encoding (time alignment)
    - Forward: `(audio [B,T,768], midi [B,L,256]) → fused [B,T,1024]`
  - [ ] `AudioOnlyFallback`: Pass-through when MIDI unavailable
- [ ] Test with dummy inputs
- [ ] Test MIDI fallback mode
- [ ] Write unit test: `tests/test_fusion.py`

### 2.4 Hierarchical Temporal Aggregation

**Time**: 2-3 hours

- [ ] Create `src/models/aggregation.py`:
  - [ ] `HierarchicalAggregator` class:
    - BiLSTM layers (2 layers, 256 hidden per direction)
    - Multi-head attention pooling (4 heads)
    - Forward: `fused [B, T, 1024] → aggregated [B, 512]`
    - Return attention weights for visualization
- [ ] Test forward pass
- [ ] Verify gradient flow through LSTM
- [ ] Write unit test: `tests/test_aggregation.py`

### 2.5 Multi-Task Learning Head

**Time**: 3-4 hours

- [x] Create `src/models/mtl_head.py`:
  - [x] `MultiTaskHead` class:
    - Shared feature extractor: `Linear(512 → 256)`
    - Per-dimension heads (8-10 dimensions):
      - `Linear(256 → 128) → ReLU → Dropout`
      - `Linear(128 → 1) → Sigmoid × 100`
    - Learnable uncertainty parameters `σ_i` (one per dimension)
    - Forward: `aggregated [B, 512] → scores [B, 10], uncertainties [10]`
- [x] Test forward pass
- [x] Verify output ranges (0-100)
- [x] Write unit test: `tests/test_mtl_head.py` (21 tests passing)

### 2.6 Uncertainty-Weighted Loss

**Time**: 2-3 hours

- [x] Create `src/losses/uncertainty_loss.py`:
  - [x] `UncertaintyWeightedLoss` class:
    - Kendall & Gal formulation: `L = Σ (1/2σ²) L_i + log(σ)`
    - Per-dimension MSE losses
    - Track and log individual task losses
- [x] Test with dummy predictions and targets
- [x] Verify gradient flow to σ parameters
- [x] Write unit test: `tests/test_uncertainty_loss.py` (16 tests passing)

### 2.7 Complete Lightning Module

**Time**: 4-5 hours

- [ ] Create `src/models/lightning_module.py`:
  - [ ] `PerformanceEvaluationModel(LightningModule)`:
    - Compose all components: MERT + MIDIBert + Fusion + Aggregation + MTL
    - `forward(audio, midi)`: Full forward pass
    - `training_step()`: Compute loss, log metrics
    - `validation_step()`: Compute validation metrics
    - `test_step()`: Compute test metrics
    - `configure_optimizers()`: AdamW with cosine schedule
      - Backbone lr: 5e-6 to 2e-5
      - Heads lr: 1e-4 to 5e-4
      - Warmup 500 steps
      - Weight decay 0.01
    - Support loading from checkpoint
- [ ] Test instantiation
- [ ] Test forward pass with dummy batch
- [ ] Test optimizer configuration
- [ ] Write integration test: `tests/test_lightning_module.py`

---

## Phase 3: Data Labeling & Pseudo-Labels

### 3.1 Pseudo-Label Generation

**Time**: 4-6 hours

- [ ] Create `scripts/generate_pseudo_labels.py`:
  - [ ] **Note Accuracy**: Compare performance MIDI to score MIDI
    - Align notes with DTW
    - Compute % correct pitches
    - Score: 0-100 based on accuracy
  - [ ] **Rhythmic Precision**: Onset timing deviations
    - Compute timing error per note
    - Score: 0-100 based on mean/std deviation
  - [ ] **Dynamics Control**: Velocity range and consistency
    - Analyze velocity distribution
    - Score: 0-100 based on dynamic range and smoothness
  - [ ] **Articulation Quality**: Attack sharpness
    - Spectral flux at note onsets
    - Score: 0-100 based on clarity
  - [ ] **Pedaling Technique**: CC64 coherence
    - Extract pedal events from MIDI
    - Score: 0-100 based on appropriate usage
  - [ ] **Tone Quality**: Spectral features
    - Spectral centroid, register balance
    - Score: 0-100 based on brightness/balance
- [ ] Run on MAESTRO subset (50-100 pieces)
- [ ] Generate ~5,000-7,500 segments with pseudo-labels
- [ ] Save to `data/maestro_pseudo_labels.jsonl`:

  ```json
  {
    "audio_path": "...",
    "midi_path": "...",
    "start_time": 0.0,
    "end_time": 10.0,
    "labels": {
      "note_accuracy": 85.2,
      "rhythmic_precision": 78.5,
      ...
    }
  }
  ```

- [ ] Validate: spot-check 10-20 segments (listen + check labels)

### 3.2 Labeling Interface

**Time**: 4-5 hours

- [ ] Create `notebooks/labeling_interface.ipynb`:
  - [ ] Audio player widget (IPython.display.Audio)
  - [ ] Waveform visualization (librosa.display)
  - [ ] MIDI score display (basic piano roll)
  - [ ] 8-10 slider widgets (0-100) per dimension:
    - Technical: note_accuracy, rhythmic_precision, dynamics_control, articulation_quality, pedaling_technique, tone_quality
    - Interpretive: phrasing, musicality, overall_quality, expressiveness
  - [ ] "Uncertain" checkbox per dimension
  - [ ] Text area for comments (optional)
  - [ ] Navigation: Previous/Next segment buttons
  - [ ] Progress tracker (X / 300 segments labeled)
  - [ ] Auto-save annotations to JSON every 10 segments
  - [ ] Load existing annotations (resume labeling)
- [ ] Test interface with 5 dummy segments
- [ ] Optimize for workflow speed (keyboard shortcuts?)
- [ ] Write instructions/guidelines document for self-reference

### 3.3 Personal Data Labeling

**Time**: 10-15 hours (spread over 1-2 weeks)

- [ ] **Prepare segments for labeling** (200-300 total):
  - Option A: Use existing recordings + MIDI scores
  - Option B: Record yourself playing + create/align MIDI
  - Segment into 20-30 second chunks
  - Ensure diversity:
    - 10-15 different pieces (various difficulty levels)
    - 3-4 composers (Baroque, Classical, Romantic, Contemporary)
    - Mix of technical quality (some flawed, some good)
- [ ] **Calibration** (1-2 hours):
  - Label 20 segments first
  - Re-label same 20 segments next day
  - Check self-consistency (correlation >0.7?)
  - Refine dimension definitions if needed
- [ ] **Main labeling** (8-12 hours):
  - Label 200-300 segments
  - ~5 minutes per segment (playback + rating)
  - Take breaks to avoid fatigue bias
  - Weekly schedule: 30-50 segments per session
- [ ] **Quality check** (1 hour):
  - Re-label 20 random segments (anchors)
  - Check consistency with original labels
  - Flag outliers or uncertain ratings
- [ ] Save annotations: `data/annotations/expert_labels.jsonl`

### 3.4 Dataset Creation

**Time**: 2-3 hours

- [ ] Create `src/data/dataset.py`:
  - [ ] `PerformanceDataset(torch.utils.data.Dataset)`:
    - Load annotations (pseudo or expert)
    - `__getitem__()`: Return `(cqt, midi_tokens, labels, metadata)`
    - Support on-the-fly CQT computation or pre-cached
    - Apply augmentation pipeline (training only)
    - Handle missing MIDI (fallback mode)
  - [ ] `create_dataloaders()`: Train/val/test splits
    - 70/15/15 split (stratified by piece, difficulty)
    - Batch collation (pad sequences to max length)
    - DataLoader with num_workers, persistent_workers
- [ ] Test dataset loading (sample 10 batches)
- [ ] Verify augmentation applied correctly
- [ ] Write unit test: `tests/test_dataset.py`

---

## Phase 4: Training Pipeline & Experiments

### 4.1 Training Configuration

**Time**: 2-3 hours

- [ ] Create `configs/pseudo_pretrain.yaml`:

  ```yaml
  data:
    train_path: data/maestro_pseudo_labels.jsonl
    val_split: 0.15
    batch_size: 8
    num_workers: 4
    augmentation:
      enabled: true
      pitch_shift: 0.3
      time_stretch: 0.3
      noise: 0.2
      # ... all augmentation configs

  model:
    audio_encoder: m-a-p/MERT-v1-95M
    midi_encoder: custom  # or HuggingFace path
    fusion_heads: 8
    lstm_layers: 2
    lstm_hidden: 256
    mtl_dimensions: 6  # Only technical for pseudo-labels
    gradient_checkpointing: true

  training:
    max_epochs: 20
    precision: 16
    optimizer:
      name: AdamW
      backbone_lr: 1e-5
      heads_lr: 1e-4
      weight_decay: 0.01
    scheduler:
      name: cosine
      warmup_steps: 500
      min_lr: 1e-6
    gradient_clip: 1.0
    accumulate_grad_batches: 4

  callbacks:
    checkpoint:
      monitor: val_loss
      mode: min
      save_top_k: 3
    early_stopping:
      monitor: val_loss
      patience: 5

  logging:
    log_every_n_steps: 50
    wandb_project: piano-eval-mvp  # optional
  ```

- [ ] Create `configs/expert_finetune.yaml`:
  - Similar to above but:
    - `train_path: data/annotations/expert_labels.jsonl`
    - `mtl_dimensions: 10` (all dimensions)
    - `max_epochs: 50`
    - `early_stopping.patience: 10`
    - `backbone_lr: 5e-6` (lower for fine-tuning)

### 4.2 Training Script

**Time**: 3-4 hours

- [ ] Create `train.py`:
  - [ ] Load config from YAML
  - [ ] Initialize model, dataloaders
  - [ ] Setup Lightning Trainer:
    - Callbacks: ModelCheckpoint, EarlyStopping, LearningRateMonitor
    - Logger: WandB
    - Gradient clipping, mixed precision
  - [ ] Support resuming from checkpoint: `--checkpoint path/to/ckpt`
  - [ ] Training loop: `trainer.fit(model, train_loader, val_loader)`
  - [ ] Save best checkpoint
- [ ] Test training script locally (1-2 epochs on CPU/M4)
- [ ] Verify checkpointing works (interrupt and resume)

### 4.3 Colab Training Notebook

**Time**: 2-3 hours

- [ ] Create `notebooks/train_on_colab.ipynb`:
  - [ ] Setup:
    - Clone repo or upload code
    - Install dependencies with uv
    - Mount Google Drive for checkpoints
    - Authenticate HuggingFace
  - [ ] Check GPU allocation: `nvidia-smi`
  - [ ] Run training:
    - Stage 2 (pseudo-labels): ~12 GPU hours
    - Stage 3 (expert labels): ~8 GPU hours
  - [ ] Monitor training:
    - Plot loss curves
    - Log GPU memory usage
    - Save checkpoints to Google Drive (backup)
  - [ ] Checkpoint management:
    - Auto-save every epoch
    - Keep top-3 by validation loss
    - Download best checkpoint
- [ ] Test on Colab free tier (verify memory limits)
- [ ] Document Colab Pro vs free differences

### 4.4 Stage 2: Pseudo-Label Pre-training

**Time**: 12 GPU hours + 1 hour monitoring

- [ ] Prepare data: Verify `maestro_pseudo_labels.jsonl` exists
- [ ] Launch training on Colab:

  ```bash
  python train.py --config configs/pseudo_pretrain.yaml
  ```

- [ ] Monitor training:
  - [ ] Check loss decreasing (expect ~20-30 → 10-15 over 20 epochs)
  - [ ] Validate on held-out MAESTRO segments
  - [ ] Check GPU memory usage (<14GB on T4)
  - [ ] Save checkpoints to Google Drive
- [ ] Evaluate pseudo-trained model:
  - [ ] Correlations with pseudo-labels (sanity check, should be high)
  - [ ] Attention visualizations (do they make sense?)
- [ ] Save best checkpoint: `checkpoints/pseudo_pretrain_best.ckpt`

### 4.5 Stage 3: Expert Fine-Tuning

**Time**: 8 GPU hours + 1 hour monitoring

- [ ] Prepare data: Verify `expert_labels.jsonl` exists (200-300 segments)
- [ ] Launch training on Colab:

  ```bash
  python train.py --config configs/expert_finetune.yaml \
                  --checkpoint checkpoints/pseudo_pretrain_best.ckpt
  ```

- [ ] Monitor training:
  - [ ] Check validation loss (expect overfitting after 20-30 epochs)
  - [ ] Early stopping should trigger around epoch 30-40
  - [ ] Monitor per-dimension losses (should be balanced via uncertainty weighting)
  - [ ] Check attention weights on validation set
- [ ] Save best checkpoint: `checkpoints/expert_finetune_best.ckpt`

---

## Phase 5: Evaluation & Documentation

### 5.1 Evaluation Script

**Time**: 3-4 hours

- [ ] Create `evaluate.py`:
  - [ ] Load trained model from checkpoint
  - [ ] Load test set (15% of expert labels)
  - [ ] Run inference on all test segments
  - [ ] Compute metrics:
    - [ ] **Per-dimension**:
      - Pearson r, Spearman ρ, Kendall τ
      - MAE, RMSE
      - Range Accuracy (α=0, 5, 10)
    - [ ] **Aggregate**:
      - Weighted average correlation
      - Overall MAE
    - [ ] **Calibration**:
      - Expected Calibration Error (ECE)
      - Reliability diagram
      - Sharpness (std of uncertainties)
    - [ ] **Baselines**:
      - Random (uniform 0-100)
      - Mean baseline (always predict training mean)
      - Linear regression (simple audio features)
  - [ ] Generate result tables (CSV, markdown)
  - [ ] Plot correlation scatter plots (predicted vs actual)
  - [ ] Plot attention heatmaps for 10 example segments
- [ ] Run evaluation:

  ```bash
  python evaluate.py --checkpoint checkpoints/expert_finetune_best.ckpt \
                     --split test \
                     --output results/
  ```

### 5.2 Ablation Studies

**Time**: 4-6 hours (mostly compute)

- [ ] Train ablation models (2-3 epochs each for speed):
  - [ ] **Audio-only**: No MIDI encoder, no fusion
  - [ ] **No pre-training**: Random MERT initialization
  - [ ] **No uncertainty weighting**: Equal task weights (1.0 each)
  - [ ] **No augmentation**: Train without data augmentation
- [ ] Compare performance:
  - [ ] Table: Correlation per dimension for each ablation
  - [ ] Expected: Multi-modal >15% better, pre-training >30% better
- [ ] Save ablation results: `results/ablations.csv`

### 5.3 Inference Demo

**Time**: 2-3 hours

- [ ] Create `inference.py`:
  - [ ] Load model from checkpoint
  - [ ] Accept audio + MIDI paths as arguments
  - [ ] Preprocess inputs (segment, compute CQT, encode MIDI)
  - [ ] Run inference
  - [ ] Output:
    - [ ] Per-dimension scores with uncertainties
    - [ ] Weighted aggregate score
    - [ ] Attention heatmap (save as PNG)
    - [ ] Temporal attention curve (which phrases are weak)
  - [ ] Pretty-print results to console
- [ ] Create `notebooks/demo_inference.ipynb`:
  - [ ] Interactive version with audio playback
  - [ ] Visualizations: waveform, attention overlay
  - [ ] Example: 3-5 test performances with predictions
- [ ] Test on unseen performances (if available)

### 5.4 Results Documentation

**Time**: 3-4 hours

- [ ] Create `results/RESULTS.md`:
  - [ ] **Executive Summary**:
    - MVP objectives met? (r>0.50 technical, r>0.35 interpretive)
    - Key findings
    - Limitations
  - [ ] **Performance Tables**:
    - Per-dimension metrics (Pearson r, MAE)
    - Comparison to baselines
    - Ablation study results
  - [ ] **Visualizations**:
    - Correlation scatter plots (predicted vs actual)
    - Loss curves (training + validation)
    - Attention heatmap examples
    - Reliability diagram (calibration)
  - [ ] **Failure Analysis**:
    - Which dimensions underperformed?
    - Which pieces/segments were hardest?
    - Error patterns (e.g., overestimate beginners, underestimate experts?)
  - [ ] **Next Steps**:
    - Recommended improvements
    - Scaling plan (2,500 labels, MERT-330M, ensemble)
- [ ] Generate all plots and tables
- [ ] Write narrative analysis

### 5.5 Hiring Pitch Document

**Time**: 2-3 hours

- [ ] Create `docs/ANNOTATOR_HIRING_PITCH.md`:
  - [ ] **Project Overview**: What we're building and why
  - [ ] **MVP Results**: Proof that architecture works
    - Show performance metrics
    - Demo predictions vs expert labels
    - Highlight multi-modal gains
  - [ ] **Scaling Plan**: What 2,500 labels will enable
    - Target performance: r=0.68-0.75 (professional-grade)
    - MERT-330M + ensemble specialists
    - Active learning to minimize labeling waste
  - [ ] **Annotation Protocol**:
    - Dimensions and definitions
    - Example segments with reference ratings
    - Expected time: 3-5 min per segment
    - Compensation: $30-50/hour
  - [ ] **Quality Assurance**:
    - Calibration training
    - Anchor segments for monitoring
    - Weekly check-ins
  - [ ] **Timeline**: 4-6 months for 2,500 segments
  - [ ] **Budget**: $75-100K for expert annotations
- [ ] Include demo notebook link
- [ ] Create 1-2 slide deck summary

### 5.6 Final Documentation Review

**Time**: 2 hours

- [ ] Update `README.md` with actual results
- [ ] Update `ARCHITECTURE.md` if design changed during implementation
- [ ] Update `CLAUDE.md` with lessons learned
- [ ] Verify all scripts have usage examples and docstrings
- [ ] Clean up notebooks (remove debug cells)
- [ ] Update `.gitignore` (exclude large files)
- [ ] Tag git commit: `v0.1.0-mvp`

---

## Success Criteria

### Minimum Viable (Must Achieve)

- [ ] **Technical dimensions**: Pearson r > 0.45, MAE < 18 points
- [ ] **Interpretive dimensions**: Pearson r > 0.30, MAE < 20 points
- [ ] **Baseline improvement**: >25% better than linear regression
- [ ] **Multi-modal gain**: >10% improvement over audio-only
- [ ] **Calibration**: ECE < 0.20

### Target Performance (Ideal)

- [ ] **Technical dimensions**: Pearson r > 0.50, MAE < 15 points
- [ ] **Interpretive dimensions**: Pearson r > 0.35, MAE < 18 points
- [ ] **Baseline improvement**: >40% better than linear regression
- [ ] **Multi-modal gain**: >15% improvement over audio-only
- [ ] **Calibration**: ECE < 0.15

### Failure Conditions (Re-evaluate Approach)

- Technical dimensions r < 0.35 (no better than simple heuristics)
- Interpretive dimensions r < 0.15 (essentially random)
- Calibration ECE > 0.30 (predictions unreliable)
- Multi-modal worse than audio-only (fusion not working)

---

## Risk Mitigation Tasks

### If Labels Are Insufficient (r < 0.40 after Stage 3)

- [ ] Generate more pseudo-labels (increase MAESTRO to 100-200 pieces)
- [ ] Extend pseudo-label pre-training (40 epochs instead of 20)
- [ ] Reduce dimensions (drop 2-3 hardest interpretive dimensions)
- [ ] Increase data augmentation (more aggressive transforms)
- [ ] Label 50-100 more segments (total 250-400)

### If Colab Sessions Time Out

- [ ] Implement auto-resume from checkpoint
- [ ] Reduce batch size / segment length (fit in shorter sessions)
- [ ] Rent 4-8 hours of A100 time (~$30-60) for critical training
- [ ] Pre-cache CQT spectrograms (faster epoch times)

### If Memory Issues on Colab

- [ ] Enable gradient checkpointing (already planned)
- [ ] Reduce batch size to 2-4
- [ ] Use 5-8 second segments instead of 10 seconds
- [ ] Clear cache between epochs: `torch.cuda.empty_cache()`
- [ ] Use Colab Pro+ (more RAM) or rent A100

### If Multi-Modal Fusion Underperforms

- [ ] Debug MIDI alignment (visual inspection)
- [ ] Try simpler fusion (concatenation instead of cross-attention)
- [ ] Pre-train MIDI encoder separately
- [ ] Use automatic transcription instead of MIDI (if alignment poor)
- [ ] Accept audio-only performance (still useful)

---

## Optional Enhancements (If Time Permits)

### Advanced Visualizations

- [ ] Interactive Gradio or Streamlit demo
- [ ] Side-by-side score-audio alignment visualization
- [ ] Temporal heatmaps over full piece waveform

### Additional Metrics

- [ ] Pairwise ranking accuracy (which performance is better?)
- [ ] Piece-level aggregation (average segment predictions)
- [ ] Performer-out cross-validation (if multiple pianists)

### Model Improvements

- [ ] Attention mechanisms: Try different fusion strategies
- [ ] Loss functions: Ordinal regression (CORAL loss) for ranking
- [ ] Ensemble: Train 2-3 variants with different seeds (if budget allows)

### Infrastructure

- [ ] Containerize with Docker (reproducibility)
- [ ] CI/CD: Automated testing on GitHub Actions
- [ ] Model versioning: MLflow or DVC

---

## Estimated Timeline (Part-Time, 15-20 hours/week)

| Week | Phase | Tasks | Hours |
|------|-------|-------|-------|
| 1 | Setup & Data Infra | 1.1-1.5 | 15-20 |
| 2 | Model Architecture | 2.1-2.7 | 18-24 |
| 3 | Pseudo-labels | 3.1 + start 3.2 | 10-15 |
| 4 | Labeling Interface + Start Personal Labeling | 3.2-3.3 | 15-20 |
| 5 | Finish Labeling + Dataset | 3.3-3.4 + 4.1-4.2 | 15-20 |
| 6 | Training Experiments | 4.3-4.5 | 20-25 (mostly compute) |
| 7 | Evaluation & Docs | 5.1-5.6 | 18-24 |

**Total**: 6-8 weeks part-time, 110-150 hours human effort + 20-25 GPU hours

---

## Deliverables Checklist

- [ ] Working PyTorch Lightning codebase
- [ ] 200-300 labeled performance segments with 8-10 dimensions
- [ ] Trained model checkpoint (`expert_finetune_best.ckpt`)
- [ ] Evaluation results (`RESULTS.md` with tables and plots)
- [ ] Demo inference notebook
- [ ] Annotator hiring pitch document
- [ ] Complete documentation (README, ARCHITECTURE, CLAUDE, TASKS)
- [ ] Unit tests for core components
- [ ] Clear scaling roadmap to 2,500 labels + MERT-330M

---

## Post-MVP: Scaling to Production

### Phase 6: Hiring & Large-Scale Annotation (4-6 months)

- Recruit 10 expert annotators (conservatory students, professional pianists)
- Set up annotation infrastructure (web platform)
- Run 2-hour calibration sessions
- Iterative active learning (8 rounds × 200 segments)
- Total: 2,500 expert-labeled segments

### Phase 7: Full System Training (2-3 months)

- Scale to MERT-330M (requires A100 GPUs)
- Implement Music Transformer hierarchical aggregation
- Train ensemble specialists (3-5 variants per interpretive dimension)
- Temperature scaling calibration
- Total compute: 1,500 GPU-hours (~$3,500-5,000)

### Phase 8: Production Deployment (1-2 months)

- Model distillation (50-100M student for inference)
- Cloud API (FastAPI serving)
- Web interface integration
- Mobile optimization (INT8 quantization)
- Monitoring and feedback collection

**Target Performance**: r=0.68-0.75, MAE <10 points, ECE <0.10
