# Data Collection Design

## Goal

Collect the data tiers (T2-T4) that the original model improvement design planned but never implemented. The audio training pipeline only uses PercePiano (T1: 1,202 segments). The symbolic side has a 14K+ pretraining corpus; the audio side has nothing comparable.

This work can run in parallel with the teacher-grounded taxonomy effort. The post-taxonomy model improvement plan depends on this being complete.

## Non-Goals

- Changing any model architecture or training code
- Running any experiments

## Data Directory Layout

```
model/data/
  percepiano_cache/            # T1: MuQ embeddings + labels + folds
  percepiano_midi/             # T1: MIDI files for symbolic finetuning
  pretrain_cache/              # Symbolic pretraining: tokens, graphs, features
    tokens/all_tokens.pt
    graphs/all_graphs.pt
    graphs/all_hetero_graphs.pt
    features/all_features.pt
  asap_cache/                  # Raw MIDI source for pretraining
  maestro_cache/               # Raw MIDI + NEW: MuQ embeddings (extend)
  atepp_cache/                 # Raw MIDI source for pretraining
  masterclass_pipeline/        # Pipeline raw outputs
    sources.yaml
    videos.jsonl
    audio/
    transcripts/
    segments/
    teaching_moments/
    state/
  masterclass_cache/           # ML-ready masterclass data
    segments/
    muq_embeddings/
    quality_scores/
  competition_cache/           # NEW: rebuilt properly (T2 collection)
    chopin2021/
      metadata.jsonl
      muq_embeddings/
  youtube_piano_cache/         # NEW: unlabeled piano audio (T4 collection)
    metadata.jsonl
    muq_embeddings/
  composite_labels/            # NEW: output from taxonomy bridge (post-taxonomy)
    taxonomy.json
    labels.json
    quote_bank.json
  checkpoints/                 # Model checkpoints
```

## T2: Competition Recordings (Ordinal Ranking Signal)

**Why:** PercePiano crowdworker labels are noisy and poorly calibrated for interpretive dimensions. Competition placements provide an independent, expert-validated ranking signal. The audit showed "interpretation" had the strongest competition correlation (rho=-0.341, p=0.052) despite the weak overall signal -- suggesting competition data helps exactly where PercePiano is weakest.

**Source:** International Chopin Piano Competition 2021 -- all performances freely available on YouTube with published round-by-round results.

**Pipeline:**

1. Scrape metadata from chopin2021.pl (performer name, piece, round, placement)
2. Match to YouTube URLs (competition channel publishes all performances)
3. Download audio via yt-dlp, resample to 24kHz mono WAV using Pedalboard (4x faster I/O than librosa)
4. Segment into 30s clips (matching PercePiano segment length)
5. Extract MuQ embeddings per segment using the existing `extract_percepiano_muq.py` pattern
6. Store metadata as `competition_cache/chopin2021/metadata.jsonl` with schema: `{recording_id, performer, piece, round, placement, segment_start, segment_end}`

**Ordinal signal:** Placement encodes a strict ordering within each round. For training: 1st place > 2nd > 3rd > semifinalist > first round eliminated. Cross-round comparisons (finalist vs first-round) are noisier but still directional.

**Estimated yield:** ~100 performers across 3 rounds, ~2,000 segments.

**Optional extension:** Add Cliburn, Leeds, or Van Cliburn Junior competitions if more data is needed. Same pipeline, different metadata source.

## T3: Cross-Performer Audio Embeddings (Contrastive Signal)

**Why:** The symbolic encoders pretrain on a 14K+ MIDI corpus from MAESTRO + ATEPP + ASAP. The audio encoders have no equivalent -- they only see PercePiano's 1,202 segments. Cross-performer contrastive learning (same piece, different performers -> positive pairs) gives the audio encoder a sense of what varies between performances of the same piece, without requiring any labels.

**Source:** MAESTRO v3 audio. We already have the MIDI and metadata; we need the audio files and MuQ embeddings.

**Pipeline:**

1. Download MAESTRO v3 audio (~200GB WAV) on Thunder Compute
2. Resample and segment each recording into 30s clips with piece and performer metadata using Pedalboard (4x faster I/O than librosa, critical at 200GB scale)
3. Extract MuQ embeddings per segment (batch processing on A100)
4. Cache embeddings to GDrive: `maestro_cache/muq_embeddings/`
5. Build piece-to-performer mapping for contrastive pair generation

**Contrastive signal:** For each piece with 2+ performers, generate positive pairs (same piece, different performer) and negative pairs (different pieces). InfoNCE loss on MuQ embeddings teaches the model what's shared (the piece) vs what varies (the performance quality).

**Estimated yield:** ~1,276 recordings across ~300 pieces, ~10,000+ segments. ~150 pieces have multiple performers.

## T4: Unlabeled Piano Audio at Scale (Augmentation Invariance Signal)

**Why:** Robustness. The model must work on phone recordings, noisy environments, different pianos, and different acoustics. Augmentation invariance training (clean embedding should match augmented embedding) requires a large corpus of diverse piano audio.

**Source:** YouTube piano channels with professional recitals and conservatory uploads.

**Pipeline:**

1. Curate list of high-quality piano channels (20-30 channels)
2. Download audio via yt-dlp, resample to 24kHz mono WAV using Pedalboard
3. Segment into 30s clips
4. Extract MuQ embeddings (clean)
5. Generate augmented versions using AudioAugmentor backed by Pedalboard (room IR convolution, noise mixing, low-pass + compression for phone sim, pitch shift, parametric EQ -- up to 300x faster than torchaudio for effect chains)
6. Extract MuQ embeddings (augmented)
7. Store clean + augmented embedding pairs

**Invariance signal:** MSE between clean and augmented embeddings. The model learns that recording conditions don't change musical quality.

**Estimated yield:** 5,000-10,000 recordings, ~50,000 segments.

**Priority:** Lower than T2 and T3. Start experiments with T1+T2+T3 and add T4 if robustness metrics are below target (augmented pairwise accuracy drop > 10% or cross-condition Pearson r < 0.9).

## LLM Distillation Scaling (Conditional)

If the distillation pilot (run during taxonomy work) returns **go**, score T2 segments with the LLM teacher before model training begins. This is the decisive experiment: T2 has an independent validation signal (competition placement) that lets us measure whether teacher labels actually help.

**T2 scoring protocol:**

1. Score all ~2,000 T2 segments using the calibrated teacher rubric (~$60)
2. Store as `competition_cache/chopin2021/teacher_labels.json` with schema: `{segment_id, dimension_scores: {dim: score}, teacher_model, rubric_version}`
3. Validate: within each round, teacher-scored quality should correlate with placement (Spearman rho > 0.2 on at least 60% of dimensions)

**If T2 validation passes** -- proceed to score T3 and T4:

- T3 (~10,000 segments, ~$300): store as `maestro_cache/teacher_labels.json`
- T4 (~50,000 segments, ~$1,500): store as `youtube_piano_cache/teacher_labels.json`
- Total distillation cost: ~$1,860

**If T2 validation fails** -- teacher labels don't correlate with competition placement. Stop. Use the original plan (bridge-only labels for T1, ordinal for T2, contrastive for T3, invariance for T4). Cost sunk: ~$100 total ($40 pilot + $60 T2).

**Confidence weights** (used in model training):

```
T1 (PercePiano composite labels):  1.0  -- human-annotated ground truth
T2 (teacher-scored + placement):   0.6  -- validated against ordinal signal
T3 (teacher-scored, no external):  0.4  -- no independent validation
T4 (teacher-scored, no external):  0.3  -- lowest confidence, most diverse
```

## Data Dependency Summary

```
T1 (PercePiano)    -- exists, 1,202 segments with labels + MuQ embeddings
T2 (Competition)   -- collect: ~2,000 segments with ordinal placement
T3 (MAESTRO audio) -- collect: ~10,000+ segments with piece-performer metadata
T4 (YouTube piano) -- collect: ~50,000 segments (lower priority)
```

Without distillation, the multi-task objective is:

```
L_total = L_regression(T1 composite) + lambda_rank * L_ranking(T1+T2)
        + lambda_contrastive * L_contrastive(T3) + lambda_invariance * L_invariance(T4)
```

With distillation (if pilot and T2 validation pass):

```
L_total = L_regression(T1+T2+T3+T4, confidence-weighted) + lambda_rank * L_ranking(T1+T2)
        + lambda_contrastive * L_contrastive(T3) + lambda_invariance * L_invariance(T4)
```

The regression loss expands from 1,202 to 60,000+ segments. All other loss terms remain unchanged.

## Validation

Before marking this work complete:

1. T2 competition metadata is scraped and MuQ embeddings are extracted
2. T3 MAESTRO MuQ embeddings are extracted and piece-performer mapping is built
3. If distillation pilot passed: T2 teacher labels scored and validated against placement

T4 is not a gate for completion -- it's additive and can be collected later. T3/T4 teacher scoring (if applicable) can happen in parallel with early model training.
