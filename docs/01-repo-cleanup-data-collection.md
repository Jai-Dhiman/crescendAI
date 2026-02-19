# Repo Cleanup & Data Collection Design

## Goal

Prepare the model/ directory for the next round of experiments by:

1. Consolidating the masterclass pipeline into model/
2. Removing stale data and old plan documents
3. Collecting the data tiers (T2-T4) that the original model improvement design planned but never implemented

This work can run in parallel with the teacher-grounded taxonomy effort. The post-taxonomy model improvement plan depends on this being complete.

## Non-Goals

- Changing any model architecture or training code
- Running any experiments
- Modifying notebook content (just reorganizing)

## Current Problems

1. The masterclass pipeline lives in `tools/masterclass-pipeline/` but its data is central to model training. It should be under `model/`.
2. `model/data/` contains stale directories that are no longer needed (rendered audio, weak competition data).
3. The audio training pipeline only uses PercePiano (T1: 1,202 segments). The original design planned four data tiers but T2-T4 were never collected. The symbolic side has a 14K+ pretraining corpus; the audio side has nothing comparable.
4. Old plan documents (`2026-02-15-model-improvement-*.md`) describe a plan that's being superseded.

## Part 1: Repo Reorganization

### Move Masterclass Pipeline

```
tools/masterclass-pipeline/src/       -> model/tools/masterclass-pipeline/src/
tools/masterclass-pipeline/Cargo.toml -> model/tools/masterclass-pipeline/Cargo.toml
tools/masterclass-pipeline/.claude/   -> model/tools/masterclass-pipeline/.claude/
```

Pipeline data consolidates into model/data/:

```
tools/masterclass-pipeline/data/sources.yaml         -> model/data/masterclass_pipeline/sources.yaml
tools/masterclass-pipeline/data/videos.jsonl          -> model/data/masterclass_pipeline/videos.jsonl
tools/masterclass-pipeline/data/audio/                -> model/data/masterclass_pipeline/audio/
tools/masterclass-pipeline/data/transcripts/          -> model/data/masterclass_pipeline/transcripts/
tools/masterclass-pipeline/data/segments/             -> model/data/masterclass_pipeline/segments/
tools/masterclass-pipeline/data/teaching_moments/     -> model/data/masterclass_pipeline/teaching_moments/
tools/masterclass-pipeline/data/state/                -> model/data/masterclass_pipeline/state/
```

After move, update the pipeline's default data directory. The pipeline should accept `--data-dir` pointing to `model/data/masterclass_pipeline/`. Delete `tools/` after verifying the move.

### Data Directory Layout (after cleanup)

```
model/data/
  percepiano_cache/            # T1: MuQ embeddings + labels + folds (keep)
  percepiano_midi/             # T1: MIDI files for symbolic finetuning (keep)
  pretrain_cache/              # Symbolic pretraining: tokens, graphs, features (keep)
    tokens/all_tokens.pt
    graphs/all_graphs.pt
    graphs/all_hetero_graphs.pt
    features/all_features.pt
  asap_cache/                  # Raw MIDI source for pretraining (keep)
  maestro_cache/               # Raw MIDI + NEW: MuQ embeddings (keep, extend)
  atepp_cache/                 # Raw MIDI source for pretraining (keep)
  masterclass_pipeline/        # NEW: pipeline raw outputs (moved from tools/)
    sources.yaml
    videos.jsonl
    audio/                     # Full video WAV files (~5.9 GB)
    transcripts/
    segments/
    teaching_moments/
    state/
  masterclass_cache/           # ML-ready masterclass data (existing, keep)
    segments/                  # Short audio clips for STOP/CONTINUE
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
  checkpoints/                 # Model checkpoints (keep)
```

### Remove Stale Data

- Delete `model/data/percepiano_pianoteq_rendered/` -- only used for MuQ embedding extraction, which is already cached in `percepiano_cache/muq_embeddings.pt`
- Delete `model/data/competition_cache/` (current) -- contains incomplete Chopin 2021 data from the audit; will be rebuilt properly in Part 2

### Remove Old Plan Documents

After the new docs are written and committed:

- Delete `docs/plans/2026-02-15-model-improvement-design.md`
- Delete `docs/plans/2026-02-15-model-improvement-implementation.md`

## Part 2: Data Collection (T2-T4)

### T2: Competition Recordings (Ordinal Ranking Signal)

**Why:** PercePiano crowdworker labels are noisy and poorly calibrated for interpretive dimensions. Competition placements provide an independent, expert-validated ranking signal. The audit showed "interpretation" had the strongest competition correlation (rho=-0.341, p=0.052) despite the weak overall signal -- suggesting competition data helps exactly where PercePiano is weakest.

**Source:** International Chopin Piano Competition 2021 -- all performances freely available on YouTube with published round-by-round results.

**Pipeline:**

1. Scrape metadata from chopin2021.pl (performer name, piece, round, placement)
2. Match to YouTube URLs (competition channel publishes all performances)
3. Download audio via yt-dlp, convert to 24kHz mono WAV
4. Segment into 30s clips (matching PercePiano segment length)
5. Extract MuQ embeddings per segment using the existing `extract_percepiano_muq.py` pattern
6. Store metadata as `competition_cache/chopin2021/metadata.jsonl` with schema: `{recording_id, performer, piece, round, placement, segment_start, segment_end}`

**Ordinal signal:** Placement encodes a strict ordering within each round. For training: 1st place > 2nd > 3rd > semifinalist > first round eliminated. Cross-round comparisons (finalist vs first-round) are noisier but still directional.

**Estimated yield:** ~100 performers across 3 rounds, ~2,000 segments.

**Optional extension:** Add Cliburn, Leeds, or Van Cliburn Junior competitions if more data is needed. Same pipeline, different metadata source.

### T3: Cross-Performer Audio Embeddings (Contrastive Signal)

**Why:** The symbolic encoders pretrain on a 14K+ MIDI corpus from MAESTRO + ATEPP + ASAP. The audio encoders have no equivalent -- they only see PercePiano's 1,202 segments. Cross-performer contrastive learning (same piece, different performers -> positive pairs) gives the audio encoder a sense of what varies between performances of the same piece, without requiring any labels.

**Source:** MAESTRO v3 audio. We already have the MIDI and metadata; we need the audio files and MuQ embeddings.

**Pipeline:**

1. Download MAESTRO v3 audio (~200GB WAV) on Thunder Compute
2. Segment each recording into 30s clips with piece and performer metadata
3. Extract MuQ embeddings per segment (batch processing on A100)
4. Cache embeddings to GDrive: `maestro_cache/muq_embeddings/`
5. Build piece-to-performer mapping for contrastive pair generation

**Contrastive signal:** For each piece with 2+ performers, generate positive pairs (same piece, different performer) and negative pairs (different pieces). InfoNCE loss on MuQ embeddings teaches the model what's shared (the piece) vs what varies (the performance quality).

**Estimated yield:** ~1,276 recordings across ~300 pieces, ~10,000+ segments. ~150 pieces have multiple performers.

### T4: Unlabeled Piano Audio at Scale (Augmentation Invariance Signal)

**Why:** Robustness. The model must work on phone recordings, noisy environments, different pianos, and different acoustics. Augmentation invariance training (clean embedding should match augmented embedding) requires a large corpus of diverse piano audio.

**Source:** YouTube piano channels with professional recitals and conservatory uploads.

**Pipeline:**

1. Curate list of high-quality piano channels (20-30 channels)
2. Download audio via yt-dlp
3. Segment into 30s clips
4. Extract MuQ embeddings (clean)
5. Generate augmented versions using AudioAugmentor (room IR, noise, phone sim, pitch shift, EQ)
6. Extract MuQ embeddings (augmented)
7. Store clean + augmented embedding pairs

**Invariance signal:** MSE between clean and augmented embeddings. The model learns that recording conditions don't change musical quality.

**Estimated yield:** 5,000-10,000 recordings, ~50,000 segments.

**Priority:** Lower than T2 and T3. Start experiments with T1+T2+T3 and add T4 if robustness metrics are below target (augmented pairwise accuracy drop > 10% or cross-condition Pearson r < 0.9).

## Data Dependency Summary

```
T1 (PercePiano)    -- exists, 1,202 segments with labels + MuQ embeddings
T2 (Competition)   -- collect: ~2,000 segments with ordinal placement
T3 (MAESTRO audio) -- collect: ~10,000+ segments with piece-performer metadata
T4 (YouTube piano) -- collect: ~50,000 segments (lower priority)
```

The post-taxonomy model improvement plan will use all four tiers with the multi-task objective:

```
L_total = L_regression(T1 composite) + lambda_rank * L_ranking(T1+T2)
        + lambda_contrastive * L_contrastive(T3) + lambda_invariance * L_invariance(T4)
```

## Validation

Before marking this work complete:

1. Pipeline builds and runs from its new location (`model/tools/masterclass-pipeline/`)
2. All data directories are correctly organized per the layout above
3. T2 competition metadata is scraped and MuQ embeddings are extracted
4. T3 MAESTRO MuQ embeddings are extracted and piece-performer mapping is built
5. Stale data directories are removed
6. Old plan documents are removed

T4 is not a gate for completion -- it's additive and can be collected later.
