# Data Reference

Complete inventory of data required for piano performance evaluation training. All heavy data processing (audio download, segmentation, MuQ extraction) runs on Thunder Compute (A100, 500GB storage). Only embeddings and metadata come back.

## Storage Strategy

| Location | Capacity | Purpose |
|---|---|---|
| Local (Mac) | 50GB free | Code, existing caches, composite labels |
| GDrive | 80GB total | Checkpoints, results, final embeddings |
| Thunder Compute | 500GB | Raw audio download, processing, MuQ extraction |

**Principle:** Raw audio lives and dies on the remote. Only `.pt` embeddings (~380KB each) and `.jsonl` metadata return.

## What Training Loads

The training code (`src/model_improvement/data.py`) loads only embeddings and metadata -- never raw audio. At training time, every tier must resolve to:

- MuQ embeddings: `{cache_dir}/muq_embeddings/{segment_id}.pt` (93x1024 float32, ~380KB)
- Segment metadata: `{cache_dir}/metadata.jsonl`
- Tier-specific signals: labels, placements, contrastive mappings, augmented pairs

## T1: PercePiano (Labeled Regression + Ranking)

**Status: READY (on disk)**

| Item | Path | Size |
|---|---|---|
| MuQ embeddings | `percepiano_cache/muq_embeddings.pt` + `_muq_file_cache/` | 4.4GB |
| Labels | `percepiano_cache/labels.json` | - |
| CV folds | `percepiano_cache/folds.json` | - |
| Piece mapping | `percepiano_cache/piece_mapping.json` | - |

**Segments:** 1,202 with 19-dimension perceptual annotations.
**Signal:** Regression (MSE on composite labels), ranking (within-piece pairs).
**Composite labels** (`composite_labels/`) are produced by the taxonomy work (doc 02) and map the 19 raw dims to N teacher-grounded dims.

**Training class:** `CompetitionPairSampler` generates within-piece ranking pairs from T1 data.

## T2: Competition Recordings (Ordinal Ranking)

**Status: CODE READY, data not collected**

| Item | Path | Est. Size |
|---|---|---|
| Segment metadata | `competition_cache/chopin2021/metadata.jsonl` | <1MB |
| MuQ embeddings | `competition_cache/chopin2021/muq_embeddings/` | ~760MB |
| Teacher labels (conditional) | `competition_cache/chopin2021/teacher_labels.json` | <1MB |

**Source:** XVIII International Chopin Piano Competition 2021 (YouTube, free).
**Segments:** ~2,000 from ~100 performers across Stage 2, 3, Final.
**Signal:** Ordinal placement within each round (1st > 2nd > ... > eliminated).
**Raw audio (remote only):** ~50GB WAV at 24kHz mono.

**metadata.jsonl schema:**
```json
{
  "segment_id": "chopin2021_performer_seg001",
  "recording_id": "chopin2021_performer",
  "performer": "Bruce Liu",
  "piece": "Ballade No. 4",
  "round": "final",
  "placement": 1,
  "competition": "chopin",
  "edition": "2021",
  "country": "CA",
  "source_url": "https://youtube.com/...",
  "segment_start": 0.0,
  "segment_end": 30.0
}
```

**Collection script:** `scripts/collect_competition_data.py`
**Training classes:** `CompetitionDataset`, `CompetitionPairSampler` (cross-round ordinal pairs).

### Remote execution plan

```bash
# On Thunder Compute (A100)
cd crescendai/model
uv run python scripts/collect_competition_data.py

# Downloads audio -> segments -> extracts MuQ -> writes metadata.jsonl
# Output: competition_cache/chopin2021/{metadata.jsonl, muq_embeddings/*.pt}
# Copy embeddings + metadata back (~760MB)
```

## T3: MAESTRO Audio (Contrastive Learning)

**Status: CODE READY, needs audio download**

| Item | Path | Est. Size |
|---|---|---|
| MAESTRO metadata | `maestro_cache/maestro-v3.0.0.json` | 83MB (on disk) |
| MIDI files | `maestro_cache/2013-2018/` | on disk |
| Segment metadata | `maestro_cache/metadata.jsonl` | <1MB |
| MuQ embeddings | `maestro_cache/muq_embeddings/` | ~3.8GB |
| Contrastive mapping | `maestro_cache/contrastive_mapping.json` | <1MB |
| Teacher labels (conditional) | `maestro_cache/teacher_labels.json` | <1MB |

**Source:** MAESTRO v3.0.0 audio (magenta.tensorflow.org/datasets/maestro).
**Segments:** ~10,000+ from 1,276 recordings across ~300 pieces, ~150 with 2+ performers.
**Signal:** Contrastive pairs (same piece, different performer = positive; different piece = negative). InfoNCE loss.
**Raw audio (remote only):** ~200GB WAV.

**Collection script:** `scripts/collect_maestro_audio.py`
**Training class:** `PairedPerformanceDataset` (contrastive pairs from `contrastive_mapping.json`).

### Remote execution plan

```bash
# On Thunder Compute (A100)
cd crescendai/model

# 1. Download MAESTRO v3 audio (~200GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip -d data/maestro_raw/

# 2. Process: segment + extract MuQ embeddings
uv run python scripts/collect_maestro_audio.py --maestro-dir data/maestro_raw/maestro-v3.0.0

# Output: maestro_cache/{metadata.jsonl, muq_embeddings/*.pt, contrastive_mapping.json}
# Copy embeddings + metadata + mapping back (~3.8GB)
# Discard raw audio
```

## T4: YouTube Piano (Augmentation Invariance)

**Status: CODE READY, lower priority**

| Item | Path | Est. Size |
|---|---|---|
| Channel list | `youtube_piano_cache/channels.yaml` | on disk (22 channels) |
| Recordings metadata | `youtube_piano_cache/recordings.jsonl` | <1MB |
| Segment metadata | `youtube_piano_cache/metadata.jsonl` | <5MB |
| Clean MuQ embeddings | `youtube_piano_cache/muq_embeddings/` | ~19GB |
| Augmented MuQ embeddings | `youtube_piano_cache/muq_embeddings_augmented/` | ~19GB |
| Teacher labels (conditional) | `youtube_piano_cache/teacher_labels.json` | <1MB |

**Source:** 22 curated YouTube channels (recital, conservatory, competition).
**Segments:** ~50,000 from 5,000-10,000 recordings.
**Signal:** Invariance -- MSE between clean and augmented MuQ embeddings. Augmentation chain: reverb, compression, low-pass, pitch shift, parametric EQ (3-band PeakFilter), pink noise mixing (10-30 dB SNR).
**Raw audio (remote only):** ~100GB+ WAV.

**Priority:** Add only if robustness metrics fall below target (augmented pairwise accuracy drop > 10% or cross-condition Pearson r < 0.9). Start experiments with T1+T2+T3 first.

**Collection script:** `scripts/collect_youtube_piano.py`
**Training class:** `AugmentedEmbeddingDataset` (clean + augmented embedding pairs).

### Remote execution plan

```bash
# On Thunder Compute (A100) -- only if needed
cd crescendai/model
uv run python scripts/collect_youtube_piano.py

# Output: youtube_piano_cache/{recordings.jsonl, metadata.jsonl,
#          muq_embeddings/*.pt, muq_embeddings_augmented/*.pt}
# Copy embeddings + metadata back (~38GB)
# Discard raw audio
```

## Symbolic Pretraining Corpus

**Status: READY (on disk)**

| Item | Path | Size |
|---|---|---|
| Tokenized MIDI | `pretrain_cache/tokens/all_tokens.pt` | 55MB |
| Score graphs | `pretrain_cache/graphs/all_graphs.pt` + shards | 29GB |
| Hetero graphs | `pretrain_cache/graphs/all_hetero_graphs.pt` + shards | (in 29GB) |
| Continuous features | `pretrain_cache/features/all_features.pt` + shards | 8.3GB |

**Source:** 14K+ MIDI files from MAESTRO + ATEPP + ASAP.
**Training classes:** `MIDIPretrainingDataset`, `ScoreGraphPretrainingDataset`, `ContinuousPretrainDataset`, `HeteroPretrainDataset`.

## Composite Labels (from Taxonomy Work)

**Status: NOT YET PRODUCED (doc 02 prerequisite)**

| Item | Path | Est. Size |
|---|---|---|
| Dimension definitions | `composite_labels/taxonomy.json` | <100KB |
| Mapped labels | `composite_labels/labels.json` | <1MB |
| Teacher quote bank | `composite_labels/quote_bank.json` | <1MB |

Produced by the teacher-grounded taxonomy work (doc 02). Maps the 19 raw PercePiano dimensions to N teacher-grounded dimensions (expected 5-8). Training cannot start without these.

## LLM Distillation (Conditional)

If the distillation pilot (during taxonomy work) returns **go**, score T2 segments with the LLM teacher to test whether teacher labels correlate with competition placement.

| Step | Segments | Cost | Validation |
|---|---|---|---|
| T2 scoring | ~2,000 | ~$60 | Spearman rho > 0.2 vs placement on 60%+ dims |
| T3 scoring (if T2 passes) | ~10,000 | ~$300 | None (no external signal) |
| T4 scoring (if T2 passes) | ~50,000 | ~$1,500 | None |

**Confidence weights for training:** T1=1.0, T2=0.6, T3=0.4, T4=0.3.

## Storage Budget

### What returns from Thunder Compute

| Tier | Artifact | Est. Size |
|---|---|---|
| T2 | embeddings + metadata | ~760MB |
| T3 | embeddings + metadata + mapping | ~3.8GB |
| T4 (if needed) | clean + augmented embeddings + metadata | ~38GB |
| **Total (T2+T3)** | | **~4.6GB** |
| **Total (T2+T3+T4)** | | **~42.6GB** |

### Local disk after collection (T2+T3 only)

```
model/data/                     current    + T2/T3
  pretrain_cache/               38 GB      38 GB
  masterclass_pipeline/         5.9 GB     5.9 GB
  percepiano_cache/             4.4 GB     4.4 GB
  competition_cache/            0          0.8 GB
  maestro_cache/                83 MB      3.9 GB
  masterclass_cache/            907 MB     907 MB
  atepp_cache/                  525 MB     525 MB
  other                         220 MB     220 MB
  TOTAL                         ~50 GB     ~54.6 GB
```

T4 embeddings (38GB) should go to GDrive or stay on Thunder Compute if local disk is tight.

## Readiness Checklist

```
[x] T1 PercePiano embeddings + labels
[x] Symbolic pretraining corpus (tokens, graphs, features)
[ ] Composite labels (blocked on doc 02 taxonomy work)
[ ] T2 competition embeddings (run on Thunder Compute)
[ ] T3 MAESTRO embeddings (run on Thunder Compute)
[ ] T4 YouTube embeddings (lower priority, run if needed)
```

**Minimum viable for first training run:** T1 + composite labels + symbolic corpus.
**Full multi-task training:** T1 + T2 + T3 + composite labels + symbolic corpus.
**With robustness:** Add T4.
