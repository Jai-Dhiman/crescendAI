# Data Inventory

Dataset inventory for piano performance evaluation training. All heavy data processing (audio download, segmentation, MuQ extraction) runs on Thunder Compute (A100). Only embeddings and metadata come back.

> **Status (2026-03-14):** T1 COMPLETE, T2 COMPLETE, T3 COMPLETE, T4 DEFERRED. Symbolic pretrain corpus COMPLETE (24,220 graphs). Composite labels COMPLETE.

## What Training Loads

The training code (`src/model_improvement/data.py`) loads only embeddings and metadata -- never raw audio. At training time, every tier must resolve to:

- MuQ embeddings: `{cache_dir}/muq_embeddings/{segment_id}.pt` (93x1024 float32, ~380KB)
- Segment metadata: `{cache_dir}/metadata.jsonl`
- Tier-specific signals: labels, placements, contrastive mappings, augmented pairs

## Datasets

### T1: PercePiano (Labeled Regression + Ranking)

| Item | Path | Size |
|---|---|---|
| MuQ embeddings | `percepiano_cache/muq_embeddings.pt` + `_muq_file_cache/` | 4.4GB |
| Labels | `percepiano_cache/labels.json` | <1MB |
| CV folds | `percepiano_cache/folds.json` | <1MB |
| Piece mapping | `percepiano_cache/piece_mapping.json` | <1MB |

- **Segments:** 1,202 with 19-dimension perceptual annotations
- **Signal:** Regression (MSE on composite labels), ranking (within-piece pairs)
- **Composite labels** in `composite_labels/` map 19 raw dims to 6 teacher-grounded dims
- **Training class:** `CompetitionPairSampler`

### T2: Competition Recordings (Ordinal Ranking)

| Item | Path | Size |
|---|---|---|
| Segment metadata | `competition_cache/chopin2021/metadata.jsonl` | <1MB |
| MuQ embeddings | `competition_cache/chopin2021/muq_embeddings/` | ~760MB |

- **Source:** XVIII International Chopin Piano Competition 2021 (YouTube)
- **Segments:** 2,293 from 11 performers across Stage 2, 3, Final
- **Signal:** Ordinal placement within each round
- **Collection script:** `scripts/collect_competition_data.py`
- **Training classes:** `CompetitionDataset`, `CompetitionPairSampler`

### T3: MAESTRO Audio (Contrastive Learning)

| Item | Path | Size |
|---|---|---|
| MAESTRO metadata | `maestro_cache/maestro-v3.0.0.json` | 83MB |
| MIDI files | `maestro_cache/2004-2018/` | on disk |
| Segment metadata | `maestro_cache/metadata.jsonl` | <1MB |
| MuQ embeddings | `maestro_cache/muq_embeddings/` | ~34GB |
| Per-recording graphs | `pretrain_cache/graphs/shards/graphs_0241-0263.pt` | ~1GB |
| Contrastive mapping | `maestro_cache/contrastive_mapping.json` | <1MB |

- **Source:** MAESTRO v3.0.0 (1,276 recordings, ~300 pieces, 204 with 2+ performers)
- **Segments:** 24,321
- **Signal:** Contrastive pairs (same piece, different performer = positive). InfoNCE loss.
- **Training class:** `PairedPerformanceDataset`

### T4: YouTube Piano (Augmentation Invariance) -- DEFERRED

| Item | Path | Size |
|---|---|---|
| Channel list | `youtube_piano_cache/channels.yaml` | on disk |

- **Source:** 22 curated YouTube channels
- **Signal:** MSE between clean and augmented MuQ embeddings
- **Priority:** Add only if robustness drops > 10% or cross-condition Pearson r < 0.9
- **Collection script:** `scripts/collect_youtube_piano.py`

### Symbolic Pretraining Corpus

| Item | Path | Size |
|---|---|---|
| Tokenized MIDI | `pretrain_cache/tokens/all_tokens.pt` | 55MB |
| Score graphs | `pretrain_cache/graphs/all_graphs.pt` + shards (0000-0263) | 29GB |
| Hetero graphs | `pretrain_cache/graphs/all_hetero_graphs.pt` + shards | (in 29GB) |
| Continuous features | `pretrain_cache/features/all_features.pt` + shards | 8.3GB |

- **Sources:** 24,220 graphs from ASAP (1,066) + ATEPP (11,697) + MAESTRO score (824) + MAESTRO recording (1,123) + PercePiano (1,202) + GIANTMIDI (8,278)
- **Training classes:** `MIDIPretrainingDataset`, `ScoreGraphPretrainingDataset`, `ShardedScoreGraphPretrainDataset`, `ContinuousPretrainDataset`, `HeteroPretrainDataset`

### Composite Labels

| Item | Path | Size |
|---|---|---|
| Dimension definitions | `composite_labels/taxonomy.json` | <100KB |
| Mapped labels | `composite_labels/labels.json` | <1MB |
| Teacher quote bank | `composite_labels/quote_bank.json` | <1MB |

Produced by the teacher-grounded taxonomy work (doc 02). Maps 19 raw PercePiano dimensions to 6 teacher-grounded dimensions.

### Other Caches

| Item | Path | Purpose |
|---|---|---|
| Intermediate YouTube | `intermediate_cache/` | 629 segments for dynamic range analysis + AMT validation |
| Masterclass moments | `masterclass_cache/` | 2,136 teaching moments for taxonomy derivation |

## Storage

| Location | Capacity | Purpose |
|---|---|---|
| Local (Mac) | 50GB free | Code, caches, composite labels |
| GDrive | 80GB total | Checkpoints, results, final embeddings |
| Thunder Compute | 500GB | Raw audio download, processing, MuQ extraction |

**Principle:** Raw audio lives and dies on the remote. Only `.pt` embeddings and `.jsonl` metadata return.

### Local Disk Usage

```
model/data/                     Size
  pretrain_cache/               38 GB
  masterclass_pipeline/         5.9 GB
  percepiano_cache/             4.4 GB
  maestro_cache/                3.9 GB
  competition_cache/            0.8 GB
  masterclass_cache/            907 MB
  atepp_cache/                  525 MB
  other                         220 MB
  TOTAL                         ~55 GB
```

## Readiness Checklist

### Current (Wave 1 -- complete)

```
[x] T1 PercePiano embeddings + labels
[x] Symbolic pretraining corpus (24,220 graphs)
[x] Composite labels (6 teacher-grounded dimensions)
[x] T2 competition embeddings (2,293 segments)
[x] T3 MAESTRO embeddings (24,321 segments) + per-recording graphs (1,123)
[x] T3 MAESTRO contrastive mapping (204 pieces)
[ ] T4 YouTube embeddings (deferred)
```

### Pipeline Roadmap Data Needs

See `04-north-star.md` for full pipeline vision and phase details.

**Phase 1: Score Infrastructure (engineering, no new training data)**

```
[ ] Score MIDI library V1 (ASAP = 242 pieces, MIDI parsing for bar structure/notes/pedal/time+key sigs)
[ ] Score MIDI library V2 (expand to MAESTRO + external sources, MusicXML for richer annotations)
[ ] Reference performance cache (per-bar statistics from MAESTRO professional recordings)
```

Sources: ASAP score MIDIs (242 pieces, V1). MAESTRO score MIDIs require external sourcing (performances exist, scores don't). IMSLP/MuseScore for expansion. MusicXML import for dynamics text, articulation marks, section labels (future enrichment).

Design spec: `docs/superpowers/specs/2026-03-14-score-midi-library-design.md`

**Phase 3: Symbolic Foundation Model (research)**

```
[ ] Expanded MIDI pretraining corpus (~370K+ performances)
    - PianoMIDI (~100K performances)
    - Lakh MIDI piano tracks (~50K)
    - MuseScore piano exports (~200K)
[ ] Reference-anchored training triples from MAESTRO
    (performance audio + performance MIDI + score MIDI, ranked by A1-Max)
```

**Phase 4: Real Audio + Expert Labels**

```
[ ] Real piano recordings (2K-5K segments)
    - University partnerships (~500), user opt-in (~1K+), YouTube (~500), commissioned (~200)
    - 3+ skill levels, 5+ piano types, 5+ environments
[ ] Expert annotations (3-5 piano teachers, 6-dim rubric with score context)
    - Active learning: prioritize uncertain segments
    - Estimated cost: ~$50-100K
```
