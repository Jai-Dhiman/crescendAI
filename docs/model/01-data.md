# Data Inventory

Dataset inventory for piano performance evaluation training. All heavy data processing (audio download, segmentation, MuQ extraction) runs on Thunder Compute (A100). Only embeddings and metadata come back.

> **Status (2026-03-18):** T1 COMPLETE, T2 COMPLETE (Chopin 2021, expansion planned), T3 COMPLETE, T5 IN PROGRESS (YouTube Skill Corpus -- 2 pieces curated, 14 remaining). Symbolic pretrain corpus COMPLETE (24,220 graphs). Composite labels COMPLETE. T4 (augmentation invariance) REPLACED by T5 (skill-level ranking).

## Directory Structure

All paths defined in `model/src/paths.py`. Organization by pipeline stage:

```
model/data/
  raw/          - Downloaded datasets (gitignored, re-downloadable)
  embeddings/   - Extracted MuQ embeddings (gitignored, regenerable)
  midi/         - Small MIDI collections (gitignored)
  pretraining/  - Symbolic pretraining corpus (gitignored)
  labels/       - Annotations and derived labels (tracked)
  manifests/    - Data source configs (tracked)
  scores/       - Score library JSON, deployed to R2 (partially tracked)
  references/   - Reference performance profiles, deployed to R2 (gitignored)
  evals/        - Evaluation data (partially tracked)
  checkpoints/  - Trained model weights (gitignored)
  results/      - Experiment results (tracked)
  calibration/  - MAESTRO calibration stats (tracked)
```

## What Training Loads

The training code (`src/model_improvement/data.py`) loads only embeddings and metadata -- never raw audio. At training time, every tier must resolve to:

- MuQ embeddings: `{cache_dir}/muq_embeddings/{segment_id}.pt` (93x1024 float32, ~380KB)
- Segment metadata: `{cache_dir}/metadata.jsonl`
- Tier-specific signals: labels, placements, contrastive mappings, augmented pairs

## Datasets

### T1: PercePiano (Labeled Regression + Ranking)

| Item | Path | Size |
|---|---|---|
| MuQ embeddings | `embeddings/percepiano/muq_embeddings.pt` + `_muq_file_cache/` | 4.4GB |
| Labels | `labels/percepiano/labels.json` | <1MB |
| CV folds | `labels/percepiano/folds.json` | <1MB |
| Piece mapping | `labels/percepiano/piece_mapping.json` | <1MB |

- **Segments:** 1,202 with 19-dimension perceptual annotations
- **Signal:** Regression (MSE on composite labels), ranking (within-piece pairs)
- **Composite labels** in `labels/composite/` map 19 raw dims to 6 teacher-grounded dims
- **Training class:** `CompetitionPairSampler`

### T2: Competition Recordings (Ordinal Ranking)

| Item | Path | Size |
|---|---|---|
| Segment metadata | `manifests/competition/metadata.jsonl` | <1MB |
| MuQ embeddings | `embeddings/competition/muq_embeddings/` | ~760MB |

- **Source:** XVIII International Chopin Piano Competition 2021 (YouTube)
- **Segments:** 2,293 from 11 performers across Stage 2, 3, Final
- **Signal:** Ordinal placement within each round
- **Collection script:** `scripts/collect_competition_data.py`
- **Training classes:** `CompetitionDataset`, `CompetitionPairSampler`

### T3: MAESTRO Audio (Contrastive Learning)

| Item | Path | Size |
|---|---|---|
| MAESTRO metadata | `raw/maestro/maestro-v3.0.0.json` | 83MB |
| MIDI files | `raw/maestro/2004-2018/` | on disk |
| Segment metadata | `embeddings/maestro/metadata.jsonl` | <1MB |
| MuQ embeddings | `embeddings/maestro/muq_embeddings/` | ~34GB |
| Per-recording graphs | `pretraining/graphs/shards/graphs_0241-0263.pt` | ~1GB |
| Contrastive mapping | `embeddings/maestro/contrastive_mapping.json` | <1MB |

- **Source:** MAESTRO v3.0.0 (1,276 recordings, ~300 pieces, 204 with 2+ performers)
- **Segments:** 24,321
- **Signal:** Contrastive pairs (same piece, different performer = positive). InfoNCE loss.
- **Training class:** `PairedPerformanceDataset`

### T4: YouTube Piano (Augmentation Invariance) -- REPLACED BY T5

Replaced by T5 (YouTube Skill Corpus). Original T4 purpose (augmentation invariance) is lower priority than skill-level discrimination. Channel list retained at `manifests/youtube/channels.yaml`.

### T5: YouTube Skill Corpus (Ordinal Skill-Level Ranking) -- IN PROGRESS

| Item | Path | Size |
|---|---|---|
| Skill eval manifests | `evals/skill_eval/{piece}/manifest.yaml` | <1MB each |
| Collection script | `apps/evals/model/skill_eval/collect.py` | on disk |
| Channel list | `manifests/youtube/channels.yaml` | on disk |

- **Source:** YouTube recordings across 5 skill levels, human-curated labels
- **Segments:** Target ~3,100 (from ~775 recordings at ~4 segments each)
- **Signal:** Ordinal skill-level ranking (5 buckets: beginner, early intermediate, intermediate, advanced, professional)
- **Pieces:** 16 target (8 core deep + 8 breadth). 2 curated: Fur Elise (28 recordings), Nocturne Op.9/2 (27 recordings)
- **Labels:** Human-curated 5-bucket classification per recording via curation UI
- **Split:** 80% training, 20% held-out eval, stratified by piece and bucket
- **Training loss:** ListMLE ranking (same as T2 competition), grouped by piece

**Why T5 exists:** A1-Max skill-level evaluation (2026-03-18) showed zero discrimination -- beginner and professional score identically (0.558 vs 0.565). PercePiano training data is 100% advanced-level. The model never learned that quality has a spectrum.

**Core pieces (10-15 recordings/bucket target):** Fur Elise, Nocturne Op.9/2, Moonlight Sonata mvt 1, Clair de Lune, Bach Prelude C WTC1, Mozart K.545 mvt 1, Chopin Waltz C#m, Liszt Liebestraum 3

**Breadth pieces (5-8 recordings/bucket target):** Chopin Etude Op.10/4, Pathetique mvt 2, Debussy Arabesque 1, Chopin Ballade 1, Rachmaninoff Prelude C#m, Schumann Traumerei, Bach Invention 1, Fantaisie-Impromptu

### Symbolic Pretraining Corpus

| Item | Path | Size |
|---|---|---|
| Tokenized MIDI | `pretraining/tokens/all_tokens.pt` | 55MB |
| Score graphs | `pretraining/graphs/all_graphs.pt` + shards (0000-0263) | 29GB |
| Hetero graphs | `pretraining/graphs/all_hetero_graphs.pt` + shards | (in 29GB) |
| Continuous features | `pretraining/features/all_features.pt` + shards | 8.3GB |

- **Sources:** 24,220 graphs from ASAP (1,066) + ATEPP (11,697) + MAESTRO score (824) + MAESTRO recording (1,123) + PercePiano (1,202) + GIANTMIDI (8,278)
- **Training classes:** `MIDIPretrainingDataset`, `ScoreGraphPretrainingDataset`, `ShardedScoreGraphPretrainDataset`, `ContinuousPretrainDataset`, `HeteroPretrainDataset`

### Composite Labels

| Item | Path | Size |
|---|---|---|
| Dimension definitions | `labels/composite/dimension_definitions.json` | <100KB |
| Mapped labels | `labels/composite/composite_labels.json` | <1MB |
| Teacher quote bank | `labels/composite/quote_bank.json` | <1MB |

Produced by the teacher-grounded taxonomy work (doc 02). Maps 19 raw PercePiano dimensions to 6 teacher-grounded dimensions.

### Other Data

| Item | Path | Purpose |
|---|---|---|
| Intermediate YouTube | `evals/intermediate/` | 629 segments for dynamic range analysis + AMT validation |
| Masterclass moments | `embeddings/masterclass/` | 2,136 teaching moments for taxonomy derivation |
| MAESTRO calibration | `calibration/maestro_stats.json` | Calibration reference stats |

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
  raw/maestro/                  ~35 GB
  pretraining/                  38 GB
  embeddings/masterclass/       7.3 GB
  embeddings/percepiano/        4.4 GB
  embeddings/competition/       3.3 GB
  raw/atepp/                    525 MB
  raw/giantmidi/                488 MB
  evals/                        850 MB
  scores/                       189 MB
  other                         220 MB
  TOTAL                         ~90 GB
```

## Readiness Checklist

### Current (Wave 1 -- complete)

```
[x] T1 PercePiano embeddings + labels
[x] Symbolic pretraining corpus (24,220 graphs)
[x] Composite labels (6 teacher-grounded dimensions)
[x] T2 competition embeddings (2,293 segments, Chopin 2021)
[x] T3 MAESTRO embeddings (24,321 segments) + per-recording graphs (1,123)
[x] T3 MAESTRO contrastive mapping (204 pieces)
```

### Phase 0: A2 Multi-Tier Training Data

```
[x] T5 YouTube Skill manifests: Fur Elise (28 recordings, curated)
[x] T5 YouTube Skill manifests: Nocturne Op.9/2 (27 recordings, curated)
[ ] T5 YouTube Skill: collect + curate remaining 14 pieces (~720 recordings)
[ ] T5 YouTube Skill: download audio + extract MuQ embeddings
[ ] T2 expansion: Chopin 2015 competition (~2,000 segments)
[ ] T2 expansion: Cliburn 2022 competition (~3,000 segments)
[ ] T2 expansion: Cliburn Amateur competition (~2,000 segments)
[ ] T2 expansion: Queen Elisabeth 2024 (~2,000 segments)
```

### Pipeline Roadmap Data Needs

See `04-north-star.md` for full pipeline vision and phase details.

**Phase 1: Score Infrastructure (COMPLETE except reference data generation)**

```
[x] Score MIDI library V1 (ASAP = 242 pieces, deployed to D1 + R2, bar-centric JSON)
[ ] Score MIDI library V2 (expand to MAESTRO + external sources, MusicXML for richer annotations)
[x] Reference performance cache script (model/src/score_library/reference_cache.py)
[ ] Reference performance data (run script on MAESTRO recordings, upload to R2)
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
