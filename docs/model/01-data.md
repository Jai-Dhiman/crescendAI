# Data Inventory

Dataset inventory for piano performance evaluation training. Heavy data processing (embedding extraction, full training runs) runs on HF Jobs (L4 $0.80/hr default, A100 $2.50/hr for Aria). Training data stored on HF Bucket.

> **Status (2026-03-18):** T1 COMPLETE, T2 COMPLETE (Chopin 2021, expansion planned), T3 COMPLETE, T5 IN PROGRESS (YouTube Skill Corpus -- 2 of 16 pieces curated: Fur Elise 28 recordings, Nocturne Op.9/2 27 recordings, 3 wrong entries removed). Graph pretraining corpus LEGACY (24,220 graphs -- replaced by Aria). Composite labels COMPLETE. T4 REPLACED by T5. **Aria-MIDI (820K piano MIDIs) available for continued pretraining.** Disk: 12 GB cleaned, 68 GB available.

## Directory Structure

All paths defined in `model/src/paths.py`. Organization by pipeline stage:

```
model/data/
  raw/          - Downloaded datasets (gitignored, re-downloadable)
  embeddings/   - Extracted MuQ embeddings (gitignored, regenerable)
  midi/         - Small MIDI collections (gitignored)
  pretraining/  - Symbolic pretraining corpus (gitignored, LEGACY)
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

For Aria (symbolic path), training also loads:
- AMT MIDI: performance MIDI transcribed from audio (via ByteDance AMT on HF endpoint)
- Score MIDI: from score library (242 ASAP pieces deployed to D1 + R2)

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
- **Folds:** Piece-stratified (verified 2026-03-18, previous segment-level folds had leak)

### T2: Competition Recordings (Ordinal Ranking)

| Item | Path | Size |
|---|---|---|
| Chopin 2021 metadata | `manifests/competition/metadata.jsonl` | <1MB |
| Chopin 2021 embeddings | `embeddings/competition/muq_embeddings/` | ~760MB |
| Cliburn 2022 metadata | `manifests/competition/cliburn_2022/metadata.jsonl` | <1MB |
| Cliburn 2022 embeddings | `manifests/competition/cliburn_2022/muq_embeddings/` | 20 GB |

- **Sources:** Chopin 2021 (11 performers) + Cliburn 2022 (30 performers)
- **Segments:** 9,059 total (2,293 Chopin 2021 + 6,766 Cliburn 2022)
- **Signal:** Ordinal placement within each round
- **RMS silence filter:** Segments with RMS < 0.002 dropped at segmentation time (1.4% of Cliburn 2022)
- **Collection scripts:** `scripts/collect_competition_data.py` (Chopin-specific), `scripts/collect_generic_competition.py` (any competition)
- **Training classes:** `CompetitionDataset`, `CompetitionPairSampler`

**T2 expansion plan (current: 9,059 segments):**

| Competition | Performers | Recordings | YouTube URLs | Est. Segments | Status |
|---|---|---|---|---|---|
| Chopin 2021 (current) | 11 | 33 | 33/33 | 2,293 | COMPLETE |
| Cliburn 2022 | 30 | 82 | 82/82 | 6,766 | COMPLETE (downloaded + embedded locally on M4 MPS) |
| Chopin 2015 | 43 | 73 | 10/73 | ~5,100 | MANIFEST READY (stage2/3 need timestamp extraction from session videos) |
| Cliburn Amateur 2022 | 39 | 65 | 0/65 | ~4,550 | MANIFEST READY (videos offline -- was on cliburn.org, now removed) |
| Queen Elisabeth 2025 | 24 | 36 | 0/36 | ~2,500 | MANIFEST READY (no YouTube -- performances on queenelisabethcompetition.be only) |

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
- **Pieces:** 16 target (8 core deep + 8 breadth). **2 of 16 curated:**
  - Fur Elise: 28 recordings (curated, verified)
  - Nocturne Op.9/2: 27 recordings (curated, 3 wrong entries removed)
- **Labels:** Human-curated 5-bucket classification per recording via curation UI
- **Split:** 80% training, 20% held-out eval, stratified by piece and bucket
- **Training loss:** ListMLE ranking (same as T2 competition), grouped by piece

**Why T5 exists:** A1-Max skill-level evaluation (2026-03-18) showed zero discrimination -- beginner and professional score identically (0.558 vs 0.565). PercePiano training data is 100% advanced-level. The model never learned that quality has a spectrum.

**Core pieces (10-15 recordings/bucket target):** Fur Elise, Nocturne Op.9/2, Moonlight Sonata mvt 1, Clair de Lune, Bach Prelude C WTC1, Mozart K.545 mvt 1, Chopin Waltz C#m, Liszt Liebestraum 3

**Breadth pieces (5-8 recordings/bucket target):** Chopin Etude Op.10/4, Pathetique mvt 2, Debussy Arabesque 1, Chopin Ballade 1, Rachmaninoff Prelude C#m, Schumann Traumerei, Bach Invention 1, Fantaisie-Impromptu

### Aria-MIDI Dataset (Available for Continued Pretraining)

| Item | Source | Size |
|---|---|---|
| Aria-MIDI | HuggingFace (EleutherAI) | 820K piano MIDI performances (~60K hours) |

- **License:** Apache 2.0
- **What it is:** The pretraining corpus for Aria, EleutherAI's 650M-param LLaMA-architecture symbolic music model. SOTA on 6 MIR benchmarks.
- **CrescendAI use:** Aria is pretrained. We fine-tune Aria on our task-specific data (T1+T2+T3+T5). The Aria-MIDI dataset is available if continued pretraining proves beneficial (e.g., piano-specific domain adaptation beyond the general pretraining).
- **Not needed for initial model v2:** Aria's existing pretraining should suffice. Only download if experiments show benefit from continued pretraining on piano-specific subset.

### Additional MIDI Expansion Sources (Available)

| Dataset | Est. Piano MIDIs | Source | License | Status |
|---|---|---|---|---|
| ADL Piano MIDI | ~11,000 | Academic dataset | Research | Available, not downloaded |
| Lakh MIDI (piano filtered) | ~50,000 | Lakh MIDI Dataset | Research | Available, requires filtering |

These are available for continued Aria pretraining or contrastive training if needed. Lower priority than task-specific fine-tuning data (T2 expansion, T5 curation).

### LEGACY: Symbolic Pretraining Corpus (Replaced by Aria)

> **LEGACY:** This graph pretraining corpus was built for CrescendAI's custom GNN symbolic encoders (S1, S2, S2H, S3). These encoders are replaced by Aria (650M params, pretrained on 820K MIDIs). The graph data is no longer needed for CrescendAI's own pretraining. Kept on disk for reference.

| Item | Path | Size |
|---|---|---|
| Tokenized MIDI | `pretraining/tokens/all_tokens.pt` | 55MB |
| Score graphs | `pretraining/graphs/all_graphs.pt` + shards (0000-0263) | 29GB |
| Hetero graphs | `pretraining/graphs/all_hetero_graphs.pt` + shards | (in 29GB) |
| Continuous features | `pretraining/features/all_features.pt` | 8.3GB |

- **Sources:** 24,220 graphs from ASAP (1,066) + ATEPP (11,697) + MAESTRO score (824) + MAESTRO recording (1,123) + PercePiano (1,202) + GIANTMIDI (8,278)
- **Training classes:** `MIDIPretrainingDataset`, `ScoreGraphPretrainingDataset`, `ShardedScoreGraphPretrainDataset`, `ContinuousPretrainDataset`, `HeteroPretrainDataset`
- **Candidate for cleanup:** 38 GB of disk. Can be deleted once Aria integration is validated.

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

## Evaluation Tiers

### Current

| Tier | Description | Status |
|---|---|---|
| E1: PercePiano pairwise | Within-piece pairwise accuracy on piece-stratified folds | Active (clean folds verified) |
| E2: Competition ranking | Spearman rho with competition placement | Active |
| E3: Skill discrimination | Monotonic score increase across 5 skill buckets (T5) | Active (primary metric for model v2) |

### Future (needed for production readiness)

| Tier | Description | Status |
|---|---|---|
| E4: Recording conditions | Model robustness across phone mic, room acoustics, piano types | NOT STARTED -- needs paired recordings (same piece, same performer, different conditions) |
| E5: Repertoire breadth | Generalization beyond PercePiano's 3 works and competition Chopin | NOT STARTED -- needs labeled recordings across 20+ composers |
| E6: Real user recordings | Accuracy on actual practice sessions from beta users | NOT STARTED -- needs opt-in user data collection pipeline |

## Storage

| Location | Capacity | Purpose |
|---|---|---|
| Local (Mac) | 68GB available | Active working set (current experiment's embeddings + T5 data) |
| HF Bucket (private) | ~92GB | Embeddings, manifests, checkpoints, T5 audio |
| GDrive (via rclone) | 80GB total | Archival backup: results, labels, final weights |
| HF Jobs (cloud) | On-demand | Full training runs, Aria fine-tuning, validation |

**Principle:** Training data lives on HF Bucket. Local disk holds only the active experiment's data. GDrive is archival backup only.

### Local Disk Usage

```
model/data/                     Size
  raw/maestro/                  ~35 GB
  pretraining/ (LEGACY)         38 GB  -- candidate for cleanup after Aria validation
  embeddings/masterclass/       7.3 GB
  embeddings/percepiano/        4.4 GB
  embeddings/competition/       3.3 GB
  raw/atepp/                    525 MB
  raw/giantmidi/                488 MB
  evals/                        850 MB
  scores/                       189 MB
  other                         220 MB
  TOTAL                         ~90 GB (~52 GB if pretraining cleaned)
```

## Readiness Checklist

### Current (Wave 1 -- complete)

```
[x] T1 PercePiano embeddings + labels
[x] T1 Piece-stratified CV folds (verified, leak-free)
[x] Composite labels (6 teacher-grounded dimensions)
[x] T2 competition embeddings (2,293 segments, Chopin 2021)
[x] T3 MAESTRO embeddings (24,321 segments) + per-recording graphs (1,123)
[x] T3 MAESTRO contrastive mapping (204 pieces)
[x] LEGACY: Symbolic pretraining corpus (24,220 graphs -- replaced by Aria)
```

### Phase 0+3: Model v2 (Aria + Multi-Tier Training)

```
[x] T5 YouTube Skill manifests: Fur Elise (28 recordings, curated)
[x] T5 YouTube Skill manifests: Nocturne Op.9/2 (27 recordings, curated, 3 wrong entries removed)
[ ] T5 YouTube Skill: collect + curate remaining 14 pieces (~720 recordings)
[ ] T5 YouTube Skill: download audio + extract MuQ embeddings + AMT MIDI
[x] T2 expansion: manifests collected (Cliburn 2022, Chopin 2015, Cliburn Amateur 2022, QE 2025)
[x] T2 expansion: Cliburn 2022 download + embed (6,766 segments from 82 recordings, 20 GB embeddings)
[ ] T2 expansion: Chopin 2015 download + embed (~700 segments from 10 final URLs; stage2/3 need timestamp work)
[ ] T2 expansion: Cliburn Amateur 2022 (videos offline, need alternative source)
[ ] T2 expansion: Queen Elisabeth 2025 (no YouTube, need queenelisabethcompetition.be access)
[ ] Aria model: download weights from HuggingFace
[ ] Aria integration: LoRA fine-tune pipeline for Aria
[ ] Contrastive pretraining: MuQ + Aria on T2+T5
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

**Phase 4: Real Audio + Expert Labels**

```
[ ] Real piano recordings (2K-5K segments)
    - University partnerships (~500), user opt-in (~1K+), YouTube (~500), commissioned (~200)
    - 3+ skill levels, 5+ piano types, 5+ environments
[ ] Expert annotations (3-5 piano teachers, 6-dim rubric with score context)
    - Active learning: prioritize uncertain segments
    - Estimated cost: ~$50-100K
```
