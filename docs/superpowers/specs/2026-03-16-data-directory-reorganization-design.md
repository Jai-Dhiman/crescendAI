# Data Directory Reorganization

Reorganize `model/data/` from 22 flat `*_cache` directories into a pipeline-stage-first structure with a central path config module.

## Goals

1. **Clarity** -- someone opening `model/data/` immediately understands what's what
2. **Pipeline hygiene** -- clean separation between raw inputs, processed artifacts, and outputs
3. **Future-proofing** -- Phase 3 (370K+ MIDI) and Phase 4 (real audio + expert labels) slot in without rethinking structure

## Target Directory Structure

```
model/data/
  raw/                          # [GITIGNORED] Downloaded/external datasets
    maestro/                    #   MAESTRO v3 audio + MIDI + maestro-v3.0.0.json
    asap/                       #   ASAP performances ({Composer}/{Piece}/...)
    atepp/                      #   ATEPP recordings + ATEPP_metadata.csv
    giantmidi/                  #   GiantMIDI piano MIDI corpus
    masterclass/                #   all_moments.jsonl, audio segments
    youtube/                    #   downloaded audio
    competition/                #   Chopin 2021 raw audio (if any retained)

  embeddings/                   # [GITIGNORED] MuQ + derived embeddings
    percepiano/                 #   muq_embeddings.pt, per-file cache/
    maestro/                    #   per-segment .pt files, metadata.jsonl, contrastive_mapping.json
    masterclass/                #   teaching moment MuQ embeddings
    competition/                #   chopin2021 MuQ embeddings

  midi/                         # [GITIGNORED] MIDI collections used directly
    percepiano/                 #   1,202 .mid files (ground truth MIDI)
    amt/                        #   AMT test set (50 files)

  pretraining/                  # [GITIGNORED] Symbolic pretraining corpus
    tokens/                     #   all_tokens.pt (55MB)
    graphs/                     #   all_graphs.pt + shards (29GB)
    features/                   #   all_features.pt + shards (8.3GB)

  labels/                       # [TRACKED] Annotations, derived labels, and classifier weights
    composite/                  #   composite_labels.json, taxonomy, quote_bank, rubric
    percepiano/                 #   folds.json, labels.json, piece_mapping.json
    stop_classifier_weights.json #  STOP classifier model weights

  manifests/                    # [TRACKED] Data source configs and metadata
    masterclass/                #   sources.yaml (YouTube masterclass manifest)
    youtube/                    #   channels.yaml (curated YouTube channel list)
    competition/                #   metadata.jsonl, recordings.jsonl (Chopin 2021)

  scores/                       # [TRACKED] Score library JSON (deployed to R2)
    titles.json                 #   Piece catalog
    seed.sql                    #   D1 seed data
                                #   242 piece .json files are gitignored (generated, deployed to R2)

  references/                   # [GITIGNORED] Reference performance profiles (deployed to R2)
    v1/                         #   Per-piece bar-level stats ({piece_id}.json)
    maestro_asap_matches.csv
    unmatched_maestro.csv

  evals/                        # [PARTIALLY TRACKED] Evaluation data
    skill_eval/                 #   [TRACKED] Manifests; [GITIGNORED] audio/, results/
    inference_cache/            #   [GITIGNORED] Model prediction JSONs
    traces/                     #   [GITIGNORED] LLM reasoning traces
    youtube_amt/                #   [GITIGNORED] Validation WAV files
    intermediate/               #   [TRACKED] Curated recording metadata

  checkpoints/                  # [GITIGNORED] Trained model weights
    model_improvement/          #   A1, A1-Max folds
    ablation/
    ... (existing structure)

  results/                      # [TRACKED] Experiment results (CSV, JSON)
    a1_max_sweep.json
    experiment4/
    competition/                #   Chopin 2021 heatmap PNGs
    teacher_voice/              #   benchmark_results, masterclass_records, synthetic_records
    ... (existing structure)

  calibration/                  # [TRACKED] Reference calibration stats
    maestro_stats.json
```

### Design Principles

- **Gitignore aligns with directory, not files.** `raw/`, `embeddings/`, `midi/`, `pretraining/`, `checkpoints/`, `references/` are entirely gitignored. `labels/`, `manifests/`, `results/`, `calibration/` are entirely tracked. `scores/` tracks only `titles.json` and `seed.sql` (generated piece JSONs are gitignored). `evals/` has mixed tracking.
- **Top-level directories mirror pipeline stages.** Reading the top level tells the pipeline story.
- **Future datasets add subdirs, not new top-level dirs.** Phase 3 adds `raw/pianomidi/`, `raw/lakh/`, `raw/musescore/` and corresponding `embeddings/` entries.

## Migration Mapping

| Old | New | Tracked? |
|-----|-----|----------|
| `maestro_cache/` | `raw/maestro/` (audio/MIDI), `embeddings/maestro/` (embeddings) | No |
| `asap_cache/` | `raw/asap/` | No |
| `atepp_cache/` | `raw/atepp/` | No |
| `giantmidi_raw/` | `raw/giantmidi/` | No |
| `percepiano_cache/muq_embeddings*` | `embeddings/percepiano/` | No |
| `percepiano_cache/{folds,labels,piece_mapping}.json` | `labels/percepiano/` | Yes (git mv) |
| `percepiano_midi/` | `midi/percepiano/` | No |
| `masterclass_cache/` | `embeddings/masterclass/` | No |
| `masterclass_pipeline/sources.yaml` | `manifests/masterclass/sources.yaml` | Yes (git mv) |
| `masterclass_pipeline/` (rest) | `raw/masterclass/` | No |
| `competition_cache/chopin2021/muq_embeddings/` | `embeddings/competition/` | No |
| `competition_cache/chopin2021/{metadata,recordings}.jsonl` | `manifests/competition/` | Yes (git mv) |
| `competition_cache/chopin2021/*.png` | `results/competition/` | Yes (git mv) |
| `youtube_piano_cache/channels.yaml` | `manifests/youtube/channels.yaml` | Yes (git mv) |
| `pretrain_cache/` | `pretraining/` | No |
| `composite_labels/` | `labels/composite/` | Yes (git mv) |
| `stop_classifier_weights.json` | `labels/stop_classifier_weights.json` | Yes (git mv) |
| `score_library/{titles.json,seed.sql}` | `scores/` | Yes (git mv) |
| `score_library/*.json` (piece files) | `scores/` | No |
| `reference_profiles/` | `references/` | No |
| `amt_cache/` | `midi/amt/` | No |
| `eval/` | `evals/` (split into subdirs) | No |
| `intermediate_cache/` | `evals/intermediate/` | Yes (git mv) |
| `skill_eval/` | `evals/skill_eval/` | Yes (git mv, manifests only) |
| `teacher_voice_eval/*.jsonl` | `results/teacher_voice/` | Yes (git mv) |
| `experiment4_results/` | `results/experiment4/` | Yes (git mv) |
| `a1_max_sweep_results.json` | `results/a1_max_sweep.json` | Yes (git mv) |
| `calibration_stats.json` (in maestro_cache) | `calibration/maestro_stats.json` | No |

## Central Path Config: `model/src/paths.py`

Single module defining all data paths. Every script imports from here instead of constructing paths inline.

```python
"""Central data path definitions for the CrescendAI model pipeline."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class Raw:
    root = DATA_ROOT / "raw"
    maestro = root / "maestro"
    asap = root / "asap"
    atepp = root / "atepp"
    giantmidi = root / "giantmidi"
    masterclass = root / "masterclass"
    youtube = root / "youtube"
    competition = root / "competition"


class Embeddings:
    root = DATA_ROOT / "embeddings"
    percepiano = root / "percepiano"
    maestro = root / "maestro"
    masterclass = root / "masterclass"
    competition = root / "competition"


class Midi:
    root = DATA_ROOT / "midi"
    percepiano = root / "percepiano"
    amt = root / "amt"


class Pretraining:
    root = DATA_ROOT / "pretraining"
    tokens = root / "tokens"
    graphs = root / "graphs"
    features = root / "features"


class Labels:
    root = DATA_ROOT / "labels"
    composite = root / "composite"
    percepiano = root / "percepiano"
    stop_classifier_weights = root / "stop_classifier_weights.json"


class Manifests:
    root = DATA_ROOT / "manifests"
    masterclass = root / "masterclass"
    youtube = root / "youtube"
    competition = root / "competition"


class Scores:
    root = DATA_ROOT / "scores"


class References:
    root = DATA_ROOT / "references"
    v1 = root / "v1"


class Evals:
    root = DATA_ROOT / "evals"
    skill_eval = root / "skill_eval"
    inference_cache = root / "inference_cache"
    traces = root / "traces"
    youtube_amt = root / "youtube_amt"
    intermediate = root / "intermediate"


class Checkpoints:
    root = DATA_ROOT / "checkpoints"
    model_improvement = root / "model_improvement"


class Results:
    root = DATA_ROOT / "results"


class Calibration:
    root = DATA_ROOT / "calibration"
```

### Usage

```python
# Before:
cache_dir = DATA_DIR / "percepiano_cache"
composite_path = DATA_DIR / "composite_labels" / "composite_labels.json"

# After:
from src.paths import Embeddings, Labels
cache_dir = Embeddings.percepiano
composite_path = Labels.composite / "composite_labels.json"
```

Design decisions:

- Classes as namespaces (not instances) -- paths resolve at import time, autocomplete works
- Paths resolve to directories, not files -- filenames stay in consuming code
- No validation at import time -- avoids failures when only a subset of data is present
- No env var override -- can be added later with one line if needed

## Gitignore Strategy

Replace scattered per-directory `.gitignore` files with a single `model/data/.gitignore`:

```gitignore
# Entirely gitignored stages (large, regenerable)
/raw/
/embeddings/
/midi/
/pretraining/
/checkpoints/
/references/

# Scores: only titles.json and seed.sql are tracked
/scores/*.json
!/scores/titles.json

# Eval outputs (manifests/metadata tracked separately)
/evals/inference_cache/
/evals/traces/
/evals/youtube_amt/
/evals/skill_eval/*/audio/
/evals/skill_eval/*/results.json
```

## Migration Strategy

### Phase 1: Non-destructive setup
1. Create `model/src/paths.py`
2. Create new directory structure (empty dirs with `.gitkeep` where needed)
3. Create new `model/data/.gitignore`

### Phase 2: Move tracked files (git mv)

All of these are confirmed tracked in git (133 total tracked files under `model/data/`):

- `composite_labels/` -> `labels/composite/` (6 files)
- `percepiano_cache/{folds,labels,piece_mapping}.json` -> `labels/percepiano/` (3 files)
- `stop_classifier_weights.json` -> `labels/stop_classifier_weights.json`
- `masterclass_pipeline/sources.yaml` -> `manifests/masterclass/sources.yaml`
- `youtube_piano_cache/channels.yaml` -> `manifests/youtube/channels.yaml`
- `competition_cache/chopin2021/{metadata,recordings}.jsonl` -> `manifests/competition/`
- `competition_cache/chopin2021/*.png` -> `results/competition/`
- `score_library/{titles.json,seed.sql}` -> `scores/`
- `skill_eval/` -> `evals/skill_eval/` (manifest YAMLs only)
- `intermediate_cache/` -> `evals/intermediate/` (6 files)
- `teacher_voice_eval/*.jsonl` -> `results/teacher_voice/` (3 files)
- `experiment4_results/` -> `results/experiment4/`
- `results/` -> `results/` (stays in place, existing structure preserved)
- `a1_max_sweep_results.json` -> `results/a1_max_sweep.json`

Use `git mv` to preserve history. Remove old per-directory `.gitignore` files.

### Phase 3: Move gitignored data (plain mv)
- `maestro_cache/` -> split `raw/maestro/` (audio/MIDI) + `embeddings/maestro/` (embeddings)
- `percepiano_cache/` (remaining: embeddings) -> `embeddings/percepiano/`
- `percepiano_midi/` -> `midi/percepiano/` (1,202 .mid files, not tracked)
- `amt_cache/` -> `midi/amt/` (50 files, not tracked)
- `pretrain_cache/` -> `pretraining/`
- `competition_cache/chopin2021/muq_embeddings/` -> `embeddings/competition/`
- `masterclass_cache/` -> `embeddings/masterclass/`
- `masterclass_pipeline/` (remaining: all_moments.jsonl, audio) -> `raw/masterclass/`
- `atepp_cache/` -> `raw/atepp/`
- `giantmidi_raw/` -> `raw/giantmidi/`
- `youtube_piano_cache/` (remaining) -> `raw/youtube/`
- `reference_profiles/` -> `references/` (not tracked, generated data)
- `score_library/*.json` (piece files) -> `scores/` (not tracked, generated)
- `calibration_stats.json` (in maestro_cache) -> `calibration/maestro_stats.json`
- `eval/` -> split into `evals/` subdirs (inference_cache, traces, youtube_amt, download script)

### Phase 4: Update all path references
- Update Python scripts to import from `src.paths`
- Update notebook cells to use `from src.paths import ...`
- Update CLI docstrings referencing old paths
- Update documentation referencing old paths (`docs/model/01-data-inventory.md`)
- Remove old per-directory `.gitignore` files

### Phase 5: Cleanup and verify
- Remove empty old directories
- Verify imports: `python -c "from src.paths import *"`
- Run existing tests
- Update `docs/model/01-data-inventory.md` with new structure

## Files That Need Path Updates

Scripts (~22 files):
- `src/model_improvement/data.py` -- main training data loader (takes `data_dir` param, migrate to paths.py)
- `src/model_improvement/datasets.py` -- MIDI file discovery (`load_all_midi_files` hardcodes old names)
- `src/model_improvement/maestro.py` -- MAESTRO metadata/embedding extraction
- `src/model_improvement/a1_max_sweep.py` -- A1-Max hyperparameter sweep
- `src/model_improvement/ablation_sweep.py` -- ablation study
- `src/model_improvement/competition.py` -- competition data processing
- `src/model_improvement/evaluation.py` -- evaluation utilities
- `src/model_improvement/youtube_piano.py` -- YouTube piano channel processing
- `src/teacher_voice/converters.py` -- teacher voice data converters
- `src/teacher_voice/benchmark.py` -- teacher voice benchmarks
- `src/teacher_voice/rate.py` -- teacher voice rating
- `src/teacher_voice/synthetic.py` -- synthetic voice data
- `src/score_library/cli.py` -- score library CLI
- `src/score_library/reference_cache.py` -- reference profile generation
- `src/skill_eval/collect.py` -- skill eval data collection
- `src/skill_eval/run_inference.py` -- skill eval inference
- `src/skill_eval/analyze.py` -- skill eval analysis
- `scripts/extract_percepiano_muq.py`
- `scripts/download_maestro_subset.py`
- `scripts/collect_competition_data.py`
- `scripts/compute_maestro_calibration.py`
- `scripts/process_maestro_recording_graphs.py`

Notebooks (~6 files):
- `notebooks/model_improvement/00_data_collection.ipynb`
- `notebooks/model_improvement/01_audio_training.ipynb`
- `notebooks/model_improvement/02_symbolic_training.ipynb`
- `notebooks/model_improvement/04_layer1_validation.ipynb`
- Other experimental notebooks

Documentation (~3 files):
- `docs/model/01-data-inventory.md`
- `docs/model/04-north-star.md`
- `CLAUDE.md` (memory references)

### Design note: `data_dir` parameter migration

Several modules (notably `data.py`, `datasets.py`) accept a `data_dir` parameter and construct paths from it. The migration should change these to import from `paths.py` directly, removing the `data_dir` parameter. For testing flexibility, individual paths can be overridden via function parameters if needed, but the default should come from `paths.py`.
