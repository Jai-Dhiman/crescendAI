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
    masterclass/                #   sources.yaml, all_moments.jsonl, audio segments
    youtube/                    #   channels.yaml, downloaded audio
    competition/                #   Chopin 2021 raw audio (if any retained)

  embeddings/                   # [GITIGNORED] MuQ + derived embeddings
    percepiano/                 #   muq_embeddings.pt, per-file cache/
    maestro/                    #   per-segment .pt files, metadata.jsonl, contrastive_mapping.json
    masterclass/                #   teaching moment MuQ embeddings
    competition/                #   chopin2021 MuQ embeddings, metadata.jsonl

  midi/                         # [TRACKED] Small MIDI collections used directly
    percepiano/                 #   1,202 .mid files (ground truth MIDI)
    amt/                        #   AMT test set (50 files)

  pretraining/                  # [GITIGNORED] Symbolic pretraining corpus
    tokens/                     #   all_tokens.pt (55MB)
    graphs/                     #   all_graphs.pt + shards (29GB)
    features/                   #   all_features.pt + shards (8.3GB)

  labels/                       # [TRACKED] Annotations and derived labels
    composite/                  #   composite_labels.json, taxonomy, quote_bank, rubric
    percepiano/                 #   folds.json, labels.json, piece_mapping.json

  scores/                       # [TRACKED] Score library JSON (deployed to R2)
    *.json                      #   242 piece files ({piece_id}.json)
    titles.json                 #   Piece catalog
    seed.sql                    #   D1 seed data

  references/                   # [TRACKED] Reference performance profiles (deployed to R2)
    v1/                         #   Per-piece bar-level stats ({piece_id}.json)
    maestro_asap_matches.csv
    unmatched_maestro.csv

  evals/                        # [PARTIALLY TRACKED] Evaluation data
    skill_eval/                 #   [TRACKED] Manifests; [GITIGNORED] audio/, results/
    inference_cache/            #   [GITIGNORED] Model prediction JSONs
    traces/                     #   [GITIGNORED] LLM reasoning traces
    youtube_amt/                #   [GITIGNORED] Validation WAV files
    intermediate/               #   [TRACKED] Curated recording metadata
    teacher_voice/              #   [GITIGNORED] Voice eval artifacts

  checkpoints/                  # [GITIGNORED] Trained model weights
    model_improvement/          #   A1, A1-Max folds
    ablation/
    ... (existing structure)

  results/                      # [TRACKED] Experiment results (CSV, JSON)
    a1_max_sweep.json
    experiment4/
    ... (existing structure)

  calibration/                  # [TRACKED] Reference calibration stats
    maestro_stats.json
```

### Design Principles

- **Gitignore aligns with directory, not files.** `raw/`, `embeddings/`, `pretraining/`, `checkpoints/` are entirely gitignored. `labels/`, `scores/`, `references/`, `results/`, `calibration/`, `midi/` are entirely tracked. Only `evals/` has mixed tracking.
- **Top-level directories mirror pipeline stages.** Reading the top level tells the pipeline story.
- **Future datasets add subdirs, not new top-level dirs.** Phase 3 adds `raw/pianomidi/`, `raw/lakh/`, `raw/musescore/` and corresponding `embeddings/` entries.

## Migration Mapping

| Old | New |
|-----|-----|
| `maestro_cache/` | `raw/maestro/` (audio/MIDI), `embeddings/maestro/` (embeddings) |
| `asap_cache/` | `raw/asap/` |
| `atepp_cache/` | `raw/atepp/` |
| `giantmidi_raw/` | `raw/giantmidi/` |
| `percepiano_cache/` | `embeddings/percepiano/` |
| `percepiano_midi/` | `midi/percepiano/` |
| `masterclass_cache/` | `embeddings/masterclass/` |
| `masterclass_pipeline/` | `raw/masterclass/` |
| `competition_cache/` | `embeddings/competition/` |
| `youtube_piano_cache/` | `raw/youtube/` |
| `pretrain_cache/` | `pretraining/` |
| `composite_labels/` | `labels/composite/` |
| `score_library/` | `scores/` |
| `reference_profiles/` | `references/` |
| `amt_cache/` | `midi/amt/` |
| `eval/` | `evals/` (split into subdirs) |
| `intermediate_cache/` | `evals/intermediate/` |
| `skill_eval/` | `evals/skill_eval/` |
| `teacher_voice_eval/` | `evals/teacher_voice/` |
| `experiment4_results/` | `results/experiment4/` |
| `a1_max_sweep_results.json` | `results/a1_max_sweep.json` |
| `calibration_stats.json` (in maestro_cache) | `calibration/maestro_stats.json` |

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
    teacher_voice = root / "teacher_voice"


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
/pretraining/
/checkpoints/

# Eval outputs (manifests/metadata tracked separately)
/evals/inference_cache/
/evals/traces/
/evals/youtube_amt/
/evals/teacher_voice/
/evals/skill_eval/*/audio/
/evals/skill_eval/*/results.json
```

## Migration Strategy

### Phase 1: Non-destructive setup
1. Create `model/src/paths.py`
2. Create new directory structure (empty dirs with `.gitkeep` where needed)
3. Create new `model/data/.gitignore`

### Phase 2: Move tracked files (git mv)
- `composite_labels/` -> `labels/composite/`
- `score_library/` -> `scores/`
- `reference_profiles/` -> `references/`
- `percepiano_midi/` -> `midi/percepiano/`
- `amt_cache/` -> `midi/amt/`
- `skill_eval/` -> `evals/skill_eval/`
- `intermediate_cache/` -> `evals/intermediate/`
- `experiment4_results/` -> `results/experiment4/`
- `a1_max_sweep_results.json` -> `results/a1_max_sweep.json`
- `calibration_stats.json` -> `calibration/maestro_stats.json`

Use `git mv` to preserve history.

### Phase 3: Move gitignored data (plain mv)
- `maestro_cache/` -> split `raw/maestro/` + `embeddings/maestro/`
- `percepiano_cache/` -> `embeddings/percepiano/`
- `pretrain_cache/` -> `pretraining/`
- `competition_cache/` -> `embeddings/competition/`
- `masterclass_cache/` -> `embeddings/masterclass/`
- `masterclass_pipeline/` -> `raw/masterclass/`
- `atepp_cache/` -> `raw/atepp/`
- `giantmidi_raw/` -> `raw/giantmidi/`
- `youtube_piano_cache/` -> `raw/youtube/`
- `eval/` -> split into `evals/` subdirs

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

Scripts (~15 files):
- `src/model_improvement/data.py` -- main training data loader
- `src/model_improvement/datasets.py` -- MIDI file discovery functions
- `src/model_improvement/maestro.py` -- MAESTRO metadata/embedding extraction
- `src/model_improvement/a1_max_sweep.py` -- A1-Max hyperparameter sweep
- `src/model_improvement/ablation_sweep.py` -- ablation study
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
