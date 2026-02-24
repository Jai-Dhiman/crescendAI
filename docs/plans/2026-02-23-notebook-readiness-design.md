# Notebook Readiness Design: Audio + Symbolic Training

Date: 2026-02-23
Status: Approved
Scope: Get audio and symbolic training/comparison notebooks fully ready for Thunder Compute runtime. Fusion deferred.

## Context

The teacher-grounded taxonomy work is complete (all 5 validation gates PASS). The LLM distillation pilot returned NO-GO (0/6 dimensions pass r > 0.5). Training proceeds on the baseline path: L_regression uses T1 composite labels only (1,202 PercePiano segments, 6 dims). Round 0 (Distillation A/B) is skipped. T2/T3/T4 contribute via ranking, contrastive, and invariance loss terms without regression labels.

All data is collected and cached in GDrive (`gdrive:crescendai_data/model_improvement/data/`):
- T1 percepiano_cache: 1,205 objects, 4.4 GB
- T2 competition_cache: 2,295 objects, 3.3 GB
- T3 maestro_cache: 24,323 objects, 34.2 GB
- Symbolic pretrain_cache: 187 objects, 37.7 GB
- asap_cache: 2,132 objects, 67 MB
- percepiano_midi: 1,202 objects, 2.7 MB

## Approach: Layered (shared foundation -> audio -> symbolic -> docs)

All logic lives in Python modules under `model/src/model_improvement/`; notebooks import and orchestrate.

## Layer 1: Shared Modules

### New: `taxonomy.py`

Single source of truth for the 6-dim label system.

```
DIMENSIONS: list[str]  # ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
NUM_DIMS: int = 6

load_composite_labels(path) -> dict[str, np.ndarray]    # segment_id -> [6] array
load_dimension_definitions(path) -> dict                 # full taxonomy metadata
load_percepiano_to_composite_mapping() -> dict           # DIM_MAPPING from bridge
get_dimension_names() -> list[str]                       # ordered names
```

### New: `evaluation.py`

Central logic imported by both comparison notebooks.

```
evaluate_audio_model(model, dataloader, fold) -> dict
evaluate_symbolic_model(model, dataloader, fold) -> dict
aggregate_folds(fold_results: list[dict]) -> dict         # mean/std across folds
run_robustness_eval(model, dataloader, noise_std=0.05, augmented_dir=None) -> dict
run_competition_eval(model, competition_data) -> dict     # graceful skip if no data
select_winner(results: dict[str, dict], primary="pairwise_accuracy") -> str
```

### Update: `datasets.py`

Replace all `num_labels=19` with import from `taxonomy.NUM_DIMS`. All dataset classes accept `num_labels` as constructor arg defaulting to `NUM_DIMS`.

### Update: `metrics.py`

Add:
- `stop_auc()` -- LOVO-CV AUC for STOP prediction
- `competition_spearman()` -- Spearman rho vs placement (graceful skip if no T2 data)
- `per_dimension_breakdown()` -- dict of metric per dimension using `taxonomy.DIMENSIONS`
- `format_comparison_table()` -- add columns for STOP AUC, competition rho

### Update: `audio_encoders.py` and `symbolic_encoders.py`

Replace all `num_labels=19` defaults with `taxonomy.NUM_DIMS`. No structural changes.

### No change: `losses.py`

`DimensionWiseRankingLoss` already parameterized by `num_dims`.

## Layer 2: Audio Path

### `01_audio_training.ipynb`

- Import `taxonomy.load_composite_labels()` instead of raw PercePiano 19-dim labels
- `num_labels=6` flows from `taxonomy.NUM_DIMS` into A1, A2, A3 constructors
- Wire T2/T3 dataloaders into training (graceful skip if cache not synced)
- Fix rclone paths to `gdrive:crescendai_data/model_improvement/data/`
- Remove Round 0 distillation A/B cell

### `03_audio_comparison.ipynb`

- Import `evaluate_audio_model()` and `aggregate_folds()` from `evaluation.py`
- Full metrics: R^2, pairwise accuracy, per-dimension breakdown (6 dims), STOP AUC
- Robustness: Gaussian noise proxy (sigma=0.05); use real T4 augmented embeddings if available
- Competition: Spearman rho vs placement if T2 data exists; graceful skip otherwise
- Winner selection: primary pairwise accuracy, tiebreak R^2, veto robustness drop > 15%
- Per-dimension bar chart with fold variance error bars

## Layer 3: Symbolic Path

### `02_symbolic_training.ipynb`

- Same label system switch as audio: `taxonomy.load_composite_labels()`, `num_labels=6`
- Two-stage training per encoder:
  - Pretrain (50 epochs on 14K corpus, 95/5 split): S1 MLM, S2 link prediction, S2H hetero link prediction, S3 contrastive masked frames
  - Finetune (150 epochs on PercePiano 4-fold CV): ranking + regression + contrastive
- S2H keeps `batch_size=1, num_workers=0` (HeteroPretrainDataset nested dict limitation)
- Fix rclone paths

### `04_symbolic_comparison.ipynb`

- Import `evaluate_symbolic_model()` from `evaluation.py`
- `evaluation.py` accepts a `preprocessor` callable per encoder type (tokens for S1, graph for S2/S2H, features for S3)
- Full metrics: same suite as audio comparison
- Score alignment (unique to symbolic): cosine similarity between performance and score-only embeddings using `asap_cache/`; graceful skip if not synced
- Winner selection: primary pairwise accuracy, tiebreak R^2, veto robustness drop > 15%, bonus score alignment (informational)

## Layer 4: Docs + Data Collection Verification

### `docs/03-model-improvement.md` -- surgical update

1. Prerequisites section: add status note (taxonomy COMPLETE, distillation NO-GO, data COMPLETE)
2. Round 0: mark as SKIPPED with one-line rationale
3. Label references: annotate "19 PercePiano dimensions" with resolved 6-dim composite state

### `docs/01-data-collection.md` -- status annotations

Add per-tier status: T1 COMPLETE (4.4 GB), T2 COMPLETE (3.3 GB), T3 COMPLETE (34.2 GB), T4 DEFERRED. Note composite labels deliverable COMPLETE.

### `docs/02-teacher-grounded-taxonomy.md` -- completion note

Add note at top: all deliverables produced, all 5 gates PASS, distillation NO-GO. Link to `model/data/composite_labels/`.

### `00_data_collection.ipynb` -- verification cell

Add a cell at top that runs `rclone lsd` / `rclone size` against each cache directory and prints a status table (PRESENT/MISSING + object count + size). Fix rclone path if needed. No changes to collection logic.

## File Change Summary

| Layer | New files | Modified files |
|-------|-----------|----------------|
| 1. Shared | `taxonomy.py`, `evaluation.py` | `datasets.py`, `metrics.py`, `audio_encoders.py`, `symbolic_encoders.py` |
| 2. Audio | -- | `01_audio_training.ipynb`, `03_audio_comparison.ipynb` |
| 3. Symbolic | -- | `02_symbolic_training.ipynb`, `04_symbolic_comparison.ipynb` |
| 4. Docs | -- | `docs/01-*.md`, `docs/02-*.md`, `docs/03-*.md`, `00_data_collection.ipynb` |
