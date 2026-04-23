# ML Training Pipeline

## Project Context

PyTorch Lightning pipeline for piano performance evaluation.

## Stack

- PyTorch Lightning 2.x
- MuQ (audio foundation model)
- nnAudio for GPU-accelerated spectrograms
- **MIDI**: mido, pretty_midi
- **Training**: HF Jobs (L4 $0.80/hr default, A100 $2.50/hr for Aria fine-tuning)
- **Experiment tracking**: Trackio (syncs to HF Space dashboard)
- **Training data**: HF Bucket (private, ~92GB, mounted in HF Jobs)
- **Local**: Apple M4, 32GB RAM (preprocessing + labeling)

## Datasets

- PercePiano: 19-dimension performance evaluation (1,202 segments)
- MAESTRO: MIDI/audio pairs for piano (200+ hours)
- ASAP: 1,067 performances of 236 scores (multi-performer)

## Notebook Conventions

Notebooks should be **minimal orchestration files** that import from `src/`. This keeps notebooks clean and code reusable.

### Setup (for cloud runtimes)

```python
# Clone repo and install
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

!curl -fsSL https://rclone.org/install.sh | sudo bash
!curl -LsSf https://astral.sh/uv/install.sh | sh

!git clone https://github.com/Jai-Dhiman/crescendai.git
%cd crescendai/model
!uv pip install -e .

# Add src to path
import sys
sys.path.insert(0, 'src')

import torch
import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')
```

## Notes

- Uses CUDA deterministic mode for reproducibility

## Research Thesis

Pretrained audio representations provide useful inductive biases for performance evaluation that are hard to learn from symbolic inputs alone given limited labeled data. The advantage reflects pretraining scale, not modality choice -- it may narrow as symbolic foundation models pretrained on large MIDI corpora become available.

## Architecture Overview

Two paths from the same underlying data: audio encoders (MuQ) and symbolic encoders (Transformer on REMI tokens, GNN on score graphs, continuous MIDI features). Fusion combines both. Training uses multiple data tiers: labeled perceptual annotations, competition ordinal rankings, cross-performer contrastive pairs, and augmentation invariance. Direction is toward replacing the 19 PercePiano dimensions with fewer teacher-grounded feedback categories derived from real masterclass teaching moments.

## Data Layout

Organized by pipeline stage. All paths defined in `src/paths.py`.

- `data/raw/` - Downloaded datasets (maestro, masterclass, youtube)
- `data/embeddings/` - Extracted MuQ embeddings (percepiano, masterclass, t5_muq, t5_aria, practice_corrupted)
- `data/midi/` - Small MIDI collections (percepiano ground truth, AMT test set)
- `data/labels/` - Annotations and derived labels (composite taxonomy, percepiano folds/labels)
- `data/manifests/` - Data source configs + `r2_offload.json` (offload registry)
- `data/scores/` - Score library JSON (mirrored to R2)
- `data/references/` - Reference performance profiles (mirrored to R2)
- `data/evals/` - Evaluation data (skill_eval, inference_cache, intermediate, ood_practice)
- `data/splits/` - Train/val/test fold assignments
- `data/checkpoints/` - Active training outputs (a1_max_sweep, contrastive_pretrain — mirrored to R2 for durability)
- `data/calibration/`, `data/results/`, `data/weights/` - Outputs

### Offload state (R2 bucket: `crescendai-bucket`)

R2 is the cold-storage tier. `paths.ensure_local()` checks `data/manifests/r2_offload.json` and raises a precise rclone-rehydrate hint when an offloaded path is accessed but missing.

| Local path | Status | R2 prefix / regen |
|---|---|---|
| `data/checkpoints/{ablation, B2_cross_soundfont, autoresearch_loss, model_improvement}` | offloaded | `checkpoints/archive/<name>` (legacy pre-Aria runs) |
| `data/evals/youtube_amt` | offloaded | `evals/youtube_amt` (845 MiB AMT validation set) |
| `data/raw/competition` | offloaded | `raw/competition` (9.2 GiB T2 origin audio, non-reproducible) |
| `data/raw/asap` | offloaded | regen via `git clone https://github.com/CPJKU/asap-dataset.git` |
| `data/embeddings/{maestro, competition}` | offloaded | regen via `extract_muq_embeddings.py` on HF Job (uplink too slow to upload) |
| `data/checkpoints/{a1_max_sweep, contrastive_pretrain}` | local primary, R2 mirror | `checkpoints/deployed/<name>` (durability backup; not auto-rehydrated) |
| Everything else under `data/` | local only | T5 labeling, percepiano, splits, midi, scores, references — all kept local for active work |

To rehydrate an offloaded path: read the exact `rclone copy ...` or `regen_command` from the FileNotFoundError message that `ensure_local()` raises, or look it up in `data/manifests/r2_offload.json`.
