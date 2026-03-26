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

- `data/raw/` - Downloaded datasets (maestro, asap, atepp, giantmidi, masterclass, youtube, competition)
- `data/embeddings/` - Extracted MuQ embeddings (percepiano, maestro, masterclass, competition)
- `data/midi/` - Small MIDI collections (percepiano ground truth, AMT test set)
- `data/pretraining/` - Symbolic pretraining corpus (tokens, graphs, features from 24K+ MIDI)
- `data/labels/` - Annotations and derived labels (composite taxonomy, percepiano folds/labels)
- `data/manifests/` - Data source configs (masterclass sources, YouTube channels, competition metadata)
- `data/scores/` - Score library JSON (deployed to R2)
- `data/references/` - Reference performance profiles (deployed to R2)
- `data/evals/` - Evaluation data (skill eval manifests, inference cache, intermediate recordings)
- `data/checkpoints/` - Trained model weights
- `data/results/` - Experiment results
- `data/calibration/` - MAESTRO calibration stats
