# Piano Performance Evaluation with Audio Foundation Models

Audio foundation models (MuQ, MERT) for automated piano performance evaluation, achieving state-of-the-art results on the PercePiano benchmark.

**Paper**: "Audio Foundation Models Outperform Symbolic Representations for Piano Performance Evaluation" (ISMIR 2026 submission)

## Key Results

| Model | R2 | 95% CI | Improvement |
|-------|-----|--------|-------------|
| **MuQ + Pianoteq** | **0.537** | [0.465, 0.575] | **+55% vs symbolic** |
| MuQ L9-12 | 0.533 | [0.514, 0.560] | +53% |
| MERT L7-12 | 0.487 | [0.460, 0.510] | +40% |
| Symbolic baseline | 0.347 | [0.315, 0.375] | - |

- Audio outperforms symbolic on **all 19 dimensions** (p < 10^-25)
- Cross-soundfont generalization: R2 = 0.534
- External validation: Spearman rho = 0.623 with piece difficulty

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/piano-eval.git
cd piano-eval/model

# Install dependencies (using uv)
uv pip install -e ".[gpu]"

# Or using pip
pip install -e ".[gpu]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- ~8GB GPU memory for inference
- ~32GB GPU memory for training

## Quick Start

### Inference

Evaluate a piano performance:

```bash
python scripts/inference.py recording.wav
```

With custom checkpoint:

```bash
python scripts/inference.py recording.wav --checkpoint path/to/model.ckpt --output results.json
```

Batch evaluation:

```bash
python scripts/inference.py ./recordings/ --batch --output results.json
```

### Python API

```python
from scripts.inference import load_model, evaluate_audio

# Load pre-trained model
model = load_model("checkpoints/muq_best.ckpt")

# Evaluate audio file
results = evaluate_audio("performance.wav", model)

print(f"Overall score: {results['overall_score']:.3f}")
for dim, score in results['scores'].items():
    print(f"  {dim}: {score:.3f}")
```

### Training

Run full 4-fold cross-validation experiment:

```python
from audio_experiments.training import run_4fold_mert_experiment
from audio_experiments.models import MuQBaseModel

results = run_4fold_mert_experiment(
    model_class=MuQBaseModel,
    experiment_name="muq_l9_12",
    layer_start=9,
    layer_end=12,
)

print(f"Mean R2: {results['mean_r2']:.3f}")
```

## Project Structure

```
model/
├── src/audio_experiments/     # Core training code
│   ├── extractors/            # MuQ/MERT feature extraction
│   ├── models/                # Model architectures
│   ├── data/                  # Dataset classes
│   └── training/              # Training runners and metrics
├── notebooks/                 # Experiment notebooks
│   ├── 01_main_experiments.ipynb
│   ├── 02_muq_fusion_experiments.ipynb
│   └── 03_mert_baseline_experiments.ipynb
├── scripts/
│   └── inference.py           # Standalone inference script
├── figures/                   # Paper figures
├── docs/                      # Documentation
│   ├── AUDIO_EXPERIMENTS_REPORT.md  # Complete results
│   └── PAPER_ROADMAP.md       # Paper planning
└── data/
    └── checkpoints/           # Model weights
```

## Pre-trained Models

Download pre-trained checkpoints:

```bash
# From Google Drive (requires gdown)
gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# Or direct download
wget https://YOUR_HOST/checkpoints/muq_best.ckpt -O data/checkpoints/muq_best.ckpt
```

Available checkpoints:
- `muq_best.ckpt` - Best MuQ model (R2=0.537)
- `mert_best.ckpt` - Best MERT model (R2=0.487)

## Evaluation Dimensions

The model predicts 19 perceptual dimensions from PercePiano:

| Category | Dimensions |
|----------|------------|
| Technical | timing |
| Articulation | length, touch |
| Pedaling | amount, clarity |
| Timbre | variety, depth, brightness, loudness |
| Dynamics | dynamic_range |
| Musical | tempo, space, balance, drama |
| Mood | valence, energy, imagination |
| Interpretation | sophistication, interpretation |

## Experiments

All experiments are documented in `docs/AUDIO_EXPERIMENTS_REPORT.md`:

- **Phase 2**: Audio baselines (13 experiments)
- **Phase 3**: Fusion experiments (12 experiments)
- **Phase 4**: MuQ layer ablations
- **Phase 5**: Statistical analysis
- **Phase 6-9**: Cross-dataset validation

Reproduce main experiments:

```bash
# Run main experiments notebook
jupyter notebook notebooks/01_main_experiments.ipynb
```

## Citation

```bibtex
@inproceedings{crescendai2026piano,
  title={Audio Foundation Models Outperform Symbolic Representations for Piano Performance Evaluation},
  author={Anonymous},
  booktitle={Proceedings of ISMIR},
  year={2026}
}
```

## License

MIT License. See `LICENSE` file.

Note: This code uses pre-trained models (MuQ, MERT) with their own licenses. See LICENSE for details.

## Acknowledgments

- [PercePiano](https://percepiano.github.io/) for the benchmark dataset
- [MuQ](https://github.com/bytedance/muq) and [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) for pre-trained audio models
- [Pianoteq](https://www.modartt.com/pianoteq) for high-quality piano synthesis
