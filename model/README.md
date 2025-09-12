# CrescendAI Model - Piano Performance Analysis

Audio Spectrogram Transformer (AST) for 19-dimensional piano performance analysis using JAX/Flax.

## Project Structure

```
crescendai_model/
├── __init__.py                 # Main package exports
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── audio_preprocessing.py  # Audio preprocessing pipeline
│   └── training.py            # Training pipeline and utilities
├── models/                     # Neural network architectures
│   ├── __init__.py
│   ├── ast_transformer.py     # Audio Spectrogram Transformer
│   ├── hybrid_ast.py          # Hybrid AST variants
│   └── ssast_pretraining.py   # Self-supervised pre-training
├── datasets/                   # Dataset loaders and processors
│   ├── __init__.py
│   ├── percepiano_dataset.py  # PercePiano dataset loader
│   ├── maestro_dataset.py     # MAESTRO dataset loader
│   └── ccmusic_piano_dataset.py  # CC Music dataset loader
├── api/                        # API contracts and interfaces
│   ├── __init__.py
│   └── contracts.py           # Pydantic models for API
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── preprocessing.py       # Preprocessing helpers
└── deployment/                 # Deployment utilities
    ├── __init__.py
    ├── modal_service.py       # Modal service deployment
    └── deploy.py              # Deployment management script

# Additional files
├── deploy.py                  # Top-level deployment entry point
├── pyproject.toml            # Package configuration and dependencies
├── README_DEPLOYMENT.md      # Detailed deployment guide
├── results/                  # Training results and models
│   └── final_finetuned_model.pkl  # Trained model (327MB)
├── PercePiano/              # Original dataset (external)
└── tests/                   # Unit tests
```

## Quick Start

### Installation
```bash
# Install dependencies using uv
uv sync

# Or install the package in development mode
uv pip install -e .
```

### Usage

#### Import the main components
```python
from crescendai_model import (
    PianoAudioPreprocessor,
    AudioSpectrogramTransformer,
    PerformanceDimensions
)

# Initialize preprocessor
preprocessor = PianoAudioPreprocessor(target_sr=22050)

# Load and preprocess audio
audio_data, sr = preprocessor.load_and_normalize_audio("piano.wav")
features = preprocessor.extract_spectral_features(audio_data, sr)
```

#### Train a model
```python
from crescendai_model.core.training import ASTTrainingPipeline

# Initialize training pipeline
config = {
    "checkpoint_dir": "./checkpoints",
    "results_dir": "./results",
    "seed": 42
}

pipeline = ASTTrainingPipeline(config)
# ... training code
```

#### Deploy to Modal
```bash
# Set up Modal authentication
modal token new

# Deploy the service
python deploy.py

# Choose option 3: "Test locally then deploy"
```

## Architecture

The model uses an Audio Spectrogram Transformer (AST) with:
- **86M parameters** for comprehensive analysis
- **16×16 patch embeddings** from mel-spectrograms  
- **12-layer transformer** with multi-head attention
- **19-dimensional output** for perceptual analysis

## Dataset Insights

**PercePiano Analysis:**

- **1202 performances** across 19 perceptual dimensions
- **22 professional performers**, classical repertoire (Schubert, Beethoven)
- **Perceptual ratings**: [0-1] normalized, mean=0.553
- **Key correlations**: Musical expression dimensions show strong inter-relationships

## Implementation Plan

**Phase 1**: AST Baseline (Current)

1. Implement Audio Spectrogram Transformer architecture
2. Train on PercePiano mel-spectrograms → 19-dimensional ratings  
3. Achieve SOTA performance (target: >0.7 correlation on key dimensions)
4. Comprehensive evaluation and comparison with baseline approaches

**Phase 2**: Research Extensions (Future)

- Cross-cultural musical perception studies
- Interpretable attention visualization
- Few-shot learning for new instruments
- Real-time performance feedback applications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff):

```bibtex
@software{piano_analysis_model,
  title = {Piano Performance Analysis Model},
  author = {Piano Analysis User},
  year = {2025},
  url = {https://github.com/username/piano-analysis-model}
}
```

## Dataset Attribution

This project uses and extends the **PercePiano dataset** for piano performance analysis:

- **Original Dataset**: Cancino-Chacón, C. E., Grachten, M., & Widmer, G. (2017). PercePiano: A Dataset for Piano Performance Analysis. *Proceedings of the International Society for Music Information Retrieval Conference*, 55-62.
- **Dataset License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Audio Source**: Classical piano performances from various composers
- **Labels**: Perceptual annotations across 19 dimensions

## Data Use and Redistribution

- **Code**: Available under MIT License - free to use, modify, and distribute
- **PercePiano Dataset**: Used under CC BY 4.0 - attribution required for any use
- **Audio Files**: Sample audio included for demonstration purposes only
- **Redistribution**: Full dataset redistribution must comply with original CC BY 4.0 terms

For questions about dataset usage or to access the complete PercePiano dataset, contact the original authors through the [ISMIR 2017 publication](https://doi.org/10.5334/tismir.17).

---
*Learning-focused implementation - building everything from scratch for deep understanding*
