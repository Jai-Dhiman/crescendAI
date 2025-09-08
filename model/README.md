# Piano Performance Analysis with Music Transformers

**State-of-the-art transformer-based approach for predicting perceptual dimensions of piano performance**

## Research Goal

Develop cutting-edge Audio Spectrogram Transformer (AST) models to predict 19 perceptual dimensions of piano performance, building on the PercePiano dataset with modern deep learning architectures for potential graduate-level research contributions.

**Perceptual Dimensions**: Timing, Articulation, Pedal, Timbre, Dynamics, Musical Expression, Emotion, Interpretation

## Approach

**Current Phase**: Music Transformer Implementation

- ✅ PercePiano dataset analysis (1202 performances, 19 dimensions)
- ✅ Audio preprocessing pipeline (librosa-based)
- ✅ Baseline neural networks (correlation analysis completed)
- 🚧 Audio Spectrogram Transformer (AST) implementation
- 🎯 Multi-task transformer training on mel-spectrograms

**Technical Foundation**: JAX/Flax implementation following SOTA transformer architectures

## Repository Structure

```
piano-analysis-model/
├── src/                    # Core implementation
│   ├── piano_cnn_jax.py   # Legacy CNN architectures
│   ├── ast_transformer.py  # Audio Spectrogram Transformer (main)
│   ├── training_pipeline.py # JAX/Flax training loop
│   └── audio_preprocessing.py # Spectrogram generation
├── PercePiano/            # Original dataset and research
├── data/                  # Preprocessed audio and labels
├── docs/                  # Project documentation and planning
├── models/                # Trained transformer checkpoints
├── results/               # Training metrics and analysis
└── notebooks/             # Research experiments and visualization
```

## Technical Architecture

### Audio Spectrogram Transformer (AST)

- **Input**: Mel-spectrograms (128 frequency bins × time frames)
- **Patch Embedding**: 16×16 patches → 768-dimensional vectors
- **Transformer**: 12-layer encoder with multi-head self-attention
- **Multi-task Head**: 19 parallel regression outputs for perceptual dimensions

### Training Pipeline

- **Framework**: JAX/Flax for high-performance training
- **Optimization**: AdamW with cosine learning rate scheduling
- **Evaluation**: Pearson correlation per dimension + cross-validation

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
