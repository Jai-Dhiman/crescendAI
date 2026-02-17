# Piano Performance Evaluation with Audio Foundation Models

Audio foundation model (MuQ) for automated piano performance evaluation, achieving state-of-the-art results on the PercePiano benchmark.

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

# ML Training Pipeline

PyTorch Lightning pipeline for piano performance evaluation.

## Stack

- PyTorch Lightning 2.x
- MuQ (audio foundation model)
- nnAudio for GPU-accelerated spectrograms
- **MIDI**: mido, pretty_midi
- **Training**: Thunder Compute (A100 GPU/80GB VRAM with 4 vCPU/32GB RAM)
- **Local**: Apple M4, 32GB RAM (preprocessing + labeling)

## Datasets

- PercePiano: 19-dimension performance evaluation (1,202 segments)
- MAESTRO: MIDI/audio pairs for piano (200+ hours)
- ASAP: 1,067 performances of 236 scores (multi-performer)
