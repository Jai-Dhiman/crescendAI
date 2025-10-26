# Piano Performance Evaluation - PyTorch Lightning MVP

Proof-of-concept deep learning system for automated piano performance assessment across 8-10 evaluation dimensions.

## Overview

Multi-modal (audio + MIDI) architecture combining MERT-95M pre-trained music encoder with MIDIBert for comprehensive performance evaluation:

- **Technical dimensions**: note accuracy, rhythm, dynamics, articulation, pedaling, tone quality
- **Interpretive dimensions**: phrasing, musicality, overall quality, expressiveness

## Key Features

- MERT-95M pre-trained music encoder (95M parameters, Colab-friendly)
- Multi-modal audio-MIDI cross-attention fusion
- Uncertainty-weighted multi-task learning (Kendall & Gal)
- Hierarchical BiLSTM temporal aggregation
- Attention-based explainability

## Dataset Requirements

- **Pseudo-labels**: 50-100 MAESTRO pieces (~10-15 hours)
- **Expert labels**: 200-300 segments @ 20-30s each (~10-15 hours labeling)
- **Format**: 44.1kHz stereo audio + aligned MIDI scores

## Training Budget

- **Compute**: 20-25 GPU hours on Colab Pro (T4/V100)
- **Storage**: ~50GB (MAESTRO subset + checkpoints)

## Expected Performance

- Technical dimensions: r=0.50-0.65 (Pearson correlation with expert ratings)
- Interpretive dimensions: r=0.35-0.50
- MAE: 10-15 points on 0-100 scale

## Documentation

- `ARCHITECTURE.md` - Detailed system design and component specifications
- `TASKS.md` - Implementation roadmap and task breakdown
- `RESEARCH.md` - Full literature review and research justification

## Next Steps

This MVP validates the architecture with minimal labeled data. After demonstrating viability:

1. Hire 10 expert annotators for 2,500 professional labels
2. Scale to MERT-330M with ensemble specialists
3. Achieve professional-grade r=0.68-0.75 correlation

## License

Research/Educational Use
