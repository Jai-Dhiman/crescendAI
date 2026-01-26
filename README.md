# CrescendAI

Audio-based piano performance evaluation using music foundation models.

## Overview

CrescendAI evaluates piano performances across 19 musical dimensions (timing, articulation, pedaling, dynamics, interpretation, etc.) using audio foundation models (MuQ, MERT). Our approach achieves **R² = 0.537** on the PercePiano benchmark, a 55% improvement over symbolic (MIDI-based) baselines.

For technical details, see our [paper](paper/arxiv/main.pdf).

## Project Structure

```
model/       PyTorch Lightning training pipeline
apps/web/    Leptos/Rust web application (Cloudflare Workers)
apps/inference/  HuggingFace inference endpoint
```

## Quick Start

### Training Pipeline

```bash
cd model
uv sync
uv run python -m audio_experiments.training.runner
```

### Web Application

```bash
cd apps/web
bun install
cargo leptos watch
```

## License

MIT
