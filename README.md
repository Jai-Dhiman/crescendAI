# CrescendAI

**A teacher for every pianist.**

iOS-first practice companion that evaluates *how* a piano performance sounds -- tone, dynamics, phrasing, pedaling -- not just note accuracy. On-device audio inference via a finetuned MuQ foundation model, with a thin cloud backend for LLM feedback and data sync.

## Key Result

**R² = 0.537** on PercePiano benchmark (55% improvement over symbolic baselines)

![Model Pipeline](model/figures/excalidraw_model_pipeline.png)

| Model | R² | 95% CI | MAE |
|-------|-----|--------|-----|
| MuQ L9-12 (ours) | 0.537 | [0.465, 0.575] | 0.067 |
| Symbolic baseline | 0.347 | [0.315, 0.375] | 0.095 |

## Architecture

| Component | Details |
|-----------|---------|
| On-device inference | Core ML MuQ (~300M params), 6-dimension output |
| iOS app | SwiftUI, AVAudioEngine, SwiftData (local-first) |
| API backend | Rust Axum on Cloudflare Workers (`api.crescend.ai`) |
| Landing page | TanStack Start + Tailwind CSS v4 (`crescend.ai`) |
| Storage | D1 (SQLite), R2 (audio), KV (cache) |
| Auth | Sign in with Apple |

## Project Structure

```
apps/ios/          Native iOS app (SwiftUI, Core ML, AVAudioEngine)
apps/api/          Rust API Worker (Axum on Cloudflare Workers)
apps/web/          TanStack Start landing page (React, Tailwind CSS v4)
apps/inference/    HuggingFace inference endpoint (cloud fallback)
model/             PyTorch Lightning training pipeline
docs/              Architecture and implementation slices
```

## Documentation

See [docs/index.md](docs/index.md) for the full documentation hub -- architecture, implementation slices with status, model training docs, and getting-started guides.

## Setup

### Training Pipeline

```bash
cd model
uv sync
uv run python -m audio_experiments.training.runner
```

### API Worker

```bash
cd apps/api
npx wrangler dev
```

### Landing Page

```bash
cd apps/web
bun install
bun run dev
```

## Paper

For technical details, see the [paper on arXiv](https://arxiv.org/abs/2601.19029).
