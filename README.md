[![sloprank](https://sloprank.io/badge/jai-dhiman/crescendai.svg)](https://sloprank.io/repo/jai-dhiman/crescendai)

# CrescendAI

**A teacher for every pianist.**

Multi-platform (iOS + web) practice companion that evaluates *how* a piano performance sounds -- dynamics, timing, pedaling, articulation, phrasing, interpretation -- not just note accuracy like MIDI-based apps. A finetuned MuQ audio foundation model scores 6 teacher-grounded dimensions from audio alone. A two-stage LLM pipeline (fast subagent analysis + quality teacher delivery) turns those scores into one actionable observation per practice moment.

## Key Result

**80.8% pairwise accuracy** on PercePiano benchmark (A1-Max 4-fold ensemble)

| Encoder | Type | Pairwise Accuracy | Notes |
|---------|------|-------------------|-------|
| A1-Max (ensemble) | Audio | 80.8% | MuQ + LoRA rank-32, ListMLE, CCC. Deployed on HF endpoint |
| S2 (GNN) | Symbolic | 71.3% | GATConv on score graphs, link prediction pretraining |

6 output dimensions: **dynamics**, **timing**, **pedaling**, **articulation**, **phrasing**, **interpretation**

## Architecture

```
+-------------------+       +-------------------+
|    iOS App        |       |    Web App         |
|  (SwiftUI,        |       |  (TanStack Start,  |
|   AVAudioEngine)  |       |   MediaRecorder)   |
+--------+----------+       +---------+----------+
         |                             |
         |  15s audio chunks (HTTPS)   |
         +----------+    +------------+
                    |    |
                    v    v
         +----------------------------+
         |  Cloudflare Workers         |
         |  api.crescend.ai            |
         |  (Rust/Axum on WASM)        |
         |                             |
         |  /api/practice/chunk        |
         |  /api/ask                   |
         |  /api/chat/send             |
         |  /api/auth/apple            |
         |  /api/sync                  |
         +--+------+------+------+----+
            |      |      |      |
            v      v      v      v
      +-------+ +-----+ +------+ +----+
      | HF    | | Groq| | Anth-| | D1 |
      | Endpt | | API | | ropic| |    |
      | (MuQ  | | sub-| | teach| | KV |
      | A1-Max)| | agent| | er  | | R2 |
      +-------+ +-----+ +------+ | DO |
                                  +----+
```

| Component | Details |
|-----------|---------|
| Cloud inference | HF Inference Endpoint -- A1-Max 4-fold ensemble + AMT + pedal CC64 extraction |
| iOS app | SwiftUI, AVAudioEngine, SwiftData (local-first) |
| Web app | TanStack Start practice companion (`crescend.ai`) -- chat, recording, real-time observations via WebSocket |
| API backend | Rust Axum on Cloudflare Workers (`api.crescend.ai`) -- inference proxy, STOP classifier, two-stage LLM pipeline, score following |
| LLM pipeline | Groq (fast subagent analysis) + Anthropic (teacher delivery), OpenRouter fallback |
| Score following | Onset+pitch DTW aligning AMT output to score MIDI -- feedback references bar numbers, not timestamps |
| Storage | D1 (SQLite), R2 (audio chunks), KV (cache), Durable Objects (practice sessions + WebSocket) |
| Auth | Sign in with Apple + Google |

## Project Structure

```
apps/ios/          Native iOS app (SwiftUI, AVAudioEngine, cloud inference)
apps/api/          Rust API Worker (Axum on Cloudflare Workers)
apps/web/          TanStack Start web practice companion (React, Tailwind CSS v4)
apps/inference/    HuggingFace inference endpoint handler (primary inference path)
model/             PyTorch Lightning training pipeline
docs/              Architecture and documentation
  docs/apps/       Apps layer (status, product vision, pipeline, memory, exercises, UI)
  docs/model/      ML layer (research timeline, data, taxonomy, encoders, north star)
```

## Documentation

See [docs/architecture.md](docs/architecture.md) for the full system diagram and documentation map.

| Area | Entry Point | Contents |
|------|-------------|----------|
| Architecture | [docs/architecture.md](docs/architecture.md) | System diagram, cross-cutting concerns (auth, sync, observability) |
| Apps status | [docs/apps/00-status.md](docs/apps/00-status.md) | Implementation dashboard for iOS, web, API, inference |
| Product vision | [docs/apps/01-product-vision.md](docs/apps/01-product-vision.md) | Target user, interaction model, UX principles |
| Pipeline | [docs/apps/02-pipeline.md](docs/apps/02-pipeline.md) | Full audio-to-observation pipeline (STOP, subagent, teacher, score following) |
| Student memory | [docs/apps/03-memory-system.md](docs/apps/03-memory-system.md) | Two-clock model, observations, synthesized facts |
| Exercises | [docs/apps/04-exercises.md](docs/apps/04-exercises.md) | Exercise database, focus mode |
| UI system | [docs/apps/05-ui-system.md](docs/apps/05-ui-system.md) | Chat interface, on-demand components |
| ML research | [docs/model/00-research-timeline.md](docs/model/00-research-timeline.md) | Research roadmap, encoder results, key decisions |
| Training data | [docs/model/01-data.md](docs/model/01-data.md) | PercePiano, Competition, MAESTRO datasets |
| Taxonomy | [docs/model/02-teacher-grounded-taxonomy.md](docs/model/02-teacher-grounded-taxonomy.md) | 19 PercePiano dims to 6 teacher-grounded dims |
| Encoders | [docs/model/03-encoders.md](docs/model/03-encoders.md) | A1-Max audio encoder, S2 symbolic encoder |
| North star | [docs/model/04-north-star.md](docs/model/04-north-star.md) | 8-stage perfect pipeline vision |

## Setup

### iOS App

```bash
open apps/ios/CrescendAI.xcodeproj
```

### Web App

```bash
cd apps/web
bun install
bun run dev
```

### API Worker

```bash
cd apps/api
npx wrangler dev
```

### Training Pipeline

```bash
cd model
uv sync
uv run python -m audio_experiments.training.runner
```

## Paper

For technical details on the audio encoder, see the [paper on arXiv](https://arxiv.org/abs/2601.19029).
