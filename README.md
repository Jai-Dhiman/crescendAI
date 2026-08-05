[![arXiv](https://img.shields.io/badge/arXiv-2601.19029-b31b1b.svg)](https://arxiv.org/abs/2601.19029)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Cloudflare Workers](https://img.shields.io/badge/Cloudflare_Workers-F38020?logo=cloudflare&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface)
![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-blue)

# CrescendAI

**A teacher for every pianist.**

Practice companion for evidence-backed piano feedback. The current inference path combines a frozen MuQ checkpoint with separately trained A1-Max heads for six research scores, plus Transkun transcription and symbolic score alignment. The V6 harness turns those signals into one actionable observation and preserves uncertainty rather than treating any model score as ground truth.

## Historical Model Benchmark

**79.85% pairwise accuracy** on PercePiano benchmark (A1-Max, 4-fold piece-stratified CV, clean folds)

| Encoder | Type | Pairwise Accuracy | Notes |
|---------|------|-------------------|-------|
| A1-Max (4-fold ensemble) | Audio | 79.85% | Historical clean-fold training result; the current loader serves stock MuQ plus the saved heads, not the fine-tuned backbone |

> **Fold integrity note:** Earlier results (80.8%, S2 GNN 71.3%) were computed with leaked folds where segments from the same piece appeared in both train and val splits. Those numbers are invalid. All current results use piece-stratified CV only.

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
         |  (TypeScript / Hono)        |
         |                             |
         |  /api/practice/chunk        |
         |  /api/chat/send (SSE)       |
         |  /api/auth/*                |
         |  /api/sync                  |
         +--+------+------+------+----+
            |      |      |      |
            v      v      v      v
      +-------+ +------+ +------+ +----+
      | HF    | | CF   | | AI   | | PG |
      | Endpt | | Cont-| | Gate-| | R2 |
      | (MuQ  | | ainer| | way  | | DO |
      | A1Max)| | (AMT)| |(LLMs)| |    |
      +-------+ +------+ +------+ +----+
```

| Component | Details |
|-----------|---------|
| Audio scoring | A1-Max inference code loads the frozen `OpenMuQ/MuQ-large-msd-iter` checkpoint plus four trained heads |
| AMT | Transkun transcription service in `apps/inference/amt/`; production deployment remains deferred |
| iOS app | SwiftUI, AVAudioEngine, SwiftData (local-first) |
| Web app | TanStack Start (`crescend.ai`) -- chat, recording, real-time observations via WebSocket |
| API backend | TypeScript/Hono on Cloudflare Workers (`api.crescend.ai`) -- inference proxy, SessionBrain DO, two-stage LLM pipeline, score following |
| LLM pipeline | V6 two-phase harness on Workers AI, with Anthropic retained as a selectable provider. Post-session synthesis runs through SessionBrain. |
| Score following | Transkun notes aligned against a known score; #133 is validating the follower on real audio before it becomes the trusted ruler |
| Storage | PlanetScale Postgres + Drizzle ORM (via Hyperdrive), R2 (audio chunks), Durable Objects (practice sessions + WebSocket) |
| Auth | better-auth -- Sign in with Apple + Google |

## Project Structure

```
apps/ios/          Native iOS app (SwiftUI, AVAudioEngine, cloud inference)
apps/api/          TypeScript/Hono API Worker (Cloudflare Workers)
apps/web/          TanStack Start web practice companion (React, Tailwind CSS v4)
apps/inference/    HuggingFace inference handler (MuQ) + AMT server (CF Containers)
evals/             E2E pipeline evaluation infrastructure (teaching quality, synthesis eval)
model/             Research pipelines, evaluation harnesses, and historical MuQ/Aria experiments
docs/              Architecture and documentation
  docs/apps/       Apps layer (status, product vision, pipeline, memory, exercises, UI)
  docs/model/      ML layer (research timeline, data, taxonomy, encoders, north star)
```

## Documentation

See [docs/architecture.md](docs/architecture.md) for the full system diagram and documentation map.

| Area | Entry Point | Contents |
|------|-------------|----------|
| Architecture | [docs/architecture.md](docs/architecture.md) | System diagram, cross-cutting concerns (auth, sync, observability) |
| Product vision | [docs/apps/01-product-vision.md](docs/apps/01-product-vision.md) | Target user, interaction model, UX principles |
| Pipeline | [docs/apps/02-pipeline.md](docs/apps/02-pipeline.md) | Full audio-to-observation pipeline (STOP, subagent, teacher, score following) |
| Student memory | [docs/apps/03-memory-system.md](docs/apps/03-memory-system.md) | Two-clock model, observations, synthesized facts |
| Exercises | [docs/apps/04-exercises.md](docs/apps/04-exercises.md) | Exercise database, focus mode |
| UI system | [docs/apps/05-ui-system.md](docs/apps/05-ui-system.md) | Chat interface, on-demand components |
| ML research | [docs/model/00-research-timeline.md](docs/model/00-research-timeline.md) | Research roadmap, encoder results, key decisions |
| Training data | [docs/model/01-data.md](docs/model/01-data.md) | PercePiano, Competition, MAESTRO datasets |
| Taxonomy | [docs/model/02-teacher-grounded-taxonomy.md](docs/model/02-teacher-grounded-taxonomy.md) | 19 PercePiano dims to 6 teacher-grounded dims |
| Encoders | [docs/model/03-encoders.md](docs/model/03-encoders.md) | A1-Max audio encoder, Aria symbolic encoder |
| North star | [docs/model/04-north-star.md](docs/model/04-north-star.md) | 8-stage perfect pipeline vision |

## Setup

Install [just](https://github.com/casey/just): `brew install just`

| Command | What it starts |
|---------|----------------|
| `just dev` | All 4 services: MuQ (8000) + AMT (8001) + API (8787) + Web (3000) |
| `just dev-muq` | MuQ + API + Web (no AMT, faster startup) |
| `just dev-light` | API + Web only (uses production HF endpoints) |
| `just muq` / `just amt` / `just api` / `just web` | Individual services |
| `just fingerprint` | Generate N-gram index + rerank features from score library |
| `just test-model` / `just test-api` / `just check-api` | Tests and type checks |
| `just deploy-api` | Deploy API worker to production |
| `just migrate-generate` / `just migrate-prod` | Drizzle migrations (generate SQL / apply to prod) |
| `just eval-e2e` | Full E2E pipeline eval (cache + pipeline + analyze) |

### iOS App

```bash
open apps/ios/CrescendAI.xcodeproj
```

### Training Pipeline

```bash
cd model
uv sync
uv run python -m audio_experiments.training.runner
```

## Paper

For technical details on the audio encoder, see the [paper on arXiv](https://arxiv.org/abs/2601.19029).
