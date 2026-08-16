[![arXiv](https://img.shields.io/badge/arXiv-2601.19029-b31b1b.svg)](https://arxiv.org/abs/2601.19029)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Cloudflare Workers](https://img.shields.io/badge/Cloudflare_Workers-F38020?logo=cloudflare&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface)
![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-blue)

# CrescendAI

**A teacher for every pianist.**

Score-first practice companion for evidence-backed piano feedback. The app is silent while you play; at natural pauses it places marks on the score, and a session review delivers one verdict plus consolidated marks. The pipeline is Transkun transcription first, then piece resolution, offline score alignment, MPM-style per-bar feature extraction, and MoonBeam-839M symbolic scoring behind baseline-gated marks.

## Historical Model Benchmark

**79.85% pairwise accuracy** on PercePiano benchmark (A1-Max, 4-fold piece-stratified CV, clean folds)

| Encoder | Type | Pairwise Accuracy | Notes |
|---------|------|-------------------|-------|
| A1-Max (4-fold ensemble) | Audio | 79.85% | Historical clean-fold training result from the arXiv paper's MuQ pipeline; MuQ no longer runs in the app request path |

> **Fold integrity note:** Earlier results (80.8%, S2 GNN 71.3%) were computed with leaked folds where segments from the same piece appeared in both train and val splits. Those numbers are invalid. All current results use piece-stratified CV only.

6 output dimensions: **dynamics**, **timing**, **pedaling**, **articulation**, **phrasing**, **interpretation**

## Architecture

```
+-------------------+
|    Web App         |
|  (TanStack Start,  |
|   MediaRecorder)   |
+---------+----------+
          |
          |  15s audio chunks (HTTPS + WebSocket)
          v
+----------------------------+
|  Cloudflare Workers         |
|  api.crescend.ai            |
|  (TypeScript / Hono)        |
|                             |
|  /api/practice/*            |
|  /api/auth/*                |
|  /api/sync                  |
+--+---------+---------+-----+
   |         |         |
   v         v         v
+------+ +--------+ +--------+
| CF   | | AI     | | PG, R2 |
| Cont-| | Gate-  | | DO     |
| ainer| | way    | |        |
| (AMT)| | (LLMs) | |        |
+------+ +--------+ +--------+
```

| Component | Details |
|-----------|---------|
| Transcription (AMT) | Transkun transcription service in `apps/inference/amt/` (CF Containers) |
| Scoring | MPM-style per-bar features + MoonBeam-839M symbolic scoring behind baseline-gated marks (mock scorer until the MoonBeam head ships) |
| Web app | TanStack Start (`crescend.ai`) -- digital music stand, marks at pauses, session review |
| API backend | TypeScript/Hono on Cloudflare Workers (`api.crescend.ai`) -- SessionBrain DO, mark gating, score following |
| Session synthesis | V6 harness on Workers AI produces the session verdict + carry-forward through SessionBrain |
| Score following | Transkun notes aligned against a known score (#133 validated the follower on real audio) |
| Storage | PlanetScale Postgres + Drizzle ORM (via Hyperdrive), R2 (audio chunks), Durable Objects (practice sessions + WebSocket) |
| Auth | better-auth -- Sign in with Apple + Google |

## Project Structure

```
apps/api/          TypeScript/Hono API Worker (Cloudflare Workers)
apps/web/          TanStack Start web practice companion (React, Tailwind CSS v4)
apps/inference/    AMT (Transkun) server (CF Containers)
apps/evals/        E2E pipeline evaluation infrastructure
model/             Research pipelines, evaluation harnesses, and historical MuQ/Aria experiments
docs/              Architecture and documentation
  docs/apps/       Apps layer (status, product vision, pipeline, memory, exercises, UI)
  docs/model/      ML layer (research timeline, data, taxonomy, encoders, north star)
```

The native iOS app was removed 2026-08-16 (pre-pivot chat-first code); it will be rebuilt against the score-first product after the web beta. History lives in git before `apps/ios` deletion.

## Documentation

See [docs/architecture.md](docs/architecture.md) for the full system diagram and documentation map.

| Area | Entry Point | Contents |
|------|-------------|----------|
| Architecture | [docs/architecture.md](docs/architecture.md) | System diagram, cross-cutting concerns (auth, sync, observability) |
| Product vision | [docs/apps/01-product-vision.md](docs/apps/01-product-vision.md) | Target user, interaction model, UX principles |
| Pipeline | [docs/apps/02-pipeline.md](docs/apps/02-pipeline.md) | Audio-to-mark pipeline (Transkun, alignment, MPM features, scoring) |
| Student memory | [docs/apps/03-memory-system.md](docs/apps/03-memory-system.md) | Two-clock model, marks, synthesized facts |
| Exercises | [docs/apps/04-exercises.md](docs/apps/04-exercises.md) | Exercise database, drill routing |
| UI system | [docs/apps/05-ui-system.md](docs/apps/05-ui-system.md) | Score-first surfaces, mark system |
| ML research | [docs/model/00-research-timeline.md](docs/model/00-research-timeline.md) | Research roadmap, encoder results, key decisions |
| Training data | [docs/model/01-data.md](docs/model/01-data.md) | PercePiano, Competition, MAESTRO datasets |
| Taxonomy | [docs/model/02-teacher-grounded-taxonomy.md](docs/model/02-teacher-grounded-taxonomy.md) | 19 PercePiano dims to 6 teacher-grounded dims |
| Encoders | [docs/model/03-encoders.md](docs/model/03-encoders.md) | A1-Max audio encoder, Aria symbolic encoder |
| North star | [docs/model/04-north-star.md](docs/model/04-north-star.md) | 8-stage perfect pipeline vision |

## Setup

Install [just](https://github.com/casey/just): `brew install just`

| Command | What it starts |
|---------|----------------|
| `just dev` | All 3 services: AMT (8001) + API (8787) + Web (3000) |
| `just dev-light` | API + Web only |
| `just amt` / `just api` / `just web` | Individual services |
| `just fingerprint` | Generate N-gram index + rerank features from score library |
| `just test-model` / `just test-api` / `just check-api` | Tests and type checks |
| `just deploy-api` | Deploy API worker to production |
| `just migrate-generate` / `just migrate-prod` | Drizzle migrations (generate SQL / apply to prod) |
| `just eval-e2e` | Full E2E pipeline eval (cache + pipeline + analyze) |

### Training Pipeline

```bash
cd model
uv sync
uv run python -m audio_experiments.training.runner
```

## Paper

For technical details on the audio encoder, see the [paper on arXiv](https://arxiv.org/abs/2601.19029).
