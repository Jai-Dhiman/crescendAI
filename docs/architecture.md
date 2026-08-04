# CrescendAI System Architecture

**"A teacher for every pianist."** Multi-platform (iOS + web) practice companion that evaluates musical expression from audio -- not note accuracy -- and delivers one actionable observation per session.

---

## System Diagram

```
+-------------------+       +-------------------+
|    iOS App        |       |    Web App         |
|  (SwiftUI,       |       |  (TanStack Start,  |
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
         |  (TypeScript/Hono;          |
         |   Rust WASM for compute)    |
         |                             |
         |  /api/practice/chunk        |
         |  /api/chat                  |
         |  /api/auth                  |
         |  /api/sync                  |
         +--+------+------+------+----+
            |      |      |      |
            v      v      v      v
      +-------+ +-----+ +------+ +----+
      | HF    | | WkAI| | Anth-| | D1 |
      | Endpt | | API | | ropic| |    |
      | (MuQ  | | sub-| | teach| | KV |
      | A1-Max)| | agent| | er  | | R2 |
      +-------+ +-----+ +------+ | DO |
                                  +----+
```

Both platforms upload 15-second audio chunks to the shared API worker. The worker orchestrates cloud inference (HF endpoint), teaching moment selection (deviation-magnitude gate), and a two-stage LLM pipeline (Workers AI subagent for analysis, Workers AI Qwen3-30B-A3B for teacher delivery by default). iOS receives observations on-demand ("How was that?"); web pushes them in real time via WebSocket.

### Platform Strategy (CEO Review 2026-03-19)

**Web-first.** The web app ships to beta users first -- it's ~90% complete, fastest to iterate (no App Store review), and shareable via URL for growth. iOS follows after web beta validates the product.

**Session intelligence.** The Durable Object that manages web practice sessions is extended to serve as the "session brain" -- a practice mode state machine (warming up / drilling / running through / winding down) with mode-aware observation pacing.

**Unified artifact system.** Rich components (exercises, score highlights, references) render as artifacts in the chat -- inline by default, expandable to viewport on demand. Teacher LLM declares artifacts via tool use. See `docs/apps/05-ui-system.md`.

**Tiered monetization.** Free (daily/weekly limits) / $5 Plus / $20 Pro / $50 Max. Free tier is the growth engine. Inference cost reduction to ~$1/session is part of the model v2 track.

---

## Four Systems

From *The runtime behind production deep agents* (Mahler wiki): "building a good agent requires both a good harness and a good runtime -- the harness shapes model behavior through prompts, tools, and skills; the runtime handles the machinery underneath." Naming these separately prevents doc drift.

### Model System (`docs/model/`)

The audio intelligence layer. A1-Max loads the frozen public `OpenMuQ/MuQ-large-msd-iter` checkpoint and four separately trained pooling/prediction heads; the serving path does not load a fine-tuned MuQ backbone. The historical clean-fold A1-Max training result was 79.85% pairwise accuracy and R2=0.336, but it is not evidence that this serving configuration generalizes to real practice. Transkun (MIT, ISMIR 2024) supplies the parallel note, offset, and pedal stream behind the frozen `/transcribe` contract. These outputs remain research signals rather than trustworthy diagnoses; the benchmark-first teacher program in #139 requires real-audio verification and calibrated abstention before broader claims.

Entry point: [`docs/model/00-research-timeline.md`](model/00-research-timeline.md)

### Harness System (`docs/harness.md`)

The behavior-shaping layer, markdown-first. Context graph (content/entity/fact), three-tier skill catalog (atoms / molecules / compounds), agent loop, student memory, eval harness. Skills, contracts, artifacts, and hook definitions are inspectable and diffable markdown. The catalog is provider-agnostic; the abandoned Qwen fine-tune was one historical consumer, not the current plan.

Entry point: [`docs/harness.md`](harness.md) | Skills: [`docs/harness/skills/`](harness/skills/)

### Runtime System

The machinery layer, invisible to skill authors. Cloudflare Workers (API request handling) + Durable Objects (per-session state + checkpointing) + D1 (relational storage) + R2 (audio blobs) + AI Gateway (provider routing + shadow eval) + Sentry (observability). Handles durable execution, checkpointing across evictions, multi-tenancy, middleware hooks (`before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`), and online eval. Middleware hooks live here because they wrap every model call uniformly across every interaction mode.

Entry point: platform docs in `apps/api/TS_STYLE.md`, runtime-level patterns documented inline in the harness doc.

### Apps System (`docs/apps/`)

Implementation detail for what the student touches and what currently runs in the API worker: audio capture, the current cloud inference pipeline, teaching moment selection (deviation gate), the two-stage subagent architecture, student memory data model, exercises, and UI components. What is built vs. planned lives in GitHub Issues and the WIP board, not in these docs. The harness and runtime systems above describe the *target* architecture this layer is being refactored toward.

Entry point: [`docs/apps/01-product-vision.md`](apps/01-product-vision.md) | Pipeline: [`docs/apps/02-pipeline.md`](apps/02-pipeline.md) | Capabilities: [`docs/apps/06-capabilities.md`](apps/06-capabilities.md) | Evaluation: [`docs/apps/07-evaluation.md`](apps/07-evaluation.md)

---

## Platform Summary

| Platform | Stack | Key Paths | Notes |
|----------|-------|-----------|-------|
| iOS | SwiftUI, AVAudioEngine, SwiftData | `apps/ios/` | On-demand observations, local-first persistence. **Follows web beta.** |
| Web | TanStack Start, React, Tailwind CSS v4 | `apps/web/` | Real-time observations via WebSocket, chat interface. **Beta-first platform.** |
| API | TypeScript/Hono on Cloudflare Workers (Rust WASM for compute-heavy modules) | `apps/api/` | Single worker: inference proxy, LLM pipeline, auth, sync |
| Inference | PyTorch, HF Inference Endpoint | `apps/inference/`, `model/` | A1-Max 4-fold ensemble, 6-dim scores |

---

## Cross-Cutting Concerns

### Auth

Sign in with Apple and Google Sign In on both platforms. The API worker validates the Apple ID token and issues a session JWT stored in iOS Keychain (native) or cookies (web). Apple provides a stable user ID and relay email for future communication. Required by App Store for account-based features.

### Sync

Local-first on iOS: all student data and sessions live in SwiftData on-device. The phone is authoritative. D1 stores copies for cross-platform backup and web access. Sync is conflict-free -- the phone pushes deltas (new sessions, updated baselines) to D1 after each session. The server is authoritative only for exercise updates. On web, D1 is the primary data store.

**Sync response payload:** The `POST /api/sync` response includes server-to-client updates:

```json
{
    "status": "ok",
    "exerciseUpdates": [
        {
            "id": "ex-ped-003",
            "title": "Legato Pedal Harmonic Changes",
            "action": "upsert"
        }
    ],
    "exerciseUpdates_since": "2026-03-14T00:00:00Z"
}
```

The `exerciseUpdates` array contains exercises added or modified since the client's last sync. iOS caches exercises in SwiftData and queries locally first; new exercises arrive via this response. The server is authoritative for exercise content. See `04-exercises.md` for the exercise schema.

### Observability

Error tracking via Sentry across all three surfaces. iOS uses `sentry-cocoa` SPM (crash reporting, error capture, breadcrumbs). Web uses `@sentry/react` (React ErrorBoundary, API errors, WebSocket errors). The API worker uses the `@sentry/cloudflare` SDK (`Sentry.captureException`, traces, breadcrumbs) with structured JSON logging. Cloudflare Workers built-in analytics covers API health and latency. Sentry org: `crescendai`, projects: `crescendai-api`, `crescendai-web`, `crescendai-ios`.

---

## Key Decisions

Durable product and architecture choices with the reasoning behind them. Current
*state* lives in GitHub Issues and the WIP board, not here — this table records
why the system is shaped the way it is, so a choice is not silently re-litigated.

| Decision | Chosen | Rationale |
|---|---|---|
| Cloud-only inference | HF endpoint for both platforms | Eliminates Core ML conversion, single deployment path, instant model updates. Trade-off: network required for scoring. |
| Two-stage LLM pipeline | Separate analysis model from delivery model | Analysis wants fast and cheap; delivery wants voice quality. Different tasks, different models. Current model IDs live in `wrangler.toml` and drift — do not hardcode them in docs. |
| Multi-provider over single gateway | Workers AI + Anthropic via CF AI Gateway | Co-located inference for the analysis stage; native prompt caching for the teacher. |
| Local-first data (iOS) | SwiftData on-device, server for backup/sync | Practice works without internet (except the LLM call). Phone is authoritative, so there is no conflict resolution. |
| Scores as reasoning inputs | Not a report card | The model is ~80% pairwise accurate. Value is in the analysis and the teacher's delivery, not raw numbers. Consequence: raw dimension scores are never sent to the client (see `buildObservationPayload`, #143). |
| Chat-first UI | Text default, components on-demand | Mirrors real teaching. Most observations are conversational; rich components only when a visual or interactive aid adds pedagogical value. |
| Piece identification | AMT fingerprint + graceful unknown | Unknown pieces still get audio-quality feedback, without bar numbers. Ask piece identity *after* the first observation, not before. Piece ID enriches but never gates. |
| Memory without vector search | Structured SQL queries, bi-temporal facts | The domain is narrow (6 dimensions, known ontology, low volume). No graph DB, no embeddings. |
| Platform strategy | Web-first, iOS follows | Web is furthest along, fastest to iterate, and gives a shareable URL. iOS catches up after beta validation. |
| Session intelligence | Durable Object as session brain | Practice-mode state machine (warming/drilling/running/winding) with mode-aware observation pacing. A single-threaded DO holds all session state. |
| Artifact system | Unified container (inline to expanded) | One `<Artifact>` component renders every rich content type. Lives in chat, expands to viewport on demand. The teacher declares artifacts via tool_use. |
| First session | Zero-config | Sign in, play anything, get an observation. Degrades gracefully when the piece is unknown. |
| Monetization | Tiered: Free / $5 Plus / $20 Pro / $50 Max | Free tier with limits as the growth engine. Decided, not yet enforced. |

## Open Questions

Unresolved design questions that are not work items. Items 1-3 are all blocked on
having real users.

1. **Deviation-gate calibration.** Does the `deviation < 0` gate behave sensibly for intermediate students on phone audio, given baselines are sparse?
2. **Minimum deviation threshold.** Should there be a *magnitude* floor, so trivially-small negative deviations produce "sounded good, keep going" instead of surfacing a tiny dip?
3. **Positive/corrective ratio.** Target is 70% corrective, 30% positive. See `docs/apps/02-pipeline.md`.
4. **When does memory synthesis become necessary?** Probably 50-100+ observations per student. Until then raw observation retrieval may suffice. See `docs/apps/03-memory-system.md`.
5. **Student-reported facts.** "I have a recital in 3 weeks" — store in `synthesized_facts` with `source_type = 'student_reported'`, or a separate table?
6. **Inference cost reduction path.** Tiered pricing needs roughly $1/session. Options: a single fused model, passage caching, serverless inference.

---

## Getting Started

```bash
# iOS app
open apps/ios/CrescendAI.xcodeproj

# Web app (crescend.ai)
cd apps/web && bun install && bun run dev

# API worker (api.crescend.ai)
cd apps/api && bun run dev   # or: just api

# ML training pipeline
cd model && uv sync && uv run python -m src.train
```

---

## Documentation Map

| Area | Entry point | Docs |
|------|-------------|------|
| Model / ML | [`docs/model/00-research-timeline.md`](model/00-research-timeline.md) | Research timeline, data, taxonomy, encoders, north star |
| Apps / Delivery | [`docs/apps/01-product-vision.md`](apps/01-product-vision.md) | Product vision, pipeline, memory, exercises, UI. Current status lives in GitHub Issues and the WIP board. |
