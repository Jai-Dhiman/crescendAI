# CrescendAI System Architecture

**"A teacher for every pianist."** A web practice companion (iOS to be rebuilt score-first after web beta) that evaluates musical expression from audio -- not note accuracy -- and delivers score-first marks, a session verdict, and one carry-forward per session.

> **Redesign in flight (epic #154, 2026-08-04):** The chat-first system and its MuQ/HF-endpoint scoring path have been replaced. Current: delivery is score-first (marks on the score/timeline, a session verdict, one carry-forward; chat survives only as passage-scoped ask threads on marks -- see `docs/apps/05-ui-system.md`), and the model path is Transkun AMT -> piece resolution -> alignment -> MPM-style per-bar features -> MoonBeam-839M scoring (a deterministic mock scorer until the MoonBeam head ships), with MuQ removed (see `docs/apps/02-pipeline.md`). Sections below are updated where they asserted permanent chat delivery or MuQ scoring; the diagram and platform/cost history remain as a record of the system being replaced.

---

## System Diagram

```
+-------------------+
|    Web App         |
|  (TanStack Start,  |
|   MediaRecorder)   |
+---------+----------+
          |
          |  15s audio chunks (HTTPS)
          v
         +----------------------------+
         |  Cloudflare Workers         |
         |  api.crescend.ai            |
         |  (TypeScript/Hono;          |
         |   Rust WASM for compute)    |
         |                             |
         |  /api/practice/chunk        |
         |  /api/auth                  |
         |  /api/sync                  |
         +--+------+------+------+----+
            |      |      |      |
            v      v      v      v
      +-------+ +-----+ +--------+ +--------+
      |Transkun| | WkAI| | Teacher| |Postgres|
      | AMT +  | | API | | LLM    | | (Hyper-|
      | mock/  | | sub-| | (wrang-| |  drive)|
      |MoonBeam| | agent| | ler.   | | R2, DO |
      |scorer  | |     | | toml)  | |        |
      +-------+ +-----+ +--------+ +--------+
```

The web app uploads 15-second audio chunks to the API worker. The worker orchestrates the Transkun AMT -> piece resolution -> alignment -> MPM feature -> scoring pipeline, teaching moment / mark selection (deviation-magnitude gate, baseline-gated), and a two-stage LLM pipeline (a Workers AI subagent for analysis, then the teacher model for the session verdict). Model IDs live in `apps/api/wrangler.toml` (`TEACHER_MODEL`, overridable per-provider) and drift -- do not hardcode them here. Marks land on the score/timeline during pauses; the session verdict and carry-forward are delivered at session review (see `docs/apps/05-ui-system.md`).

**iOS.** The pre-pivot iOS app (SwiftUI, chat-first) was deleted 2026-08-16 (#164); iOS will be rebuilt score-first from git history after web beta validates the product.

### Platform Strategy (CEO Review 2026-03-19)

**Web-first.** The web app ships to beta users first -- it is furthest along, fastest to iterate (no App Store review), and shareable via URL for growth. iOS is rebuilt score-first after web beta validates the product (see note above; the original "iOS follows web beta" rationale still holds, iOS itself no longer exists in the tree).

**Session intelligence.** The Durable Object that manages web practice sessions is extended to serve as the "session brain" -- a practice mode state machine (warming up / drilling / running through / winding down) with mode-aware observation pacing.

**Unified artifact system (historical, superseded by #154/#164).** Rich components (exercises, score highlights, references) used to render as artifacts in the chat -- inline by default, expandable to viewport on demand -- with the teacher LLM declaring artifacts via tool use. The chat-container artifact system, the web chat UI, and the API chat routes were removed 2026-08-16 (#164); the score-first design delivers marks, a session verdict, and drills instead. See `docs/apps/05-ui-system.md`.

**Tiered monetization.** Free (daily/weekly limits) / $5 Plus / $20 Pro / $50 Max. Free tier is the growth engine. Inference cost reduction to ~$1/session is part of the model v2 track.

---

## Four Systems

From *The runtime behind production deep agents* (Mahler wiki): "building a good agent requires both a good harness and a good runtime -- the harness shapes model behavior through prompts, tools, and skills; the runtime handles the machinery underneath." Naming these separately prevents doc drift.

### Model System (`docs/model/`)

The audio intelligence layer, retargeted under #162 (epic #154, merged to local main 2026-08-15). Historically, A1-Max loaded the frozen public `OpenMuQ/MuQ-large-msd-iter` checkpoint and four separately trained pooling/prediction heads (clean-fold result: 79.85% pairwise accuracy, R2=0.336, not evidence this serving configuration generalizes to real practice). MuQ has been removed from the request path (`apps/inference/muq/` deleted, no MuQ calls remain in `session-brain.ts`): the path is Transkun (MIT, ISMIR 2024) AMT first, then piece resolution, offline alignment, MPM-style per-bar feature extraction, and MoonBeam-839M symbolic scoring behind baseline-gated marks. These outputs remain research signals rather than trustworthy diagnoses; the benchmark-first teacher program in #139 requires real-audio verification and calibrated abstention before broader claims.

Entry point: [`docs/model/00-research-timeline.md`](model/00-research-timeline.md)

### Harness System (`docs/harness.md`)

The behavior-shaping layer, markdown-first. Context graph (content/entity/fact), three-tier skill catalog (atoms / molecules / compounds), agent loop, student memory, eval harness. Skills, contracts, artifacts, and hook definitions are inspectable and diffable markdown. The catalog is provider-agnostic; the abandoned Qwen fine-tune was one historical consumer, not the current plan.

Entry point: [`docs/harness.md`](harness.md) | Skills: [`docs/harness/skills/`](harness/skills/)

### Runtime System

The machinery layer, invisible to skill authors. Cloudflare Workers (API request handling) + Durable Objects (per-session state + checkpointing) + Postgres via Hyperdrive (relational storage) + R2 (audio blobs and score assets) + AI Gateway (provider routing + shadow eval) + Sentry (observability). Handles durable execution, checkpointing across evictions, multi-tenancy, middleware hooks (`before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`), and online eval. Middleware hooks live here because they wrap every model call uniformly across every interaction mode.

Entry point: platform docs in `apps/api/TS_STYLE.md`, runtime-level patterns documented inline in the harness doc.

### Apps System (`docs/apps/`)

Implementation detail for what the student touches and what currently runs in the API worker: audio capture, the current cloud inference pipeline, teaching moment selection (deviation gate), the two-stage subagent architecture, student memory data model, exercises, and UI components. What is built vs. planned lives in GitHub Issues and the WIP board, not in these docs. The harness and runtime systems above describe the *target* architecture this layer is being refactored toward.

Entry point: [`docs/apps/01-product-vision.md`](apps/01-product-vision.md) | Pipeline: [`docs/apps/02-pipeline.md`](apps/02-pipeline.md) | Capabilities: [`docs/apps/06-capabilities.md`](apps/06-capabilities.md) | Evaluation: [`docs/apps/07-evaluation.md`](apps/07-evaluation.md)

---

## Platform Summary

| Platform | Stack | Key Paths | Notes |
|----------|-------|-----------|-------|
| iOS | (deleted 2026-08-16, #164) | -- | Pre-pivot chat-first app removed. Rebuilt score-first from git history after web beta. |
| Web | TanStack Start, React, Tailwind CSS v4 | `apps/web/` | Score-first delivery: marks on the score/timeline, session verdict, carry-forward (#154/#162). **Beta-first platform.** |
| API | TypeScript/Hono on Cloudflare Workers (Rust WASM for compute-heavy modules) | `apps/api/` | Single worker: AMT/scoring pipeline, LLM pipeline, auth, sync |
| Inference | Transkun AMT + MPM features + MoonBeam-839M (mock scorer until the head ships) | `apps/inference/`, `model/` | See `docs/apps/02-pipeline.md`; MuQ removed from the request path (#162) |

---

## Cross-Cutting Concerns

### Auth

Sign in with Apple and Google Sign In. The API worker validates the Apple ID token and issues a session JWT stored in cookies (web). Apple provides a stable user ID and relay email for future communication. When iOS is rebuilt, the session JWT is expected to move to iOS Keychain there, as it did pre-pivot.

### Sync

On web, the server database is the primary and authoritative data store; there is no client-side cache to reconcile. (Historical: the pre-pivot iOS app was local-first -- SwiftData on-device, phone authoritative, conflict-free delta push to the server after each session, server authoritative only for exercise updates. iOS was deleted 2026-08-16, #164; this local-first sync design is expected to be revisited, not assumed unchanged, when iOS is rebuilt score-first.)

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

The `exerciseUpdates` array contains exercises added or modified since the client's last sync. The server is authoritative for exercise content. See `04-exercises.md` for the exercise schema.

### Observability

Error tracking via Sentry. Web uses `@sentry/react` (React ErrorBoundary, API errors, WebSocket errors). The API worker uses the `@sentry/cloudflare` SDK (`Sentry.captureException`, traces, breadcrumbs) with structured JSON logging. Cloudflare Workers built-in analytics covers API health and latency. Sentry org: `crescendai`, projects: `crescendai-api`, `crescendai-web` (the `crescendai-ios` project is dormant since iOS was deleted 2026-08-16, #164).

---

## Key Decisions

Durable product and architecture choices with the reasoning behind them. Current
*state* lives in GitHub Issues and the WIP board, not here — this table records
why the system is shaped the way it is, so a choice is not silently re-litigated.

| Decision | Chosen | Rationale |
|---|---|---|
| Cloud-only inference (SERVING PATH SUPERSEDED by #162) | HF endpoint (MuQ) at decision time; now Transkun AMT + MPM features + MoonBeam-839M, all server-side | Eliminates Core ML conversion, single deployment path, instant model updates. Trade-off: network required for scoring. The "cloud-only, no on-device inference" principle stands; the specific HF/MuQ endpoint was retired with MuQ (#162). |
| Two-stage LLM pipeline | Separate analysis model from delivery model | Analysis wants fast and cheap; delivery wants voice quality. Different tasks, different models. Current model IDs live in `wrangler.toml` and drift — do not hardcode them in docs. |
| Multi-provider over single gateway | Workers AI + Anthropic via CF AI Gateway | Co-located inference for the analysis stage; native prompt caching for the teacher. Since #172 (Aug 2026 unification): Workers AI runs on the unified `AI` binding / unified REST API (`apps/api/src/lib/ai-gateway.ts`); Anthropic stays on the provider-native gateway URL because BYOK keys are not supported through the unified binding. |
| Local-first data (iOS) (HISTORICAL — iOS deleted #164) | SwiftData on-device, server for backup/sync | Practice works without internet (except the LLM call). Phone is authoritative, so there is no conflict resolution. Decision record for the pre-pivot app; revisit when iOS is rebuilt score-first, don't assume unchanged. |
| Scores as reasoning inputs | Not a report card | The model is ~80% pairwise accurate. Value is in the analysis and the teacher's delivery, not raw numbers. Consequence: raw dimension scores are never sent to the client (see `buildObservationPayload`, #143). |
| Chat-first UI (SUPERSEDED by #154) | Text default, components on-demand | Mirrors real teaching. Most observations are conversational; rich components only when a visual or interactive aid adds pedagogical value. Replaced by score-first delivery (marks + verdict + carry-forward, chat scoped to passage asks); see `docs/apps/05-ui-system.md`. |
| Piece identification | AMT fingerprint + graceful unknown | Unknown pieces still get audio-quality feedback, without bar numbers. Ask piece identity *after* the first observation, not before. Piece ID enriches but never gates. |
| Memory without vector search | Structured SQL queries, bi-temporal facts | The domain is narrow (6 dimensions, known ontology, low volume). No graph DB, no embeddings. |
| Platform strategy | Web-first, iOS rebuilt after | Web is furthest along, fastest to iterate, and gives a shareable URL. iOS was deleted 2026-08-16 (#164) and is rebuilt score-first from git history after web beta validation, rather than catching up in place. |
| Session intelligence | Durable Object as session brain | Practice-mode state machine (warming/drilling/running/winding) with mode-aware observation pacing. A single-threaded DO holds all session state. |
| Artifact system (SUPERSEDED by #154/#164) | Unified container (inline to expanded) | One `<Artifact>` component renders every rich content type. Lives in chat, expands to viewport on demand. The teacher declares artifacts via tool_use. Chat-container artifacts are removed under #164; exercise/segment-loop machinery survives as "drills" launched from marks. |
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
# iOS app deleted 2026-08-16 (#164); rebuilt from git history after web beta

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
