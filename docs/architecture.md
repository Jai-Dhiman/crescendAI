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

Both platforms upload 15-second audio chunks to the shared API worker. The worker orchestrates cloud inference (HF endpoint), STOP classification, teaching moment selection, and a two-stage LLM pipeline (Groq subagent for analysis, Anthropic for teacher delivery). iOS receives observations on-demand ("How was that?"); web pushes them in real time via WebSocket.

### Platform Strategy (CEO Review 2026-03-19)

**Web-first.** The web app ships to beta users first -- it's ~90% complete, fastest to iterate (no App Store review), and shareable via URL for growth. iOS follows after web beta validates the product.

**Session intelligence.** The Durable Object that manages web practice sessions is extended to serve as the "session brain" -- a practice mode state machine (warming up / drilling / running through / winding down) with mode-aware observation pacing.

**Unified artifact system.** Rich components (exercises, score highlights, references) render as artifacts in the chat -- inline by default, expandable to viewport on demand. Teacher LLM declares artifacts via tool use. See `docs/apps/05-ui-system.md`.

**Tiered monetization.** Free (daily/weekly limits) / $5 Plus / $20 Pro / $50 Max. Free tier is the growth engine. Inference cost reduction to ~$1/session is part of the model v2 track.

---

## Two Systems

### Model System (`docs/model/`)

The audio intelligence layer. A finetuned MuQ foundation model (A1-Max) outputs 6 teacher-grounded dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation. Deployed as a 4-fold ensemble on HuggingFace Inference Endpoints (80.8% pairwise accuracy). The model taxonomy, encoder architecture, training pipeline, and research roadmap live here.

Entry point: [`docs/model/00-research-timeline.md`](model/00-research-timeline.md)

### Apps System (`docs/apps/`)

Everything from microphone to student-facing observation. Audio capture, the cloud inference pipeline, STOP classification, teaching moment selection, the two-stage subagent architecture, student memory, exercises, and UI components. Each implementation slice has a status header tracking what is built vs. planned.

Entry point: [`docs/apps/00-status.md`](apps/00-status.md) | Pipeline: [`docs/apps/02-pipeline.md`](apps/02-pipeline.md) | Product vision: [`docs/apps/01-product-vision.md`](apps/01-product-vision.md)

---

## Platform Summary

| Platform | Stack | Key Paths | Notes |
|----------|-------|-----------|-------|
| iOS | SwiftUI, AVAudioEngine, SwiftData | `apps/ios/` | On-demand observations, local-first persistence. **Follows web beta.** |
| Web | TanStack Start, React, Tailwind CSS v4 | `apps/web/` | Real-time observations via WebSocket, chat interface. **Beta-first platform.** |
| API | Rust/Axum on Cloudflare Workers (WASM) | `apps/api/` | Single worker: inference proxy, LLM pipeline, auth, sync |
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

Error tracking via Sentry across all three surfaces. iOS uses `sentry-cocoa` SPM (crash reporting, error capture, breadcrumbs). Web uses `@sentry/react` (React ErrorBoundary, API errors, WebSocket errors). The API worker uses Cloudflare Workers Observability with OTLP drain to Sentry -- no SDK in the Rust/WASM binary, just `console_error!` capture and invocation traces. Cloudflare Workers built-in analytics covers API health and latency. Sentry org: `crescendai`, projects: `crescendai-api`, `crescendai-web`, `crescendai-ios`.

---

## Getting Started

```bash
# iOS app
open apps/ios/CrescendAI.xcodeproj

# Web app (crescend.ai)
cd apps/web && bun install && bun run dev

# API worker (api.crescend.ai)
cd apps/api && npx wrangler dev

# ML training pipeline
cd model && uv sync && uv run python -m src.train
```

---

## Documentation Map

| Area | Entry point | Docs |
|------|-------------|------|
| Model / ML | [`docs/model/00-research-timeline.md`](model/00-research-timeline.md) | Research timeline, data, taxonomy, encoders, north star |
| Apps / Delivery | [`docs/apps/00-status.md`](apps/00-status.md) | Status, product vision, pipeline, memory, exercises, UI |
