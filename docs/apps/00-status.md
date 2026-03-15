# Apps & Delivery System Status

> **Status (2026-03-14):** Two-stage LLM pipeline IMPLEMENTED (subagent + teacher, Groq + Anthropic). HF inference endpoint DEPLOYED (A1-Max 4-fold ensemble, 80.8% pairwise, 6-dim). iOS audio capture COMPLETE. Web practice companion IN PROGRESS (chat, recording, auth). Auth + sync COMPLETE. STOP classifier NOT STARTED. Teaching moment selection NOT STARTED. Exercise system NOT STARTED. UI components DESIGNED.

*Core loop: student plays, cloud inference scores 6 dimensions, STOP classifier identifies teaching moments, two-stage subagent pipeline reasons about what matters, teacher LLM delivers one specific observation.*

Target user: Sarah -- intermediate self-learner, no teacher, wants to know the one thing to work on next. North star metric: one useful observation per practice session, delivered in under 3 seconds.

---

## Current Implementation Status (2026-03-14)

### iOS App

| Component | Status | Key Files | Notes |
|---|---|---|---|
| Audio capture (AVAudioEngine) | COMPLETE | `apps/ios/` | 24kHz mono, ring buffer, 15s chunking, background mode |
| Cloud inference client | STUB | `apps/ios/` | Code ready, needs API integration |
| SwiftData models | COMPLETE | `apps/ios/` | Student, PracticeSession, ChunkResult, Observation |
| Sign in with Apple | COMPLETE | `apps/ios/` | Native one-tap, JWT stored in Keychain |
| D1 sync service | COMPLETE | `apps/ios/` | Background push after session, conflict-free (phone authoritative) |
| Student model service | COMPLETE | `apps/ios/` | Baselines, level inference, goal extraction |
| PracticeView | PARTIAL | `apps/ios/` | Basic session screen, not fully wired to pipeline |
| Chat interface | NOT STARTED | -- | Planned SwiftUI chat view with inline cards |
| Focus mode | NOT STARTED | -- | Depends on exercise DB and teaching moment detection |

### Web App (crescend.ai)

| Component | Status | Key Files | Notes |
|---|---|---|---|
| Chat interface | IN PROGRESS | `apps/web/` | TanStack Start + React, streaming LLM responses |
| Audio recording | IN PROGRESS | `apps/web/` | MediaRecorder, Opus/WebM, 15s chunks, waveform visualizer |
| WebSocket observations | IN PROGRESS | `apps/web/` | Real-time observation push during recording |
| Sign in with Apple (web) | IN PROGRESS | `apps/web/` | JS SDK popup flow |
| Durable Object sessions | IN PROGRESS | `apps/api/` | Practice session state management |
| On-demand UI components | DESIGNED | -- | Score highlight, keyboard guide, exercise set, reference browser |

Stack: TanStack Start, Tailwind CSS v4, Web Audio API, MediaRecorder, WebSocket.

### API Worker (api.crescend.ai)

| Endpoint / Service | Status | Key Files | Notes |
|---|---|---|---|
| `POST /api/auth/apple` | COMPLETE | `apps/api/src/` | Validates Apple ID token, issues session JWT |
| `POST /api/sync` | COMPLETE | `apps/api/src/` | Receives student model delta from iOS, upserts to D1 |
| `POST /api/extract-goals` | COMPLETE | `apps/api/src/` | Extracts student goals from conversation |
| `POST /api/ask` | IMPLEMENTED | `apps/api/src/services/ask.rs` | Two-stage pipeline (subagent + teacher), provider routing |
| `POST /api/practice/start` | IN PROGRESS | `apps/api/src/` | Creates Durable Object session (web path) |
| `POST /api/practice/chunk` | IN PROGRESS | `apps/api/src/` | Uploads audio, triggers HF inference |
| `WS /api/practice/ws/:sessionId` | IN PROGRESS | `apps/api/src/` | Real-time observation delivery (web path) |
| `POST /api/chat/send` | IN PROGRESS | `apps/api/src/` | Streaming teacher chat (web path) |
| D1 schema (students, sessions) | COMPLETE | `apps/api/` | Students, sessions, observations tables |
| D1 schema (exercises) | DEFINED | `apps/api/` | Tables defined in architecture, not migrated |
| STOP classifier | NOT STARTED | -- | 6-weight logistic regression, weights extracted from sklearn |
| Teaching moment selection | NOT STARTED | -- | STOP filter + blind-spot detection + ranking |
| Synthesized facts | NOT STARTED | -- | Background synthesis from observation traces |
| Exercise endpoints | NOT STARTED | -- | `GET /api/exercises`, exercise tracking |

Bindings: D1 (students, sessions, exercises), KV (JWTs, rate limits), R2 (audio chunks), DO (practice sessions).

### HF Inference Endpoint

| Component | Status | Notes |
|---|---|---|
| A1-Max 4-fold ensemble | DEPLOYED | 80.8% pairwise accuracy, R2=0.50, 6 dimensions |
| Inference latency | ~1-2s | HF endpoint round-trip |
| Handler | DEPLOYED | `apps/inference/handler.py` |
| MAESTRO calibration | COMPLETE | `model/data/maestro_cache/calibration_stats.json` |

---

## Critical Path: End-to-End Feedback Loop

The feedback loop is not yet closed. Four gates block the path from "student plays" to "student hears useful feedback" in production.

| Gate | Component | Status | Blocks | Effort |
|---|---|---|---|---|
| 1 | STOP classifier in cloud worker | NOT STARTED | Teaching moment selection cannot trigger without it | Small (6-weight logistic regression, ~1 day) |
| 2 | Teaching moment selection | NOT STARTED | Subagent receives no filtered moments to reason over | Medium (selection algorithm, blind-spot detection, ~1 week) |
| 3 | Web real-time observations | IN PROGRESS | Web users cannot receive feedback during practice | Medium (WebSocket plumbing, DO state, ~2 weeks) |
| 4 | Exercise system | NOT STARTED | System can observe problems but cannot prescribe fixes | Large (DB migration, seed data, endpoints, focus mode, ~3-4 weeks) |

**What works today:** The `/api/ask` endpoint accepts a pre-built teaching moment payload and returns a teacher observation via the two-stage pipeline. The inference endpoint scores audio chunks. Auth and sync are complete. The gap is the middle: nothing in production selects which chunk matters and why.

---

## Apps Documentation Map

| Doc | Title | What It Covers |
|---|---|---|
| `01-product-vision.md` | Product Vision | Target users, ideal practice session, UX principles, platform strategy, student model concept |
| `02-pipeline.md` | Audio-to-Observation Pipeline | Full technical pipeline: capture, inference, STOP classification, teaching moment selection, two-stage subagent, provider architecture |
| `03-memory-system.md` | Student Memory System | Two-clock model (state + event), observations table, synthesized facts, retrieval strategy, eval framework |
| `04-exercises.md` | Exercises and Focused Practice | Exercise DB schema, curated seed data, LLM-generated exercises, focus mode session flow |
| `05-ui-system.md` | UI System | Chat-first interface, on-demand components (score highlight, keyboard guide, exercise set, reference browser), three-stage pipeline extension |

For model/ML documentation, see `docs/model/00-research-timeline.md`.
For system architecture, see `docs/architecture.md`.

---

## Development Roadmap

### Phase 1: Close the Feedback Loop

**Goal:** A student can play, and the system tells them the one thing that matters. End-to-end on at least one platform.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| Deploy STOP classifier in cloud worker | Trained weights from `model/src/masterclass_experiments/` | 1-2 days | P0 |
| Implement teaching moment selection | STOP classifier | 1 week | P0 |
| Wire iOS cloud inference client | HF endpoint (deployed) | 1 week | P0 |
| Complete web recording + WebSocket observation flow | Teaching moment selection, DO sessions | 2 weeks | P0 |
| Score alignment V1 (student-reported piece + bar) | Teaching moment selection | 1 week | P1 |

### Phase 2: Memory and Exercises

**Goal:** The system remembers what it has said and can prescribe fixes, not just observations.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| Migrate exercise tables to D1 | D1 schema (complete) | 1-2 days | P1 |
| Seed 20-30 curated exercises | Exercise schema | 1-2 weeks | P1 |
| Build `GET /api/exercises` endpoint | Exercise tables | 3-5 days | P1 |
| Implement synthesized facts (background synthesis) | Observation data from Phase 1 | 1-2 weeks | P2 |
| Focus mode session flow | Exercises, teaching moment selection | 2-3 weeks | P2 |
| Exercise sync to iOS (via `/api/sync` response) | Exercise endpoints | 3-5 days | P2 |

### Phase 3: Rich UI and Polish

**Goal:** The teacher can show, not just tell. Components render inline in the chat.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| UI subagent (stage 3, Groq) | Teacher LLM modality output | 1 week | P2 |
| Score highlight component (V1, static) | Score alignment, notation library | 2-3 weeks | P2 |
| Exercise set component | Exercise DB | 1 week | P2 |
| Reference browser component | YouTube/Apple Music search | 1-2 weeks | P3 |
| Keyboard guide component (V1, static) | Score data | 2 weeks | P3 |
| iOS chat interface | Web chat (share learnings) | 3-4 weeks | P3 |

---

## Key Decisions

| Decision | Chosen | Rationale |
|---|---|---|
| Cloud-only inference | HF endpoint for both platforms | Eliminates Core ML conversion, single deployment path, instant model updates. Trade-off: network required for scoring. |
| Two-stage LLM pipeline | Subagent (Groq/Llama 70B) + Teacher (Anthropic/Sonnet 4.6) | Separates analysis (fast, cheap, ~0.3s) from delivery (quality voice, ~1.5s). Different tasks need different models. |
| Multi-provider over single gateway | Groq + Anthropic direct APIs, OpenRouter fallback | ~0.3-0.5s latency savings, native prompt caching, Groq LPU speed. OpenRouter remains as fallback. |
| Local-first data (iOS) | SwiftData on-device, D1 for backup/sync | Practice works without internet (except LLM call). Phone is authoritative. No conflict resolution needed. |
| Sign in with Apple | Single auth provider, both platforms | Zero friction, App Store requirement, stable cross-device identity. |
| Scores as reasoning inputs | Not a report card | Model is ~80% pairwise accurate. Value is in the subagent analysis and teacher delivery, not raw numbers. |
| STOP classifier Option B first | 6-dim scores in worker (0.845 AUC) | Simplest to deploy. Upgrade to Option A (MuQ embeddings, 0.936 AUC) if accuracy gap matters. |
| Chat-first UI | Text default, components on-demand (~30%) | Mirrors real teaching. Most observations are conversational. Rich components only when visual/interactive aid adds pedagogical value. |
| Student-reported piece context V1 | No automatic piece identification | Simple, no infrastructure. Auto-identification is a separate ML problem with limited ROI. |
| Memory without vector search | Structured D1 queries, bi-temporal facts | Domain is narrow (6 dimensions, known ontology, low volume). No graph DB, no embeddings needed. |

---

## Open Questions

### Pipeline

1. **STOP classifier generalization.** Trained on masterclass audio (professional students, concert pianos). Will it generalize to intermediate students on upright pianos with phone audio? Likely needs recalibration.
2. **Minimum STOP threshold.** Below what threshold should the system say "sounded good, keep going" instead of always finding something to critique?
3. **Positive/corrective ratio.** Target: 70% corrective, 30% positive. Validate with real users. See `02-pipeline.md`.

### Memory

4. **When does synthesis become necessary?** At 50-100+ observations per student (months of use). Until then, raw observation retrieval may suffice. See `03-memory-system.md`.
5. **Student-reported facts.** "I have a recital in 3 weeks" -- store in `synthesized_facts` with `source_type = 'student_reported'` or a separate table?

### Exercises

6. **Notation rendering library.** VexFlow (lightweight) vs. OpenSheetMusicDisplay (most capable, heaviest) vs. native Swift renderer. WebView adds latency but notation rendering is hard. See `05-ui-system.md`.
7. **Curated exercise count for V1.** 20-30 is the target. When does the LLM-generated path become more important than expanding the curated set?

### Cross-Platform

8. **iOS vs. web parity.** Web has real-time WebSocket observations; iOS has on-demand "how was that?" trigger. Are these fundamentally different products or converging interaction models?
9. **Continuous inference cost.** Background MuQ inference on every 15s chunk. Per-session cost at scale needs measurement.

---

## Getting Started

### iOS App

```bash
cd apps/ios
open CrescendAI.xcodeproj
# Requires Xcode 15+, iOS 17+ target
# Sign in with Apple requires Apple Developer account
```

### Web App

```bash
cd apps/web
bun install
bun run dev
# Runs at localhost:5173, TanStack Start dev server
```

### API Worker

```bash
cd apps/api
# Rust/WASM Cloudflare Worker
wrangler dev
# Runs at localhost:8787
# Requires wrangler.toml with D1/KV/R2/DO bindings
```

### HF Inference Endpoint

```bash
# Deployed on HuggingFace Inference Endpoints
# Handler: apps/inference/handler.py
# Model: A1-Max 4-fold ensemble
# Test locally:
cd apps/inference
python handler.py --test
```

### Model Training

```bash
cd model
uv sync
# See docs/model/00-research-timeline.md for training details
```
