# CrescendAI Documentation

**"A teacher for every pianist."**

Multi-platform (iOS + web) practice companion that evaluates *how* a piano performance sounds -- tone, dynamics, phrasing, pedaling -- not just note accuracy. Cloud audio inference via a finetuned MuQ foundation model (HF endpoint), with a Cloudflare Workers backend for STOP classification, teaching moment selection, LLM feedback, and data sync.

**Target user:** Sarah -- 3 years playing, no teacher, records on her phone, wants direction on what to work on next.

**North star:** Give Sarah one piece of useful feedback on one passage she's working on. Not perfect. Not comprehensive. One thing a teacher would actually say after hearing her play.

**Architecture:** Two platforms, shared backend. Both iOS and web upload audio chunks to the API, which runs cloud inference (HF endpoint), STOP classification, and teaching moment selection. Student data is local-first on iOS (SwiftData) with D1 sync. Web uses real-time observations via WebSocket. Both feed a two-stage LLM pipeline (fast subagent + quality teacher) that generates one natural observation. See [architecture.md](architecture.md) for the full system design.

---

## Documentation Map

### Architecture & Product

| Doc | Description |
|-----|-------------|
| [architecture.md](architecture.md) | Full system architecture -- cloud inference pipeline, API, data models, sync protocol, observability (source of truth) |
| [00-practice-companion.md](apps/00-practice-companion.md) | Product spec -- core interaction model, student model, exercise database, infrastructure |
| [design-system.md](design-system.md) | Visual design system -- colors, typography, spacing, component patterns for iOS and web |
| [landing-page-design.md](landing-page-design.md) | Landing page design spec for crescend.ai |
| [plans/2026-03-09-web-recording-mode-design.md](plans/2026-03-09-web-recording-mode-design.md) | Web practice companion design -- recording, real-time observations, Durable Objects architecture |

### Implementation Slices

| Slice | Doc | Status | Description |
|-------|-----|--------|-------------|
| 01 | [Phone Audio Validation](apps/01-phone-audio-validation.md) | NOT STARTED | Validate MuQ on phone-recorded piano audio |
| 02 | [iOS Audio Capture](apps/02-ios-audio-capture.md) | COMPLETE | AVAudioEngine, ring buffer, chunking, background mode |
| 03 | [Chunked Inference Pipeline](apps/03-chunked-inference-pipeline.md) | IN PROGRESS | Cloud inference pipeline (stub -- needs API integration) |
| 04 | [Teaching Moment Detection](apps/04-teaching-moment-detection.md) | NOT STARTED | STOP classifier, blind spot detection, teaching moment selection |
| 05 | [Student Model + Auth](apps/05-student-model-and-auth.md) | COMPLETE | Sign in with Apple, SwiftData models, D1 sync, goals, check-ins |
| 06 | [Teacher LLM Prompt](apps/06-teacher-llm-prompt.md) | DESIGNED | Teacher persona prompt (stage 2 of pipeline). Superseded as standalone by 06a |
| 06a | [Subagent Architecture](apps/06a-subagent-architecture.md) | DESIGNED | Two-stage pipeline: fast subagent + quality teacher LLM |
| 06c | [Student Memory System](apps/06c-memory-system.md) | IMPLEMENTED | Bi-temporal synthesized facts, teaching approach tracking, memory retrieval |
| 07 | [Exercise Database](apps/07-exercise-database.md) | NOT STARTED | D1 exercises, curated + LLM-generated |
| 08 | [Focus Mode](apps/08-focus-mode.md) | NOT STARTED | Guided practice targeting weak dimensions. Depends on 04 + 07 |
| 09 | [iOS Frontend](apps/09-ios-frontend.md) | IN PROGRESS | SwiftUI screens: Practice, Observation, Review, Focus, Profile |
| 10 | [On-Demand UI](apps/10-on-demand-ui.md) | DESIGNED | Chat-first interface with inline interactive component cards |
| 11 | [Teacher Voice Fine-Tuning](apps/11-teacher-voice-finetuning.md) | RESEARCH | Fine-tuning strategy, provider architecture (Groq + Anthropic), dataset plan |

### Model / ML

| Doc | Description |
|-----|-------------|
| [model/00-research-timeline.md](model/00-research-timeline.md) | Research roadmap, decision log, current status (evolving) |
| [model/01-data.md](model/01-data.md) | Dataset inventory -- what exists, paths, sizes (reference) |
| [model/02-teacher-grounded-taxonomy.md](model/02-teacher-grounded-taxonomy.md) | 6-dimension derivation from masterclass data (COMPLETE -- all gates pass) |
| [model/03-encoders.md](model/03-encoders.md) | Audio (A1-Max) + Symbolic (S2 GNN) encoder status, results, next experiments (evolving) |
| [model/04-north-star.md](model/04-north-star.md) | Perfect model vision -- fusion rationale, score conditioning, long-term research (evolving) |

### Agent Instructions (CLAUDE.md files)

| File | Scope |
|------|-------|
| [/CLAUDE.md](../CLAUDE.md) | Project-wide conventions, architecture summary, package managers |
| [apps/CLAUDE.md](../apps/CLAUDE.md) | Apps layer -- iOS stack, API endpoints, web stack, feedback tone |
| [apps/ios/CLAUDE.md](../apps/ios/CLAUDE.md) | iOS-specific conventions, directory structure, patterns |
| [model/CLAUDE.md](../model/CLAUDE.md) | ML pipeline -- stack, datasets, training infrastructure, research thesis |

---

## Implementation Status

*Last verified: 2026-03-14*

### iOS App (`apps/ios/`)

| Component | Status | Key Files | Notes |
|-----------|--------|-----------|-------|
| Design system | COMPLETE | `DesignSystem/Tokens/`, `Components/`, `Theme.swift` | Colors, typography, spacing tokens + CrescendButton, CrescendCard |
| Audio capture | COMPLETE | `Services/AudioEngine/` (5 files) | AVAudioEngine 24kHz mono, ring buffer, chunk producer |
| Cloud inference client | STUB | `Services/Inference/` (3 files) | Provider code ready, needs API integration for cloud inference |
| Auth (Sign in with Apple) | COMPLETE | `Services/Auth/AuthService.swift`, `Features/Auth/SignInView.swift` | JWT stored in Keychain |
| SwiftData models | COMPLETE | `Models/` (7 files) | Student, PracticeSession, ChunkResult, Observation, CheckIn, AudioChunk |
| D1 sync | COMPLETE | `Services/Auth/SyncService.swift` | Post-session + launch sync |
| Student model service | COMPLETE | `Services/StudentModelService.swift`, `CheckInService.swift` | Baselines, goals, check-ins |
| Goal extraction | COMPLETE | `Services/GoalExtractionService.swift` | Workers AI-powered |
| STOP classifier | NOT STARTED | -- | Needs masterclass model extraction |
| Teaching moment selection | NOT STARTED | -- | Depends on STOP classifier |
| Practice UI | PARTIAL | `Features/Practice/PracticeView.swift` | Basic session screen |
| Observation/Review UI | NOT STARTED | -- | |
| Focus mode UI | NOT STARTED | -- | Depends on Slices 4, 7, 8 |

### API Worker (`apps/api/`)

| Component | Status | Key Files | Notes |
|-----------|--------|-----------|-------|
| Auth endpoint | COMPLETE | `src/auth/mod.rs`, `jwt.rs` | `POST /api/auth/apple` |
| Sync endpoint | COMPLETE | `src/services/sync.rs` | `POST /api/sync` |
| Goal extraction | COMPLETE | `src/services/goals.rs` | `POST /api/extract-goals` (Workers AI) |
| D1 schema | COMPLETE | `migrations/0003_student_model.sql` | students, sessions, check_ins tables |
| Teacher LLM endpoint | NOT STARTED | -- | `POST /api/ask` (needs OpenRouter) |
| Exercise tables/endpoint | NOT STARTED | -- | Schema defined but not migrated |
| Legacy v1 endpoints | PRESENT | `src/server.rs`, `src/services/` | analyze, chat, upload, performances -- to be removed |

### Model (`model/`)

| Component | Status | Key Files | Notes |
|-----------|--------|-----------|-------|
| Taxonomy (6 dims) | COMPLETE | `data/composite_labels/` | Validated via 5-gate process |
| Audio training | COMPLETE | `notebooks/model_improvement/01_audio_training.ipynb` | A1-Max deployed, 80.8% ensemble pairwise |
| Symbolic training | COMPLETE | `notebooks/model_improvement/02_symbolic_training.ipynb` | S2 GNN winner, 71.3% pairwise |
| Fusion experiments | DEFERRED | -- | Failed in ISMIR paper (r=0.738 error correlation). See 04-north-star.md |
| Layer 1 validation | COMPLETE | `notebooks/model_improvement/04_layer1_validation.ipynb` | All gates pass. YouTube AMT: 79.9% agreement |
| Cloud inference | DEPLOYED | `apps/inference/handler.py` | A1-Max 4-fold ensemble on HF endpoint |

### Web App (`apps/web/`)

| Component | Status | Notes |
|-----------|--------|-------|
| Landing page | COMPLETE | TanStack Start + Tailwind CSS v4, deployed to crescend.ai |
| Auth (Sign in with Apple) | COMPLETE | Apple JS SDK popup flow, HttpOnly cookie JWT |
| Chat interface | IN PROGRESS | Streaming LLM responses, conversation history, markdown rendering |
| Practice recording | IN PROGRESS | MediaRecorder 15s Opus/WebM chunks, Web Audio API waveform |
| Real-time observations | IN PROGRESS | WebSocket connection, observation toasts during recording |
| Session summaries | IN PROGRESS | Post-recording summary posted to chat |

### HF Inference (`apps/inference/`)

| Component | Status | Notes |
|-----------|--------|-------|
| Cloud inference endpoint | DEPLOYED | A1-Max 4-fold ensemble, 6-dim output (80.8% pairwise) |

---

## Critical Path

The end-to-end feedback loop (record -> infer -> detect teaching moment -> generate observation) is not yet connected. The critical gates are:

1. **Cloud inference integration** -- Connect iOS/web chunk upload to HF endpoint via API worker. Cloud inference is deployed; needs API plumbing (Slice 3).
2. **STOP classifier in cloud worker** -- Extract from masterclass data, implement in API worker. Blocks teaching moment detection (Slice 4).
3. **Teacher LLM endpoint** -- Build `POST /api/ask` with LLM providers. Blocks observation generation (Slices 6/6a).
4. **Observation UI** -- Display teacher's observation in-app. Part of Slice 9.

---

## Getting Started

### iOS App

```bash
# Open in Xcode
open apps/ios/CrescendAI.xcodeproj
# See apps/ios/CLAUDE.md for conventions
```

### API Worker

```bash
cd apps/api
npx wrangler dev
# See apps/CLAUDE.md for endpoint documentation
```

### Web App

```bash
cd apps/web
bun install
bun run dev
```

### ML Training Pipeline

```bash
cd model
uv sync
# See model/CLAUDE.md for training instructions
# Notebooks in model/notebooks/model_improvement/
```

### HF Inference Endpoint

```bash
cd apps/inference
# See handler.py for the endpoint implementation
# Deployed as a HuggingFace Inference Endpoint
```
