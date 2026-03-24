# Apps & Delivery System Status

> **Status (2026-03-19):** Two-stage LLM pipeline IMPLEMENTED with bar-aligned musical analysis (subagent + teacher, Groq + Anthropic). HF inference endpoint DEPLOYED (A1-Max 4-fold ensemble + AMT + pedal CC64). STOP classifier IMPLEMENTED. Teaching moment selection IMPLEMENTED. Score following (DTW) IMPLEMENTED. Bar-aligned analysis engine IMPLEMENTED (all 6 dims, Tier 1/2/3). Durable Object practice sessions IMPLEMENTED. Web practice companion IN PROGRESS. Exercise system IMPLEMENTED. iOS audio capture COMPLETE. Auth + sync COMPLETE. Web Beta roadmap defined (CEO review). Platform strategy: web-first. Session brain (DO state machine), unified artifact container, zero-config first session, tiered monetization (Free/$5/$20/$50) decided.

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
| Chat interface | COMPLETE | `apps/web/` | TanStack Start + React, streaming LLM responses |
| Audio recording | COMPLETE | `apps/web/` | MediaRecorder, Opus/WebM, 15s chunks, waveform visualizer |
| WebSocket observations | COMPLETE | `apps/web/` | Real-time observation push during recording |
| Sign in with Apple (web) | COMPLETE | `apps/web/` | JS SDK popup flow |
| Durable Object sessions | COMPLETE | `apps/api/` | Practice session state management |
| On-demand UI components | COMPLETE | -- | Artifact container system COMPLETE (unified inline-to-expanded pattern). Teacher LLM declares artifacts via Anthropic tool_use (tool_choice: auto). Hybrid catalog lookup + generated fallback. Exercise artifact type for beta. |

Stack: TanStack Start, Tailwind CSS v4, Web Audio API, MediaRecorder, WebSocket.

### API Worker (api.crescend.ai)

| Endpoint / Service | Status | Key Files | Notes |
|---|---|---|---|
| `POST /api/auth/apple` | COMPLETE | `apps/api/src/` | Validates Apple ID token, issues session JWT |
| `POST /api/sync` | COMPLETE | `apps/api/src/` | Receives student model delta from iOS, upserts to D1 |
| `POST /api/extract-goals` | COMPLETE | `apps/api/src/` | Extracts student goals from conversation |
| `POST /api/ask` | IMPLEMENTED | `apps/api/src/services/ask.rs` | Two-stage pipeline (subagent + teacher), provider routing |
| `POST /api/practice/start` | COMPLETE | `apps/api/src/` | Creates Durable Object session (web path) |
| `POST /api/practice/chunk` | COMPLETE | `apps/api/src/` | Uploads audio, triggers HF inference |
| `WS /api/practice/ws/:sessionId` | COMPLETE | `apps/api/src/` | Real-time observation delivery (web path) |
| `POST /api/chat/send` | COMPLETE | `apps/api/src/` | Streaming teacher chat (web path) |
| D1 schema (students, sessions) | COMPLETE | `apps/api/` | Students, sessions, observations tables |
| D1 schema (observations) | COMPLETE | `apps/api/` | Observations table ships with /api/ask pipeline |
| D1 schema (exercises) | DEFINED | `apps/api/` | Tables defined in architecture, not migrated |
| STOP classifier | IMPLEMENTED | `apps/api/src/services/stop.rs` | 6-weight logistic regression, AUC 0.845 |
| Teaching moment selection | IMPLEMENTED | `apps/api/src/services/teaching_moments.rs` | STOP filter + blind-spot detection + positive moments + dedup |
| Score following (DTW) | IMPLEMENTED | `apps/api/src/practice/score_follower.rs` | Onset+pitch subsequence DTW, cross-chunk continuity, re-anchoring |
| Bar-aligned analysis engine | IMPLEMENTED | `apps/api/src/practice/analysis.rs` | All 6 dims, Tier 1/2/3 degradation, reference comparison |
| Fuzzy piece matching | IMPLEMENTED | `apps/api/src/practice/piece_match.rs` | Bigram Dice against 242-piece catalog, demand tracking |
| Score context loading | IMPLEMENTED | `apps/api/src/practice/score_context.rs` | R2 score + reference fetch, D1 catalog, piece request logging |
| D1 schema (piece_requests) | COMPLETE | `apps/api/migrations/0005_piece_requests.sql` | Demand tracking for catalog expansion |
| Synthesized facts | COMPLETE | `apps/api/src/services/memory.rs` | Background synthesis trigger (DO finalization + HTTP endpoint), observation counting fix, `SynthesisResult` observability |
| Exercise endpoints | IMPLEMENTED | `apps/api/src/services/exercises.rs` | `GET /api/exercises`, exercise tracking |

Bindings: D1 (students, sessions, exercises), KV (JWTs, rate limits), R2 (audio chunks), DO (practice sessions).

### HF Inference Endpoint

| Component | Status | Notes |
|---|---|---|
| A1-Max 4-fold ensemble | DEPLOYED | 80.8% pairwise accuracy, R2=0.50, 6 dimensions |
| ByteDance AMT transcription | DEPLOYED | Notes + pedal CC64 events, sequential after MuQ |
| Inference latency | ~1-2s | HF endpoint round-trip (MuQ + AMT sequential) |
| Handler | DEPLOYED | `apps/inference/handler.py` (returns predictions + midi_notes + pedal_events) |
| MAESTRO calibration | COMPLETE | `model/data/maestro_cache/calibration_stats.json` |

---

## Critical Path: End-to-End Feedback Loop

The core feedback loop is now wired end-to-end on the web platform. The pipeline: student plays -> HF scores + AMT -> DO runs STOP classifier -> teaching moment selection -> score following (DTW) -> bar-aligned analysis -> enriched subagent prompt -> teacher observation delivered via WebSocket.

| Gate | Component | Status | Notes |
|---|---|---|---|
| 1 | STOP classifier | COMPLETE | 6-weight logistic regression, AUC 0.845 |
| 2 | Teaching moment selection | COMPLETE | STOP + blind-spot + positive moments + dedup |
| 3 | Web real-time observations | COMPLETE | DO orchestration, WebSocket delivery, bar-aligned analysis |
| 4 | Score following + analysis | COMPLETE | DTW + Tier 1/2/3 analysis, all 6 dimensions |
| 5 | Exercise system | PARTIAL | DB + endpoints implemented, focus mode NOT STARTED |

**What works today:** A student can record on the web, chunks are scored by MuQ + transcribed by AMT, the DO runs STOP classification and teaching moment selection, score following maps to bar numbers (if piece is identified), the analysis engine produces per-dimension musical facts, and the enriched subagent prompt generates a bar-specific teacher observation delivered via WebSocket. Three-tier degradation: Tier 1 (full bar-aligned with score+reference), Tier 2 (absolute MIDI for unknown pieces), Tier 3 (scores only if AMT fails).

**Remaining gaps:** Session brain state machine (DO practice mode detection), artifact container system, first-session zero-config flow, session arc (opening/closing), free tier gating, landing page.

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

### Phase 1: Close the Feedback Loop -- COMPLETE (2026-03-15)

**Goal:** A student can play, and the system tells them the one thing that matters. End-to-end on at least one platform.

| Task | Status | Notes |
|---|---|---|
| Deploy STOP classifier in cloud worker | COMPLETE | `stop.rs`, 6-weight logistic regression, AUC 0.845 |
| Implement teaching moment selection | COMPLETE | `teaching_moments.rs`, STOP + blind-spot + positive + dedup |
| Complete web recording + WebSocket observation flow | COMPLETE | DO orchestration, chunk upload, WebSocket delivery |
| Score following (Phase 1c) | COMPLETE | `score_follower.rs`, onset+pitch DTW, cross-chunk continuity |
| Bar-aligned analysis engine (Phase 1d) | COMPLETE | `analysis.rs`, all 6 dims, Tier 1/2/3 degradation |
| Reference cache script (Phase 1e) | COMPLETE | `reference_cache.py`, data generation pending |
| Fuzzy piece matching + demand tracking | COMPLETE | `piece_match.rs`, `score_context.rs`, `piece_requests` table |
| Enrich subagent prompt with musical analysis | COMPLETE | `prompts.rs`, `<musical_analysis>` per-dimension facts |
| Wire iOS cloud inference client | NOT STARTED | HF endpoint deployed, iOS code needs API integration |

### Phase 2: Web Beta -- Core Loop + First Session Magic (~4 weeks)

**Goal:** A pianist opens crescend.ai, plays anything, gets their first useful observation in under 60 seconds. Session intelligence makes observations feel like a teacher in the room.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| Session brain state machine in DO | Existing DO session code | 1 week | P0 |
| Observation pacing (mode-aware throttle) | Session brain | 3 days | P0 |
| First-session zero-config flow (AMT piece ID) | Existing AMT + fuzzy match | 1 week | P0 |
| Artifact container component (inline/expanded) | -- | 1 week | P0 |
| Exercise artifact renderer | Artifact container | 3 days | P0 |
| ~~Session opening context (memory retrieval)~~ | ~~Existing memory system~~ | ~~3 days~~ | ~~P0~~ (NOT NEEDED -- memory already flows into every request via build_memory_context) |
| Session closing synthesis | Session brain | 3 days | P0 |
| Memory retrieval wired E2E | Existing D1 queries | 3 days | P0 |
| Free tier cap (hardcoded session limit) | -- | 1 day | P1 |
| Landing page with value prop | -- | 3 days | P1 |
| Google Sign In on web | Existing API endpoint | 1 day | P1 |

### Phase 3: iOS + Payment + Rich Artifacts

**Goal:** iOS goes end-to-end. Stripe enables paid tiers. Additional artifact types ship.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| iOS cloud inference wiring (stub to real API) | Phase 2 API stability | 1-2 weeks | P1 |
| Stripe integration (subscription management) | -- | 1-2 weeks | P1 |
| Usage tracking + tier enforcement | Stripe | 1 week | P1 |
| Score highlight artifact renderer | Artifact container, score rendering lib | 2 weeks | P2 |
| Reference browser artifact renderer | Artifact container | 1 week | P2 |
| Session review artifact | Artifact container | 1 week | P2 |
| Passage repetition comparison | Session brain DTW | 2 weeks | P2 |
| Focus mode (multi-exercise sequences) | Exercise artifact | 2-3 weeks | P2 |
| Analytics dashboard | Usage tracking | 1 week | P3 |

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
| Piece identification | AMT fingerprint + graceful unknown | Auto-detect via AMT MIDI fingerprint against 242-piece score library. Unknown pieces get audio-quality feedback without bar numbers. Ask piece identity AFTER first observation, not before. |
| Memory without vector search | Structured D1 queries, bi-temporal facts | Domain is narrow (6 dimensions, known ontology, low volume). No graph DB, no embeddings needed. |
| Platform strategy | Web-first, iOS follows | Web is ~90% complete, fastest to iterate, shareable URL for growth. iOS catches up after beta validation. |
| Session intelligence | Durable Object as session brain | Practice mode state machine (warming/drilling/running/winding) with mode-aware observation pacing. Single-threaded DO holds all session state. |
| Artifact system | Unified container (inline to expanded) | One `<Artifact>` component renders all rich content types. Lives in chat, expands to viewport on demand. Teacher LLM declares artifacts via Anthropic tool_use (tool_choice: auto). Hybrid catalog lookup + generated fallback. COMPLETE. |
| First session | Zero-config magic | Sign in, play anything, first observation in <60s. AMT fingerprint for piece ID; graceful degradation if unknown. Piece ID enriches but never gates. |
| Monetization | Tiered: Free / $5 Plus / $20 Pro / $50 Max | Free tier with daily/weekly limits as growth engine. Inference cost reduction to ~$1/session via model v2. |

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

### Architecture

8. **Inference cost reduction path.** Current $6/session must reach ~$1/session for tiered pricing to work. Optimization via single fused model, passage caching, serverless inference.

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
