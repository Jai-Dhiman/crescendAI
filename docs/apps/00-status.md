# Apps & Delivery System Status

> **Status (2026-06-04):** Two-stage LLM pipeline IMPLEMENTED with bar-aligned musical analysis (subagent + teacher, Workers AI + Anthropic). HF inference endpoint DEPLOYED (A1-Max 4-fold ensemble); AMT + pedal CC64 wired LOCAL ONLY (prod `AMT_ENDPOINT` unset -> prod = Tier 3; full pipeline verified locally via `just dev`, prod deploy deferred pre-beta -- see #9). Teaching moment selection IMPLEMENTED (deviation-magnitude gate: worst dimension below baseline, with positive-moment fallback). Score following (chroma-DTW) IMPLEMENTED. Bar-aligned analysis engine IMPLEMENTED (all 6 dims, Tier 1/2/3). Durable Object practice sessions IMPLEMENTED with state persistence (survives DO eviction). Web practice companion IMPLEMENTED (chat, recording, WebSocket observations, session synthesis, landing page v2 with ProofCard demo). Exercise system IMPLEMENTED (25 curated exercises seeded). iOS audio capture COMPLETE. Auth COMPLETE (Apple + Google on both API and web). AI Gateway COMPLETE (Anthropic + Workers AI). Zero-config piece ID CODE COMPLETE (merged, pending AMT container deploy). Session synthesis COMPLETE (alarm-triggered, all exit paths, deferred recovery; honest DO-path eval baseline locked (#22) and cold-start within-session synthesis for first-session/no-baseline students (#24)). Unified artifact container COMPLETE. **V6 harness loop SHIPPED** (hook-driven two-phase compound loop, flag-gated via HARNESS_V6_ENABLED). **V6 atoms SHIPPED** (14 ToolDefinition objects + ALL_ATOMS barrel). **V6 molecules SHIPPED** (9 ToolDefinition molecules + ALL_MOLECULES barrel; wires atoms into compound-registry for Phase 1 tool dispatch). **V6 Integration (Plan 4) COMPLETE** (ALL_ATOMS wired into OnSessionEnd, HookContext.digest populated with chunks/baselines/cohort_tables/session_history/past_diagnoses, diagnosis_artifacts table added, E2E test passes; HARNESS_V6_ENABLED=true safe to flip). **iOS auth refactored** (cookie sessions → Keychain JWT). **V8a SHIPPED** (`assign_segment_loop` action atom live: segment_loops DB table, lifecycle service + routes, ASSIGN_SEGMENT_LOOP_TOOL in both OnChatMessage and OnSessionEnd bindings, PassageLoopDetector in DO, SegmentLoopArtifactCard in web with accept/dismiss; SynthesisArtifact carries assigned_loops; migration 0003 required before deploy). **own_passage_loop playback SHIPPED (#45)** (`ExerciseSetCard` redesigned: score-first layout, `LoopTransport` UI, `LoopPlayer` audio orchestrator with smplr piano + metronome + `LoopClock`, `useLoopPlayer` hook, animated `ScoreCursor`; `tempoFactor` from the prescription drives transport). **iOS score rendering SHIPPED (#57)** (Verovio WASM + smplr in WKWebView scorehost bundle via WKURLSchemeHandler; ArtifactRenderer dispatches score_highlight + play_passage via ScoreHostBridge; `just build-scorehost` produces `apps/web/dist-scorehost/`). **V6 teacher on Workers AI SHIPPED (#61)** (V6 harness Phase 1+2 now route through `gateway-client.callModel` with `TEACHER_PROVIDER` toggle; default = Workers AI `@cf/qwen/qwen3-30b-a3b-fp8`; Anthropic selectable; gateway var/auth drift fixed; dead env vars removed). **Persisted-session e2e UI test SHIPPED (#68)** (`just e2e-ui-session` drives a real recording through MuQ+AMT -> glm-4.7-flash V6 synthesis -> DO state persistence, then Playwright asserts session artifacts appear in the web UI; 27 offline unit tests + live orchestrator in `apps/evals/e2e_ui_session.py`). **Chat teacher migrated to glm-4.7-flash@WorkersAI SHIPPED (#69)** (callWorkersAIStream + parseOpenAIStream streaming path; provider branch in runPhase1Streaming; system blocks incl. <student_memory> translated for Workers AI; tool_choice none + stopReason normalized; keep-tools with text-only fallback; Anthropic selectable via TEACHER_PROVIDER=anthropic). **Grounded molecule layer SHIPPED (#99)** (7 molecules self-fetch real signal via `resolveMoleculeContext`; `buildGroundedDigest` deep module; Phase-1+Phase-2 compact_signal_summary fixes glm 131K context overflow; cold-start `reference_mode` guardrail restored on V6 path; articulation-clarity-check deferred pending score-articulation capability). **Visible headed-Chrome full-eval harness SHIPPED (#70)** (`just e2e-full-session` drives the complete loop: session recording -> synthesis render -> scripted live glm chat turns -> memory seed (Postgres) + recall assertion -> tool action verification (TOOL_RENDERED/TEXT_ONLY) -> screenshots + PASS/FAIL. `memory_seeder.py` seeds canary synthesized_facts; `run_chat_turns` bounded poll with hard deadline; `data-testid=assistant-message` on ChatMessages for Playwright selectors. Use `--reply-timeout 180000` for cold glm; `--headless` for CI. Live memory-recall requires `:3000` serving the current build + MuQ up for the recording step.). Platform strategy: web-first. Tiered monetization (Free/$5/$20/$50) decided, not yet enforced.

*Core loop: student plays, cloud inference scores 6 dimensions, teaching-moment selection flags the chunks where the student fell below baseline (deviation-magnitude gate; falls back to a positive moment when nothing is below baseline), two-stage subagent pipeline reasons about what matters, teacher LLM delivers one specific observation.*

Target user: Sarah -- intermediate self-learner, no teacher, wants to know the one thing to work on next. North star metric: one useful observation per practice session, delivered in under 3 seconds.

---

## Current Implementation Status (2026-03-28)

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
| Score rendering (WKWebView) | COMPLETE | `apps/ios/`, `apps/web/dist-scorehost/` | Verovio WASM + smplr sampler hosted in a standalone scorehost bundle served via WKURLSchemeHandler (scorehost://app/). ArtifactRenderer dispatches score_highlight (ArtifactScoreView) and play_passage (ArtifactPlayPassageView) via ScoreHostBridge. Build: `just build-scorehost`. Score bytes from GET /api/scores/:id/data with offline bundled-piece fallback. (#57) |
| Chat interface | NOT STARTED | -- | Planned SwiftUI chat view with inline cards |
| Focus mode | NOT STARTED | -- | Depends on exercise DB and teaching moment detection |

### Web App (crescend.ai)

| Component | Status | Key Files | Notes |
|---|---|---|---|
| Chat interface | COMPLETE | `apps/web/` | TanStack Start + React, streaming LLM responses |
| Audio recording | COMPLETE | `apps/web/` | MediaRecorder, Opus/WebM, 15s chunks, waveform visualizer |
| WebSocket observations | COMPLETE | `apps/web/` | Real-time observation push during recording |
| Session synthesis | COMPLETE | `apps/web/` | Alarm-triggered on all exit paths, deferred recovery, WebSocket delivery |
| Sign in with Apple (web) | COMPLETE | `apps/web/` | JS SDK popup flow |
| Google Sign In (web) | COMPLETE | `apps/web/` | GSI client, custom-styled button, API token verification |
| Landing page v2 | COMPLETE | `apps/web/src/routes/index.tsx` | ExerciseProofBlock with live ProofCard demo, FinalCtaSection ("Your playing. Heard clearly."), LandingFooter. ProofCard: scroll autoplay, BarScoreChip bar-tap drill-down, keyboard nav, reduced-motion support, graceful degradation on missing audio/scoreIR. |
| Durable Object sessions | COMPLETE | `apps/api/` | Practice session state management with DO storage persistence (survives eviction) |
| On-demand UI components | COMPLETE | -- | Artifact container system COMPLETE (unified inline-to-expanded pattern). Teacher LLM declares artifacts via tool_use (tool_choice: auto; glm-4.7-flash@WorkersAI default). Hybrid catalog lookup + generated fallback. Exercise artifact type for beta. `play_passage` replaces `reference_browser` stub. |
| SegmentLoopArtifactCard | COMPLETE | `apps/web/src/components/cards/SegmentLoopArtifact.tsx` | Renders pending (Accept/Skip), active (attempt counter/Dismiss), completed states. Wired into InlineCard switch. api.ts has typed accept/decline/dismiss methods. usePracticeSession handles `segment_loop_status` and `loop_attempt` WS events. |
| PlayPassageCard | COMPLETE | `apps/web/src/components/cards/PlayPassageCard.tsx` | Plays a bar-bounded slice of the student's recording. Fetches PassageManifest, decodes WebM chunks via Web Audio API, sequences AudioBufferSourceNode with startOffsetSec/endOffsetSec trim, RAF cursor ticks over score SVG clip. AudioContext closed on unmount. |

Stack: TanStack Start, Tailwind CSS v4, Web Audio API, MediaRecorder, WebSocket.

### API Worker (api.crescend.ai)

| Endpoint / Service | Status | Key Files | Notes |
|---|---|---|---|
| `POST /api/auth/apple` | COMPLETE | `apps/api/src/` | Validates Apple ID token, issues session JWT |
| `POST /api/auth/google` | COMPLETE | `apps/api/src/auth/mod.rs` | Validates Google ID token via tokeninfo endpoint, issues session JWT |
| `POST /api/sync` | COMPLETE | `apps/api/src/` | Receives student model delta from iOS, upserts to D1 |
| `POST /api/extract-goals` | COMPLETE | `apps/api/src/` | Extracts student goals from conversation |
| `POST /api/ask` | IMPLEMENTED | `apps/api/src/services/ask.rs` | Two-stage pipeline (subagent + teacher), provider routing |
| `POST /api/practice/start` | COMPLETE | `apps/api/src/` | Creates Durable Object session (web path) |
| `POST /api/practice/chunk` | COMPLETE | `apps/api/src/` | Uploads audio, triggers HF inference |
| `GET /api/practice/chunk` | COMPLETE | `apps/api/src/routes/practice.ts` | Auth + session-ownership gated R2 read; returns WebM audio for PlayPassageCard. |
| `GET /api/sessions/:id/passage` | COMPLETE | `apps/api/src/routes/sessions.ts` | Auth + ownership gated. Talks to SessionBrain DO `/passage` handler; returns PassageManifest JSON or 409 if alignment missing. |
| `WS /api/practice/ws/:sessionId` | COMPLETE | `apps/api/src/` | Real-time observation delivery (web path) |
| `POST /api/chat/send` | COMPLETE | `apps/api/src/` | Streaming teacher chat via glm-4.7-flash@WorkersAI (Anthropic selectable via TEACHER_PROVIDER=anthropic) (web path) |
| D1 schema (students, sessions) | COMPLETE | `apps/api/` | Students, sessions, observations tables |
| D1 schema (observations) | COMPLETE | `apps/api/` | Observations table ships with /api/ask pipeline |
| D1 schema (exercises) | COMPLETE | `apps/api/migrations/0004_exercises.sql` | Tables migrated, 25 curated exercises seeded |
| `search_catalog` structured retrieval | COMPLETE | `apps/api/src/services/tool-processor.ts`, `apps/api/src/services/catalog-parse.ts` | Exact integer match on `opus_number`/`piece_number` — disambiguates "Op. 64 No. 2" vs "No. 3". Pieces table has `opus_number`, `piece_number`, `catalogue_type` columns. Backfill: 159/242 pieces. |
| `segment_loops` DB table + service | COMPLETE | `apps/api/src/db/schema/segment-loops.ts`, `apps/api/src/services/segment-loops.ts` | Lifecycle: pending → active → completed/dismissed/superseded. Unique index enforces one active loop per (student, piece). Migration: `0003_segment_loops.sql` (apply before deploy). |
| Segment loop routes | COMPLETE | `apps/api/src/routes/segment-loops.ts` | `POST /api/segment-loops/:id/accept\|decline\|dismiss` — all auth-guarded. |
| `assign_segment_loop` action atom | COMPLETE | `apps/api/src/harness/atoms/assign-segment-loop.ts` | Registered in OnChatMessage (trigger=chat → pending) and OnSessionEnd (trigger=synthesis → active) compound bindings. Zod-validated input, throws ToolPreconditionError if no piece. |
| PassageLoopDetector (DO) | COMPLETE | `apps/api/src/do/passage-loop-detector.ts` | Strict ±1-bar tolerance, 2s debounce. DO holds per-instance detector in WeakMap; hydrates active assignment on WS connect; broadcasts `segment_loop_status` and `loop_attempt` events. |
| Teaching moment selection | IMPLEMENTED | `apps/api/src/wasm/score-analysis/src/teaching_moments.rs` | Deviation-magnitude gate (worst dimension below baseline) + blind-spot ranking + positive-moment fallback + dedup |
| Score following (chroma-DTW) | IMPLEMENTED | `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` | Chroma-based DTW alignment; MuQ extracts 12-bin chroma, Rust aligns audio chroma against score chroma, emits chunk_bar_map events |
| Bar-aligned analysis engine | IMPLEMENTED | `apps/api/src/practice/analysis.rs` | All 6 dims, Tier 1/2/3 degradation, reference comparison |
| Fuzzy piece matching | IMPLEMENTED | `apps/api/src/practice/piece_match.rs` | Bigram Dice against 242-piece catalog, demand tracking |
| Score context loading | IMPLEMENTED | `apps/api/src/practice/score_context.rs` | R2 score + reference fetch, D1 catalog, piece request logging |
| D1 schema (piece_requests) | COMPLETE | `apps/api/migrations/0005_piece_requests.sql` | Demand tracking for catalog expansion |
| Synthesized facts | COMPLETE | `apps/api/src/services/memory.rs` | Background synthesis trigger (DO finalization + HTTP endpoint), observation counting fix, `SynthesisResult` observability |
| Exercise endpoints | IMPLEMENTED | `apps/api/src/services/exercises.rs` | `GET /api/exercises`, assign, complete; 18 tests |
| Zero-config piece ID | COMPLETE | `apps/api/src/practice/piece_identify.rs` | N-gram + rerank + DTW, merged to main (pending AMT container deploy) |
| Session synthesis | COMPLETE | `apps/api/src/practice/synthesis.rs` | Alarm-triggered, all exit paths, deferred recovery, 963 lines |
| V6 harness loop | COMPLETE | `apps/api/src/harness/loop/` | Hook-driven two-phase compound execution loop. Phase 1 dispatches molecule atoms as Anthropic tools; Phase 2 forced-writes a Zod-validated SynthesisArtifact. `HARNESS_V6_ENABLED=true` in prod. |
| Spec X: chat harness migration | COMPLETE | `apps/api/src/harness/`, `apps/api/src/routes/chat.ts` | chatV6 routes through runStreamingHook("OnChatMessage"); legacy chat() and buildChatBinding deleted. HARNESS_V6_CHAT_ENABLED flag removed (route always uses chatV6). |
| V6 atoms | COMPLETE | `apps/api/src/harness/skills/atoms/` | 14 ToolDefinition objects (compute-dimension-delta, align-performance-to-score, compute-ioi-correlation, compute-key-overlap-ratio, compute-onset-drift, compute-pedal-overlap-ratio, compute-velocity-curve, detect-passage-repetition, extract-bar-range-signals, fetch-reference-percentile, fetch-session-history, fetch-similar-past-observation, fetch-student-baseline, prioritize-diagnoses) + `ALL_ATOMS` barrel. |
| V6 molecules | COMPLETE | `apps/api/src/harness/skills/molecules/` | 7 ToolDefinition molecules (voicing-diagnosis, pedal-triage, rubato-coaching, phrasing-arc-analysis, tempo-stability-triage, dynamic-range-audit, cross-modal-contradiction-check) + `ALL_MOLECULES` barrel. articulation-clarity-check removed (deferred: needs MusicXML/score-articulation capability). exercise-proposal removed; exercise routing is now handled by the prescribe_exercise tool + pending_exercises table. Each molecule is selectors-only: it declares its required signal keys and self-fetches real data server-side via `resolveMoleculeContext` (tiered baseline, bar-range signals, cohort stats) rather than consuming pre-hydrated context. Both Phase 1 and Phase 2 render a `compact_signal_summary` (not the raw digest) to stay within the 131K glm context window. Cold-start `reference_mode` carried through the full grounded path. Deferred: per-note alignment arrays are always `[]` in production until a WASM BarMap follow-up populates IOI data. (#99) |
| V6 Integration (Plan 4) | COMPLETE | `apps/api/src/harness/` | ALL_ATOMS wired into OnSessionEnd, HookContext.digest populated with chunks/baselines/cohort_tables/session_history/past_diagnoses, diagnosis_artifacts table added, E2E test passes. HARNESS_V6_ENABLED=true is safe to set in production. `buildGroundedDigest` (grounded-digest.ts) and `resolveMoleculeContext` (resolve-molecule-context.ts) added as deep modules: digest injected into synthesizeV6; past_diagnoses grounded server-side. Phase-1 and Phase-2 context overflow fixed via compact_signal_summary (no raw digest dump). (#99) |
| DO state persistence | COMPLETE | `apps/api/src/practice/session.rs` | Persists to state.storage(), reloads on eviction at all async boundaries |
| AI Gateway | COMPLETE | `apps/api/src/services/llm.ts`, `apps/api/src/harness/loop/gateway-client.ts`, `apps/api/src/harness/loop/simplify-schema.ts` | Unified authenticated CF AI Gateway (`crescendai`). Teacher defaults to Workers AI `@cf/zai-org/glm-4.7-flash` (131K ctx, ~$0.011/sess); overrideable via `TEACHER_MODEL` wrangler var. Anthropic selectable via `TEACHER_PROVIDER=anthropic`. `simplifyConstrainedSchema` merges anyOf-of-objects into single nullable object to fix Workers AI tool-grammar incompatibility. Subagent unchanged (Workers AI). (#61, #62) |
| Practice mode detection | COMPLETE | `apps/api/src/practice/session.rs` | 4-state machine (warming/drilling/running/winding) |

Bindings: D1 (students, sessions, exercises), KV (JWTs, rate limits), R2 (audio chunks, fingerprints), DO (practice sessions).

### Content Engine (`apps/content-engine/`)

Python pipeline that produces 3 YouTube Shorts/week from piano performance clips, cross-posted to TikTok and Reels. Drives B2C installs from beta rollout.

| Component | Status | Key Files | Notes |
|---|---|---|---|
| Episode state machine | SHIPPED | `content_engine/pipeline/states.py` | 17 states, validated transitions, FAILED_* as dead-ends |
| SQLite episode store | SHIPPED | `content_engine/store/episode_store.py` | Atomic transitions, versioned config store |
| LLM gateway | SHIPPED | `content_engine/adapters/llm_gateway.py` | Workers AI (selector) + Claude CLI (narrator/critic), retry on 5xx |
| ClipScout | SHIPPED | `content_engine/agents/clip_scout.py` | YT + TikTok backends, duration filter, source-weight ranking |
| ObservationSelector | SHIPPED | `content_engine/agents/observation_selector.py` | JSON schema + clip-bounds validation |
| Narrator | SHIPPED | `content_engine/agents/narrator.py` | ≤45s scripts (120-word cap) via Claude CLI |
| CriticTruthfulness | SHIPPED | `content_engine/agents/critic_truthfulness.py` | PASS/KILL binary via Claude CLI, human override endpoint |
| Renderer | SHIPPED | `content_engine/render/renderer.py` | 9:16 mp4, ffmpeg, bitexact flags for determinism |
| Postiz scheduler | SHIPPED | `content_engine/adapters/scheduler.py` | Cross-posts YT/TikTok/IG |
| Analytics ingestor | SHIPPED | `content_engine/adapters/analytics_ingestor.py` | YT Data API + Postiz metrics; raises on auth/5xx errors |
| FeedbackScorer | SHIPPED | `content_engine/feedback/scorer.py` | Updates source-type weights from install conversion |
| Flask swipe UI | SHIPPED | `content_engine/ui/server.py` | Approve/reject/override-critic/record-complete endpoints |
| Typer CLI | SHIPPED | `content_engine/cli.py` | `tick`, `scout`, `ui` commands |
| Sentry observability | SHIPPED | `content_engine/observability.py` | DSN-gated init |
| Test suite | SHIPPED | `tests/` | 57 unit + property + e2e tests |

Run: `cd apps/content-engine && uv run python -m content_engine.cli tick`

---

### HF Inference Endpoint

| Component | Status | Notes |
|---|---|---|
| A1-Max 4-fold ensemble | DEPLOYED | 79.85% pairwise accuracy (clean folds), R2=0.336, 6 dimensions |
| Aria-AMT transcription | LOCAL ONLY | Replaces ByteDance. Notes + pedal CC64 events. MAESTRO F1 0.86. Prod `AMT_ENDPOINT=""` (unset) -> prod sessions degrade to Tier 3; local `.dev.vars` -> `localhost:8001`. Prod deploy deferred (pre-beta, local-first verification). |
| Inference latency | ~1-2s | HF endpoint round-trip (MuQ on HF; AMT local-only via `localhost:8001`) |
| MuQ Handler | DEPLOYED | `apps/inference/muq/handler.py` (returns predictions + midi_notes + pedal_events) |
| AMT Container | CODE COMPLETE | `apps/inference/amt/` -- ONNX + PyTorch server, pending CF Container deploy |
| MAESTRO calibration | COMPLETE | `model/data/maestro_cache/calibration_stats.json` |

---

## Critical Path: End-to-End Feedback Loop

The core feedback loop is now wired end-to-end on the web platform. The pipeline: student plays -> HF scores + AMT -> DO runs teaching-moment selection (deviation-magnitude gate) -> score following (DTW) -> bar-aligned analysis -> enriched subagent prompt -> teacher observation delivered via WebSocket.

| Gate | Component | Status | Notes |
|---|---|---|---|
| 1 | Teaching-moment gate | COMPLETE | Worst-dimension `deviation < 0` with positive-moment fallback |
| 2 | Teaching moment selection | COMPLETE | deviation gate + blind-spot ranking + positive-moment fallback + dedup |
| 3 | Web real-time observations | COMPLETE | DO orchestration, WebSocket delivery, bar-aligned analysis |
| 4 | Score following + analysis | COMPLETE | DTW + Tier 1/2/3 analysis, all 6 dimensions |
| 5 | Exercise system | PARTIAL | DB + endpoints implemented, focus mode NOT STARTED |

**What works today:** A student can record on the web, chunks are scored by MuQ + transcribed by AMT, the DO runs teaching-moment selection (deviation-magnitude gate), score following maps to bar numbers (if piece is identified), the analysis engine produces per-dimension musical facts, and the enriched subagent prompt generates a bar-specific teacher observation delivered via WebSocket. Three-tier degradation: Tier 1 (full bar-aligned with score+reference), Tier 2 (absolute MIDI for unknown pieces), Tier 3 (scores only if AMT fails).

**Remaining gaps:** Free tier gating, AMT container deployment (enables zero-config piece ID + bar-aligned analysis in production), observation pacing tuning, eval validation (synthesis quality).

---

## Apps Documentation Map

| Doc | Title | What It Covers |
|---|---|---|
| `01-product-vision.md` | Product Vision | Target users, ideal practice session, UX principles, platform strategy, student model concept |
| `02-pipeline.md` | Audio-to-Observation Pipeline | Full technical pipeline: capture, inference, teaching moment selection, two-stage subagent, provider architecture |
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
| Implement teaching moment selection | COMPLETE | `teaching_moments.rs`, deviation gate + blind-spot + positive fallback + dedup |
| Complete web recording + WebSocket observation flow | COMPLETE | DO orchestration, chunk upload, WebSocket delivery |
| Score following (Phase 1c) | COMPLETE | `chroma_dtw.rs`, chroma-based DTW, cross-chunk continuity |
| Bar-aligned analysis engine (Phase 1d) | COMPLETE | `analysis.rs`, all 6 dims, Tier 1/2/3 degradation |
| Reference cache script (Phase 1e) | COMPLETE | `reference_cache.py`, data generation pending |
| Fuzzy piece matching + demand tracking | COMPLETE | `piece_match.rs`, `score_context.rs`, `piece_requests` table |
| Enrich subagent prompt with musical analysis | COMPLETE | `prompts.rs`, `<musical_analysis>` per-dimension facts |
| Wire iOS cloud inference client | NOT STARTED | HF endpoint deployed, iOS code needs API integration |

### Phase 2: Web Beta -- Core Loop + First Session Magic (~4 weeks)

**Goal:** A pianist opens crescend.ai, plays anything, gets their first useful observation in under 60 seconds. Session intelligence makes observations feel like a teacher in the room.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| Session brain state machine in DO | Existing DO session code | 1 week | P0 | COMPLETE -- practice mode detection + DO state persistence (commit 312a1db) |
| Observation pacing (mode-aware throttle) | Session brain | 3 days | P0 | Implemented, thresholds need tuning with real sessions |
| First-session zero-config flow (AMT piece ID) | Existing AMT + fuzzy match | 1 week | P0 | CODE COMPLETE (merged 2026-03-22) -- pending AMT container deploy |
| Artifact container component (inline/expanded) | -- | 1 week | P0 | COMPLETE -- unified artifact system with tool_use |
| Exercise artifact renderer | Artifact container | 3 days | P0 | COMPLETE -- 25 curated exercises seeded |
| ~~Session opening context (memory retrieval)~~ | ~~Existing memory system~~ | ~~3 days~~ | ~~P0~~ (NOT NEEDED -- memory already flows into every request via build_memory_context) |
| Session closing synthesis | Session brain | 3 days | P0 | COMPLETE -- alarm-triggered synthesis, all exit paths, deferred recovery (synthesis.rs, 963 lines) |
| Memory retrieval wired E2E | Existing D1 queries | 3 days | P0 | COMPLETE -- flows into chat and practice session paths |
| Free tier cap (hardcoded session limit) | -- | 1 day | P1 | NOT STARTED |
| Landing page with value prop | -- | 3 days | P1 | COMPLETE -- landing v2 shipped: ProofCard live demo, ExerciseProofBlock, FinalCTA, BarScoreChip bar-tap |
| Google Sign In on web | Existing API endpoint | 1 day | P1 | COMPLETE -- GSI client, API endpoint, env vars configured (commit 2047caa) |

### Phase 3: iOS + Payment + Rich Artifacts

**Goal:** iOS goes end-to-end. Stripe enables paid tiers. Additional artifact types ship.

| Task | Depends On | Effort | Priority |
|---|---|---|---|
| iOS cloud inference wiring (stub to real API) | Phase 2 API stability | 1-2 weeks | P1 |
| Stripe integration (subscription management) | -- | 1-2 weeks | P1 |
| Usage tracking + tier enforcement | Stripe | 1 week | P1 |
| Score highlight artifact renderer (iOS) | Artifact container | -- | P2 | COMPLETE (#57) -- ArtifactScoreView via WKWebView scorehost bundle |
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
| Two-stage LLM pipeline | Subagent (Workers AI/Gemma 4 26B) + Teacher (Workers AI/glm-4.7-flash; Anthropic/Sonnet 4.6 via TEACHER_PROVIDER=anthropic) | Separates analysis (fast, cheap) from delivery (quality voice, ~1.5s). Different tasks need different models. |
| Multi-provider over single gateway | Workers AI + Anthropic via CF AI Gateway | Co-located inference for subagent; native prompt caching for teacher. |
| Local-first data (iOS) | SwiftData on-device, D1 for backup/sync | Practice works without internet (except LLM call). Phone is authoritative. No conflict resolution needed. |
| Sign in with Apple | Single auth provider, both platforms | Zero friction, App Store requirement, stable cross-device identity. |
| Scores as reasoning inputs | Not a report card | Model is ~80% pairwise accurate. Value is in the subagent analysis and teacher delivery, not raw numbers. |
| Chat-first UI | Text default, components on-demand (~30%) | Mirrors real teaching. Most observations are conversational. Rich components only when visual/interactive aid adds pedagogical value. |
| Piece identification | AMT fingerprint + graceful unknown | Auto-detect via AMT MIDI fingerprint against 242-piece score library. Unknown pieces get audio-quality feedback without bar numbers. Ask piece identity AFTER first observation, not before. |
| Memory without vector search | Structured D1 queries, bi-temporal facts | Domain is narrow (6 dimensions, known ontology, low volume). No graph DB, no embeddings needed. |
| Platform strategy | Web-first, iOS follows | Web is ~90% complete, fastest to iterate, shareable URL for growth. iOS catches up after beta validation. |
| Session intelligence | Durable Object as session brain | Practice mode state machine (warming/drilling/running/winding) with mode-aware observation pacing. Single-threaded DO holds all session state. |
| Artifact system | Unified container (inline to expanded) | One `<Artifact>` component renders all rich content types. Lives in chat, expands to viewport on demand. Teacher LLM declares artifacts via tool_use (tool_choice: auto; glm-4.7-flash@WorkersAI default). Hybrid catalog lookup + generated fallback. COMPLETE. |
| First session | Zero-config magic | Sign in, play anything, first observation in <60s. AMT fingerprint for piece ID; graceful degradation if unknown. Piece ID enriches but never gates. |
| Monetization | Tiered: Free / $5 Plus / $20 Pro / $50 Max | Free tier with daily/weekly limits as growth engine. Inference cost reduction to ~$1/session via model v2. |

---

## Open Questions

### Pipeline

1. **Deviation-gate calibration.** Does the `deviation < 0` gate behave sensibly on intermediate students with phone audio, given baselines are still sparse? Validate with real sessions.
2. **Minimum deviation threshold.** The deviation gate already returns a positive moment when no chunk falls below baseline. Open question: should there also be a *magnitude* floor (ignore trivially-small negative deviations) so the system says "sounded good, keep going" instead of surfacing tiny dips?
3. **Positive/corrective ratio.** Target: 70% corrective, 30% positive. Validate with real users. See `02-pipeline.md`.

### Memory

4. **When does synthesis become necessary?** At 50-100+ observations per student (months of use). Until then, raw observation retrieval may suffice. See `03-memory-system.md`.
5. **Student-reported facts.** "I have a recital in 3 weeks" -- store in `synthesized_facts` with `source_type = 'student_reported'` or a separate table?

### Exercises

6. ~~**Notation rendering library.**~~ RESOLVED — Verovio WASM in a Web Worker (Phase 1). Phase 2 SHIPPED: ScoreIR intermediate representation (`score-ir.ts`), ScoreCursor rAF playback cursor (`score-cursor.ts`), and ScoreRenderer worker API with `load/getIR/getPage/getClip` (`score-renderer.ts`, `score-worker.ts`). 85 tests passing. See `05-ui-system.md`.
7. ~~**Curated exercise count for V1.**~~ RESOLVED -- 25 curated exercises seeded in migration `0004_exercises.sql`, covering all 6 dimensions and 3 difficulty levels. LLM-generated path deferred to post-beta.

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
