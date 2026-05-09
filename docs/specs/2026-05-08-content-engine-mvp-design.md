# Content Engine MVP Design

**Goal:** Ship a Python-based content pipeline that produces 3 single-observation YouTube Shorts per week (cross-posted to TikTok + Reels), driving B2C app installs starting at crescendai's beta rollout.

**Not in scope:**
- Long-form YouTube content
- TTS / voice cloning (Jai records voiceover manually for MVP)
- Stage-2 craft critic agent (Jai handles craft assessment in swipe review at MVP)
- Audience-submission CTA flow (CTA Phase C is config-deferred)
- Recruited collaborator-teacher student clip sourcing (deferred until 1-3 collaborators are signed; not MVP)
- Per-platform render reformatting (Postiz cross-posts identical 9:16 asset)
- Cloud deployment (MVP runs locally on Jai's machine; cloud migration deferred)
- Founder-authored manual posts that bypass the pipeline
- Replacement of any other crescendai surface (this is a standalone app)

## Problem

crescendai is pre-launch with no audience. The product evaluates piano performance expression — a category Simply Piano / Flowkey / Piano Marvel do not address. With no install funnel running, beta launch arrives to silence: the product ships into a market with no top-of-funnel traffic, and B2C install conversion depends entirely on paid ads or word-of-mouth, both of which are slow and expensive at zero scale.

The market gap is not arbitrage (the article framing) — there is no busy competitive content market in piano-performance-AI to arbitrage from. The actual gap is **creation**: nobody is producing music-AI content at all. Producing it ourselves is a moat only crescendai can fill, because only crescendai can run the model and narrate what it heard.

Without a content engine running by beta launch, three things break:
1. **No top-of-funnel.** Installs depend on paid ads from day 0, which is the worst possible time (no install attribution data to optimize against).
2. **No moat content compounding.** Reaction/observation episodes are inventory — every shipped episode is a permanent install asset. Starting at beta means losing 4-8 weeks of compounding distribution.
3. **No clip-discovery flywheel.** Public piano clips with model-observable issues are abundant; without a scout running, candidate inventory is zero.

## Solution (from the user's perspective)

Jai operates the engine via three touchpoints, ~15 minutes/day total:

1. **Swipe review (5-10 min/day).** Opens a local web UI. Sees a queue of candidate clips, each annotated with the model's selected observation and a generated voiceover script. Approves or rejects with a tap. Approved candidates advance through the pipeline asynchronously.
2. **Voiceover recording (3-5 min/day).** For approved + critic-passed episodes, records a ≤45-second voiceover from a script the engine wrote. Browser mic capture, single-take, attached to the episode.
3. **Final approval (1-2 min/day).** Scrubs through the rendered 9:16 video, approves or rejects. Approved videos auto-schedule via Postiz to YouTube Shorts (lead) + TikTok + Reels with phase-appropriate CTA.

The engine handles the rest: scraping candidate clips nightly, running them through crescendai inference, picking the most concrete + audible observation per clip, drafting voiceover scripts, fact-checking the model's observation against the audio (binary brand-safety gate), rendering 9:16 video assets, scheduling cross-posts, ingesting per-post analytics, and adjusting clip-scout ranking weights based on which past episodes converted.

End state at week 4 of operation: ~12 published episodes, ~0.5-2K total views (Shorts cold-start), early UTM-attributed install signal informing future scout weights. End state at week 12 (post-beta): ~36 episodes, established channel with algorithmic distribution warming up, threshold for CTA Phase A→B upgrade reachable.

## Design

### Approach

Build a local-first Python worker that drives a single state machine over an SQLite-backed `Episode` record. Four agents perform the LLM-heavy work; five thin adapters hide external systems; one renderer composes the asset. Jai is in the loop at three checkpoints (candidate approval, voiceover capture, final asset approval) and can override the brand-safety critic when the LLM is wrong.

### Key decisions and trade-offs

| Decision | Choice | Trade-off accepted |
|---|---|---|
| Format | Single-observation episode (clip + one model observation + audio proof) over transformation arc | No multi-week journey content; gain: no lead-time problem, day-0-of-beta = day-0-of-content |
| Platform priority | YouTube Shorts lead, TikTok + Reels cross-post | Slower cold-start than TikTok-lead; gain: best Shorts→install conversion mechanic, audience demographic match |
| Cadence | 3 episodes/week sustained | Below "5/week creator wisdom"; gain: sustainable for solo founder, no week-6 burnout collapse |
| Architecture | Build fresh, open-source content-workflow repo as reference only | ~2-3 weeks scaffolding vs ~3 days fork; gain: clean abstractions for clip-and-observation pipeline |
| Quality gate | Two-stage critic — Stage 1 truthfulness binary kill (LLM + Jai override), Stage 2 craft deferred (Jai swipe-review at MVP) | Less automated quality assurance; gain: ship 4-agent MVP in weeks not months |
| CTA strategy | Phased A (passive) → B (soft + landing page) → C (submission flywheel), config-swappable | Lowest install conversion at MVP (~0.05-0.2% view→install); gain: algorithm-friendly while building audience, upgrade is config not code |
| LLM routing | Workers AI for `observation_selector`; Claude Code CLI for `narrator` + `critic_truthfulness` | ToS exposure on CLI, no structured output mode for CLI calls; gain: subscription cost leverage, single deep adapter (`llm_gateway`) makes API-fallback a one-file swap |
| Runtime | Local Python worker + SQLite + launchd cron | Not durable if Jai's machine is off; gain: zero infra cost, fastest path to first episode, simplest dev loop |
| Sourcing | Jai's own playing + scraped public clips; collaborator-students phased in later | Brand drifts toward "Jai-centric" if Jai-clips dominate; mitigation: cap Jai's-own-playing at ≤30% of feed |
| Clip length cap | ≤20 sec for scraped clips | Some interesting passages excluded; gain: strongest fair-use commentary posture |

### Pipeline state machine

```
candidate → curated → analyzed → observation_selected
   → script_drafted → critic_passed → recorded → rendered
   → scheduled → published → measured
```

Failure states `failed_<stage>` and `killed_truthfulness` are surfaced in the swipe-review UI for Jai recovery. Every transition persists artifacts before advancing — a crashed worker resumes mid-pipeline; nothing recomputes that already succeeded.

### Data flow

1. `clip_scout` runs nightly (launchd cron at 06:00 local). Searches YouTube + TikTok per `source_criteria` config. Emits ~20-50 candidates → `state=candidate`.
2. Jai opens swipe UI, approves/rejects. Approved → `state=curated`.
3. `model_runner` calls crescendai HF inference endpoint, persists 6-dim ModelOutput → `state=analyzed`.
4. `observation_selector` (Workers AI via `crescendai-background` gateway) emits one `Observation{dimension, time_range, plain_english}` → `state=observation_selected`.
5. `narrator` (Claude Code CLI) writes ScriptText (≤45 sec spoken). Reads current `cta_config` phase → `state=script_drafted`.
6. `critic_truthfulness` (Claude Code CLI) returns `Verdict(pass|kill, reason)`. Kill → `state=killed_truthfulness` surfaced for Jai. Pass → `state=critic_passed`.
7. Jai reviews script + critic verdict, may override critic, records voiceover audio → `state=recorded`.
8. `renderer` composes 9:16 video (clip audio + waveform overlay + model-output text overlay + voiceover, with CTA template per current phase) → `state=rendered`.
9. Jai final-approves render → `scheduler` cross-posts via Postiz → `state=scheduled`, then `state=published`.
10. `analytics_ingestor` runs daily (07:00 local), pulls views + UTM-tracked installs → `state=measured`.
11. `feedback_scorer` runs weekly, updates `source_criteria` ranking weights based on which past episode-types converted. Next `clip_scout` run picks up the new weights.

### CTA phases

| Phase | Description | Upgrade trigger |
|---|---|---|
| A (MVP) | No in-video CTA. Channel description + bio link with UTM (`?utm_source=shorts&utm_medium=organic&utm_campaign=ce`). Brand watermark on render. | Default at launch |
| B | End-card "crescend.ai" + landing page `crescend.ai/shorts` with one-line value-prop + App Store buttons. UTM-tagged. | 1K subs OR 10K cumulative views (whichever first) |
| C | Spoken submission CTA in voiceover ("send your clip — crescend.ai/submit"). Submission flow ingests clip → episode candidate. | Beta product supports clip-upload + analysis flow + Phase B retired thresholds met |

CTA phase is a row in `cta_config` table. Render template binds to phase at render time. Phase upgrade is a manual config change, not a code change.

### Error handling philosophy

Per project convention (CLAUDE.md): explicit exception handling, no silent fallbacks. Specifically:

- LLM agent failures retry once with backoff, then transition episode to `failed_<agent>` for Jai recovery. **No template fallback** — would silently degrade content quality.
- `critic_truthfulness` infra failure does NOT default-pass. Default-pass would be a backdoor through the brand-safety gate.
- `renderer` failure transitions to `failed_render` with the ffmpeg error captured; re-render is byte-deterministic and replaces prior output.
- `scheduler` partial failures (e.g., posted to YT, rejected by TikTok) record per-platform status; Jai sees partial-publish in UI.
- `analytics_ingestor` missing-data is acceptable; mark partial, don't kill the loop.
- All exceptions captured by Sentry (project: `crescendai-content-engine`).

## Modules

### Deep modules

**`clip_scout`** (`content_engine/agents/clip_scout.py`)
- Interface: `search(criteria: SourceCriteria, count: int) → list[Candidate]`
- Hides: YouTube Data API + TikTok scraping logic, license inference (CC vs label-recorded vs amateur-public), per-platform query construction, dedup against past episodes, ranking by `source_criteria.weights`.
- Tested through: golden-set tests over fixed mock API responses; property tests on filter correctness and rank ordering.

**`observation_selector`** (`content_engine/agents/observation_selector.py`)
- Interface: `select(model_output: ModelOutput, metadata: ClipMetadata) → Observation`
- Hides: scoring function over (concreteness × audibility × dimension-priority × novelty-vs-recent-episodes); Workers AI gateway call with structured-JSON output mode; prompt construction. **This is the engine's core IP.**
- Tested through: golden eval set of (model_output, expected_dimension); property tests on output schema validity and time_range bounds.

**`narrator`** (`content_engine/agents/narrator.py`)
- Interface: `write_script(observation: Observation, cta_config: CtaConfig, style_examples: list[Example]) → ScriptText`
- Hides: Claude Code CLI invocation (subprocess + output parsing), prompt construction, length budgeting (≤120 words ≈ 45 sec spoken), hook-template selection, voice-consistency enforcement.
- Tested through: property tests on script length, hook position, CTA-phase match; golden-set sample review for voice consistency.

**`critic_truthfulness`** (`content_engine/agents/critic_truthfulness.py`)
- Interface: `verify(clip_path: Path, observation: Observation) → Verdict`
- Hides: Claude Code CLI invocation, prompt construction with audio context, reasoning-trace capture for observability, kill/pass decision schema parsing. **Highest-stakes module — false-negatives ship wrong observations.**
- Tested through: golden eval set including deliberately false observations; precision and recall tracked; **FN rate < 5% as deploy gate**.

**`llm_gateway`** (`content_engine/adapters/llm_gateway.py`)
- Interface: `complete(prompt: str, mode: LlmMode, schema: JsonSchema | None = None) → LlmResponse`
- Hides: routing decision (CLI subprocess for `narrator`/`critic` modes, Workers AI HTTP for `selector` mode); subprocess management with timeout + retry; output parsing including stripping any output-style decorations from CLI; structured-JSON validation when schema provided. **The CLI ↔ API swap point — entire LLM provider strategy is one file's surface.**
- Tested through: unit tests with mocked subprocess and mocked HTTP; verifies correct provider selected per mode, correct retry on transient failures, correct schema validation on Workers AI mode.

**`renderer`** (`content_engine/render/renderer.py`)
- Interface: `render(episode: Episode, cta_config: CtaConfig) → VideoPath`
- Hides: ffmpeg subprocess orchestration, audio mixing (clip audio + voiceover ducking), waveform/spectrogram generation, text overlay positioning, CTA template binding, output validation (9:16 dimensions, expected duration ±0.5s, audio present).
- Tested through: determinism property test (identical inputs → byte-identical output); output-validity assertions; CTA template-binding unit tests.

**`episode_store`** (`content_engine/store/episode_store.py`)
- Interface: `get(id) → Episode; save(episode) → None; transition(id, new_state) → Episode; list_by_state(state) → list[Episode]`
- Hides: SQLite connection management, schema migrations, state-machine transition validation (rejects invalid transitions), serialization of nested JSON columns (model_output, observation, posts).
- Tested through: unit tests on transition validity (raises on invalid), persistence (save then get round-trips), state filtering.

**`config_store`** (`content_engine/store/config_store.py`)
- Interface: `get(key, version=latest) → Config; create_version(key, value) → version_id; list_versions(key) → list[ConfigRow]`
- Hides: versioned-config storage, version resolution semantics, immutability of historical versions.
- Tested through: unit tests on version creation immutability, latest-resolution correctness.

**`feedback_scorer`** (`content_engine/feedback/scorer.py`)
- Interface: `update_weights(metrics_since: datetime) → version_id`
- Hides: scoring formula (per-episode install-conversion as feedback signal, weight delta computation, normalization).
- Tested through: unit tests with synthetic metrics → expected weight delta.

### Shallow modules (justified)

| Module | Why shallow |
|---|---|
| `pipeline.orchestrator` | **Intentional.** Must remain a dumb dispatcher to prevent god-class drift. State machine + per-state dispatch only. |
| `pipeline.states`, `pipeline.episode` | Pure data primitives (enum, dataclass). |
| `adapters.model_runner`, `adapters.scheduler`, `adapters.analytics_ingestor` | Thin HTTP adapters. Per design Section 2 rule: adapters MUST NOT grow business logic — that belongs in deep modules. |
| `render.templates` | Pure data (template dataclasses). |
| `ui.server` | Thin HTTP routing; logic delegated to stores and agents. |
| `cli` | typer entrypoint dispatcher. |
| `observability` | One-line Sentry init. |

## File Changes

| File | Change | Type |
|---|---|---|
| `apps/content-engine/pyproject.toml` | uv-managed Python project: deps include `httpx`, `typer`, `flask`, `apscheduler`, `sentry-sdk`, `ffmpeg-python` (optional helper), `pytest`, `pytest-asyncio`, `yt-dlp` (or `youtube-search-python` for metadata-only) | New |
| `apps/content-engine/.gitignore` | ignore `data/` (SQLite + media), `.env`, `__pycache__/` | New |
| `apps/content-engine/.env.example` | document required env vars: `CRESCENDAI_INFERENCE_URL`, `CRESCENDAI_INFERENCE_TOKEN`, `CF_AI_GATEWAY_URL`, `CF_API_TOKEN`, `POSTIZ_URL`, `POSTIZ_TOKEN`, `YOUTUBE_API_KEY`, `SENTRY_DSN_CONTENT_ENGINE`, `CLAUDE_CODE_BIN` (path to claude CLI) | New |
| `apps/content-engine/content_engine/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/pipeline/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/pipeline/states.py` | `State` enum (12 states), `is_valid_transition(from, to) → bool` | New |
| `apps/content-engine/content_engine/pipeline/episode.py` | `Episode` dataclass: `id`, `candidate_url`, `source_type`, `model_output`, `observation`, `script_text`, `voiceover_path`, `render_path`, `posts`, `analytics`, `state`, `config_versions`, `created_at`, `updated_at` | New |
| `apps/content-engine/content_engine/pipeline/orchestrator.py` | `Orchestrator.tick()` — reads pending episodes, dispatches to right component per state, persists transitions. Logic-less. | New |
| `apps/content-engine/content_engine/agents/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/agents/clip_scout.py` | YT/TikTok candidate discovery | New |
| `apps/content-engine/content_engine/agents/observation_selector.py` | ModelOutput → Observation via Workers AI structured JSON | New |
| `apps/content-engine/content_engine/agents/narrator.py` | Observation → ScriptText via Claude Code CLI | New |
| `apps/content-engine/content_engine/agents/critic_truthfulness.py` | (clip, observation) → Verdict via Claude Code CLI | New |
| `apps/content-engine/content_engine/adapters/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/adapters/llm_gateway.py` | Single LLM swap point: `complete(prompt, mode, schema?)` routing CLI vs Workers AI | New |
| `apps/content-engine/content_engine/adapters/model_runner.py` | crescendai HF inference HTTP call | New |
| `apps/content-engine/content_engine/adapters/scheduler.py` | Postiz HTTP cross-post (YT Shorts + TikTok + Reels) | New |
| `apps/content-engine/content_engine/adapters/analytics_ingestor.py` | Per-platform analytics pull (YouTube Data API + TikTok + IG Graph) | New |
| `apps/content-engine/content_engine/render/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/render/renderer.py` | ffmpeg-based 9:16 composition (clip + waveform + text overlay + voiceover + CTA) | New |
| `apps/content-engine/content_engine/render/templates.py` | CTA template dataclasses for phase A/B/C | New |
| `apps/content-engine/content_engine/store/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/store/episode_store.py` | SQLite-backed episode CRUD + transitions | New |
| `apps/content-engine/content_engine/store/config_store.py` | Versioned configs (cta_config, source_criteria, ranking_weights) | New |
| `apps/content-engine/content_engine/store/schema/001_init.sql` | DDL: `episode`, `cta_config`, `source_criteria`, `ranking_weights`, `migrations` tables | New |
| `apps/content-engine/content_engine/feedback/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/feedback/scorer.py` | Analytics → ranking-weight version updater | New |
| `apps/content-engine/content_engine/ui/__init__.py` | package marker | New |
| `apps/content-engine/content_engine/ui/server.py` | Flask app: swipe-review UI + voiceover record + final-approve UI | New |
| `apps/content-engine/content_engine/ui/templates/swipe.html` | candidate review template | New |
| `apps/content-engine/content_engine/ui/templates/record.html` | voiceover capture template | New |
| `apps/content-engine/content_engine/ui/templates/approve.html` | final render approve template | New |
| `apps/content-engine/content_engine/cli.py` | typer entrypoints: `scout`, `tick`, `ui`, `migrate`, `feedback`, `ingest` | New |
| `apps/content-engine/content_engine/observability.py` | Sentry init for `crescendai-content-engine` project | New |
| `apps/content-engine/tests/__init__.py` | package marker | New |
| `apps/content-engine/tests/conftest.py` | shared fixtures (in-memory SQLite, mock LLM gateway, sample model output) | New |
| `apps/content-engine/tests/unit/test_state_transitions.py` | episode_store transition validity | New |
| `apps/content-engine/tests/unit/test_pipeline_dispatch.py` | orchestrator dispatch correctness per state | New |
| `apps/content-engine/tests/unit/test_renderer_determinism.py` | byte-identical output property | New |
| `apps/content-engine/tests/unit/test_renderer_validity.py` | output dimensions/duration/audio | New |
| `apps/content-engine/tests/unit/test_cta_template_binding.py` | phase A/B/C → correct render template | New |
| `apps/content-engine/tests/unit/test_scheduler_adapter.py` | Postiz contract with mocked HTTP | New |
| `apps/content-engine/tests/unit/test_model_runner_adapter.py` | inference adapter contract | New |
| `apps/content-engine/tests/unit/test_analytics_adapter.py` | analytics pull contract | New |
| `apps/content-engine/tests/unit/test_feedback_scorer.py` | known metrics → expected weight delta | New |
| `apps/content-engine/tests/unit/test_llm_gateway.py` | mode routing, subprocess mock, schema validation | New |
| `apps/content-engine/tests/unit/test_config_store.py` | versioning immutability, latest resolution | New |
| `apps/content-engine/tests/property/test_clip_scout_properties.py` | filter correctness, ranking properties | New |
| `apps/content-engine/tests/property/test_observation_selector_properties.py` | output schema, time_range bounds | New |
| `apps/content-engine/tests/property/test_narrator_properties.py` | length, hook position, CTA-phase match | New |
| `apps/content-engine/tests/eval/golden_observations.json` | seed eval set: 10 (model_output, expected_dimension) cases | New |
| `apps/content-engine/tests/eval/golden_critic.json` | seed eval set: 15 cases (10 true observations + 5 deliberately false) | New |
| `apps/content-engine/tests/eval/test_observation_selector_eval.py` | golden-set accuracy ≥80% on dimension category | New |
| `apps/content-engine/tests/eval/test_critic_eval.py` | precision tracking, recall tracking, **FN rate < 5% deploy gate** | New |
| `apps/content-engine/tests/e2e/test_pipeline_smoke.py` | full pipeline with mocked externals → state=scheduled | New |
| `apps/content-engine/scripts/launchd/com.crescendai.content-engine.plist` | macOS launchd plist for ticker (every 5 min) + scout (06:00) + analytics (07:00) | New |

## Open Questions

- **Q:** Which YouTube data source API is permissible at scale — official Data API (quota-limited, 10K units/day) or yt-dlp metadata mode? **Default:** start with YouTube Data API for search; fall back to yt-dlp for metadata enrichment if quota becomes a bottleneck. Document the quota cost in `clip_scout` so consumption is observable.
- **Q:** Does Postiz's TikTok integration require an approved TikTok Business account? **Default:** verify during /build before scheduler implementation; if blocked, MVP ships YouTube + Reels only and TikTok is added in a follow-up.
- **Q:** Where does the rendered video file live? Local disk (MVP) vs. R2 from day 1? **Default:** local disk under `apps/content-engine/data/renders/`, gitignored. Migration to R2 happens with cloud migration.
- **Q:** Should the swipe UI authenticate? **Default:** no auth at MVP — it binds to `127.0.0.1` only and requires Jai's machine to be reachable. Add auth at cloud migration.
- **Q:** What is the exact `crescendai_inference_url` for MuQ? **Default:** read from `.env` at boot; resolution deferred to /build (likely points at the existing HF Inference Endpoint per `apps/inference/muq/handler.py`).
