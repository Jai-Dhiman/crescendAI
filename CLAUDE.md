# CrescendAI

**"A teacher for every pianist."**

Multi-platform (iOS + web) practice companion for pianists. Cloud audio inference via HF endpoint, with a Cloudflare Workers backend for LLM, STOP classification, and data sync.

## What CrescendAI Does

- Evaluates *how* a piano performance sounds (tone, dynamics, phrasing, pedaling) -- not just note accuracy like MIDI-based apps
- Uses dual encoders -- MuQ (audio, finetuned) and Aria (symbolic, 650M-param pretrained from EleutherAI) -- with gated fusion, outputting 6 teacher-grounded dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation
- Target users: self-learners (B2C), music educators (B2B), institutions (B2B)
- Competitors (Simply Piano, Flowkey, Piano Marvel) check note accuracy via MIDI; CrescendAI evaluates musical expression from audio

## Architecture

See `docs/architecture.md` for the full system design. Key points:

- **Web (beta-first):** Browser audio capture (MediaRecorder), cloud inference (HF endpoint), real-time observations via WebSocket, chat-first teacher interface with unified artifact system
- **iOS (follows web beta):** Audio capture (AVAudioEngine), chunk upload to API, SwiftData local-first persistence. Inference client needs cloud wiring.
- **Cloud backend:** Cloudflare Workers (split inference: MuQ endpoint + Aria-AMT endpoint, parallel dispatch from DO + STOP classifier + teaching moment selection + session brain DO + LLM proxy + D1 data sync), Sign in with Apple + Google auth
- **Zero-config piece ID:** Multi-signal MIDI-based piece identification (N-gram recall + statistical rerank + DTW confirmation). Student plays, system identifies the piece automatically from 242-score library.
- **Session intelligence:** Durable Object as session brain with practice mode state machine (warming/drilling/running/winding), mode-aware observation pacing
- **Artifact system:** Unified container for rich components (exercises, score highlights, references). Lives inline in chat, expands to viewport. Teacher LLM declares via tool use (pattern TBD)
- **Teacher pipeline:** Two-stage subagent (fast analysis) + teacher LLM (quality voice + artifact tool use). Score-conditioned quality assessment when score MIDI available. Model scores are reasoning inputs, not ground truth. See `docs/apps/02-pipeline.md`
- **Monetization:** Tiered pricing (Free / $5 Plus / $20 Pro / $50 Max). Free tier with daily/weekly limits as growth engine.
- **Apps docs:** `docs/apps/00-status.md` through `docs/apps/05-ui-system.md` (each has status header)
- **Documentation entry point:** `docs/architecture.md` -- system diagram + pointers to model and apps docs

## Model Strategy

- **Model v2:** MuQ (audio, pretrained on 160K hrs) + Aria (symbolic, pretrained on 820K MIDIs) with gated fusion
- **Score conditioning:** delta = z_perf - z_score when score MIDI available
- **Training:** PercePiano anchor (20%) + ordinal-dominated (80%) with T2 competition + T5 YouTube Skill data
- **ALL previous pairwise numbers are INVALID** -- fold leak discovered, folds now fixed
- **S2 GNN is LEGACY** -- Aria replaces all custom symbolic encoders
- **Phase 3 (Symbolic FM) ELIMINATED** -- Aria IS the symbolic foundation model

## Development

Uses `just` (justfile) for dev commands. Install: `brew install just`.

| Command | What it starts |
|---------|---------------|
| `just dev` | All 4 services: MuQ (8000) + AMT (8001) + API (8787) + Web (3000) |
| `just dev-muq` | MuQ + API + Web (no AMT, faster startup) |
| `just dev-light` | API + Web only (uses production HF endpoints) |
| `just muq` / `just amt` / `just api` / `just web` | Individual services |
| `just fingerprint` | Generate N-gram index + rerank features from score library |
| `just test-model` / `just test-api` / `just check-api` | Tests and checks |
| `just deploy-api` | Deploy API worker to production |
| `just migrate-local` / `just migrate-prod` | Apply D1 migrations |

## Package Managers

- Python: `uv` (not pip)
- JavaScript: `bun` (not npm)

## Coding Standards

- Explicit exception handling over silent fallbacks
- No backup files when making fixes
- No emojis unless explicitly requested
- **Rust (API):** Follow `apps/api/RUST_STYLE.md` for all Rust code. Key rules: `thiserror` error enums (not `Result<T, String>`), `#[serde(rename_all = "camelCase")]` on API types, no `.unwrap()` in handlers, `RefCell` not `Mutex`, trait objects over generics in cold paths for binary size.

## Observability

- **Error tracking:** Sentry across all surfaces (iOS, web, API)
  - iOS: `sentry-cocoa` SPM -- crash reporting, error capture, breadcrumbs
  - Web: `@sentry/react` -- client-side errors, API errors, WebSocket errors
  - API (Rust/WASM): Cloudflare Workers Observability OTLP drain to Sentry (no SDK)
- Sentry org: `crescendai` with projects: `crescendai-api`, `crescendai-web`, `crescendai-ios`
- Error logging: `console_error!` (Rust), `Sentry.captureException` (web), `SentrySDK.capture(error:)` (iOS)

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

### `/plan-ceo-review` -- Product Vision Review

**What:** CEO/founder-mode plan review. Rethinks the problem from first principles, challenges premises, maps the 12-month ideal, finds the 10-star product version. Three modes: SCOPE EXPANSION (dream big), HOLD SCOPE (maximum rigor), SCOPE REDUCTION (strip to essentials).
**When:** Before starting a major new feature or initiative. When deciding *what* to build, not *how* to build it. When you need to challenge whether the plan is ambitious enough or too ambitious.
**How:** Describe what you want to build, then invoke `/plan-ceo-review`. Select a mode. Answer one question at a time. The review walks through architecture, error handling, security, edge cases, tests, performance, observability, deployment, and long-term trajectory.

### `/plan-eng-review` -- Engineering Execution Review

**What:** Eng manager-mode plan review. Locks in the execution plan: architecture decisions, data flow diagrams, edge cases, test coverage, performance characteristics. Walks through issues interactively with opinionated recommendations.
**When:** After you have a plan and before you start coding. When deciding *how* to build something, not *what* to build. Use after `/plan-ceo-review` if you did product visioning first.
**How:** Share an implementation plan, then invoke `/plan-eng-review`. It stress-tests architecture, identifies ambiguities, and surfaces decisions that need to be made before implementation starts.

### `/review` -- Pre-Landing PR Review

**What:** Analyzes diff against main for SQL safety, LLM trust boundary violations, conditional side effects, and structural issues.
**When:** Before creating a PR or merging. After implementation is complete and tests pass.
**How:** Invoke `/review` with your changes staged or committed. It reads the diff and produces a structured review.

### `/ship` -- Full Ship Workflow

**What:** Merge main, run tests, review diff, bump VERSION, update CHANGELOG, commit, push, create PR. One command, full ceremony.
**When:** When a feature is complete, reviewed, and ready to land.
**How:** Invoke `/ship`. It handles the entire merge-to-PR workflow.

### Other gstack skills

- `/browse` -- Headless browser for QA testing. Navigate, interact, screenshot, verify. Use for all web browsing.
- `/qa` -- Systematic QA testing. Four modes: diff-aware, full, quick, regression. Produces report with health score.
- `/setup-browser-cookies` -- Import cookies from your real browser for authenticated QA testing.
- `/retro` -- Weekly engineering retrospective from commit history.

## Superpowers

### `/superpowers:brainstorming` -- Creative Exploration

**What:** Explores user intent, requirements, and design before implementation. Prevents jumping straight to code by forcing you to think through what you're actually building.
**When:** Before any creative work: creating features, building components, adding functionality, modifying behavior. Use before `/superpowers:writing-plans`.
**How:** Describe what you want to build. The brainstorm explores alternatives, trade-offs, and design options before committing to an approach.

### `/superpowers:dispatching-parallel-agents` -- Fan Out Independent Work

**What:** Launches 2+ parallel subagents for independent tasks that don't share state or have sequential dependencies.
**When:** When your plan has independent components that can be built simultaneously. Example: building a data model, an API endpoint, and a UI component that don't depend on each other.
**How:** Describe the independent tasks. Each gets its own agent that works autonomously and reports back.

### `/superpowers:using-git-worktrees` -- Isolated Feature Work

**What:** Creates isolated git worktrees for feature development. Each worktree is a full copy of the repo on its own branch, preventing conflicts when multiple work streams are active.
**When:** Before starting feature work that needs isolation. Before executing implementation plans with multiple parallel tracks. When you want to experiment without affecting your current working directory.
**How:** Invoke before starting work. It creates a worktree with smart directory selection and safety verification.

### `/superpowers:test-driven-development` -- TDD Workflow

**What:** Write tests first, then implement. Enforces the red-green-refactor cycle.
**When:** When implementing any feature or bugfix, before writing implementation code. Especially important for algorithmic work (score following, rubato detection) where correctness is critical.
**How:** Invoke before writing implementation code. It guides you through writing failing tests first, then implementing to make them pass.

### `/superpowers:verification-before-completion` -- Verify Before Claiming Done

**What:** Requires running verification commands and confirming output before making any success claims. Evidence before assertions.
**When:** When about to claim work is complete, fixed, or passing. Before committing or creating PRs.
**How:** Invoke when you think you're done. It forces you to actually run tests, check output, and confirm everything works.

### Other superpowers skills

- `/superpowers:writing-plans` -- Create structured implementation plans from specs/requirements. Use after brainstorming.
- `/superpowers:executing-plans` -- Execute plans step by step with review checkpoints.
- `/superpowers:subagent-driven-development` -- Execute plans with independent tasks using subagents.
- `/superpowers:systematic-debugging` -- Structured debugging before proposing fixes.
- `/superpowers:requesting-code-review` -- Self-review before merging.
- `/superpowers:receiving-code-review` -- Handle code review feedback with technical rigor.
- `/superpowers:finishing-a-development-branch` -- Guide completion: merge, PR, or cleanup.

## Frontend Design

### `/frontend-design` -- Production-Grade UI

**What:** Creates distinctive, production-grade frontend interfaces with high design quality. Generates creative, polished code that avoids generic AI aesthetics.
**When:** When building web components, pages, or applications. Use for any UI work on the web practice companion.
**How:** Describe the component or page you need. It produces polished, opinionated code with strong visual design.

### UI Review (`userinterface-wiki` skill)

**What:** 152 rules across 12 categories (animation, timing, exit animations, CSS, audio, UX laws, typography, visual design) for reviewing frontend code quality. Complements `/frontend-design` -- one generates, the other reviews.
**When:** After building or modifying web UI components. During `/review` of PRs that touch `apps/web/`. When checking animation timing, interaction design, or typography.
**How:** Reference rules by prefix ID (e.g., `timing-under-300ms`, `ux-doherty-under-400ms`). Full rules with code examples in `~/.claude/skills/userinterface-wiki/rules/`. Compiled reference in `~/.claude/skills/userinterface-wiki/AGENTS.md`.
**Key rules for CrescendAI:** `ux-doherty-under-400ms` (inference feedback), `ux-progressive-disclosure` (chat interface), `ux-peak-end-finish-strong` (post-session), `spring-for-gestures` (recording UI), `timing-under-300ms` (interactions).

## Hugging Face Skills

### `/hugging-face-cli` -- Hub Operations

**What:** Execute HF Hub operations: download models/datasets, upload files, create repos, manage cache, run compute jobs.
**When:** When deploying model updates to HF endpoint, managing inference handler, downloading datasets.

### `/hugging-face-datasets` -- Dataset Management

**What:** Create and manage datasets on HF Hub. Supports streaming row updates and SQL-based querying.
**When:** When publishing training data, creating evaluation datasets, or querying existing datasets.

### `/hugging-face-evaluation` -- Model Evaluation

**What:** Add and manage evaluation results in model cards. Run custom evaluations with vLLM/lighteval.
**When:** When benchmarking model performance, comparing encoder variants, publishing results.

### `/hugging-face-jobs` -- Cloud Compute

**What:** Run workloads on HF Jobs infrastructure. UV scripts, Docker jobs, hardware selection, cost estimation.
**When:** When running training, batch inference, data processing, or any GPU workload without local setup.

## Autoresearch

### `/autoresearch` -- Autonomous Improvement Loop

**What:** Iteratively improves any artifact (skill, prompt, code, config) against a frozen metric. Modify one thing, measure, keep or revert, repeat. Git-based tracking with full changelog.
**When:** When optimizing skill prompts, code performance, content quality, or any artifact with a measurable "better." When you want to run 10-30 experiments unattended.
**How:** Define Goal, Scope (files that can change), Metric (how to measure), Verify (command to run), and Guard (regression check). Then invoke `/autoresearch` and let it loop.

## Recommended Workflow

### Full Feature Cycle

```
 IDEATION
   /office-hours                             -- new product bets only (e.g., "add sight-reading?")
   /superpowers:brainstorming                -- every feature (always)

 PLANNING
   /superpowers:writing-plans                -- draft the plan
   /autoplan                                 -- review gauntlet (CEO + design + eng, auto-decided)
     OR pick individual reviews:
       /plan-ceo-review                      -- when scope/ambition is in question
       /plan-design-review                   -- when UI/UX is involved
       /plan-eng-review                      -- always (architecture is never optional)

 EXECUTION
   /superpowers:using-git-worktrees          -- isolate the work
   /superpowers:test-driven-development      -- tests first
   /superpowers:executing-plans              -- step by step with checkpoints
   /superpowers:dispatching-parallel-agents  -- when tasks are independent
   /superpowers:verification-before-completion -- evidence before claims

 REVIEW
   /review                                   -- diff safety (always before PR)
   /superpowers:requesting-code-review       -- requirements/intent check
   /design-review                            -- UI work only (live site visual QA + auto-fix)
   /cso                                      -- security-sensitive work (auth, API, data boundaries)

 SHIP & VERIFY
   /ship                                     -- merge, changelog, PR
   /qa or /browse                            -- verify in production
   /retro                                    -- end of sprint

 CROSS-CUTTING (as needed)
   /superpowers:systematic-debugging         -- code bugs
   /investigate                              -- production incidents, deep root cause
   /autoresearch                             -- measurable optimization loops (model, prompts, perf)
```

### Shortcuts

- **Small bugfix:** debug -> fix -> `/review` -> `/ship`
- **UI-only change:** brainstorm -> plan -> `/plan-design-review` -> build -> `/design-review` -> `/ship`
- **Model/ML work:** brainstorm -> plan -> `/plan-eng-review` -> TDD -> `/autoresearch` -> `/ship`
- **Security-touching:** add `/cso` before `/ship`

## Project Structure

- `apps/ios/` - Native iOS app (SwiftUI, AVAudioEngine, cloud inference)
- `apps/api/` - Rust API Worker at api.crescend.ai (Axum on Cloudflare Workers)
- `apps/web/` - TanStack Start web practice companion at crescend.ai (React, Tailwind CSS v4, chat + recording)
- `apps/inference/` - HF Inference Endpoint handlers + local dev servers
  - `handler.py` - MuQ quality scoring endpoint (6-dim scores)
  - `amt_handler.py` - Aria-AMT transcription endpoint (MIDI notes + pedal)
  - `muq_local_server.py` - Local MuQ dev server (port 8000)
  - `amt_local_server.py` - Local AMT dev server (port 8001)
- `model/` - PyTorch Lightning ML training pipeline
- `Justfile` - Dev commands (`just dev`, `just test-model`, etc.)

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming -> invoke office-hours
- Bugs, errors, "why is this broken", 500 errors -> invoke investigate
- Ship, deploy, push, create PR -> invoke ship
- QA, test the site, find bugs -> invoke qa
- Code review, check my diff -> invoke review
- Update docs after shipping -> invoke document-release
- Weekly retro -> invoke retro
- Design system, brand -> invoke design-consultation
- Visual audit, design polish -> invoke design-review
- Architecture review -> invoke plan-eng-review
  