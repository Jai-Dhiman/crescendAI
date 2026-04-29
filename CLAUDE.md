# CrescendAI

**"A teacher for every pianist."**

Multi-platform (iOS + web) practice companion for pianists.

## What CrescendAI Does

- Evaluates *how* a piano performance sounds (tone, dynamics, phrasing, pedaling) -- not just note accuracy like MIDI-based apps
- Uses dual encoders -- MuQ (audio, finetuned) and Aria (symbolic, 650M-param pretrained from EleutherAI) -- with gated fusion, outputting 6 teacher-grounded dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation
- Target users: self-learners (B2C), music educators (B2B), institutions (B2B)
- Competitors (Simply Piano, Flowkey, Piano Marvel) check note accuracy via MIDI; CrescendAI evaluates musical expression from audio

## Architecture

See `docs/architecture.md` for the full system design. Key points:
**Apps docs:** `docs/apps/00-status.md` through `docs/apps/05-ui-system.md` (each has status header)

## Model Strategy

- **Model v2:** MuQ (audio, pretrained on 160K hrs) + Aria (symbolic, pretrained on 820K MIDIs) with gated fusion
- **Score conditioning:** delta = z_perf - z_score when score MIDI available
- **Training:** PercePiano anchor (20%) + ordinal-dominated (80%) with T2 competition + T5 YouTube Skill data

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
| `just migrate-generate` / `just migrate-prod` | Drizzle migrations (generate SQL / apply to prod) |

## Package Managers

- Python: `uv` (not pip)
- JavaScript: `bun` (not npm)

## Coding Standards

- ALWAYS use model: "Sonnet 4.6" when creating and using subagents for search, reivew, or subagent driven development
- Explicit exception handling over silent fallbacks
- No backup files when making fixes
- No emojis unless explicitly requested
- **TypeScript (API):** Follow `apps/api/TS_STYLE.md` for all code in `apps/api/`. Key rules: never destructure `c.env`, ServiceContext for DI, domain errors in services (no HTTPException), chain `.route()` for Hono RPC types, Zod validation with JSON error hooks, state versioning in DOs across awaits, `console.log(JSON.stringify({...}))` for logging.

## Observability

- **Error tracking:** Sentry across all surfaces (iOS, web, API)
  - iOS: `sentry-cocoa` SPM -- crash reporting, error capture, breadcrumbs
  - Web: `@sentry/react` -- client-side errors, API errors, WebSocket errors
  - API: `@sentry/cloudflare` SDK -- traces, breadcrumbs, error capture
- Sentry org: `crescendai` with projects: `crescendai-api`, `crescendai-web`, `crescendai-ios`
- Error logging: `console_error!` (Rust), `Sentry.captureException` (web), `SentrySDK.capture(error:)` (iOS)


## Swift in `apps/ios/`

codedb (see global CLAUDE.md) does not index Swift. Use Grep/Read directly for files under `apps/ios/`.
