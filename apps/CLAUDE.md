# Apps

## iOS App (`ios/`)

Native iOS practice companion built with SwiftUI. Cloud inference via HF endpoint.

See `docs/architecture.md` for the full system design.
See `ios/CLAUDE.md` for iOS-specific conventions.

### Stack

- UI: SwiftUI (iOS 17+)
- Audio: AVAudioEngine (24kHz mono, ring buffer, background mode)
- Inference: Cloud via HF endpoint (A1-Max 4-fold ensemble, 6-dim output)
- Persistence: SwiftData (local-first)
- Auth: Sign in with Apple
- Networking: URLSession (sync to Workers, LLM calls)
- Observability: Sentry (`sentry-cocoa` SPM) -- crash handler, error capture, breadcrumbs

### Key Directories

- `ios/CrescendAI/App/` - Application entry point
- `ios/CrescendAI/DesignSystem/` - Tokens and reusable components
- `ios/CrescendAI/Features/` - Feature modules (Practice, Auth)
- `ios/CrescendAI/Networking/` - API client and models

## API Worker (`api/`)

TypeScript API backend deployed to Cloudflare Workers at `api.crescend.ai`.

**TypeScript Style Guide:** See `api/TS_STYLE.md` for all coding standards, patterns, and conventions.
Always follow that guide when editing files under `api/src/`.

### Stack

- Runtime: Cloudflare Workers (Hono + TypeScript)
- Routing: Hono (chained `.route()` for RPC type inference)
- Database: PlanetScale Postgres via Cloudflare Hyperdrive, Drizzle ORM
- Auth: better-auth (Apple + Google social providers, cookie sessions)
- WASM: Rust modules for compute-heavy algorithms (DTW, N-gram, STOP classifier)
- Real-time: Durable Object `SessionBrain` with WebSocket Hibernation API
- LLM: Groq (subagent) + Anthropic (teacher) via CF AI Gateway
- Config: `wrangler.toml` defines all bindings
- Observability: `@sentry/cloudflare` SDK, structured JSON logging

### API Endpoints

- `GET /health` - Health check
- `POST /api/auth/*` - better-auth handles all auth routes (Apple, Google, session, signout)
- `GET /api/scores` - List piece catalog
- `GET /api/scores/:pieceId` - Get piece metadata
- `GET /api/scores/:pieceId/data` - Get score MIDI data from R2
- `GET /api/exercises` - List exercises (auth required)
- `POST /api/exercises/assign` - Assign exercise to student
- `POST /api/exercises/complete` - Mark exercise complete
- `GET /api/conversations` - List conversations
- `GET /api/conversations/:id` - Get conversation with messages
- `DELETE /api/conversations/:id` - Delete conversation
- `POST /api/chat` - SSE streaming chat with Anthropic
- `POST /api/sync` - iOS delta sync (student baselines + sessions)
- `POST /api/waitlist` - Email waitlist signup
- `POST /api/extract-goals` - Extract goals from message (Groq)
- `POST /api/practice/start` - Start practice session
- `POST /api/practice/chunk` - Upload audio chunk to R2
- `GET /api/practice/ws/:sessionId` - WebSocket upgrade to SessionBrain DO
- `GET /api/practice/needs-synthesis` - Check deferred synthesis
- `POST /api/practice/synthesize` - Run deferred synthesis

### Key Directories

- `api/src/index.ts` - App composition + route mounting
- `api/src/routes/` - Hono route modules
- `api/src/services/` - Business logic (chat, ask, synthesis, llm, memory, inference, goals)
- `api/src/do/` - Durable Object (SessionBrain)
- `api/src/db/schema/` - Drizzle table definitions
- `api/src/middleware/` - Auth session, DB, error handler, logger, sentry
- `api/src/wasm/` - Rust WASM crates (score-analysis, piece-identify)

## Web App (`web/`)

Web practice companion deployed to Cloudflare Workers at `crescend.ai`. Chat-first teacher interface with live recording, real-time observations, and session summaries.

### Stack

- Framework: TanStack Start (React 19, SSR)
- Styling: Tailwind CSS v4
- Audio: MediaRecorder (15s Opus/WebM chunks), Web Audio API (waveform visualization)
- Real-time: WebSocket for practice session observations
- State: TanStack React Query + Zustand stores
- Auth: Sign in with Apple (Apple JS SDK popup flow, HttpOnly cookie JWT)
- Package manager: bun
- Deployment: Cloudflare Workers via `@cloudflare/vite-plugin`
- Observability: `@sentry/react` client SDK, OTLP drain for SSR, sourcemaps via `@sentry/vite-plugin`

### Key Files

- `web/src/lib/api.ts` - API client (credentials: include, typed auth methods)
- `web/src/lib/auth.tsx` - Auth context (AuthProvider, useAuth hook)
- `web/src/hooks/usePracticeSession.ts` - Recording state machine, WebSocket, chunk upload
- `web/src/components/AppChat.tsx` - Chat interface with streaming LLM responses
- `web/src/components/ChatInput.tsx` - Text input + record button
- `web/src/components/RecordingBar.tsx` - Recording overlay (waveform, timer, observation toasts)
- `web/src/components/WaveformVisualizer.tsx` - Real-time audio waveform
- `web/src/stores/` - Zustand stores (sidebar, toasts)
- `web/src/routes/signin.tsx` - Sign in with Apple page
- `web/src/routes/app.tsx` - Protected app shell with sidebar and conversation history
- `web/src/types/apple.d.ts` - Apple JS SDK type declarations

## Feedback Tone (Both Platforms)

Warm and encouraging, specific to actual musical elements, actionable practice strategies. Celebrate strengths before suggesting improvements. Frame as observations, not absolute judgments. Feedback framing (correction/recognition/encouragement/question) adapts to learning arc position and session context. See `docs/apps/02-pipeline.md`.
