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

Rust API backend deployed to Cloudflare Workers at `api.crescend.ai`.

### Stack

- Runtime: Cloudflare Workers (Rust compiled to WASM)
- Routing: Axum (path matching in `#[event(fetch)]` handler)
- Storage: D1 (SQLite)
- Auth: Sign in with Apple (JWT via HMAC-SHA256, HttpOnly cookies for web, Bearer header for iOS)
- Schema: Provider-agnostic (student_id UUID + auth_identities table)
- Sync: D1 student/session delta sync
- LLM: Groq (subagent), Anthropic (teacher) via HTTP
- Config: `wrangler.toml` defines all bindings
- Observability: Cloudflare Workers OTLP drain to Sentry, `console_error!` for error paths

### API Endpoints (current)

- `POST /api/auth/apple` - Validate Apple identity token, issue JWT (HttpOnly cookie + response body)
- `POST /api/auth/google` - Validate Google ID token via tokeninfo, issue JWT (HttpOnly cookie + response body)
- `GET /api/auth/me` - Validate JWT (cookie or Bearer), return user profile
- `POST /api/auth/signout` - Clear auth cookie
- `POST /api/sync` - Receive student/session deltas from iOS, return exercise updates
- `POST /api/extract-goals` - Extract goals from student message (Workers AI)
- `POST /api/ask` - Two-stage teacher pipeline: send teaching moment context, receive LLM observation (Groq subagent + Anthropic teacher)
- `POST /api/ask/elaborate` - "Tell me more" follow-up for a previous observation
- `POST /api/auth/debug` - Dev-only login bypassing Apple Sign In (returns 404 in production)
- `GET /health` - Health check

### API Endpoints (planned -- not yet implemented)

- `GET /api/exercises` - Fetch exercise catalog

### Key Directories

- `api/src/server.rs` - Entry point and route handling
- `api/src/auth/` - Apple Sign in auth, JWT generation/verification
- `api/src/services/` - Business logic (ask, chat, goals, llm, memory, sync)

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
