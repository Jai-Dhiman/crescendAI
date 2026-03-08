# Apps

## iOS App (`ios/`)

The primary product. Native iOS practice companion built with SwiftUI.

See `docs/architecture.md` for the full system design.
See `ios/CLAUDE.md` for iOS-specific conventions.

### Stack

- UI: SwiftUI (iOS 17+)
- Audio: AVAudioEngine (24kHz mono, ring buffer, background mode)
- Inference: Core ML (finetuned MuQ, 6-dimension output)
- Persistence: SwiftData (local-first)
- Auth: Sign in with Apple
- Networking: URLSession (sync to Workers, LLM calls)

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

### API Endpoints (current)

- `POST /api/auth/apple` - Validate Apple identity token, issue JWT (HttpOnly cookie + response body)
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

## Landing Page (`web/`)

TanStack Start web app deployed to Cloudflare Workers at `crescend.ai`.

### Stack

- Framework: TanStack Start (React, SSR)
- Styling: Tailwind CSS v4
- Auth: Sign in with Apple (Apple JS SDK popup flow, HttpOnly cookie JWT)
- Package manager: bun
- Deployment: Cloudflare Workers via `@cloudflare/vite-plugin`

### Key Files

- `web/src/lib/api.ts` - API client (credentials: include, typed auth methods)
- `web/src/lib/auth.tsx` - Auth context (AuthProvider, useAuth hook)
- `web/src/routes/signin.tsx` - Sign in with Apple page
- `web/src/routes/app.tsx` - Protected app shell
- `web/src/types/apple.d.ts` - Apple JS SDK type declarations

## Feedback Tone (Both Platforms)

Warm and encouraging, specific to actual musical elements, actionable practice strategies. Celebrate strengths before suggesting improvements. Frame as observations, not absolute judgments. Feedback framing (correction/recognition/encouragement/question) adapts to learning arc position and session context. See `docs/apps/06a-subagent-architecture.md`.
