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
- Storage: D1 (SQLite), R2 (audio uploads), KV (cache)
- Auth: Sign in with Apple (JWT via HMAC-SHA256)
- Sync: D1 student/session delta sync
- AI: Workers AI (goal extraction via Llama 3.3), HuggingFace Inference Endpoint (MuQ cloud fallback)
- Config: `wrangler.toml` defines all bindings

### API Endpoints (current)

- `POST /api/auth/apple` - Validate Apple identity token, issue JWT
- `POST /api/sync` - Receive student/session deltas from iOS, return exercise updates
- `POST /api/extract-goals` - Extract goals from student message (Workers AI)
- `GET /health` - Health check

### API Endpoints (legacy v1 -- to be removed)

- `POST /api/analyze/:id` - Performance analysis with RAG feedback
- `POST /api/chat` - Chat Q&A with RAG
- `POST /api/upload` - Audio file upload to R2
- `GET /api/performances` - List demo performances
- `GET /api/performances/:id` - Get single performance
- `GET /r2/:key` - Serve R2 audio files

### API Endpoints (planned -- not yet implemented)

- `POST /api/ask` - Send teaching moment context, receive LLM observation (two-stage pipeline via OpenRouter)
- `GET /api/exercises` - Fetch exercise catalog

### Key Directories

- `api/src/server.rs` - Entry point and route handling
- `api/src/auth/` - Apple Sign in auth, JWT generation/verification
- `api/src/api/` - Axum route handlers
- `api/src/services/` - Business logic (sync, goals, HF inference, legacy RAG services)
- `api/src/models/` - Data models (student, performance, analysis, pedagogy)

## Landing Page (`web/`)

TanStack Start landing page deployed to Cloudflare Workers at `crescend.ai`.

### Stack

- Framework: TanStack Start (React, SSR)
- Styling: Tailwind CSS v4
- Package manager: bun
- Deployment: Cloudflare Workers via `@cloudflare/vite-plugin`

## Feedback Tone (Both Platforms)

Warm and encouraging, specific to actual musical elements, actionable practice strategies. Celebrate strengths before suggesting improvements. Frame as observations, not absolute judgments. Feedback framing (correction/recognition/encouragement/question) adapts to learning arc position and session context. See `docs/apps/06a-subagent-architecture.md`.
