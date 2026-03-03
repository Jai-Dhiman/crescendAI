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
- `ios/CrescendAI/Features/` - Feature modules (Listening, Recording, Analysis, Chat)
- `ios/CrescendAI/Networking/` - API client and models

## API Worker (`api/`)

Rust API backend deployed to Cloudflare Workers at `api.crescend.ai`.

### Stack

- Runtime: Cloudflare Workers (Rust compiled to WASM)
- Routing: Axum (path matching in `#[event(fetch)]` handler)
- Storage: R2 (audio uploads), D1 (SQLite), KV (cache)
- AI: Workers AI (embeddings, reranking, LLM), HuggingFace Inference Endpoint (MuQ model)
- Search: Hybrid BM25 (D1 FTS5) + Vectorize + cross-encoder reranking
- Config: `wrangler.toml` defines all bindings

### API Endpoints

- `POST /api/analyze/:id` - Full performance analysis with RAG feedback
- `POST /api/chat` - Chat Q&A with RAG
- `POST /api/upload` - Audio file upload to R2
- `GET /api/performances` - List demo performances
- `GET /api/performances/:id` - Get single performance
- `GET /r2/:key` - Serve R2 audio files
- `GET /health` - Health check

### Key Directories

- `api/src/server.rs` - Entry point and route handling
- `api/src/api/` - Axum route handlers (performances)
- `api/src/services/` - Business logic (HF inference, RAG, R2, feedback)
- `api/src/models/` - Data models (performance, analysis, pedagogy)

## Landing Page (`web/`)

TanStack Start landing page deployed to Cloudflare Workers at `crescend.ai`.

### Stack

- Framework: TanStack Start (React, SSR)
- Styling: Tailwind CSS v4
- Package manager: bun
- Deployment: Cloudflare Workers via `@cloudflare/vite-plugin`

## Feedback Tone (Both Platforms)

Warm and encouraging, specific to actual musical elements, actionable practice strategies. Celebrate strengths before suggesting improvements. Frame as observations, not absolute judgments.
