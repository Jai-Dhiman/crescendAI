# Backend/Frontend Architecture Split

**Date:** 2026-03-02
**Status:** Approved

## Decision

Split the current monolithic Rust/Leptos Cloudflare Worker into two deployments:

1. **api.crescend.ai** -- Rust API Worker (Cloudflare Workers)
2. **crescend.ai** -- TanStack Start landing page (Cloudflare Pages)

The iOS app calls `api.crescend.ai` exclusively. The landing page is a separate, independently deployed marketing site.

## Why

The existing `apps/web/` is a single Rust/Leptos isomorphic app that serves both a marketing landing page (Leptos SSR + WASM hydration) and API endpoints. This creates friction:

- Adding auth, sync, and LLM proxy endpoints to a Leptos SSR app is unnecessarily complex
- The landing page and API have different deployment cadences and concerns
- The Leptos frontend is overbuilt for a marketing page
- The iOS app needs a clean JSON API, not a full-stack web framework

## Architecture

```
crescend.ai (Cloudflare Pages)        api.crescend.ai (Cloudflare Workers)
+------------------------------+      +-----------------------------------+
| TanStack Start               |      | Rust API Worker                   |
| - Landing page (SSR)         |      |                                   |
| - Marketing content          |      | Existing:                         |
| - Tailwind CSS               |      |   POST /api/analyze/:id           |
| - Static + interactive       |      |   POST /api/upload                |
|                              |      |   POST /api/chat                  |
| No D1/KV/R2 access.         |      |   GET  /r2/:key                   |
| Calls api.crescend.ai       |      |   GET  /api/performances          |
| for any dynamic data.        |      |                                   |
+------------------------------+      | New (Slice 5+):                   |
                                      |   POST /api/auth/apple            |
+------------------------------+      |   POST /api/sync                  |
| iOS App                      |      |   POST /api/ask                   |
| calls api.crescend.ai       |      |   GET  /api/exercises             |
+------------------------------+      |                                   |
                                      | Bindings: D1, KV, R2,            |
                                      |   Vectorize, AI                   |
                                      | CORS: crescend.ai allowed         |
                                      +-----------------------------------+
```

## Project Structure

```
apps/
  api/       Rust API Worker (evolved from current apps/web/)
             Leptos frontend stripped, API-only
             wrangler.toml -> api.crescend.ai
  web/       TanStack Start (fresh project)
             Cloudflare Pages -> crescend.ai
  ios/       iOS app (unchanged)
             APIEndpoints -> api.crescend.ai
```

## Rust API Worker (apps/api/)

### What stays from the existing codebase

- `src/server.rs` -- API route handlers (stripped of Leptos router)
- `src/services/` -- HF inference, R2, RAG, feedback, vectorize
- `src/models/` -- Performance, PerformanceDimensions, AnalysisResult, pedagogy models
- `src/api/` -- performance list/detail handlers
- `src/state.rs` -- AppState (simplified, no LeptosOptions)
- `migrations/` -- D1 schema
- `wrangler.toml` -- bindings (D1, KV, R2, Vectorize, AI)

### What gets removed

- `src/lib.rs` (hydrate feature gate)
- `src/app.rs` (Leptos routing)
- `src/shell.rs` (HTML shell)
- `src/client.rs` (WASM hydration)
- `src/pages/` (all Leptos page components)
- `src/components/` (all UI components)
- `dist/`, `public/` (static assets)
- Cargo.toml: leptos, leptos_router, leptos_meta, leptos_axum, wasm-bindgen, gloo-*, web-sys, js-sys
- `build.sh` simplified: no WASM client build, no Tailwind, just `worker-build --release`

### What gets added

- `POST /api/auth/apple` -- validate Apple ID token, issue JWT
- `POST /api/sync` -- receive student model + session data, upsert to D1
- `POST /api/ask` -- build teacher prompt, call OpenRouter, return observation
- `GET /api/exercises` -- serve exercises from D1
- JWT middleware for protected endpoints
- CORS middleware (allow `https://crescend.ai`)
- New D1 tables: `students`, `sessions`, `student_check_ins`

### wrangler.toml changes

- Remove `[assets]` section
- Change custom domain from `crescend.ai` to `api.crescend.ai`
- Add OpenRouter API key to secrets
- Add Apple Sign In configuration to secrets
- Simplify build command (no WASM client build)

## TanStack Start Landing Page (apps/web/)

### Stack

- TanStack Start (React, SSR, file-based routing)
- Tailwind CSS
- Cloudflare Pages deployment
- bun as package manager

### Design system

- Warm cream/dark gray palette (background `#FDF8F0`, foreground `#2D2926`)
- Lora serif font
- Premium, minimal aesthetic

### Pages

- `/` -- Hero, value prop, demo/preview, call-to-action
- Future: `/about`, `/privacy`, `/terms`

### No backend logic

The landing page does not access D1, KV, or R2 directly. Any dynamic data comes from `api.crescend.ai`. For V1, the landing page is static content.

## CORS

The Rust API Worker serves these headers:

- `Access-Control-Allow-Origin: https://crescend.ai`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization`
- `OPTIONS` preflight handler on all routes

The iOS app does not need CORS (native HTTP client).

## iOS Integration

`APIEndpoints.swift` base URL changes from `https://crescend.ai` to `https://api.crescend.ai`.

## Migration Order

1. Move existing Rust code to `apps/api/`, strip Leptos, deploy to `api.crescend.ai`
2. Verify existing API endpoints work at new domain
3. Create TanStack Start project in `apps/web/`, deploy to Cloudflare Pages at `crescend.ai`
4. Update iOS app to point to `api.crescend.ai`
5. Proceed with Slice 5 (auth, student model, sync) on the Rust API Worker

## D1 Database

Only the Rust API Worker binds D1 (by database ID in wrangler.toml). The landing page on Cloudflare Pages has no direct database access. Shared state is exposed through API endpoints only.
