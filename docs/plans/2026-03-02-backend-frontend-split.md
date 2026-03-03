# Backend/Frontend Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the monolithic Rust/Leptos Worker into a Rust API Worker at api.crescend.ai and a TanStack Start landing page at crescend.ai.

**Architecture:** Move existing Rust code to `apps/api/`, strip all Leptos frontend code, add CORS, update wrangler.toml for `api.crescend.ai`. Create a fresh TanStack Start project in `apps/web/` deployed to `crescend.ai` via Cloudflare Workers (TanStack Start uses `@cloudflare/vite-plugin` which deploys as a Worker with native static asset serving -- equivalent to Pages with CDN caching).

**Tech Stack:** Rust (worker crate + axum) for API, TanStack Start (React, SSR) + Tailwind CSS v4 + bun for landing page, Cloudflare Workers for both deployments.

**Design doc:** `docs/plans/2026-03-02-backend-frontend-split-design.md`

---

## Phase 1: Rust API Worker

### Task 1: Copy apps/web/ to apps/api/

Copy the existing Rust codebase to its new home. The original stays until we verify everything works.

**Step 1: Copy the directory**

```bash
cp -r apps/web apps/api
```

**Step 2: Remove build artifacts from the copy**

```bash
rm -rf apps/api/dist apps/api/build apps/api/target apps/api/pkg apps/api/node_modules apps/api/.wrangler
```

**Step 3: Commit**

```bash
git add apps/api/
git commit -m "copy apps/web/ to apps/api/ as starting point for API Worker"
```

---

### Task 2: Strip Leptos frontend code from apps/api/

Remove all Leptos SSR, WASM hydration, page components, and UI code. Keep only API route handlers and services.

**Files to delete:**

```
apps/api/src/lib.rs          # hydrate/ssr feature gate
apps/api/src/app.rs           # Leptos routing
apps/api/src/shell.rs         # HTML shell template
apps/api/src/client.rs        # WASM hydration entry
apps/api/src/pages/           # all Leptos page components (entire directory)
apps/api/src/components/      # all UI components (entire directory)
apps/api/public/              # static assets (entire directory)
apps/api/dist/                # build output (if still present)
apps/api/tailwind.config.js   # Tailwind config
apps/api/tailwind.css         # Tailwind entry
apps/api/package.json         # JS deps (Tailwind only)
apps/api/bun.lock             # JS lockfile
apps/api/build.sh             # complex build script (replaced by wrangler.toml build command)
```

**Step 1: Delete the frontend files**

```bash
cd apps/api
rm -f src/lib.rs src/app.rs src/shell.rs src/client.rs
rm -rf src/pages src/components
rm -rf public dist
rm -f tailwind.config.js tailwind.css package.json bun.lock build.sh
```

**Step 2: Verify remaining src/ structure**

Expected:
```
apps/api/src/
  server.rs         # API route handlers
  state.rs          # AppState
  api/
    mod.rs          # re-exports
    handlers.rs     # performance list/detail
  services/
    mod.rs
    huggingface.rs
    r2.rs
    rag.rs
    feedback.rs
    vectorize.rs
    embedding.rs
    reranker.rs
    vectorize_binding.rs
  models/
    mod.rs
    performance.rs
    analysis.rs
    pedagogy.rs
```

**Step 3: Commit**

```bash
git add -A apps/api/
git commit -m "strip Leptos frontend code from API Worker"
```

---

### Task 3: Rewrite server.rs as API-only entry point

Replace the Leptos+API hybrid server with a pure API router. This file becomes the sole entry point.

**File:** `apps/api/src/server.rs`

**Step 1: Read the existing server.rs to understand all API routes**

The current `server.rs` has these direct API handlers (outside of Leptos):
- `POST /api/analyze/:id` -> `handle_full_analyze()`
- `POST /api/chat` -> `handle_chat()`
- `POST /api/upload` -> `handle_upload()`
- `GET /r2/:key` -> `handle_r2_serve()`
- `GET /api/performances` -> `api::list_performances`
- `GET /api/performances/:id` -> `api::get_performance`
- `GET /health` -> returns "OK"

All Leptos SSR routes (/, /analyze, /analyze/:id, /demo/:id) are being deleted.

**Step 2: Rewrite server.rs**

Replace the entire file. The new version:
- Uses `#[event(fetch)]` directly (no Leptos router)
- Keeps all existing API handlers (analyze, chat, upload, R2 serve, performances)
- Adds CORS headers to all responses
- Handles OPTIONS preflight requests
- Removes all Leptos imports and router setup

Key changes:
- Remove `use leptos::prelude::*` and all leptos_axum imports
- Remove `fn router()` that builds the Leptos router
- Remove `generate_route_list(App)` and `.leptos_routes_with_context()`
- The `#[event(fetch)]` handler now routes directly to API handlers
- Add a `with_cors()` wrapper function for all responses
- Add OPTIONS handler returning 204 with CORS headers

The existing handler functions (`handle_full_analyze`, `handle_chat`, `handle_upload`, `handle_r2_serve`, `find_uploaded_performance`, `generate_model_variants`, `parse_multipart`) remain unchanged -- they are pure API logic.

Remove the `use crate::{... shell::shell, ... App}` import line. Replace with just the modules that still exist.

Remove the `router()` function entirely (it built the Leptos router).

In `#[event(fetch)]`, after the existing path-based routing, instead of falling through to `app.oneshot(req)` (which was the Leptos router), return 404 for unmatched routes.

Add CORS wrapper. This adds `Access-Control-Allow-Origin: *` to every response (will be tightened to `https://crescend.ai` later). Handle OPTIONS preflight at the top of the fetch handler, before any route matching.

**Step 3: Update state.rs**

Remove `LeptosOptions` from `AppState`. The state becomes just the `Env`:

```rust
use worker::Env;

#[derive(Clone)]
pub struct AppState {
    pub env: Env,
}

impl AppState {
    pub fn new(env: Env) -> Self {
        Self { env }
    }
}
```

Remove the helper methods that accessed Leptos-specific bindings. Keep any D1/KV/R2 helpers.

**Step 4: Update src/api/mod.rs and handlers.rs**

Check if these reference any Leptos types. Remove any Leptos imports. The handlers return axum `Json<T>` responses which are independent of Leptos.

**Step 5: Check src/services/ and src/models/ for Leptos imports**

Grep for `leptos` in all remaining files:

```bash
grep -r "leptos" apps/api/src/
```

Remove any Leptos imports found. The services and models should be pure Rust with serde -- they should not depend on Leptos.

**Step 6: Verify compilation direction**

At this point the code will NOT compile yet (Cargo.toml still has Leptos deps and feature flags). That is expected -- Task 4 fixes Cargo.toml.

**Step 7: Commit**

```bash
git add apps/api/src/
git commit -m "rewrite server.rs as API-only entry point, remove Leptos routing"
```

---

### Task 4: Simplify Cargo.toml and build configuration

Strip Leptos and frontend dependencies. Keep only what the API Worker needs.

**File:** `apps/api/Cargo.toml`

**Step 1: Read the existing Cargo.toml**

Understand all current dependencies and feature flags.

**Step 2: Rewrite Cargo.toml**

Keep:
- `worker` with features `["http", "axum", "d1"]`
- `worker-macros` with features `["http"]`
- `axum` (update to match worker crate expectations, use `default-features = false`)
- `tower-service`
- `serde`, `serde_json`
- `console_error_panic_hook`
- `getrandom` with `["js"]` feature
- `http` (for StatusCode, etc.)
- `http-body-util` (used by existing handlers for body collection)
- `regex` (used by RAG service)

Remove:
- `leptos`, `leptos_router`, `leptos_meta`, `leptos_axum`
- `wasm-bindgen`, `wasm-bindgen-futures`
- `gloo-net`, `gloo-timers`
- `web-sys`, `js-sys`
- `futures`
- All `[features]` sections (no more `hydrate`/`ssr` feature flags)

Change:
- `crate-type` from `["cdylib", "rlib"]` to `["cdylib"]`
- Remove all `#[cfg(feature = "ssr")]` guards from service files (everything is now server-side)

**Step 3: Remove cfg(feature) guards from services**

Check each file in `apps/api/src/services/` for `#[cfg(feature = "ssr")]` guards. These were needed when code was shared between client and server. Now everything is server-only, so remove the guards and keep the code.

```bash
grep -rn 'cfg(feature' apps/api/src/
```

Remove all `#[cfg(feature = "ssr")]` and `#[cfg(feature = "hydrate")]` attributes.

**Step 4: Update wrangler.toml**

```toml
name = "crescendai-api"
main = "build/worker/shim.mjs"
compatibility_date = "2025-01-01"

[build]
command = "cargo install -q worker-build && worker-build --release"

# No [assets] section -- API-only, no static files

[[kv_namespaces]]
binding = "KV"
id = "a20cea1f48d9415da9d7ca6418563025"

[[r2_buckets]]
binding = "BUCKET"
bucket_name = "crescendai-bucket"

[[d1_databases]]
binding = "DB"
database_name = "crescendai-db"
database_id = "659755a8-4e9e-4581-a2bd-d34b6f912c3a"

[[vectorize]]
binding = "VECTORIZE"
index_name = "crescendai-piano-pedagogy"

[ai]
binding = "AI"

[[routes]]
pattern = "api.crescend.ai"
custom_domain = true

[vars]
ENVIRONMENT = "development"
HF_INFERENCE_ENDPOINT = "https://u1oxy7egvq7rnm8z.us-east-1.aws.endpoints.huggingface.cloud"
PUBLIC_URL = "https://api.crescend.ai"
```

Changes from original:
- `name` changed to `"crescendai-api"`
- Removed `[assets]` section
- Route pattern changed from `crescend.ai` to `api.crescend.ai`
- `PUBLIC_URL` changed to `https://api.crescend.ai`
- Build command simplified (no build.sh)

**Step 5: Try to build**

```bash
cd apps/api
cargo install -q worker-build && worker-build --release
```

Fix any compilation errors. Common issues:
- Missing imports after removing Leptos
- `#[cfg]` guards that referenced removed features
- Type mismatches from axum version changes

**Step 6: Commit**

```bash
git add apps/api/
git commit -m "simplify Cargo.toml: remove Leptos deps, API-only build"
```

---

### Task 5: Verify API endpoints work locally

**Step 1: Start the dev server**

```bash
cd apps/api
npx wrangler dev
```

(wrangler is a global tool, doesn't need the JS package.json)

**Step 2: Test health endpoint**

```bash
curl http://localhost:8787/health
```

Expected: `OK`

**Step 3: Test CORS preflight**

```bash
curl -X OPTIONS http://localhost:8787/api/performances \
  -H "Origin: https://crescend.ai" \
  -H "Access-Control-Request-Method: GET" \
  -v
```

Expected: 204 response with `Access-Control-Allow-Origin` header.

**Step 4: Test performances list**

```bash
curl http://localhost:8787/api/performances
```

Expected: JSON array of demo performances.

**Step 5: Test 404 for unknown routes**

```bash
curl http://localhost:8787/nonexistent -v
```

Expected: 404 Not Found.

**Step 6: Commit if any fixes were needed**

```bash
git add apps/api/
git commit -m "fix: resolve compilation issues in API Worker"
```

---

### Task 6: Update apps/ CLAUDE.md

**File:** `apps/CLAUDE.md`

Update the documentation to reflect the new structure:
- `apps/api/` is the Rust API Worker at api.crescend.ai
- `apps/web/` will be the TanStack Start landing page at crescend.ai (not yet created)
- `apps/ios/` is unchanged

**Step 1: Update apps/CLAUDE.md**

Replace the "Web App" section with the new API Worker description. Add a placeholder for the new landing page.

**Step 2: Commit**

```bash
git add apps/CLAUDE.md
git commit -m "update apps/CLAUDE.md for backend/frontend split"
```

---

## Phase 2: TanStack Start Landing Page

### Task 7: Scaffold TanStack Start project

Create the new `apps/web/` with TanStack Start, Tailwind CSS v4, and Cloudflare Workers deployment.

**Note:** TanStack Start deploys to Cloudflare Workers (not Pages) via `@cloudflare/vite-plugin`. Workers now serve static assets natively with CDN caching, so the result is equivalent to Pages.

**Step 1: Scaffold with create-cloudflare**

```bash
cd apps
bunx create-cloudflare@latest web --framework=tanstack-start
```

Follow the prompts. This creates a pre-configured TanStack Start project with Cloudflare integration.

**Step 2: Add Tailwind CSS v4**

```bash
cd apps/web
bun add -d tailwindcss @tailwindcss/vite
```

**Step 3: Configure Tailwind in vite.config.ts**

Add the `@tailwindcss/vite` plugin. The plugin order must be:
1. `tailwindcss()`
2. `tsConfigPaths()`
3. `cloudflare()`
4. `tanstackStart()`
5. `viteReact()` (last)

**Step 4: Create src/styles/app.css**

```css
@import 'tailwindcss' source('../');

@theme {
  --color-cream: #FDF8F0;
  --color-ink: #2D2926;
  --color-ink-60: rgba(45, 41, 38, 0.6);
  --color-ink-12: rgba(45, 41, 38, 0.12);
  --color-ink-5: rgba(45, 41, 38, 0.05);

  --font-display: 'Lora', serif;
}
```

These match the existing iOS design system tokens from `apps/ios/CrescendAI/DesignSystem/Tokens/Colors.swift`.

**Step 5: Import the stylesheet in __root.tsx**

```tsx
import appCss from '~/styles/app.css?url'

// In head():
links: [{ rel: 'stylesheet', href: appCss }]
```

**Step 6: Update wrangler.jsonc**

Set the name and custom domain:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "crescendai-web",
  "compatibility_date": "2026-03-02",
  "compatibility_flags": ["nodejs_compat"],
  "main": "@tanstack/react-start/server-entry",
  "routes": [
    { "pattern": "crescend.ai", "custom_domain": true }
  ]
}
```

**Step 7: Verify dev server runs**

```bash
cd apps/web
bun run dev
```

Open http://localhost:3000 in browser. Should see the default TanStack Start page with Tailwind CSS working.

**Step 8: Commit**

```bash
git add apps/web/
git commit -m "scaffold TanStack Start landing page with Tailwind CSS v4"
```

---

### Task 8: Build minimal landing page

Replace the default scaffolded content with a minimal CrescendAI landing page. This is a placeholder -- real marketing content comes later. The goal is to prove the deployment works.

**Step 1: Update __root.tsx**

- Set page title to "CrescendAI"
- Apply the cream background and Lora font via Tailwind classes
- Add Google Fonts link for Lora (or include the font files from the iOS app)

**Step 2: Update index.tsx**

Create a simple hero section:
- "A teacher for every pianist." headline
- Brief description (2-3 sentences from the product spec)
- "Coming soon to iOS" call-to-action
- Dark ink text on cream background
- Lora serif font for headings

Keep it minimal. No images, no animations, no complex layout. Just text that proves the design system carries over from iOS.

**Step 3: Verify locally**

```bash
bun run dev
```

Confirm the page renders with cream background, Lora font, and the correct color palette.

**Step 4: Commit**

```bash
git add apps/web/
git commit -m "add minimal CrescendAI landing page with design system tokens"
```

---

### Task 9: Add Lora font files

The iOS app bundles Lora-Regular, Lora-Medium, Lora-SemiBold, Lora-Bold as .ttf files. For the web, we can either use Google Fonts or self-host.

**Step 1: Check if font files exist in the iOS app**

```bash
ls apps/ios/CrescendAI/Resources/Fonts/
```

**Step 2: Copy font files to apps/web/public/fonts/**

```bash
mkdir -p apps/web/public/fonts
cp apps/ios/CrescendAI/Resources/Fonts/Lora-*.ttf apps/web/public/fonts/
```

**Step 3: Add @font-face declarations in app.css**

```css
@font-face {
  font-family: 'Lora';
  src: url('/fonts/Lora-Regular.ttf') format('truetype');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
/* ... Medium (500), SemiBold (600), Bold (700) */
```

**Step 4: Verify fonts load in dev**

```bash
bun run dev
```

Check browser dev tools -> Network tab to confirm font files load.

**Step 5: Commit**

```bash
git add apps/web/
git commit -m "add Lora font files to landing page"
```

---

## Phase 3: iOS Integration

### Task 10: Update iOS APIEndpoints to use api.crescend.ai

**File:** `apps/ios/CrescendAI/Networking/APIEndpoints.swift`

**Step 1: Change base URL**

```swift
// Before:
static let baseURL = URL(string: "https://crescend.ai")!

// After:
static let baseURL = URL(string: "https://api.crescend.ai")!
```

**Step 2: Commit**

```bash
git add apps/ios/CrescendAI/Networking/APIEndpoints.swift
git commit -m "update iOS API base URL to api.crescend.ai"
```

---

## Phase 4: Cleanup

### Task 11: Remove the old apps/web/ Rust/Leptos code

Now that both the API Worker and the landing page are set up, remove the original Rust/Leptos app.

**Step 1: Verify apps/api/ and apps/web/ are committed and working**

```bash
git status
git log --oneline -10
```

**Step 2: The old apps/web/ is already replaced by the TanStack Start project**

Since we scaffolded TanStack Start directly into `apps/web/` (Task 7), the old Rust code is already gone. If not, remove any remaining Rust artifacts:

```bash
# Only if old files remain:
rm -rf apps/web/src apps/web/Cargo.toml apps/web/Cargo.lock apps/web/target
```

**Step 3: Update root-level CLAUDE.md if needed**

Ensure the project description reflects the new structure.

**Step 4: Final commit**

```bash
git add -A
git commit -m "complete backend/frontend split: Rust API + TanStack Start landing page"
```

---

## Deployment Notes (Manual Steps)

These require Cloudflare dashboard or wrangler CLI with production credentials:

1. **Set up api.crescend.ai DNS**: Add CNAME record pointing to the Workers custom domain
2. **Deploy API Worker**: `cd apps/api && wrangler deploy`
3. **Deploy landing page**: `cd apps/web && bun run deploy`
4. **Move D1 database bindings**: Both Workers can reference the same D1 database by ID
5. **Set secrets on API Worker**: `wrangler secret put JWT_SECRET`, `wrangler secret put OPENROUTER_API_KEY`, etc.
6. **Remove old crescend.ai Worker route**: After verifying both new deployments work

---

## What Comes Next

After this split is complete, Slice 5 (Student Model + Auth) can be implemented on the Rust API Worker at `apps/api/`. The new endpoints (`/api/auth/apple`, `/api/sync`) get added to the clean API-only codebase.
