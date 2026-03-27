# API Refactor: Rust Style Guide Alignment + Axum Router Migration

**Date:** 2026-03-27
**Status:** Approved
**Scope:** `apps/api/` -- full refactor of 36 Rust files (~16,300 LOC)

## Summary

Big-bang refactor of the CrescendAI API to align with `apps/api/RUST_STYLE.md` and migrate from manual path dispatch to proper Axum Router with extractors, service layer, typed errors, and standardized serde conventions. Pre-beta, pre-user -- zero deployment risk.

## Motivation

Current codebase debt:
- 45 `Result<T, String>` return signatures across 11 files
- 138 `Response::builder()` chains across 14 files (heavy boilerplate)
- 176 `.unwrap()` calls across 19 files (panic risk in WASM)
- 520-line `server.rs` with manual `if path == ...` dispatch
- No clippy config, no rustfmt config, no typed error enums
- Raw `String` for student_id, session_id everywhere (no compile-time distinction)
- Inconsistent serde: no `rename_all`, no `skip_serializing_if`, mixed JSON casing
- `&Env` passed to every function (no service abstraction)

## Architecture Decisions

### 1. Axum Router (not manual dispatch)

**Decision:** Migrate to `axum::Router` with `tower_service::Service::call()` bridge.

**Rationale:**
- `worker` 0.7+ has first-class Axum support (`HttpRequest` IS `http::Request<worker::Body>`)
- `#[worker::send]` macro solves the Send bound problem for handlers touching D1/R2/KV
- Already paying binary size cost for Axum (in Cargo.toml) without using Router
- Eliminates 520 lines of manual dispatch, gains type-safe routing + extractors
- `CorsLayer` from tower-http compiles to WASM -- replaces manual CORS

**Carve-outs (bypass Router):**
- WebSocket upgrade (`/api/practice/ws/*`) -- needs `WebSocketPair::new()`, incompatible with Axum ws
- Streaming chat (`/api/chat` POST) -- needs `worker::Response` for true streaming

Carve-out functions keep raw `&Env` (not service layer) since they run before `AppState` construction and return `worker::Response` directly. Auth for these paths is validated manually as today.

### 2. Service Layer (not raw Env passing)

**Decision:** `AppState` with domain service objects, not raw `&Env`.

```
AppState
  auth: AuthService       -- JWT sign/verify, user lookup/creation, secrets
  db: DbService           -- D1 operations (memory, conversations, messages, sync)
  inference: InferenceService  -- MuQ + AMT endpoint calls
  practice: PracticeService    -- DO namespace access
```

Each service wraps `SendWrapper<Env>` and exposes typed methods. Handlers never see raw `Env`.

### 3. thiserror Error Enums (not Result<T, String>)

**Decision:** Two error types -- `ApiError` for HTTP handlers, `PracticeError` for DO internals.

### 4. AuthUser Custom Extractor (not manual header parsing)

**Decision:** `AuthUser` implements `FromRequestParts`. Reads JWT from cookie (web) or Bearer header (iOS). Handlers add `auth: AuthUser` to get authenticated. `Option<AuthUser>` for optional auth comes free from Axum.

### 5. Practice Module Restructure (nested by concern)

**Decision:** Nest `practice/` into `handlers/`, `session/`, `analysis/` submodules.

### 6. Keep Manual Dispatch in Durable Object

**Decision:** DO routing stays as match-on-path. DO is simple enough (2-3 internal routes) that Axum adds overhead without benefit.

---

## Error System

### `src/error.rs` -- API errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("unauthorized")]
    Unauthorized,

    #[error("forbidden")]
    Forbidden,

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("internal: {0}")]
    Internal(String),

    #[error("external service: {0}")]
    ExternalService(String),
}

pub type Result<T> = std::result::Result<T, ApiError>;
```

`IntoResponse` impl maps each variant to the correct HTTP status code + JSON body:
- `NotFound` -> 404
- `BadRequest` -> 400
- `Unauthorized` -> 401
- `Forbidden` -> 403
- `InferenceFailed` -> 502
- `Internal` -> 500
- `ExternalService` -> 502

Response body format: `{"error": "<variant_snake>", "message": "<display string>"}`

Server errors (5xx) are logged via `console_error!` in the `IntoResponse` impl.

`From` impls for common conversions:
- `serde_json::Error` -> `BadRequest`
- `worker::Error` -> `Internal`

### `src/practice/session/error.rs` -- DO errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum PracticeError {
    #[error("storage: {0}")]
    Storage(String),

    #[error("inference: {0}")]
    Inference(String),

    #[error("piece identification: {0}")]
    PieceId(String),

    #[error("synthesis: {0}")]
    Synthesis(String),

    #[error("websocket: {0}")]
    WebSocket(String),
}
```

No `IntoResponse` -- DO errors are logged and communicated over WebSocket as JSON messages.

---

## Axum Router + Entry Point

### `src/server.rs`

```rust
#[event(fetch)]
async fn fetch(req: HttpRequest, env: Env, _ctx: Context)
    -> Result<http::Response<axum::body::Body>>
{
    console_error_panic_hook::set_once();
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // Carve-outs: routes that can't go through Axum
    if path.starts_with("/api/practice/ws/") {
        return handle_ws_upgrade(req, &env).await;
    }
    if path == "/api/chat" && method == http::Method::POST {
        return handle_chat_stream(req, &env).await;
    }

    // Everything else through Axum Router
    let state = AppState::from_env(env);
    let app = router(state).layer(cors_layer(&allowed_origins));
    Ok(app.call(req).await?)
}
```

Allowed origins derived from `env.var("ALLOWED_ORIGIN")` as today.

### `src/routes.rs`

```rust
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        // Auth
        .route("/api/auth/apple", post(auth::handlers::handle_apple))
        .route("/api/auth/google", post(auth::handlers::handle_google))
        .route("/api/auth/me", get(auth::handlers::handle_me))
        .route("/api/auth/signout", post(auth::handlers::handle_signout))
        .route("/api/auth/debug", post(auth::handlers::handle_debug))
        // Practice
        .route("/api/practice/start", post(practice::handlers::handle_start))
        .route("/api/practice/upload", post(practice::handlers::handle_upload))
        // Services
        .route("/api/ask", post(services::ask::handle_ask))
        .route("/api/ask/elaborate", post(services::ask::handle_elaborate))
        .route("/api/conversations", get(services::chat::list_conversations))
        .route("/api/conversations/:id/messages", get(services::chat::get_messages))
        .route("/api/sync", post(services::sync::handle_sync))
        .route("/api/extract-goals", post(services::goals::handle_goals))
        .route("/api/exercises", get(services::exercises::handle_exercises))
        .route("/api/scores", get(services::scores::handle_scores))
        .route("/api/waitlist", post(services::waitlist::handle_waitlist))
        .with_state(state)
}
```

---

## Service Layer

### `src/state.rs`

```rust
#[derive(Clone)]
pub struct AppState {
    pub auth: AuthService,
    pub db: DbService,
    pub inference: InferenceService,
    pub practice: PracticeService,
}

impl AppState {
    pub fn from_env(env: Env) -> Self {
        let env = SendWrapper(env);
        Self {
            auth: AuthService::new(env.clone()),
            db: DbService::new(env.clone()),
            inference: InferenceService::new(env.clone()),
            practice: PracticeService::new(env),
        }
    }
}
```

### Service responsibilities

**AuthService** (`SendWrapper<Env>`):
- `jwt_secret()` -> `Result<String>` (from env.secret)
- `verify_jwt(token)` -> `Result<Claims>`
- `sign_jwt(student_id)` -> `Result<String>`
- `find_or_create_user(provider, provider_id, email, name)` -> `Result<(StudentId, bool)>`
- `get_user_profile(student_id)` -> `Result<UserProfile>`
- Apple/Google token validation helpers

**DbService** (`SendWrapper<Env>`):
- `d1()` -> `Result<D1Database>` (binding access)
- `query_memory(student_id)` -> `Result<Vec<SynthesizedFact>>`
- `store_message(conversation_id, role, content, ...)` -> `Result<()>`
- `list_conversations(student_id)` -> `Result<Vec<Conversation>>`
- `get_messages(conversation_id)` -> `Result<Vec<MessageRow>>`
- `sync_student_data(student_id, deltas)` -> `Result<SyncResponse>`
- All D1 read/write operations

**InferenceService** (`SendWrapper<Env>`):
- `call_llm(provider, messages, tools)` -> `Result<LlmResponse>`
- `generate_observation(student_id, context)` -> `Result<Observation>`
- `generate_elaboration(student_id, observation_id)` -> `Result<Observation>`
- Groq (subagent) + Anthropic (teacher) client logic

**PracticeService** (`SendWrapper<Env>`):
- `do_namespace()` -> `Result<DurableObjectNamespace>`
- `create_session(student_id, piece_id)` -> `Result<SessionId>`
- DO stub creation and request forwarding

---

## Newtypes

### `src/types.rs`

```rust
// All newtypes follow this pattern:
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StudentId(String);

impl StudentId {
    pub fn new() -> Self { Self(generate_uuid_v4()) }
    pub fn as_str(&self) -> &str { &self.0 }
}
impl fmt::Display for StudentId { /* delegates to inner */ }
impl From<String> for StudentId { /* wraps */ }
impl AsRef<str> for StudentId { /* delegates */ }
```

Types: `StudentId`, `SessionId`, `ConversationId`, `PieceId`.

`#[serde(transparent)]` ensures wire-format compatibility -- JSON clients see plain strings.

---

## Auth Extractor

### `src/auth/extractor.rs`

```rust
pub struct AuthUser {
    pub student_id: StudentId,
}

impl<S: Send + Sync> FromRequestParts<S> for AuthUser
where
    AppState: FromRef<S>,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self> {
        let app_state = AppState::from_ref(state);

        // Try cookie first (web), then Bearer header (iOS)
        let token = extract_token_from_cookie(parts)
            .or_else(|| extract_token_from_bearer(parts))
            .ok_or(ApiError::Unauthorized)?;

        let claims = app_state.auth.verify_jwt(&token)?;
        Ok(AuthUser {
            student_id: StudentId::from(claims.sub),
        })
    }
}
```

---

## Serde Conventions

### API-facing types (request/response)

- `#[serde(rename_all = "camelCase")]` on all request and response structs
- `#[serde(deny_unknown_fields)]` on request structs only
- `#[serde(default)]` on optional request fields
- `#[serde(skip_serializing_if = "Option::is_none")]` on optional response fields

### Internal types (D1 rows, DO state)

- NO `rename_all` -- field names match D1 column names (snake_case)
- `#[serde(default)]` on optional fields for backward compatibility

---

## Handler Pattern

### Before (current)

```rust
pub async fn handle_ask(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    // ~15 lines: parse auth token from headers
    // ~5 lines: verify JWT
    // ~5 lines: parse JSON body manually
    // ~10 lines: call service logic
    // ~10 lines: build Response::builder() chain
    // ~10 lines: error handling with more Response::builder() chains
}
```

### After (new)

```rust
#[worker::send]
async fn handle_ask(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(payload): Json<AskRequest>,
) -> Result<Json<AskResponse>> {
    let response = state.inference
        .generate_observation(&auth.student_id, &payload)
        .await?;
    Ok(Json(response))
}
```

Auth extraction, body parsing, JSON deserialization, error-to-HTTP conversion all handled by Axum.

---

## Module Layout

```
src/
  lib.rs                          -- pub mod + pub use re-exports
  error.rs                        -- ApiError + IntoResponse
  types.rs                        -- StudentId, SessionId, ConversationId, PieceId
  state.rs                        -- AppState + AuthService, DbService, InferenceService, PracticeService
  routes.rs                       -- fn router(AppState) -> Router
  server.rs                       -- #[event(fetch)] + WS/streaming carve-outs
  auth/
    mod.rs                        -- re-exports
    extractor.rs                  -- AuthUser FromRequestParts
    handlers.rs                   -- apple, google, me, signout, debug
    jwt.rs                        -- HMAC-SHA256 sign/verify
  practice/
    mod.rs                        -- re-exports
    dims.rs                       -- DIMS_6 constant
    handlers/
      mod.rs
      start.rs                    -- POST /api/practice/start
      upload.rs                   -- POST /api/practice/upload
    session/
      mod.rs                      -- PracticeSession DO + WebSocket dispatch
      error.rs                    -- PracticeError enum
      state.rs                    -- SessionState + constants
      inference.rs                -- HF endpoint calls
      processing.rs               -- MuQ + AMT result handling
      finalization.rs             -- end-of-session synthesis
      accumulator.rs              -- SessionAccumulator
      practice_mode.rs            -- state machine
      synthesis.rs                -- teacher LLM call
    analysis/
      mod.rs
      piece_identify.rs           -- N-gram + rerank
      piece_match.rs              -- DTW matching
      score_follower.rs           -- onset+pitch DTW
      score_context.rs            -- score MIDI loading
      session_piece_id.rs         -- piece ID orchestration
  services/
    mod.rs                        -- re-exports
    ask.rs                        -- /api/ask + /api/ask/elaborate
    chat.rs                       -- streaming chat + conversation queries
    exercises.rs                  -- /api/exercises
    goals.rs                      -- /api/extract-goals
    llm.rs                        -- Groq + Anthropic clients
    memory.rs                     -- student memory (D1 queries)
    prompts.rs                    -- LLM prompt templates
    scores.rs                     -- /api/scores
    stop.rs                       -- STOP classifier
    sync.rs                       -- /api/sync
    teaching_moments.rs           -- teaching moment selection
    teaching_moment_handler.rs    -- handler wiring
    waitlist.rs                   -- /api/waitlist
```

---

## Dependency Changes

### Cargo.toml additions

```toml
thiserror = "2.0"
tower-service = "0.3"
tower-http = { version = "0.6", default-features = false, features = ["cors"] }
```

### Cargo.toml updates

```toml
# Enable extractor features
axum = { version = "0.8.8", default-features = false, features = ["json", "query", "matched-path"] }

# Binary size optimization
[profile.release]
lto = true
opt-level = 'z'
codegen-units = 1
strip = true          # NEW
panic = "abort"       # NEW
```

### Clippy + formatting config

```toml
# In Cargo.toml
[lints.clippy]
pedantic = { priority = -1, level = "warn" }
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
unwrap_used = "warn"
expect_used = "warn"
todo = "warn"
dbg_macro = "warn"
print_stdout = "warn"
print_stderr = "warn"

[lints.rust]
unsafe_code = "deny"
```

Note: `#[worker::send]` generates `unsafe impl Send` internally. The `unsafe_code = "deny"` lint applies to hand-written unsafe. The macro-generated unsafe is in expanded code and does not trigger the lint. If it does, add `#[allow(unsafe_code)]` on the specific handler functions, not module-wide.

```toml
# rustfmt.toml
edition = "2021"
max_width = 100
use_field_init_shorthand = true
```

---

## Smoke Test

### `apps/api/tests/smoke_test.py`

HTTP-level smoke test run against local dev server (`just api` at localhost:8787).

**Unauthenticated tests:**
- `GET /health` -> 200
- `GET /api/auth/me` (no token) -> 401 + `{"error": ...}`
- `POST /api/auth/apple` (invalid token) -> 401
- `POST /api/ask` (no auth) -> 401
- `POST /api/sync` (no auth) -> 401
- `OPTIONS /api/auth/me` -> 204 + CORS headers
- `GET /api/nonexistent` -> 404

**Authenticated flow** (via debug login):
1. `POST /api/auth/debug` -> JWT
2. `GET /api/auth/me` + JWT -> 200 + profile
3. `POST /api/practice/start` + JWT -> 200 + session
4. `GET /api/conversations` + JWT -> 200 + list

**Execution:**
- Run before refactor: `uv run python tests/smoke_test.py` -> saves baseline
- Run after refactor: same command -> compares against baseline
- Part of `just test-api` command

---

## What Does NOT Change

- Durable Object internal logic (score following, piece ID algorithms, accumulator, practice mode state machine) -- restructured into submodules but logic untouched
- LLM prompt templates (`services/prompts.rs`)
- STOP classifier coefficients (`services/stop.rs`)
- Teaching moment selection algorithm (`services/teaching_moments.rs`)
- D1 schema / migrations
- wrangler.toml bindings
- Wire format (JSON field names stay the same for web/iOS clients)
- WebSocket message protocol
- Streaming chat response format

---

## Success Criteria

1. `cargo build --target wasm32-unknown-unknown --release` succeeds
2. `cargo clippy` passes with new lint config (zero warnings)
3. `cargo fmt --check` passes
4. Smoke test passes (all endpoints return same status codes and response shapes)
5. Zero `.unwrap()` in handler/service code (clippy enforces)
6. Zero `Result<T, String>` signatures
7. Zero manual `Response::builder()` chains in handlers (all through `IntoResponse`)
8. Binary size within 10% of pre-refactor baseline
