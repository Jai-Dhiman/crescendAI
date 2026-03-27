# Rust Style Guide

## CrescendAI API -- Cloudflare Workers (WASM + Axum)

Grounded in: Rust API Guidelines (RFC 430), Google Comprehensive Rust, Cloudflare Workers patterns,
community consensus from thiserror/axum/serde maintainers.

This document defines coding standards for `apps/api/`. Claude Code must follow these rules when
editing any file under `apps/api/src/`.

---

## 1. WASM Hard Constraints

Target: `wasm32-unknown-unknown`. Every dependency and pattern must respect these.

### What does NOT exist

- No filesystem (`std::fs`)
- No threads (`std::thread`, `std::sync::Mutex` is pointless -- use `RefCell`)
- No OS networking (`std::net`)
- No tokio runtime, `async_std`, or any threaded executor
- No `std::time::Instant` (use `js_sys::Date::now()`)
- No `jemalloc`

### Async model

All async runs cooperatively on the JS event loop via `wasm-bindgen-futures`.
`futures::join!` for concurrent dispatch. No `tokio::spawn`, no `spawn_local` unless fire-and-forget.

### Dependency hygiene

- `default-features = false` on every dependency. Opt in to only needed features.
- Before adding any crate: verify it compiles to `wasm32-unknown-unknown`.
- Audit transitive deps with `cargo tree`. Watch for `mio`, `socket2`, `libc` leaking via features.
- `getrandom` requires the `js` feature.

### Binary size

Every kilobyte affects cold start time.

```toml
[profile.release]
lto = true
opt-level = 'z'
codegen-units = 1
strip = true            # add if not present
panic = "abort"         # add -- reduces panic infrastructure
```

Additional rules:
- Prefer trait objects (`&dyn Trait`, `Box<dyn Trait>`) over generics in cold paths to reduce monomorphization bloat.
- Avoid `format!()` in error paths when a static string suffices.
- Index slices with `.get()` instead of `[]` -- indexing pulls in panic infrastructure.
- Avoid `regex` crate (~500kb). Use manual parsing.
- Profile with `twiggy` when investigating binary size regressions.

---

## 2. Error Handling

### Use `thiserror` for error types

Define explicit error enums. Map variants to HTTP status codes via `IntoResponse`.
Do NOT use `anyhow` in handler/service code -- callers need to match on variants.
Do NOT use `Result<T, String>` -- it discards structure and prevents pattern matching.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
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

### IntoResponse for errors

```rust
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            ApiError::Forbidden => (StatusCode::FORBIDDEN, "forbidden"),
            ApiError::InferenceFailed(_) => (StatusCode::BAD_GATEWAY, "inference_failed"),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
            ApiError::ExternalService(_) => (StatusCode::BAD_GATEWAY, "external_error"),
        };
        // Log all server errors
        if status.is_server_error() {
            console_error!("{}: {}", error_type, &self);
        }
        let body = serde_json::json!({ "error": error_type, "message": self.to_string() });
        (status, Json(body)).into_response()
    }
}
```

### Error context with `.map_err()`

```rust
// Good: explicit conversion with context
let db = env.d1("DB")
    .map_err(|e| ApiError::Internal(format!("D1 binding: {e}")))?;

// Good: From impl for common conversions
impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::BadRequest(format!("JSON: {e}"))
    }
}
```

### Never panic in handlers

- Never `.unwrap()` or `.expect()` in handler/service code.
- Use `?` with proper error conversion everywhere.
- Index slices with `.get()` instead of `[]`.
- Use checked arithmetic for division.

---

## 3. Naming Conventions

Follow RFC 430 and the Rust API Guidelines.

| Item | Convention | Example |
|------|-----------|---------|
| Modules | `snake_case` | `audio_processing` |
| Types (structs, enums, traits) | `UpperCamelCase` | `SessionState` |
| Functions, methods | `snake_case` | `process_audio` |
| Local variables | `snake_case` | `sample_rate` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_RETRIES` |
| Type parameters | Short `UpperCamelCase` | `T`, `E` |
| Lifetimes | Short lowercase | `'a`, `'ctx` |
| Acronyms | One word | `Uuid` not `UUID`, `Midi` not `MIDI` |

### Conversion methods

- `as_*` -- cheap, borrowed view (`&self -> &T`)
- `to_*` -- expensive conversion (`&self -> T`)
- `into_*` -- consuming conversion (`self -> T`)
- `from_*` -- constructor from another type

### Getters

No `get_` prefix. Use the field name directly:

```rust
fn name(&self) -> &str { &self.name }     // Good
fn get_name(&self) -> &str { &self.name }  // Bad
```

---

## 4. Type System

### Newtypes for domain concepts

Wrap primitive types to prevent mixing up semantically different values:

```rust
pub struct StudentId(String);
pub struct SessionId(String);
pub struct ConversationId(String);
pub struct Score(f32);
```

Use `#[serde(transparent)]` so newtypes serialize as their inner type.

### Enums over booleans for function parameters

```rust
// Bad: what does true mean?
fn process(audio: &[f32], normalize: bool) { ... }

// Good: self-documenting
enum Normalization { Enabled, Disabled }
fn process(audio: &[f32], normalization: Normalization) { ... }
```

### Eagerly derive standard traits

Every public type should derive applicable traits:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session { ... }
```

Checklist:
- `Debug` -- always (required for good error messages and logging)
- `Clone` -- unless the type manages a unique resource
- `PartialEq` / `Eq` -- for value types
- `Serialize` / `Deserialize` -- for any type crossing an API boundary
- `Default` -- when a sensible default exists
- `Display` -- for types shown to users or in logs

---

## 5. Serialization (Serde)

### JSON naming: `camelCase` at the API boundary

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    pub piece_id: String,
    pub sample_rate: u32,
    #[serde(default)]
    pub options: SessionOptions,
}
```

### Request types

- `#[serde(rename_all = "camelCase")]` -- consistent casing for JS clients
- `#[serde(deny_unknown_fields)]` -- catch typos from clients
- `#[serde(default)]` -- for optional fields with sensible defaults

### Response types

- `#[serde(rename_all = "camelCase")]` -- match request convention
- `#[serde(skip_serializing_if = "Option::is_none")]` -- omit null fields for compact JSON
- Do NOT use `deny_unknown_fields` on response types

### Internal types (D1 rows, DO state)

- Use `snake_case` to match D1 column names directly (no rename)
- `#[serde(default)]` on optional fields for backward compatibility

### Anti-patterns

- `serde_json::Value` as a struct field for "anything goes" -- define the actual shape
- `#[serde(flatten)]` with `#[serde(deny_unknown_fields)]` -- they conflict
- `#[serde(untagged)]` without `#[serde(expecting = "...")]` -- error messages are unusable

---

## 6. Module Organization

### File layout within a module

1. Module-level documentation (`//!`)
2. Imports (grouped: std, external crates, internal modules)
3. Constants
4. Type definitions (structs, enums)
5. Trait implementations (`Display`, `From`, standard traits)
6. Inherent implementations (`impl MyType`)
7. Free functions
8. Tests (`#[cfg(test)]` at bottom)

### Import ordering

Group imports with blank lines between groups:

```rust
// 1. Standard library
use std::collections::HashMap;

// 2. External crates
use axum::body::Body;
use http::{Response, StatusCode};
use serde::{Deserialize, Serialize};
use worker::{console_error, console_log, Env};

// 3. Internal modules
use crate::error::{ApiError, Result};
use crate::practice::session::SessionState;
```

### Visibility

- `pub` -- HTTP handlers called from `server.rs`, DTO types crossing module boundaries
- `pub(crate)` -- internal helpers, state types used across modules within the crate
- private (no modifier) -- implementation details within a single module

### Re-exports

Keep `lib.rs` as a clean index:

```rust
pub mod auth;
pub mod error;
pub mod practice;
pub mod server;
pub mod services;

pub use error::{ApiError, Result};
```

---

## 7. Handler Patterns

### Signature convention

Handlers receive `&Env`, optional `&HeaderMap`, optional `&[u8]` body, and return `Response`:

```rust
pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body>
```

### Keep handlers thin

Extract, validate, delegate to service functions, return response. Business logic belongs in
`services/`, not in the handler body.

### Response construction

Use helper functions to eliminate repeated `Response::builder()` chains:

```rust
fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(body).unwrap_or_default()))
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap()
        })
}

fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    json_response(status, &serde_json::json!({ "error": message }))
}
```

---

## 8. State Management

### Env passing

`worker::Env` is the root state container. Pass as `&Env` to all handlers and services.
Access bindings through it:

- `env.d1("DB")` -- D1 database
- `env.bucket("CHUNKS")` -- R2 storage
- `env.durable_object("PRACTICE_SESSION")` -- DO namespace
- `env.secret("SECRET_NAME")` -- secrets (never log these)
- `env.var("VAR_NAME")` -- configuration variables

### Durable Object interior mutability

Use `RefCell` (not `Mutex` -- single-threaded in WASM):

```rust
pub struct PracticeSession {
    state: State,
    env: Env,
    inner: RefCell<SessionState>,
}
```

### Durable Object state persistence

Critical state (session_id, student_id, conversation_id) must be persisted to `state.storage()`
and reloaded on resume. In-memory `RefCell` state is lost when the DO is evicted during
long async operations.

---

## 9. Observability

### Logging

- `console_log!()` for informational events (auth success, sync complete, chunk received)
- `console_error!()` for all error paths -- always log before returning an error response
- Include structured context: IDs, counts, timing

```rust
// Good: structured context
console_log!("chunk_scores[{}]: dyn={:.2} tim={:.2}", index, scores[0], scores[1]);
console_error!("D1 query failed for student {}: {:?}", student_id, e);

// Bad: generic messages
console_log!("done");
console_error!("error occurred");
```

### Never log secrets

Do not log values from `env.secret()`, auth tokens, or JWT payloads even at debug level.

---

## 10. Performance

### Minimize allocations

- Prefer `&str` over `String` in function parameters when you only read
- Take `String` ownership only when you need to store it
- Use `Cow<'_, str>` when a function sometimes borrows, sometimes owns
- Return references when the data lives in `self`

### Avoid unnecessary clones

```rust
fn process(data: &str) { ... }  // Good: borrow when you only read
fn store(data: String) { ... }  // Good: own when you need to store
fn bad(data: String) { ... }    // Bad: owns but only reads
```

### Static strings in error paths

```rust
// Good: no allocation
ApiError::Unauthorized

// Acceptable: allocation with context
ApiError::Internal(format!("D1 binding: {e}"))

// Bad: unnecessary allocation
ApiError::Internal("something failed".to_string())  // use a static variant instead
```

---

## 11. Clippy & Linting

Add to `Cargo.toml`:

```toml
[lints.clippy]
pedantic = { priority = -1, level = "warn" }

# Pedantic overrides (too noisy for this codebase)
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"

# Restriction lints (opt-in)
unwrap_used = "warn"
expect_used = "warn"
todo = "warn"
dbg_macro = "warn"
print_stdout = "warn"
print_stderr = "warn"

[lints.rust]
unsafe_code = "deny"
```

### Formatting

Add `rustfmt.toml` in `apps/api/`:

```toml
edition = "2021"
max_width = 100
use_field_init_shorthand = true
```

---

## 12. Security

### Input validation at the boundary

Validate all incoming data in handlers before passing to services:

```rust
fn validate_sample_rate(rate: u32) -> Result<u32> {
    if !(8000..=192000).contains(&rate) {
        return Err(ApiError::BadRequest(
            format!("sample rate {rate} out of range [8000, 192000]")
        ));
    }
    Ok(rate)
}
```

### Secrets

Access via `env.secret("NAME")`, never hardcode. Never log secret values.

### CORS

Configure CORS explicitly per route. Never use `CorsLayer::very_permissive()` in production.

---

## 13. Quick Reference

| Do | Don't |
|----|-------|
| `thiserror` error enums with `IntoResponse` | `Result<T, String>` or `anyhow` in handlers |
| `?` with `.map_err()` for context | `.unwrap()` or `.expect()` |
| `Debug` on every public type | Types without `Debug` |
| `default-features = false` on deps | Full feature sets without auditing |
| `#[serde(rename_all = "camelCase")]` on API types | Inconsistent JSON casing |
| `#[serde(skip_serializing_if = "Option::is_none")]` | Null fields in every response |
| Helper fns for response construction | Repeated `Response::builder()` chains |
| `console_error!` with structured context | Generic "error occurred" messages |
| `.get()` for slice indexing | `[]` indexing (panic infrastructure) |
| `RefCell` for interior mutability | `Mutex` (single-threaded WASM) |
| Trait objects in cold paths | Unbounded generic monomorphization |
| `&str` params when only reading | `String` params when ownership not needed |
| `env.secret()` for sensitive values | Hardcoded secrets or logging secret values |
