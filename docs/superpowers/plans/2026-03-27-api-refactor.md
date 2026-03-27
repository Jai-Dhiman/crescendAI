# API Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the CrescendAI API (`apps/api/`) from manual dispatch + `Result<T, String>` to proper Axum Router with typed errors, service layer, extractors, and standardized serde -- per `apps/api/RUST_STYLE.md`.

**Architecture:** Axum Router with `tower_service::Service::call()` bridge to Cloudflare Workers `#[event(fetch)]`. Service layer (`AppState` with `AuthService`, `DbService`, `InferenceService`, `PracticeService`) replaces raw `&Env` passing. `thiserror` enums replace `String` errors. `AuthUser` custom extractor replaces manual JWT parsing.

**Tech Stack:** Rust, axum 0.8 (no defaults + json/query/matched-path), worker 0.7, thiserror 2.0, tower-http 0.6 (cors), tower-service 0.3, serde, wasm32-unknown-unknown target.

**Spec:** `docs/superpowers/specs/2026-03-27-api-refactor-design.md`

**Style Guide:** `apps/api/RUST_STYLE.md`

---

## File Map

### New files
- `src/error.rs` -- ApiError enum + IntoResponse impl
- `src/types.rs` -- StudentId, SessionId, ConversationId, PieceId newtypes
- `src/state.rs` -- AppState + AuthService, DbService, InferenceService, PracticeService
- `src/routes.rs` -- fn router(AppState) -> Router
- `src/auth/extractor.rs` -- AuthUser FromRequestParts impl
- `src/auth/handlers.rs` -- auth HTTP handlers (extracted from auth/mod.rs)
- `src/practice/handlers/mod.rs` -- re-exports
- `src/practice/handlers/start.rs` -- moved from practice/start.rs
- `src/practice/handlers/upload.rs` -- moved from practice/upload.rs
- `src/practice/session/mod.rs` -- PracticeSession DO (moved from practice/session.rs)
- `src/practice/session/error.rs` -- PracticeError enum
- `src/practice/session/state.rs` -- SessionState + constants (extracted from session.rs)
- `src/practice/session/inference.rs` -- moved from session_inference.rs
- `src/practice/session/processing.rs` -- moved from session_processing.rs
- `src/practice/session/finalization.rs` -- moved from session_finalization.rs
- `src/practice/session/accumulator.rs` -- moved from practice/accumulator.rs
- `src/practice/session/practice_mode.rs` -- moved from practice/practice_mode.rs
- `src/practice/session/synthesis.rs` -- moved from practice/synthesis.rs
- `src/practice/analysis/mod.rs` -- re-exports
- `src/practice/analysis/piece_identify.rs` -- moved from practice/piece_identify.rs
- `src/practice/analysis/piece_match.rs` -- moved from practice/piece_match.rs
- `src/practice/analysis/score_follower.rs` -- moved from practice/score_follower.rs
- `src/practice/analysis/score_context.rs` -- moved from practice/score_context.rs
- `src/practice/analysis/session_piece_id.rs` -- moved from practice/session_piece_id.rs
- `apps/api/rustfmt.toml` -- formatting config
- `apps/api/tests/smoke_test.py` -- HTTP smoke test

### Modified files
- `Cargo.toml` -- dependencies, features, lints, profile
- `src/lib.rs` -- new module declarations + re-exports
- `src/server.rs` -- rewritten: Axum Router bridge + carve-outs
- `src/auth/mod.rs` -- slimmed: re-exports only (handlers moved to handlers.rs)
- `src/auth/jwt.rs` -- Result<Claims, ApiError> instead of Result<Claims, String>
- `src/practice/mod.rs` -- updated submodule declarations
- `src/practice/dims.rs` -- unchanged (already clean)
- `src/practice/teaching_moment.rs` -- unchanged (already clean)
- `src/services/mod.rs` -- unchanged
- `src/services/ask.rs` -- Axum handler signature, ApiError, serde attributes
- `src/services/chat.rs` -- Axum handler signatures (non-streaming), ApiError, serde
- `src/services/exercises.rs` -- Axum handler signatures, ApiError, serde
- `src/services/goals.rs` -- Axum handler signature, ApiError, serde
- `src/services/llm.rs` -- ApiError instead of String, serde attributes
- `src/services/memory.rs` -- ApiError instead of String, serde attributes, handler signatures
- `src/services/prompts.rs` -- unchanged (string constants)
- `src/services/scores.rs` -- Axum handler signature, ApiError, serde
- `src/services/stop.rs` -- unchanged (pure function)
- `src/services/sync.rs` -- Axum handler signature, ApiError, serde
- `src/services/teaching_moments.rs` -- unchanged (pure function)
- `src/services/teaching_moment_handler.rs` -- ApiError
- `src/services/waitlist.rs` -- Axum handler signature, ApiError, serde

### Deleted files (after moves)
- `src/practice/start.rs` -- moved to practice/handlers/start.rs
- `src/practice/upload.rs` -- moved to practice/handlers/upload.rs
- `src/practice/session.rs` -- moved to practice/session/mod.rs
- `src/practice/accumulator.rs` -- moved to practice/session/accumulator.rs
- `src/practice/practice_mode.rs` -- moved to practice/session/practice_mode.rs
- `src/practice/synthesis.rs` -- moved to practice/session/synthesis.rs
- `src/practice/session_inference.rs` -- moved to practice/session/inference.rs
- `src/practice/session_processing.rs` -- moved to practice/session/processing.rs
- `src/practice/session_finalization.rs` -- moved to practice/session/finalization.rs
- `src/practice/piece_identify.rs` -- moved to practice/analysis/piece_identify.rs
- `src/practice/piece_match.rs` -- moved to practice/analysis/piece_match.rs
- `src/practice/score_follower.rs` -- moved to practice/analysis/score_follower.rs
- `src/practice/score_context.rs` -- moved to practice/analysis/score_context.rs
- `src/practice/session_piece_id.rs` -- moved to practice/analysis/session_piece_id.rs
- `src/practice/analysis.rs` -- logic absorbed into analysis/ submodule

---

## Task 1: Smoke Test Baseline + Dependencies

**Files:**
- Create: `apps/api/tests/smoke_test.py`
- Create: `apps/api/rustfmt.toml`
- Modify: `apps/api/Cargo.toml`

This task captures the pre-refactor baseline and adds all new dependencies + config so subsequent tasks can use them.

- [ ] **Step 1: Record pre-refactor binary size**

```bash
cd apps/api
cargo build --target wasm32-unknown-unknown --release 2>/dev/null
ls -la target/wasm32-unknown-unknown/release/*.wasm | awk '{print $5, $NF}'
# Save this number -- success criterion is within 10%
```

- [ ] **Step 2: Create the smoke test**

Create `apps/api/tests/smoke_test.py`:

```python
#!/usr/bin/env python3
"""
Smoke test for CrescendAI API.
Run against local dev server (just api) at http://localhost:8787.
Validates all endpoints return expected status codes and response shapes.

Usage:
    uv run python tests/smoke_test.py              # run tests
    uv run python tests/smoke_test.py --baseline    # save baseline
    uv run python tests/smoke_test.py --compare     # compare to baseline
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://localhost:8787"
BASELINE_FILE = Path(__file__).parent / "smoke_baseline.json"

# (method, path, body_dict_or_None, expected_status_or_list, response_checks)
UNAUTH_TESTS = [
    ("GET", "/health", None, 200, {"has_key": "status"}),
    ("GET", "/api/auth/me", None, 401, {"has_key": "error"}),
    ("POST", "/api/auth/apple", {"identityToken": "invalid"}, 401, None),
    ("POST", "/api/ask", {}, 401, None),
    ("POST", "/api/sync", {}, 401, None),
    ("POST", "/api/waitlist", {"email": "smoke@test.com"}, [200, 400, 409], None),
    ("GET", "/api/nonexistent", None, [404, 405], None),
]


def make_request(method: str, path: str, body=None, headers=None):
    """Make an HTTP request and return (status_code, response_body_dict, response_headers)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req)
        body_bytes = resp.read()
        try:
            body_json = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        return resp.status, body_json, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            body_json = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        return e.code, body_json, dict(e.headers)


def run_unauth_tests():
    """Run unauthenticated endpoint tests."""
    results = []
    for method, path, body, expected, checks in UNAUTH_TESTS:
        status, resp_body, resp_headers = make_request(method, path, body)
        expected_list = expected if isinstance(expected, list) else [expected]
        passed = status in expected_list
        if passed and checks:
            if "has_key" in checks:
                passed = checks["has_key"] in resp_body
            if "header" in checks:
                passed = passed and checks["header"].lower() in {
                    k.lower() for k in resp_headers
                }
        result = {
            "method": method,
            "path": path,
            "status": status,
            "expected": expected_list,
            "passed": passed,
        }
        results.append(result)
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {method} {path} -> {status} (expected {expected_list})")
    return results


def run_auth_flow():
    """Run authenticated flow via debug login."""
    results = []

    # Step 1: Debug login
    status, body, headers = make_request(
        "POST", "/api/auth/debug", {"email": "smoke@test.com"}
    )
    if status == 404:
        print("  [SKIP] Debug auth not available (production mode)")
        return results

    passed = status == 200 and "student_id" in body
    results.append(
        {"test": "debug_login", "status": status, "passed": passed}
    )
    print(f"  [{'PASS' if passed else 'FAIL'}] POST /api/auth/debug -> {status}")
    if not passed:
        return results

    # Extract token from Set-Cookie or response body
    token = body.get("token", "")
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Step 2: Auth me
    status, body, _ = make_request("GET", "/api/auth/me", headers=auth_headers)
    passed = status == 200
    results.append({"test": "auth_me", "status": status, "passed": passed})
    print(f"  [{'PASS' if passed else 'FAIL'}] GET /api/auth/me -> {status}")

    # Step 3: List conversations
    status, body, _ = make_request(
        "GET", "/api/conversations", headers=auth_headers
    )
    passed = status == 200
    results.append(
        {"test": "list_conversations", "status": status, "passed": passed}
    )
    print(
        f"  [{'PASS' if passed else 'FAIL'}] GET /api/conversations -> {status}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="API Smoke Test")
    parser.add_argument(
        "--baseline", action="store_true", help="Save results as baseline"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare to saved baseline"
    )
    args = parser.parse_args()

    print("\n=== CrescendAI API Smoke Test ===\n")

    print("Unauthenticated tests:")
    unauth_results = run_unauth_tests()

    print("\nAuthenticated flow:")
    auth_results = run_auth_flow()

    all_results = {
        "unauth": unauth_results,
        "auth": auth_results,
    }

    total = len(unauth_results) + len(auth_results)
    passed = sum(1 for r in unauth_results if r["passed"]) + sum(
        1 for r in auth_results if r["passed"]
    )
    print(f"\n--- {passed}/{total} tests passed ---\n")

    if args.baseline:
        BASELINE_FILE.write_text(json.dumps(all_results, indent=2))
        print(f"Baseline saved to {BASELINE_FILE}")

    if args.compare and BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
        diffs = []
        for test_type in ["unauth", "auth"]:
            for i, result in enumerate(all_results.get(test_type, [])):
                if i < len(baseline.get(test_type, [])):
                    base = baseline[test_type][i]
                    if result["status"] != base["status"]:
                        diffs.append(
                            f"  {test_type}[{i}]: {base['status']} -> {result['status']}"
                        )
        if diffs:
            print("REGRESSIONS:")
            for d in diffs:
                print(d)
            sys.exit(1)
        else:
            print("No regressions detected vs baseline.")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update Cargo.toml with new dependencies, lints, and profile**

In `apps/api/Cargo.toml`, make these changes:

Update axum line:
```toml
axum = { version = "0.8.8", default-features = false, features = ["json", "query", "matched-path"] }
```

Add after the `console_error_panic_hook` line:
```toml
# Error handling
thiserror = "2.0"

# Router bridge
tower-service = "0.3"
tower-http = { version = "0.6", default-features = false, features = ["cors"] }
```

Add at end of `[profile.release]`:
```toml
strip = true
panic = "abort"
```

Add new sections at end of file:
```toml
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

- [ ] **Step 4: Create rustfmt.toml**

Create `apps/api/rustfmt.toml`:
```toml
edition = "2021"
max_width = 100
use_field_init_shorthand = true
```

- [ ] **Step 5: Verify new dependencies compile to WASM**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | tail -5
```

Expected: succeeds (warnings OK for now, errors are blockers).

- [ ] **Step 6: Commit**

```bash
cd apps/api
git add tests/smoke_test.py rustfmt.toml Cargo.toml Cargo.lock
git commit -m "chore: add smoke test baseline, dependencies, and lint config for API refactor"
```

---

## Task 2: Error Types + Newtypes Foundation

**Files:**
- Create: `apps/api/src/error.rs`
- Create: `apps/api/src/types.rs`
- Modify: `apps/api/src/lib.rs`

These are the foundational types that every subsequent task depends on. No existing code changes yet -- just adding new files.

- [ ] **Step 1: Create `src/error.rs`**

```rust
//! Centralized API error types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use worker::console_error;

/// API-level error type. Maps to HTTP status codes via `IntoResponse`.
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

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ApiError>;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            Self::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            Self::Forbidden => (StatusCode::FORBIDDEN, "forbidden"),
            Self::InferenceFailed(_) => (StatusCode::BAD_GATEWAY, "inference_failed"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
            Self::ExternalService(_) => (StatusCode::BAD_GATEWAY, "external_error"),
        };

        if status.is_server_error() {
            console_error!("{}: {}", error_type, &self);
        }

        let body = serde_json::json!({
            "error": error_type,
            "message": self.to_string(),
        });

        (status, axum::Json(body)).into_response()
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        Self::BadRequest(format!("JSON: {e}"))
    }
}

impl From<worker::Error> for ApiError {
    fn from(e: worker::Error) -> Self {
        Self::Internal(format!("worker: {e}"))
    }
}
```

- [ ] **Step 2: Create `src/types.rs`**

```rust
//! Domain newtypes for compile-time distinction of IDs.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Generate a v4 UUID using getrandom.
fn generate_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).unwrap_or_default();
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

macro_rules! define_id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new() -> Self {
                Self(generate_uuid_v4())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

define_id_type!(StudentId);
define_id_type!(SessionId);
define_id_type!(ConversationId);
define_id_type!(PieceId);
```

- [ ] **Step 3: Update `src/lib.rs`**

Replace contents of `src/lib.rs` with:

```rust
pub mod auth;
pub mod error;
pub mod practice;
pub mod routes;
pub mod server;
pub mod services;
pub mod state;
pub mod types;

pub use error::{ApiError, Result};
pub use types::{ConversationId, PieceId, SessionId, StudentId};

/// Truncate a string at a UTF-8 safe boundary, returning at most `max_bytes` bytes.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
```

- [ ] **Step 4: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -20
```

Expected: succeeds. The new modules exist but nothing references them yet (routes.rs and state.rs don't exist yet, which is fine -- lib.rs declares them but cargo check will warn, not error, if they're empty). If cargo errors on missing modules, create empty placeholder files:

```bash
touch src/routes.rs src/state.rs
```

- [ ] **Step 5: Commit**

```bash
git add src/error.rs src/types.rs src/lib.rs src/routes.rs src/state.rs
git commit -m "feat: add ApiError, PracticeError, and domain newtypes (StudentId, SessionId, etc.)"
```

---

## Task 3: Service Layer + AppState

**Files:**
- Create: `apps/api/src/state.rs`
- Modify: `apps/api/src/auth/jwt.rs` (Result type change)

This task builds the service layer that all refactored handlers will use.

- [ ] **Step 1: Create `src/state.rs`**

```rust
//! Application state and service layer.
//!
//! Each service wraps `SendWrapper<Env>` and exposes typed methods.
//! Handlers access services via `State<AppState>` extractor.

use crate::error::{ApiError, Result};
use crate::types::StudentId;
use serde::{Deserialize, Serialize};
use worker::send::SendWrapper;
use worker::{console_error, Env};

/// Shared application state passed to all Axum handlers via `State<AppState>`.
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

// ---------------------------------------------------------------------------
// AuthService
// ---------------------------------------------------------------------------

/// Handles JWT operations and user management.
#[derive(Clone)]
pub struct AuthService {
    env: SendWrapper<Env>,
}

impl AuthService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn jwt_secret(&self) -> Result<Vec<u8>> {
        self.env
            .secret("JWT_SECRET")
            .map(|s| s.to_string().into_bytes())
            .map_err(|e| ApiError::Internal(format!("JWT_SECRET: {e}")))
    }

    pub fn verify_jwt(&self, token: &str) -> Result<crate::auth::jwt::Claims> {
        let secret = self.jwt_secret()?;
        crate::auth::jwt::verify(token, &secret)
    }

    pub fn sign_jwt(&self, student_id: &StudentId) -> Result<String> {
        let secret = self.jwt_secret()?;
        crate::auth::jwt::sign_for_student(student_id.as_str(), &secret)
    }
}

// ---------------------------------------------------------------------------
// DbService
// ---------------------------------------------------------------------------

/// Wraps D1 database access.
#[derive(Clone)]
pub struct DbService {
    env: SendWrapper<Env>,
}

impl DbService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn d1(&self) -> Result<worker::D1Database> {
        self.env
            .d1("DB")
            .map_err(|e| ApiError::Internal(format!("D1 binding: {e}")))
    }
}

// ---------------------------------------------------------------------------
// InferenceService
// ---------------------------------------------------------------------------

/// Wraps LLM and inference endpoint access.
#[derive(Clone)]
pub struct InferenceService {
    env: SendWrapper<Env>,
}

impl InferenceService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }
}

// ---------------------------------------------------------------------------
// PracticeService
// ---------------------------------------------------------------------------

/// Wraps Durable Object namespace and R2 bucket access.
#[derive(Clone)]
pub struct PracticeService {
    env: SendWrapper<Env>,
}

impl PracticeService {
    pub fn new(env: SendWrapper<Env>) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn do_namespace(&self) -> Result<worker::DurableObjectNamespace> {
        self.env
            .durable_object("PRACTICE_SESSION")
            .map_err(|e| ApiError::Internal(format!("DO namespace: {e}")))
    }

    pub fn chunks_bucket(&self) -> Result<worker::Bucket> {
        self.env
            .bucket("CHUNKS")
            .map_err(|e| ApiError::Internal(format!("R2 CHUNKS: {e}")))
    }

    pub fn scores_bucket(&self) -> Result<worker::Bucket> {
        self.env
            .bucket("SCORES")
            .map_err(|e| ApiError::Internal(format!("R2 SCORES: {e}")))
    }
}
```

- [ ] **Step 2: Update `src/auth/jwt.rs` to return `ApiError`**

Read the current file first. Then change the two function signatures from `Result<T, String>` to use the crate error type. The key changes:

Change `sign` to:
```rust
use crate::error::{ApiError, Result};

pub fn sign(claims: &Claims, secret: &[u8]) -> Result<String> {
    // ... existing logic, but change error returns from
    // Err("message".to_string()) to Err(ApiError::Internal("message".into()))
}

pub fn verify(token: &str, secret: &[u8]) -> Result<Claims> {
    // ... existing logic, but change error returns from
    // Err("message".to_string()) to Err(ApiError::Unauthorized)
    // (verification failures are auth errors, not internal errors)
}

/// Convenience: sign a JWT for a student ID with standard expiry.
pub fn sign_for_student(student_id: &str, secret: &[u8]) -> Result<String> {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    let claims = Claims {
        sub: student_id.to_string(),
        iat: now,
        exp: now + 30 * 24 * 60 * 60, // 30 days
    };
    sign(&claims, secret)
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -20
```

- [ ] **Step 4: Commit**

```bash
git add src/state.rs src/auth/jwt.rs
git commit -m "feat: add AppState service layer and update jwt to use ApiError"
```

---

## Task 4: Auth Extractor + Auth Handlers

**Files:**
- Create: `apps/api/src/auth/extractor.rs`
- Create: `apps/api/src/auth/handlers.rs`
- Modify: `apps/api/src/auth/mod.rs`

Extract auth handlers into their own file with Axum signatures, create the `AuthUser` extractor.

- [ ] **Step 1: Create `src/auth/extractor.rs`**

```rust
//! AuthUser extractor for Axum handlers.

use axum::extract::FromRequestParts;
use http::request::Parts;

use crate::error::ApiError;
use crate::state::AppState;
use crate::types::StudentId;

/// Authenticated user extracted from JWT in cookie or Bearer header.
///
/// Add to any handler signature to require authentication.
/// Use `Option<AuthUser>` for optional authentication.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub student_id: StudentId,
}

impl FromRequestParts<AppState> for AuthUser {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let token = extract_token_from_cookie(&parts.headers)
            .or_else(|| extract_token_from_bearer(&parts.headers))
            .ok_or(ApiError::Unauthorized)?;

        let claims = state.auth.verify_jwt(&token)?;
        Ok(Self {
            student_id: StudentId::from(claims.sub),
        })
    }
}

fn extract_token_from_cookie(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|cookie_str| {
            cookie_str.split(';').find_map(|pair| {
                let pair = pair.trim();
                if let Some(value) = pair.strip_prefix("session=") {
                    Some(value.to_string())
                } else {
                    None
                }
            })
        })
}

fn extract_token_from_bearer(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
        .map(|token| token.to_string())
}
```

- [ ] **Step 2: Create `src/auth/handlers.rs`**

This file contains the auth HTTP handlers with Axum signatures. Read the current `src/auth/mod.rs` (962 lines) and migrate each handler. The handlers are large -- preserve the existing business logic but change:
- Signature from `(env: &Env, ...)` to `(State(state): State<AppState>, ...)`
- Error handling from `Response::builder()` to `Result<impl IntoResponse>`
- Body parsing from manual `serde_json::from_slice` to `Json<T>` extractor
- Auth checking from manual to `AuthUser` extractor (for `handle_auth_me`)

The private helper functions (`find_or_create_student`, `build_auth_cookie`, `parse_apple_token_claims`, etc.) move into this file too.

Key signatures after migration:
```rust
use axum::extract::State;
use axum::Json;
use crate::state::AppState;
use crate::error::Result;
use crate::auth::extractor::AuthUser;

#[worker::send]
pub async fn handle_apple(
    State(state): State<AppState>,
    Json(payload): Json<AppleAuthRequest>,
) -> Result<impl axum::response::IntoResponse> { ... }

#[worker::send]
pub async fn handle_google(
    State(state): State<AppState>,
    Json(payload): Json<GoogleAuthRequest>,
) -> Result<impl axum::response::IntoResponse> { ... }

#[worker::send]
pub async fn handle_me(
    State(state): State<AppState>,
    auth: AuthUser,
) -> Result<Json<AuthResponse>> { ... }

pub async fn handle_signout(
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse { ... }

#[worker::send]
pub async fn handle_debug(
    State(state): State<AppState>,
) -> Result<impl axum::response::IntoResponse> { ... }
```

Note: `handle_apple` and `handle_google` return `impl IntoResponse` (not `Json`) because they set `Set-Cookie` headers. Build the response with:
```rust
let mut headers = http::HeaderMap::new();
headers.insert("set-cookie", cookie_value.parse().map_err(|_| ApiError::Internal("cookie header".into()))?);
Ok((headers, Json(response_body)))
```

Add serde attributes to request/response structs:
```rust
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppleAuthRequest {
    pub identity_token: String,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthResponse {
    pub student_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub is_new_user: bool,
}
```

- [ ] **Step 3: Update `src/auth/mod.rs` to re-export**

Replace the entire contents of `src/auth/mod.rs` with:

```rust
pub mod extractor;
pub mod handlers;
pub mod jwt;

pub use extractor::AuthUser;
```

The old `verify_auth` and `verify_auth_header` functions are no longer needed -- the `AuthUser` extractor replaces them. The carve-out paths (WS upgrade, streaming chat) will do their own manual auth check using `AuthService::verify_jwt` directly in `server.rs`.

- [ ] **Step 4: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

This step will likely show warnings about unused code in handlers.rs (since nothing calls the new handlers yet). That's expected.

- [ ] **Step 5: Commit**

```bash
git add src/auth/
git commit -m "feat: add AuthUser extractor and migrate auth handlers to Axum signatures"
```

---

## Task 5: Practice Module Restructure

**Files:**
- Create: `apps/api/src/practice/handlers/mod.rs`, `start.rs`, `upload.rs`
- Create: `apps/api/src/practice/session/mod.rs`, `error.rs`, `state.rs`, `inference.rs`, `processing.rs`, `finalization.rs`, `accumulator.rs`, `practice_mode.rs`, `synthesis.rs`
- Create: `apps/api/src/practice/analysis/mod.rs`, `piece_identify.rs`, `piece_match.rs`, `score_follower.rs`, `score_context.rs`, `session_piece_id.rs`
- Modify: `apps/api/src/practice/mod.rs`
- Delete: old flat files

This is a structural move -- logic stays the same, just relocated into the nested structure.

- [ ] **Step 1: Create directory structure**

```bash
cd apps/api/src/practice
mkdir -p handlers session analysis
```

- [ ] **Step 2: Create `session/error.rs`**

```rust
//! Durable Object error types.

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

pub type Result<T> = std::result::Result<T, PracticeError>;
```

- [ ] **Step 3: Move files to new locations**

Move each file, updating `crate::practice::X` imports to `crate::practice::session::X` or `crate::practice::analysis::X` as appropriate. The key moves:

```bash
# Session internals
git mv src/practice/session.rs src/practice/session/mod.rs
git mv src/practice/accumulator.rs src/practice/session/accumulator.rs
git mv src/practice/practice_mode.rs src/practice/session/practice_mode.rs
git mv src/practice/synthesis.rs src/practice/session/synthesis.rs
git mv src/practice/session_inference.rs src/practice/session/inference.rs
git mv src/practice/session_processing.rs src/practice/session/processing.rs
git mv src/practice/session_finalization.rs src/practice/session/finalization.rs

# Analysis
git mv src/practice/piece_identify.rs src/practice/analysis/piece_identify.rs
git mv src/practice/piece_match.rs src/practice/analysis/piece_match.rs
git mv src/practice/score_follower.rs src/practice/analysis/score_follower.rs
git mv src/practice/score_context.rs src/practice/analysis/score_context.rs
git mv src/practice/session_piece_id.rs src/practice/analysis/session_piece_id.rs
git mv src/practice/analysis.rs src/practice/analysis/bar_analysis.rs

# Handlers
git mv src/practice/start.rs src/practice/handlers/start.rs
git mv src/practice/upload.rs src/practice/handlers/upload.rs
```

- [ ] **Step 4: Create submodule `mod.rs` files**

Create `src/practice/handlers/mod.rs`:
```rust
pub mod start;
pub mod upload;
```

Create `src/practice/analysis/mod.rs`:
```rust
pub mod bar_analysis;
pub mod piece_identify;
pub mod piece_match;
pub mod score_context;
pub mod score_follower;
pub mod session_piece_id;
```

Add submodule declarations to `src/practice/session/mod.rs` (at top of the existing file):
```rust
pub mod accumulator;
pub mod error;
pub mod finalization;
pub mod inference;
pub mod practice_mode;
pub mod processing;
pub mod synthesis;

// ... rest of existing session.rs code (PracticeSession struct, etc.)
```

Create `src/practice/session/state.rs` by extracting `SessionState` and constants from session/mod.rs. Move `SessionState`, `ALARM_DURATION_MS`, `HF_RETRY_DELAYS_MS`, `HF_RETRY_DELAYS_ENDING_MS`, `base64_encode`, `sleep_ms` into this file.

- [ ] **Step 5: Update `src/practice/mod.rs`**

Replace with:
```rust
pub mod analysis;
pub mod dims;
pub mod handlers;
pub mod session;
pub mod teaching_moment;
```

- [ ] **Step 6: Fix all import paths**

Search for `crate::practice::accumulator` and replace with `crate::practice::session::accumulator`. Do the same for all moved modules. Key renames:

| Old path | New path |
|----------|----------|
| `crate::practice::accumulator` | `crate::practice::session::accumulator` |
| `crate::practice::practice_mode` | `crate::practice::session::practice_mode` |
| `crate::practice::synthesis` | `crate::practice::session::synthesis` |
| `crate::practice::session_inference` | `crate::practice::session::inference` |
| `crate::practice::session_processing` | `crate::practice::session::processing` |
| `crate::practice::session_finalization` | `crate::practice::session::finalization` |
| `crate::practice::piece_identify` | `crate::practice::analysis::piece_identify` |
| `crate::practice::piece_match` | `crate::practice::analysis::piece_match` |
| `crate::practice::score_follower` | `crate::practice::analysis::score_follower` |
| `crate::practice::score_context` | `crate::practice::analysis::score_context` |
| `crate::practice::session_piece_id` | `crate::practice::analysis::session_piece_id` |
| `crate::practice::analysis` | `crate::practice::analysis::bar_analysis` |
| `crate::practice::start` | `crate::practice::handlers::start` |
| `crate::practice::upload` | `crate::practice::handlers::upload` |

- [ ] **Step 7: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

Fix any remaining import path errors iteratively.

- [ ] **Step 8: Commit**

```bash
git add -A src/practice/
git commit -m "refactor: restructure practice module into handlers/, session/, analysis/ submodules"
```

---

## Task 6: Router + Server Entry Point

**Files:**
- Create: `apps/api/src/routes.rs`
- Rewrite: `apps/api/src/server.rs`

This is the core architectural change -- replacing the 520-line manual dispatcher with the Axum Router.

- [ ] **Step 1: Create `src/routes.rs`**

```rust
//! Axum router definition.

use axum::routing::{get, post};
use axum::Router;

use crate::auth;
use crate::practice;
use crate::services;
use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        // Health
        .route("/health", get(health))
        // Auth
        .route("/api/auth/apple", post(auth::handlers::handle_apple))
        .route("/api/auth/google", post(auth::handlers::handle_google))
        .route("/api/auth/me", get(auth::handlers::handle_me))
        .route("/api/auth/signout", post(auth::handlers::handle_signout))
        .route("/api/auth/debug", post(auth::handlers::handle_debug))
        // Practice
        .route(
            "/api/practice/start",
            post(practice::handlers::start::handle_start),
        )
        .route(
            "/api/practice/upload",
            post(practice::handlers::upload::handle_upload),
        )
        // Services
        .route("/api/ask", post(services::ask::handle_ask))
        .route("/api/ask/elaborate", post(services::ask::handle_elaborate))
        .route(
            "/api/conversations",
            get(services::chat::list_conversations),
        )
        .route(
            "/api/conversations/:id/messages",
            get(services::chat::get_messages),
        )
        .route("/api/sync", post(services::sync::handle_sync))
        .route(
            "/api/extract-goals",
            post(services::goals::handle_goals),
        )
        .route("/api/exercises", get(services::exercises::handle_exercises))
        .route(
            "/api/exercises/assign",
            post(services::exercises::handle_assign),
        )
        .route(
            "/api/exercises/complete",
            post(services::exercises::handle_complete),
        )
        .route("/api/scores", get(services::scores::handle_scores))
        .route("/api/waitlist", post(services::waitlist::handle_waitlist))
        // Memory (eval/benchmark endpoints)
        .route(
            "/api/memory/extract-chat",
            post(services::memory::handle_extract_chat),
        )
        .route(
            "/api/memory/store-facts",
            post(services::memory::handle_store_facts),
        )
        .route(
            "/api/memory/search",
            post(services::memory::handle_search_facts),
        )
        .route(
            "/api/memory/clear-benchmark",
            post(services::memory::handle_clear_benchmark),
        )
        .route(
            "/api/memory/synthesize",
            post(services::memory::handle_synthesize),
        )
        .route(
            "/api/memory/seed-observations",
            post(services::memory::handle_seed_observations),
        )
        .with_state(state)
}

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}
```

- [ ] **Step 2: Rewrite `src/server.rs`**

Replace the entire 520-line file with:

```rust
//! Cloudflare Workers entry point.
//!
//! Routes WebSocket upgrades and streaming chat directly (carve-outs),
//! everything else through the Axum Router.

use axum::body::Body;
use http_body_util::BodyExt;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_service::Service;
use worker::{event, console_error, Context, Env, HttpRequest};

use crate::routes::router;
use crate::state::AppState;

#[event(fetch)]
async fn fetch(
    req: HttpRequest,
    env: Env,
    _ctx: Context,
) -> worker::Result<http::Response<Body>> {
    console_error_panic_hook::set_once();

    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // --- Carve-out 1: WebSocket upgrade (bypasses Axum) ---
    if path.starts_with("/api/practice/ws/") {
        return handle_ws_upgrade(&path, &env, req).await;
    }

    // --- Carve-out 2: Streaming chat (bypasses Axum) ---
    if path == "/api/chat" && method == http::Method::POST {
        return handle_chat_stream(&env, req).await;
    }

    // --- Everything else through Axum Router ---
    let allowed_origin = env
        .var("ALLOWED_ORIGIN")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "http://localhost:3000".to_string());

    let cors = CorsLayer::new()
        .allow_origin(
            allowed_origin
                .parse::<http::HeaderValue>()
                .map(AllowOrigin::exact)
                .unwrap_or_else(|_| AllowOrigin::any()),
        )
        .allow_methods([
            http::Method::GET,
            http::Method::POST,
            http::Method::OPTIONS,
            http::Method::DELETE,
        ])
        .allow_headers(Any)
        .allow_credentials(true);

    let state = AppState::from_env(env);
    let mut app = router(state).layer(cors);

    Ok(app.call(req).await.unwrap_or_else(|err| {
        // Infallible in practice, but handle gracefully
        let body = format!("{{\"error\":\"internal\",\"message\":\"{err}\"}}");
        http::Response::builder()
            .status(http::StatusCode::INTERNAL_SERVER_ERROR)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap_or_default()
    }))
}

/// WebSocket upgrade for practice sessions. Auth validated manually.
async fn handle_ws_upgrade(
    path: &str,
    env: &Env,
    req: HttpRequest,
) -> worker::Result<http::Response<Body>> {
    // Extract session ID from path: /api/practice/ws/{sessionId}
    let session_id = path
        .strip_prefix("/api/practice/ws/")
        .unwrap_or_default();

    if session_id.is_empty() {
        return Ok(http::Response::builder()
            .status(http::StatusCode::BAD_REQUEST)
            .body(Body::from(r#"{"error":"missing session ID"}"#))
            .unwrap_or_default());
    }

    // Parse query params for student_id, conversation_id, is_eval
    let query = req.uri().query().unwrap_or_default();
    let params: Vec<(&str, &str)> = query
        .split('&')
        .filter_map(|p| p.split_once('='))
        .collect();

    let student_id = params
        .iter()
        .find(|(k, _)| *k == "studentId")
        .map(|(_, v)| *v)
        .unwrap_or_default();

    // Auth check: verify the student has a valid token
    // (extracted from headers, same cookie/bearer logic)
    let headers = req.headers();
    let token = extract_token(headers);
    if let Some(token) = &token {
        let secret = env
            .secret("JWT_SECRET")
            .map(|s| s.to_string().into_bytes())
            .map_err(|e| worker::Error::RustError(format!("JWT_SECRET: {e}")))?;
        if crate::auth::jwt::verify(token, &secret).is_err() {
            return Ok(http::Response::builder()
                .status(http::StatusCode::UNAUTHORIZED)
                .body(Body::from(r#"{"error":"unauthorized"}"#))
                .unwrap_or_default());
        }
    }

    // Forward to Durable Object
    let namespace = env.durable_object("PRACTICE_SESSION")?;
    let stub = namespace.id_from_name(session_id)?.get_stub()?;

    // Build worker::Request from the http::Request
    let worker_url = format!(
        "https://dummy/ws?sessionId={session_id}&studentId={student_id}&{}",
        query
    );
    let worker_req = worker::Request::new(&worker_url, worker::Method::Get)?;
    // Copy upgrade headers
    let worker_resp = stub.fetch_with_request(worker_req).await?;

    // Convert worker::Response to http::Response
    // This is the WebSocket upgrade response -- pass through directly
    Ok(worker_resp.into())
}

/// Streaming chat handler. Auth validated manually. Returns worker::Response for streaming.
async fn handle_chat_stream(
    env: &Env,
    req: HttpRequest,
) -> worker::Result<http::Response<Body>> {
    let headers = req.headers().clone();

    // Auth check
    let token = extract_token(&headers);
    let student_id = if let Some(token) = &token {
        let secret = env
            .secret("JWT_SECRET")
            .map(|s| s.to_string().into_bytes())
            .map_err(|e| worker::Error::RustError(format!("JWT_SECRET: {e}")))?;
        match crate::auth::jwt::verify(token, &secret) {
            Ok(claims) => claims.sub,
            Err(_) => {
                return Ok(http::Response::builder()
                    .status(http::StatusCode::UNAUTHORIZED)
                    .body(Body::from(r#"{"error":"unauthorized"}"#))
                    .unwrap_or_default());
            }
        }
    } else {
        return Ok(http::Response::builder()
            .status(http::StatusCode::UNAUTHORIZED)
            .body(Body::from(r#"{"error":"unauthorized"}"#))
            .unwrap_or_default());
    };

    // Collect body
    let body_bytes = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();

    // Delegate to existing streaming handler
    let resp = crate::services::chat::handle_chat_stream(env, &headers, &body_bytes).await;

    // Add CORS headers to the worker::Response
    let allowed_origin = env
        .var("ALLOWED_ORIGIN")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "http://localhost:3000".to_string());

    // Convert worker::Response -> http::Response<Body>
    Ok(resp.into())
}

fn extract_token(headers: &http::HeaderMap) -> Option<String> {
    // Try cookie first
    headers
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|c| {
            c.split(';')
                .find_map(|p| p.trim().strip_prefix("session=").map(String::from))
        })
        .or_else(|| {
            // Then Bearer header
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|a| a.strip_prefix("Bearer "))
                .map(String::from)
        })
}
```

Note: The WS upgrade and streaming chat carve-outs are approximate -- they'll need adjustment during implementation based on the exact `worker::Response` to `http::Response` conversion available in worker 0.7. The key principle is: these two paths bypass the Axum Router and handle auth + CORS manually.

- [ ] **Step 3: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

This will likely show errors for handler functions that haven't been migrated yet (they still have old signatures). That's expected -- we'll fix them in the next tasks.

- [ ] **Step 4: Commit**

```bash
git add src/routes.rs src/server.rs
git commit -m "feat: add Axum Router and rewrite server.rs entry point"
```

---

## Task 7: Migrate Service Handlers (Part 1 -- Small Handlers)

**Files:**
- Modify: `apps/api/src/services/waitlist.rs`
- Modify: `apps/api/src/services/goals.rs`
- Modify: `apps/api/src/services/scores.rs`
- Modify: `apps/api/src/services/sync.rs`
- Modify: `apps/api/src/services/exercises.rs`

Migrate the simpler service handlers to Axum signatures. Each handler changes from `(env: &Env, headers: &HeaderMap, body: &[u8]) -> Response<Body>` to `(State(state): State<AppState>, auth: AuthUser, Json(payload): Json<T>) -> Result<Json<R>>`.

- [ ] **Step 1: Migrate `services/waitlist.rs`**

Read the current file. Change:
- Signature: `pub async fn handle_waitlist(env: &Env, body: &[u8])` -> `#[worker::send] pub async fn handle_waitlist(State(state): State<AppState>, Json(payload): Json<WaitlistRequest>) -> Result<Json<serde_json::Value>>`
- No auth required (unauthenticated endpoint)
- Add serde attributes to `WaitlistRequest`
- Replace `Response::builder()` with `Result` + `Json`
- Replace `env.d1("DB")` with `state.db.d1()?`
- Replace `.map_err(|e| format!(...))` with `.map_err(|e| ApiError::Internal(format!(...)))`

- [ ] **Step 2: Migrate `services/goals.rs`**

Read the current file. Change:
- Signature: add `State(state)`, `auth: AuthUser`, `Json(payload)`
- Replace `verify_auth_header` call with `auth.student_id`
- Replace `Response::builder()` with `Result` + `Json`
- Add serde attributes

- [ ] **Step 3: Migrate `services/scores.rs`**

Same pattern. Read the current file and apply the same transformations.

- [ ] **Step 4: Migrate `services/sync.rs`**

Same pattern. The sync endpoint has a complex request body -- add `#[serde(rename_all = "camelCase")]` to `SyncRequest`, `StudentDelta`, `SessionDelta`, and `SyncResponse`.

- [ ] **Step 5: Migrate `services/exercises.rs`**

Three handlers here: `handle_exercises`, `handle_assign_exercise`, `handle_complete_exercise`. All get the same treatment.

- [ ] **Step 6: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

- [ ] **Step 7: Commit**

```bash
git add src/services/waitlist.rs src/services/goals.rs src/services/scores.rs src/services/sync.rs src/services/exercises.rs
git commit -m "refactor: migrate small service handlers to Axum signatures"
```

---

## Task 8: Migrate Service Handlers (Part 2 -- Complex Handlers)

**Files:**
- Modify: `apps/api/src/services/ask.rs`
- Modify: `apps/api/src/services/chat.rs`
- Modify: `apps/api/src/services/memory.rs`
- Modify: `apps/api/src/services/llm.rs`
- Modify: `apps/api/src/services/teaching_moment_handler.rs`

These are the larger, more complex handlers.

- [ ] **Step 1: Migrate `services/llm.rs`**

This file doesn't have HTTP handlers -- it has internal functions called by other handlers. Change:
- All `Result<T, String>` return types to `Result<T>` (using `crate::error::Result`)
- Replace `Err(format!(...))` with `Err(ApiError::ExternalService(format!(...)))`
- Keep function signatures internal (not Axum handlers)

- [ ] **Step 2: Migrate `services/ask.rs`**

Read the current file (839 lines). This is the two-stage LLM pipeline.
- `handle_ask` and `handle_elaborate` get Axum handler signatures
- `handle_ask_inner` stays as an internal function but returns `Result<AskInnerResponse>`
- Add serde attributes to `AskRequest`, `AskResponse`, `ElaborateRequest`, `ElaborateResponse`
- Replace all `env: &Env` parameters with `state: &AppState` in internal functions
- Replace `crate::services::llm::call_groq(env, ...)` with `crate::services::llm::call_groq(state.inference.env(), ...)`

- [ ] **Step 3: Migrate `services/chat.rs`**

This file has both HTTP handlers and the streaming handler.
- `list_conversations` and `get_messages` get Axum handler signatures
- `delete_conversation` gets an Axum handler signature
- `handle_chat_stream` keeps its old signature (it's a carve-out, called directly from server.rs with raw `&Env`)
- Add serde attributes to `ConversationSummary`, `ConversationDetail`, `MessageRow`, etc.

- [ ] **Step 4: Migrate `services/memory.rs`**

This is the largest service file (1950 lines). It has both HTTP handlers and internal query functions.
- HTTP handlers (`handle_extract_chat`, `handle_store_facts`, `handle_search_facts`, `handle_clear_benchmark`, `handle_synthesize`, `handle_seed_observations`) get Axum handler signatures
- Internal query functions (`query_active_facts`, `query_recent_observations_with_engagement`, etc.) change from `Result<T, String>` to `Result<T>`
- Replace all `env.d1("DB").map_err(...)` with proper `ApiError` conversions

- [ ] **Step 5: Migrate `services/teaching_moment_handler.rs`**

Small file. Change `Result<T, String>` to `Result<T>`.

- [ ] **Step 6: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

- [ ] **Step 7: Commit**

```bash
git add src/services/
git commit -m "refactor: migrate complex service handlers to Axum signatures and ApiError"
```

---

## Task 9: Migrate Practice Handlers

**Files:**
- Modify: `apps/api/src/practice/handlers/start.rs`
- Modify: `apps/api/src/practice/handlers/upload.rs`
- Modify: `apps/api/src/practice/session/mod.rs` (DO internal error handling)
- Modify: `apps/api/src/practice/session/*.rs` (Result<T, String> -> PracticeError)

- [ ] **Step 1: Migrate `practice/handlers/start.rs`**

Change signature to Axum:
```rust
#[worker::send]
pub async fn handle_start(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(payload): Json<StartRequest>,
) -> Result<Json<StartResponse>> { ... }
```

- [ ] **Step 2: Migrate `practice/handlers/upload.rs`**

Change signature to Axum. Note: upload uses path params for session_id and chunk_index -- use `Path` extractor:
```rust
#[worker::send]
pub async fn handle_upload(
    State(state): State<AppState>,
    auth: AuthUser,
    Path((session_id, chunk_index)): Path<(String, String)>,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>> { ... }
```

Update `routes.rs` to use a path parameter:
```rust
.route("/api/practice/upload/:session_id/:chunk_index", post(practice::handlers::upload::handle_upload))
```

- [ ] **Step 3: Migrate DO session internals to `PracticeError`**

In `practice/session/inference.rs`, `processing.rs`, `finalization.rs`, `synthesis.rs`:
- Change `Result<T, String>` to `Result<T, PracticeError>` (using the session-local error type)
- Replace `Err(format!(...))` with `Err(PracticeError::Inference(format!(...)))` etc.

The DO's WebSocket handlers catch these errors and log them via `console_error!` -- they don't need to convert to HTTP responses.

- [ ] **Step 4: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -30
```

- [ ] **Step 5: Commit**

```bash
git add src/practice/
git commit -m "refactor: migrate practice handlers to Axum + PracticeError in DO internals"
```

---

## Task 10: Eliminate `.unwrap()` and Apply Clippy

**Files:**
- All files under `apps/api/src/`

- [ ] **Step 1: Run clippy and capture all warnings**

```bash
cd apps/api
cargo clippy --target wasm32-unknown-unknown 2>&1 | grep "warning\[" | sort | uniq -c | sort -rn | head -30
```

This shows the most common warning categories. Focus on:
- `clippy::unwrap_used` -- replace with `?` or `.unwrap_or_default()` or `.ok_or(ApiError::...)?`
- `clippy::expect_used` -- same treatment
- Other pedantic warnings -- fix as appropriate

- [ ] **Step 2: Fix all `unwrap_used` warnings in handler/service code**

For each `.unwrap()`:
- If on `Response::builder()` -- these should all be gone after the migration to `IntoResponse`. If any remain, replace with `.unwrap_or_default()`.
- If on `serde_json::to_string()` -- replace with `?` (now that handlers return `Result`)
- If on `getrandom` -- `.unwrap_or_default()` is acceptable
- If on `JsValue` conversions -- `.map_err(|e| ApiError::Internal(...))?`

- [ ] **Step 3: Fix remaining clippy pedantic warnings**

Address warnings category by category. Common fixes:
- `needless_pass_by_value` -- change `String` params to `&str` where applicable
- `redundant_closure_for_method_calls` -- simplify closures
- `cast_possible_truncation` -- add explicit casts with comments
- `similar_names` -- rename variables for clarity

- [ ] **Step 4: Run clippy clean**

```bash
cd apps/api
cargo clippy --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -5
```

Expected: zero warnings.

- [ ] **Step 5: Run rustfmt**

```bash
cd apps/api
cargo fmt
```

- [ ] **Step 6: Verify fmt is clean**

```bash
cargo fmt -- --check
```

Expected: no diff.

- [ ] **Step 7: Commit**

```bash
git add -A src/
git commit -m "refactor: eliminate unwrap() calls and fix clippy pedantic warnings"
```

---

## Task 11: Serde Standardization Pass

**Files:**
- All struct definitions with `Serialize`/`Deserialize` in `services/`, `auth/`, `practice/handlers/`

- [ ] **Step 1: Add `rename_all = "camelCase"` to all API-facing request/response structs**

Search for all `#[derive(` lines that include `Serialize` or `Deserialize` in:
- `src/auth/handlers.rs`
- `src/services/ask.rs`
- `src/services/chat.rs` (non-streaming types only)
- `src/services/exercises.rs`
- `src/services/goals.rs`
- `src/services/memory.rs` (HTTP handler request/response types only, NOT D1 row types)
- `src/services/scores.rs`
- `src/services/sync.rs`
- `src/services/waitlist.rs`

Add `#[serde(rename_all = "camelCase")]` to each.

**Do NOT add `rename_all` to:**
- D1 row structs (e.g., `MessageRow` which reads from D1 with snake_case columns)
- DO state structs (internal, stored in durable storage)
- Types in `practice/session/` (internal)

- [ ] **Step 2: Add `deny_unknown_fields` to request structs**

Add `#[serde(deny_unknown_fields)]` to all `Deserialize`-only structs that represent incoming API requests.

- [ ] **Step 3: Add `skip_serializing_if` to optional response fields**

Add `#[serde(skip_serializing_if = "Option::is_none")]` to all `Option<T>` fields on `Serialize` structs.

- [ ] **Step 4: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -10
```

- [ ] **Step 5: Commit**

```bash
git add -A src/
git commit -m "refactor: standardize serde attributes (camelCase, deny_unknown_fields, skip_serializing_if)"
```

---

## Task 12: Propagate Newtypes

**Files:**
- All files that use `student_id: String`, `session_id: String`, `conversation_id: String`

- [ ] **Step 1: Replace `String` with `StudentId` in handler/service signatures**

Search for `student_id: &str` and `student_id: String` in function parameters. Replace with `student_id: &StudentId` or `student_id: StudentId`. Use `student_id.as_str()` where a `&str` is needed (e.g., D1 query bindings).

Start with `AuthService`, `DbService`, then propagate to handler functions.

- [ ] **Step 2: Replace `String` with `SessionId` and `ConversationId`**

Same treatment for `session_id` and `conversation_id` throughout.

- [ ] **Step 3: Verify it compiles**

```bash
cd apps/api
cargo check --target wasm32-unknown-unknown 2>&1 | head -10
```

- [ ] **Step 4: Commit**

```bash
git add -A src/
git commit -m "refactor: propagate StudentId, SessionId, ConversationId newtypes"
```

---

## Task 13: Final Verification

**Files:**
- None modified -- verification only

- [ ] **Step 1: Full build**

```bash
cd apps/api
cargo build --target wasm32-unknown-unknown --release 2>&1 | tail -5
```

Expected: success.

- [ ] **Step 2: Clippy clean**

```bash
cargo clippy --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -5
```

Expected: zero warnings.

- [ ] **Step 3: Format clean**

```bash
cargo fmt -- --check
```

Expected: no diff.

- [ ] **Step 4: Binary size check**

```bash
ls -la target/wasm32-unknown-unknown/release/*.wasm | awk '{print $5, $NF}'
```

Compare to baseline from Task 1. Must be within 10%.

- [ ] **Step 5: Verify zero `Result<T, String>` in handler/service code**

```bash
cd apps/api
grep -rn "Result<.*String>" src/services/ src/auth/handlers.rs src/auth/extractor.rs src/practice/handlers/ | grep -v "// legacy" | head -20
```

Expected: zero matches.

- [ ] **Step 6: Verify zero `Response::builder()` in handlers**

```bash
grep -rn "Response::builder()" src/services/ src/auth/handlers.rs src/practice/handlers/ | head -20
```

Expected: zero matches (may still exist in `server.rs` carve-outs, which is acceptable).

- [ ] **Step 7: Run smoke test against dev server**

Start the dev server in one terminal:
```bash
just api
```

In another terminal:
```bash
cd apps/api
uv run python tests/smoke_test.py --compare
```

Expected: "No regressions detected vs baseline."

- [ ] **Step 8: Commit any remaining fixes**

```bash
git add -A
git commit -m "fix: final verification fixes for API refactor"
```

---

## Execution Order and Dependencies

```
Task 1: Smoke Test + Dependencies
  |
Task 2: Error Types + Newtypes
  |
Task 3: Service Layer + AppState
  |
Task 4: Auth Extractor + Auth Handlers
  |
Task 5: Practice Module Restructure
  |
Task 6: Router + Server Entry Point
  |
  +-- Task 7: Small Service Handlers (waitlist, goals, scores, sync, exercises)
  |
  +-- Task 8: Complex Service Handlers (ask, chat, memory, llm)
  |
Task 9: Practice Handlers
  |
Task 10: Eliminate .unwrap() + Clippy
  |
Task 11: Serde Standardization
  |
Task 12: Propagate Newtypes
  |
Task 13: Final Verification
```

Tasks 7 and 8 can run in parallel (they touch different files). All other tasks are sequential.
