//! Axum Router definition.
//!
//! Auth routes use new Axum-style handlers directly (from Task 4).
//! Service and practice routes use thin wrappers that delegate to existing
//! old-style handlers -- these wrappers will be removed in Tasks 7-9.

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router(state: AppState) -> Router {
    Router::new()
        // Health
        .route("/health", get(health))
        // Auth (new Axum handlers -- Task 4)
        .route("/api/auth/apple", post(crate::auth::handlers::handle_apple))
        .route(
            "/api/auth/google",
            post(crate::auth::handlers::handle_google),
        )
        .route("/api/auth/me", get(crate::auth::handlers::handle_me))
        .route(
            "/api/auth/signout",
            post(crate::auth::handlers::handle_signout),
        )
        .route(
            "/api/auth/debug",
            post(crate::auth::handlers::handle_debug),
        )
        // Services (temporary wrappers -- Tasks 7-9 will replace)
        .route("/api/waitlist", post(wrap_waitlist))
        .route("/api/extract-goals", post(wrap_extract_goals))
        .route("/api/sync", post(wrap_sync))
        .route("/api/conversations", get(wrap_list_conversations))
        .route(
            "/api/conversations/{id}",
            get(wrap_get_conversation).delete(wrap_delete_conversation),
        )
        .route("/api/exercises", get(wrap_exercises))
        .route("/api/exercises/assign", post(wrap_assign_exercise))
        .route("/api/exercises/complete", post(wrap_complete_exercise))
        // Memory
        .route("/api/memory/extract-chat", post(wrap_extract_chat))
        .route("/api/memory/store-facts", post(wrap_store_facts))
        .route("/api/memory/search", post(wrap_search_facts))
        .route("/api/memory/clear-benchmark", post(wrap_clear_benchmark))
        .route("/api/memory/synthesize", post(wrap_memory_synthesize))
        .route(
            "/api/memory/seed-observations",
            post(wrap_seed_observations),
        )
        // Practice
        .route("/api/practice/start", post(wrap_practice_start))
        .route("/api/practice/chunk", post(wrap_upload_chunk))
        .route(
            "/api/practice/needs-synthesis",
            get(wrap_check_needs_synthesis),
        )
        .route(
            "/api/practice/synthesize",
            post(wrap_deferred_synthesis),
        )
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[allow(clippy::unused_async)]
async fn health() -> impl IntoResponse {
    "OK"
}

// ---------------------------------------------------------------------------
// Wrapper helpers: service handlers (env, headers, body) pattern
// ---------------------------------------------------------------------------

/// Temporary wrapper -- will be removed when services/waitlist.rs is migrated (Task 7).
#[worker::send]
async fn wrap_waitlist(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::waitlist::handle_waitlist(state.db.env(), &body).await
}

/// Temporary wrapper -- will be removed when services/goals.rs is migrated (Task 7).
#[worker::send]
async fn wrap_extract_goals(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::goals::handle_extract_goals(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/sync.rs is migrated (Task 7).
#[worker::send]
async fn wrap_sync(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::sync::handle_sync(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/chat.rs is migrated (Task 8).
#[worker::send]
async fn wrap_list_conversations(
    State(state): State<AppState>,
    headers: http::HeaderMap,
) -> http::Response<axum::body::Body> {
    crate::services::chat::handle_list_conversations(state.db.env(), &headers).await
}

/// Temporary wrapper -- will be removed when services/chat.rs is migrated (Task 8).
#[worker::send]
async fn wrap_get_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: http::HeaderMap,
) -> http::Response<axum::body::Body> {
    crate::services::chat::handle_get_conversation(state.db.env(), &headers, &id).await
}

/// Temporary wrapper -- will be removed when services/chat.rs is migrated (Task 8).
#[worker::send]
async fn wrap_delete_conversation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: http::HeaderMap,
) -> http::Response<axum::body::Body> {
    crate::services::chat::handle_delete_conversation(state.db.env(), &headers, &id).await
}

/// Temporary wrapper -- will be removed when services/exercises.rs is migrated (Task 7).
#[worker::send]
async fn wrap_exercises(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    query: axum::extract::RawQuery,
) -> http::Response<axum::body::Body> {
    let qs = query.0.unwrap_or_default();
    crate::services::exercises::handle_exercises(state.db.env(), &headers, &qs).await
}

/// Temporary wrapper -- will be removed when services/exercises.rs is migrated (Task 7).
#[worker::send]
async fn wrap_assign_exercise(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::exercises::handle_assign_exercise(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/exercises.rs is migrated (Task 7).
#[worker::send]
async fn wrap_complete_exercise(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::exercises::handle_complete_exercise(state.db.env(), &headers, &body).await
}

// ---------------------------------------------------------------------------
// Memory wrappers (all env, headers, body)
// ---------------------------------------------------------------------------

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_extract_chat(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_extract_chat(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_store_facts(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_store_facts(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_search_facts(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_search_facts(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_clear_benchmark(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_clear_benchmark(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_memory_synthesize(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_synthesize(state.db.env(), &headers, &body).await
}

/// Temporary wrapper -- will be removed when services/memory.rs is migrated (Task 8).
#[worker::send]
async fn wrap_seed_observations(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::services::memory::handle_seed_observations(state.db.env(), &headers, &body).await
}

// ---------------------------------------------------------------------------
// Practice wrappers
// ---------------------------------------------------------------------------

/// Temporary wrapper -- will be removed when practice handlers are migrated (Task 9).
#[worker::send]
async fn wrap_practice_start(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    crate::practice::handlers::start::handle_start(state.practice.env(), &headers, &body).await
}

/// Query params for chunk upload.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkParams {
    session_id: Option<String>,
    chunk_index: Option<String>,
}

/// Temporary wrapper -- will be removed when practice handlers are migrated (Task 9).
#[worker::send]
async fn wrap_upload_chunk(
    State(state): State<AppState>,
    Query(params): Query<ChunkParams>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    let session_id = params.session_id.unwrap_or_default();
    if session_id.is_empty() {
        return http::Response::builder()
            .status(http::StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(
                r#"{"error":"Missing sessionId"}"#,
            ))
            .expect("response builder");
    }
    let chunk_index = params.chunk_index.unwrap_or_else(|| "0".to_string());
    crate::practice::handlers::upload::handle_upload_chunk(
        state.practice.env(),
        &headers,
        body.to_vec(),
        &session_id,
        &chunk_index,
    )
    .await
}

/// Query params for needs-synthesis check.
#[derive(serde::Deserialize)]
struct NeedsSynthesisParams {
    conversation_id: Option<String>,
}

/// Temporary wrapper -- will be removed when practice handlers are migrated (Task 9).
#[worker::send]
async fn wrap_check_needs_synthesis(
    State(state): State<AppState>,
    Query(params): Query<NeedsSynthesisParams>,
    headers: http::HeaderMap,
) -> http::Response<axum::body::Body> {
    let conv_id = params.conversation_id.unwrap_or_default();
    crate::practice::session::synthesis::handle_check_needs_synthesis(
        state.practice.env(),
        &headers,
        &conv_id,
    )
    .await
}

/// Temporary wrapper -- will be removed when practice handlers are migrated (Task 9).
#[worker::send]
async fn wrap_deferred_synthesis(
    State(state): State<AppState>,
    headers: http::HeaderMap,
    body: axum::body::Bytes,
) -> http::Response<axum::body::Body> {
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    crate::practice::session::synthesis::handle_deferred_synthesis(
        state.practice.env(),
        &headers,
        &body_json,
    )
    .await
}
