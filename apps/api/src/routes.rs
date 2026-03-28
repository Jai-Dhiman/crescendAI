//! Axum Router definition.
//!
//! Auth and service routes use native Axum-style handlers (Tasks 4, 7, 8).
//! Practice routes still use thin wrappers -- Task 9 will migrate.

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;

use crate::state::AppState;

use crate::services::{ask, chat, exercises, goals, memory, scores, sync, teaching_moment_handler, waitlist};

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
        // Services -- migrated to native Axum handlers (Task 7)
        .route("/api/waitlist", post(waitlist::handle_waitlist))
        .route("/api/extract-goals", post(goals::handle_extract_goals))
        .route("/api/sync", post(sync::handle_sync))
        .route("/api/scores", get(scores::handle_list_pieces))
        .route("/api/scores/{piece_id}", get(scores::handle_get_piece))
        .route(
            "/api/scores/{piece_id}/data",
            get(scores::handle_get_piece_data),
        )
        .route("/api/exercises", get(exercises::handle_exercises))
        .route(
            "/api/exercises/assign",
            post(exercises::handle_assign_exercise),
        )
        .route(
            "/api/exercises/complete",
            post(exercises::handle_complete_exercise),
        )
        // Ask / Elaborate (Task 8)
        .route("/api/ask", post(ask::handle_ask))
        .route("/api/ask/elaborate", post(ask::handle_elaborate))
        // Teaching moment (Task 8)
        .route(
            "/api/practice/teaching-moment",
            post(teaching_moment_handler::handle_teaching_moment),
        )
        // Conversations (Task 8)
        .route("/api/conversations", get(chat::handle_list_conversations))
        .route(
            "/api/conversations/{id}",
            get(chat::handle_get_conversation).delete(chat::handle_delete_conversation),
        )
        // Memory (Task 8)
        .route("/api/memory/extract-chat", post(memory::handle_extract_chat))
        .route("/api/memory/store-facts", post(memory::handle_store_facts))
        .route("/api/memory/search", post(memory::handle_search_facts))
        .route("/api/memory/clear-benchmark", post(memory::handle_clear_benchmark))
        .route("/api/memory/synthesize", post(memory::handle_synthesize))
        .route(
            "/api/memory/seed-observations",
            post(memory::handle_seed_observations),
        )
        // Practice -- temporary wrappers (Task 9 will replace)
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
// Practice wrappers (Task 9 will remove)
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
