//! Axum Router definition.
//!
//! Auth, service, and practice routes all use native Axum-style handlers.

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
        // Auth (Task 4)
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
        // Services (Task 7)
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
        // Practice (Task 9)
        .route(
            "/api/practice/start",
            post(crate::practice::handlers::start::handle_start),
        )
        .route(
            "/api/practice/chunk",
            post(crate::practice::handlers::upload::handle_upload_chunk),
        )
        .route(
            "/api/practice/needs-synthesis",
            get(crate::practice::session::synthesis::handle_check_needs_synthesis),
        )
        .route(
            "/api/practice/synthesize",
            post(crate::practice::session::synthesis::handle_deferred_synthesis),
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
