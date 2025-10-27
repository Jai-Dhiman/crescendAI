pub mod annotations;
pub mod auth;
pub mod chat;
pub mod health;
pub mod knowledge;
pub mod projects;
pub mod relationships;

use axum::{middleware, Router};

use crate::state::AppState;

/// Creates the main API router with all routes
pub fn create_router(state: AppState) -> Router {
    Router::new().nest("/api", api_routes(state))
}

/// API routes under /api prefix
fn api_routes(state: AppState) -> Router {
    // Public routes
    let public = Router::new()
        .merge(health::routes())
        .route("/auth/register", axum::routing::post(auth::register))
        .route("/auth/login", axum::routing::post(auth::login))
        .route("/auth/refresh", axum::routing::post(auth::refresh));

    // Protected routes (require authentication)
    let protected = Router::new()
        .route("/auth/me", axum::routing::get(auth::me))
        .route(
            "/relationships",
            axum::routing::post(relationships::create_relationship)
                .get(relationships::list_relationships),
        )
        .route(
            "/relationships/:id",
            axum::routing::delete(relationships::delete_relationship),
        )
        // Knowledge base routes
        .route(
            "/knowledge",
            axum::routing::post(knowledge::create_knowledge_doc)
                .get(knowledge::list_knowledge_docs),
        )
        .route(
            "/knowledge/:id",
            axum::routing::get(knowledge::get_knowledge_doc)
                .delete(knowledge::delete_knowledge_doc),
        )
        .route(
            "/knowledge/:id/process",
            axum::routing::post(knowledge::process_knowledge_doc),
        )
        .route(
            "/knowledge/:id/status",
            axum::routing::get(knowledge::get_processing_status),
        )
        // Project routes
        .route(
            "/projects",
            axum::routing::post(projects::create_project)
                .get(projects::list_projects),
        )
        .route(
            "/projects/:id",
            axum::routing::get(projects::get_project)
                .patch(projects::update_project)
                .delete(projects::delete_project),
        )
        .route(
            "/projects/:id/confirm",
            axum::routing::post(projects::confirm_upload),
        )
        .route(
            "/projects/:id/access",
            axum::routing::post(projects::grant_access)
                .get(projects::list_project_access),
        )
        .route(
            "/projects/:id/access/:user_id",
            axum::routing::delete(projects::revoke_access),
        )
        // Annotation routes
        .route(
            "/annotations",
            axum::routing::post(annotations::create_annotation)
                .get(annotations::list_annotations),
        )
        .route(
            "/annotations/:id",
            axum::routing::get(annotations::get_annotation)
                .patch(annotations::update_annotation)
                .delete(annotations::delete_annotation),
        )
        // Chat/RAG routes
        .route(
            "/chat/query",
            axum::routing::post(chat::rag_query),
        )
        .route(
            "/chat/health",
            axum::routing::get(chat::chat_health),
        )
        .route(
            "/chat/sessions",
            axum::routing::post(chat::create_session)
                .get(chat::list_sessions),
        )
        .route(
            "/chat/sessions/:id",
            axum::routing::get(chat::get_session)
                .delete(chat::delete_session),
        )
        .route(
            "/chat/messages",
            axum::routing::post(chat::store_message),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            crate::auth::middleware::auth_required,
        ));

    public.merge(protected).with_state(state)
}
