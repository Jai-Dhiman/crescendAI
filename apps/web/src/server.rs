use axum::{
    routing::{get, post},
    Router,
};
use leptos::prelude::*;
use leptos_axum::{generate_route_list, LeptosRoutes};
use tower::ServiceExt;
use worker::{event, Context, Env, HttpRequest, Result};

use crate::{api, shell::shell, state::AppState, App};

fn router(app_state: AppState) -> Router {
    let leptos_options = app_state.leptos_options.clone();
    let routes = generate_route_list(App);

    // API routes with our AppState
    let api_router = Router::new()
        .route("/api/performances", get(api::list_performances))
        .route("/api/performances/:id", get(api::get_performance))
        .route("/api/analyze/:id", post(api::analyze_performance))
        .route("/health", get(|| async { "OK" }))
        .with_state(app_state.clone());

    // Leptos SSR routes - the context provides our state to components
    let leptos_router = Router::new()
        .leptos_routes_with_context(
            &leptos_options,
            routes,
            {
                let state = app_state.clone();
                move || provide_context(state.clone())
            },
            {
                let leptos_options = leptos_options.clone();
                move || shell(leptos_options.clone())
            },
        )
        .with_state(leptos_options);

    // Merge API routes first (so they take priority), then Leptos routes
    api_router.merge(leptos_router)
}

#[event(fetch)]
async fn fetch(
    req: HttpRequest,
    env: Env,
    _ctx: Context,
) -> Result<http::Response<axum::body::Body>> {
    console_error_panic_hook::set_once();

    let leptos_options = LeptosOptions::builder()
        .output_name("crescendai")
        .site_pkg_dir("pkg")
        .build();

    let state = AppState::new(env, leptos_options);
    let app = router(state);

    Ok(app.oneshot(req).await?)
}
