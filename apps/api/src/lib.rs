use worker::*;

mod handlers;
mod models;
mod services;
mod utils;

use handlers::{handle_analyze, handle_get_performance, handle_list_performances};
use utils::response::cors_preflight;

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Create router
    let router = Router::new();

    router
        // CORS preflight for all routes
        .options("/api/*catchall", |_req, _ctx| cors_preflight())
        // Performance endpoints
        .get_async("/api/performances", handle_list_performances)
        .get_async("/api/performances/:id", handle_get_performance)
        // Analysis endpoint
        .post_async("/api/analyze/:id", handle_analyze)
        // Health check
        .get("/health", |_req, _ctx| {
            Response::ok("OK")
        })
        // 404 fallback
        .or_else_any_method("/*catchall", |_req, _ctx| {
            utils::error_response("Not found", 404)
        })
        .run(req, env)
        .await
}
