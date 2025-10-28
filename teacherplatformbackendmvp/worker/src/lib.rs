use worker::*;

mod cache;
mod routes;
mod utils;

#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Initialize panic hook for better error messages
    utils::set_panic_hook();

    // CORS headers for development
    let cors = Cors::new()
        .with_origins(vec!["*"])
        .with_methods(vec![Method::Get, Method::Post, Method::Put, Method::Delete, Method::Options])
        .with_allowed_headers(vec!["Content-Type", "Authorization"]);

    // Create router with Env data
    let router = Router::with_data(env.clone());

    let response = router
        // Health check
        .get("/health", |_, _ctx| {
            Response::ok("Worker is healthy")
        })

        // RAG query endpoint (with caching)
        .post_async("/api/chat/query", routes::rag_query)

        // Project streaming (R2 binding - zero latency)
        .get_async("/api/projects/:id/stream", routes::stream_project)

        // Embedding endpoint (with caching)
        .post_async("/api/embeddings/generate", routes::generate_embedding)

        // Proxy to GCP API for all other operations
        .on_async("/api/*", routes::proxy_to_gcp)

        // 404 handler
        .or_else_any_method("/*", |_, _| {
            Response::error("Not Found", 404)
        })
        .run(req, env)
        .await?;

    Ok(response.with_cors(&cors)?)
}
