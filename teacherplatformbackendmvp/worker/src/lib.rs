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

    // Create router
    let router = Router::with_data(env);

    router
        // Health check
        .get("/health", |_, ctx| {
            Response::ok("Worker is healthy")
        })

        // RAG query endpoint (with caching)
        .post_async("/api/chat/query", routes::rag_query)

        // Project presigned URLs (R2 direct access)
        .post_async("/api/projects/upload-url", routes::generate_upload_url)
        .get_async("/api/projects/:id/download-url", routes::generate_download_url)

        // Embedding endpoint (with caching)
        .post_async("/api/embeddings/generate", routes::generate_embedding)

        // Proxy to GCP API for complex operations
        .on_async("/api/*", routes::proxy_to_gcp)

        // 404 handler
        .or_else_any_method("/*", |_, _| {
            Response::error("Not Found", 404)
        })
        .run(req, env)
        .await
        .map(|mut res| {
            res.with_cors(&cors)
        })
}
