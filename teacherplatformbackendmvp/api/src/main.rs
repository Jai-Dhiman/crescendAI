mod ai;
mod auth;
mod cache;
mod config;
mod db;
mod errors;
mod ingestion;
mod llm;
mod models;
mod routes;
mod search;
mod state;
mod storage;
mod utils;

use anyhow::Result;
use std::time::Duration;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging
    init_tracing();

    tracing::info!("Starting Piano API server...");

    // Load configuration
    let config = config::Config::from_env()?;
    tracing::info!(
        "Loaded configuration: server={}:{}",
        config.server.host,
        config.server.port
    );

    // Create database connection pool
    let pool = db::create_pool(&config.database.url, config.database.max_connections).await?;

    // Clean up stale processing documents on startup
    // Any documents in "processing" state when server starts are orphaned from crashed jobs
    let stale_count = sqlx::query(
        r#"
        UPDATE knowledge_base_docs
        SET status = 'failed',
            error_message = 'Processing interrupted by server restart'
        WHERE status = 'processing'
        "#,
    )
    .execute(&pool)
    .await?
    .rows_affected();

    if stale_count > 0 {
        tracing::warn!(
            "Marked {} stale processing documents as failed on startup",
            stale_count
        );
    }

    // Initialize Workers AI client if credentials are available
    let workers_ai = if let (Some(account_id), Some(api_token)) = (
        config.cloudflare.account_id.clone(),
        config.cloudflare.workers_ai_api_token.clone(),
    ) {
        tracing::info!("Initializing Cloudflare Workers AI client");
        Some(ai::workers_ai::WorkersAIClient::new(account_id, api_token))
    } else {
        tracing::warn!("Cloudflare Workers AI credentials not found - embeddings and LLM will not work");
        None
    };

    // Initialize cache service (optional, requires KV namespace IDs)
    let cache = if let (Some(account_id), Some(api_token)) = (
        config.cloudflare.account_id.clone(),
        config.cloudflare.workers_ai_api_token.clone(),
    ) {
        // Try to get KV namespace IDs from environment
        let embedding_ns = std::env::var("CLOUDFLARE_KV_EMBEDDING_NAMESPACE_ID").ok();
        let search_ns = std::env::var("CLOUDFLARE_KV_SEARCH_NAMESPACE_ID").ok();
        let llm_ns = std::env::var("CLOUDFLARE_KV_LLM_NAMESPACE_ID").ok();

        if let (Some(emb), Some(search), Some(llm)) = (embedding_ns, search_ns, llm_ns) {
            let kv_client = cache::KVClient::new(account_id, api_token, emb, search, llm);
            tracing::info!("Workers KV caching enabled");
            cache::CacheService::new(Some(kv_client), config.cache.clone())
        } else {
            tracing::warn!("Workers KV namespace IDs not found - caching disabled");
            cache::CacheService::new(None, config.cache.clone())
        }
    } else {
        tracing::info!("Caching disabled (no Cloudflare credentials)");
        cache::CacheService::new(None, config.cache.clone())
    };

    // Initialize R2 client for presigned URL generation
    tracing::info!("Initializing R2 client for presigned URL generation");
    let r2_client = storage::R2Client::new(&config.cloudflare).await?;
    tracing::info!("R2 client initialized successfully");

    // Create app state
    let state = state::AppState::new(pool, config.clone(), workers_ai, cache, r2_client);

    // Build router with middleware
    let app = routes::create_router(state)
        .layer(
            ServiceBuilder::new()
                // Logging layer
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                        .on_response(DefaultOnResponse::new().level(Level::INFO)),
                )
                // CORS layer
                .layer(CorsLayer::permissive()) // TODO: Configure CORS properly for production
                // Compression layer
                .layer(CompressionLayer::new()),
        );

    // Start server
    let addr = config.server_address();
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("Health check available at http://{}/api/health", addr);

    axum::serve(listener, app)
        .await?;

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "piano_api=debug,tower_http=debug,axum=trace".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}
