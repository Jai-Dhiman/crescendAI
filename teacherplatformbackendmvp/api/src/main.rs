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

    // Create app state
    let state = state::AppState::new(pool, config.clone());

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
