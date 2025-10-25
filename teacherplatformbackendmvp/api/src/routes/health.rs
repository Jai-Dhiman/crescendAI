use axum::{extract::State, routing::get, Json, Router};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{errors::AppError, state::AppState};

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    status: String,
    timestamp: String,
    database: String,
    version: String,
}

/// Health check endpoint
///
/// Returns the health status of the API and its dependencies
async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, AppError> {
    // Check database connectivity
    let db_status = match crate::db::health_check(&state.pool).await {
        Ok(_) => "connected".to_string(),
        Err(e) => {
            tracing::error!("Database health check failed: {:?}", e);
            "disconnected".to_string()
        }
    };

    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        timestamp: Utc::now().to_rfc3339(),
        database: db_status,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

/// Readiness probe for load balancers
///
/// Returns 200 if the service is ready to accept traffic
async fn readiness(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    // Check if database is accessible
    crate::db::health_check(&state.pool).await?;

    Ok(Json(serde_json::json!({
        "ready": true
    })))
}

/// Liveness probe for orchestration systems
///
/// Returns 200 if the service is alive
async fn liveness() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "alive": true
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health_check))
        .route("/readiness", get(readiness))
        .route("/liveness", get(liveness))
}
