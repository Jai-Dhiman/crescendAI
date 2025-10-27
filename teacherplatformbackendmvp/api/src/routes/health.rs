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

/// Health check
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
async fn readiness(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    crate::db::health_check(&state.pool).await?;

    Ok(Json(serde_json::json!({
        "ready": true
    })))
}

/// Liveness probe for orchestration systems
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
