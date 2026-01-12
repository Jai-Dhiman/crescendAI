use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::json;

use crate::models::Performance;

pub async fn list_performances() -> impl IntoResponse {
    let performances = Performance::get_demo_performances();
    Json(performances)
}

pub async fn get_performance(Path(id): Path<String>) -> impl IntoResponse {
    match Performance::find_by_id(&id) {
        Some(performance) => (StatusCode::OK, Json(json!(performance))),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Performance not found"})),
        ),
    }
}
