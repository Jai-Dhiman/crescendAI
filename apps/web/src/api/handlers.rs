use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::json;

use crate::models::{AnalysisResult, Performance};
use crate::services::{generate_teacher_feedback, get_performance_dimensions, get_practice_tips};

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

pub async fn analyze_performance(Path(id): Path<String>) -> impl IntoResponse {
    let performance = match Performance::find_by_id(&id) {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Performance not found"})),
            )
        }
    };

    let dimensions = get_performance_dimensions(&id).await;
    let practice_tips = get_practice_tips(&performance, &dimensions).await;
    let teacher_feedback = generate_teacher_feedback(&performance, &dimensions).await;

    let result = AnalysisResult {
        performance_id: id,
        dimensions,
        teacher_feedback,
        practice_tips,
    };

    (StatusCode::OK, Json(json!(result)))
}
