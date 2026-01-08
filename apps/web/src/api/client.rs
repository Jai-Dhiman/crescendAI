use crate::models::{AnalysisResult, Performance};
use gloo_net::http::Request;
use serde::{Deserialize, Serialize};

const API_BASE_URL: &str = "/api";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ApiError {
    NetworkError(String),
    ParseError(String),
    NotFound,
    ServerError(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ApiError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ApiError::NotFound => write!(f, "Not found"),
            ApiError::ServerError(msg) => write!(f, "Server error: {}", msg),
        }
    }
}

/// Fetch all performances from the API
pub async fn fetch_performances() -> Result<Vec<Performance>, ApiError> {
    let url = format!("{}/performances", API_BASE_URL);

    let response = Request::get(&url)
        .send()
        .await
        .map_err(|e| ApiError::NetworkError(e.to_string()))?;

    if !response.ok() {
        return Err(ApiError::ServerError(format!(
            "HTTP {}: {}",
            response.status(),
            response.status_text()
        )));
    }

    response
        .json::<Vec<Performance>>()
        .await
        .map_err(|e| ApiError::ParseError(e.to_string()))
}

/// Fetch a single performance by ID
pub async fn fetch_performance_by_id(id: &str) -> Result<Performance, ApiError> {
    let url = format!("{}/performances/{}", API_BASE_URL, id);

    let response = Request::get(&url)
        .send()
        .await
        .map_err(|e| ApiError::NetworkError(e.to_string()))?;

    if response.status() == 404 {
        return Err(ApiError::NotFound);
    }

    if !response.ok() {
        return Err(ApiError::ServerError(format!(
            "HTTP {}: {}",
            response.status(),
            response.status_text()
        )));
    }

    response
        .json::<Performance>()
        .await
        .map_err(|e| ApiError::ParseError(e.to_string()))
}

/// Analyze a performance by ID
pub async fn analyze_performance(id: &str) -> Result<AnalysisResult, ApiError> {
    let url = format!("{}/analyze/{}", API_BASE_URL, id);

    let response = Request::post(&url)
        .send()
        .await
        .map_err(|e| ApiError::NetworkError(e.to_string()))?;

    if response.status() == 404 {
        return Err(ApiError::NotFound);
    }

    if !response.ok() {
        return Err(ApiError::ServerError(format!(
            "HTTP {}: {}",
            response.status(),
            response.status_text()
        )));
    }

    response
        .json::<AnalysisResult>()
        .await
        .map_err(|e| ApiError::ParseError(e.to_string()))
}

/// Loading messages to cycle through during analysis
pub fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Analyzing articulation patterns...",
        "Evaluating pedal technique...",
        "Measuring dynamic range...",
        "Assessing timbral qualities...",
        "Examining phrasing and tempo...",
        "Detecting expressive nuances...",
        "Analyzing harmonic balance...",
        "Evaluating interpretive choices...",
        "Generating personalized feedback...",
    ]
}
