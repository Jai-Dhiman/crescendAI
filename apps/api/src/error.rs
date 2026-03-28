//! Centralized API error types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use worker::console_error;

/// API-level error type. Maps to HTTP status codes via `IntoResponse`.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("unauthorized")]
    Unauthorized,

    #[error("forbidden")]
    Forbidden,

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("internal: {0}")]
    Internal(String),

    #[error("external service: {0}")]
    ExternalService(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ApiError>;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            Self::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            Self::Forbidden => (StatusCode::FORBIDDEN, "forbidden"),
            Self::InferenceFailed(_) => (StatusCode::BAD_GATEWAY, "inference_failed"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
            Self::ExternalService(_) => (StatusCode::BAD_GATEWAY, "external_error"),
        };

        if status.is_server_error() {
            console_error!("{}: {}", error_type, &self);
        }

        let body = serde_json::json!({
            "error": error_type,
            "message": self.to_string(),
        });

        (status, axum::Json(body)).into_response()
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        Self::BadRequest(format!("JSON: {e}"))
    }
}

impl From<worker::Error> for ApiError {
    fn from(e: worker::Error) -> Self {
        Self::Internal(format!("worker: {e}"))
    }
}
