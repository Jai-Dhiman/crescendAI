//! Comprehensive Error Handling System
//!
//! This module provides a centralized error handling system with detailed error types,
//! structured error responses, and logging capabilities.

use worker::*;
use serde::{Serialize, Deserialize};

// ============================================================================
// Application Error Types
// ============================================================================

/// Comprehensive application error type
#[derive(Debug, Clone)]
pub enum AppError {
    /// Database operation failed
    DatabaseError(String),

    /// Storage operation failed (R2)
    StorageError(String),

    /// Input validation failed
    ValidationError(String),

    /// Dedalus API error
    DedalusError(String),

    /// Resource not found
    NotFound(String),

    /// Unauthorized access
    Unauthorized(String),

    /// Rate limit exceeded
    RateLimitExceeded,

    /// Internal server error
    InternalError(String),

    /// External service error
    ExternalServiceError(String),

    /// Configuration error
    ConfigurationError(String),

    /// Serialization/Deserialization error
    SerializationError(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            AppError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::DedalusError(msg) => write!(f, "Dedalus API error: {}", msg),
            AppError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AppError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            AppError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            AppError::InternalError(msg) => write!(f, "Internal server error: {}", msg),
            AppError::ExternalServiceError(msg) => write!(f, "External service error: {}", msg),
            AppError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AppError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

// ============================================================================
// Error Conversion Implementations
// ============================================================================

/// Convert worker::Error to AppError
impl From<worker::Error> for AppError {
    fn from(err: worker::Error) -> Self {
        match err {
            worker::Error::RouteNoDataError => {
                AppError::NotFound("Route not found".to_string())
            }
            worker::Error::BindingError(msg) => {
                AppError::ConfigurationError(format!("Binding error: {}", msg))
            }
            worker::Error::Json((_, code)) => {
                AppError::SerializationError(format!("JSON error: code {}", code))
            }
            _ => AppError::InternalError(format!("Worker error: {:?}", err)),
        }
    }
}

/// Convert DbError to AppError
impl From<crate::db::DbError> for AppError {
    fn from(err: crate::db::DbError) -> Self {
        match err {
            crate::db::DbError::NotFound(msg) => AppError::NotFound(msg),
            crate::db::DbError::InvalidInput(msg) => AppError::ValidationError(msg),
            crate::db::DbError::DatabaseError(msg) => AppError::DatabaseError(msg),
            crate::db::DbError::SerializationError(msg) => AppError::SerializationError(msg),
        }
    }
}

// ============================================================================
// Error Response Structure
// ============================================================================

/// Structured error response for API endpoints
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,

    /// Error code for programmatic handling
    pub code: String,

    /// HTTP status code
    pub status: u16,

    /// Request ID for debugging (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Additional error details (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>, code: impl Into<String>, status: u16) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            status,
            request_id: None,
            details: None,
        }
    }

    /// Add request ID to error response
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Add additional details to error response
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

// ============================================================================
// Error Response Conversion
// ============================================================================

impl AppError {
    /// Convert AppError to HTTP Response
    pub fn to_response(&self) -> Result<Response> {
        let (status, code) = match self {
            AppError::DatabaseError(_) => (500, "DATABASE_ERROR"),
            AppError::StorageError(_) => (500, "STORAGE_ERROR"),
            AppError::ValidationError(_) => (400, "VALIDATION_ERROR"),
            AppError::DedalusError(_) => (502, "AI_SERVICE_ERROR"),
            AppError::NotFound(_) => (404, "NOT_FOUND"),
            AppError::Unauthorized(_) => (403, "UNAUTHORIZED"),
            AppError::RateLimitExceeded => (429, "RATE_LIMIT_EXCEEDED"),
            AppError::InternalError(_) => (500, "INTERNAL_ERROR"),
            AppError::ExternalServiceError(_) => (502, "EXTERNAL_SERVICE_ERROR"),
            AppError::ConfigurationError(_) => (500, "CONFIGURATION_ERROR"),
            AppError::SerializationError(_) => (400, "SERIALIZATION_ERROR"),
        };

        let error_response = ErrorResponse::new(
            self.to_string(),
            code,
            status,
        );

        // Log error with appropriate level
        match status {
            400..=499 => {
                console_log!("[{}] Client error: {}", code, self);
            }
            500..=599 => {
                console_error!("[{}] Server error: {}", code, self);
            }
            _ => {
                console_log!("[{}] Error: {}", code, self);
            }
        }

        Ok(Response::from_json(&error_response)?
            .with_status(status))
    }

    /// Convert AppError to HTTP Response with request ID
    pub fn to_response_with_id(&self, request_id: &str) -> Result<Response> {
        let (status, code) = match self {
            AppError::DatabaseError(_) => (500, "DATABASE_ERROR"),
            AppError::StorageError(_) => (500, "STORAGE_ERROR"),
            AppError::ValidationError(_) => (400, "VALIDATION_ERROR"),
            AppError::DedalusError(_) => (502, "AI_SERVICE_ERROR"),
            AppError::NotFound(_) => (404, "NOT_FOUND"),
            AppError::Unauthorized(_) => (403, "UNAUTHORIZED"),
            AppError::RateLimitExceeded => (429, "RATE_LIMIT_EXCEEDED"),
            AppError::InternalError(_) => (500, "INTERNAL_ERROR"),
            AppError::ExternalServiceError(_) => (502, "EXTERNAL_SERVICE_ERROR"),
            AppError::ConfigurationError(_) => (500, "CONFIGURATION_ERROR"),
            AppError::SerializationError(_) => (400, "SERIALIZATION_ERROR"),
        };

        let error_response = ErrorResponse::new(
            self.to_string(),
            code,
            status,
        ).with_request_id(request_id);

        // Log error with request ID
        match status {
            400..=499 => {
                console_log!("[{}][request_id={}] Client error: {}", code, request_id, self);
            }
            500..=599 => {
                console_error!("[{}][request_id={}] Server error: {}", code, request_id, self);
            }
            _ => {
                console_log!("[{}][request_id={}] Error: {}", code, request_id, self);
            }
        }

        Ok(Response::from_json(&error_response)?
            .with_status(status))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a request ID for error tracking
pub fn generate_request_id() -> String {
    crate::db::generate_id()
}

/// Create a standardized error response
pub fn error_response(error: impl Into<String>, status: u16) -> Result<Response> {
    let code = match status {
        400 => "BAD_REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        429 => "RATE_LIMIT_EXCEEDED",
        500 => "INTERNAL_ERROR",
        502 => "BAD_GATEWAY",
        503 => "SERVICE_UNAVAILABLE",
        _ => "UNKNOWN_ERROR",
    };

    let error_response = ErrorResponse::new(error, code, status);

    Ok(Response::from_json(&error_response)?
        .with_status(status))
}

// ============================================================================
// Result Type Alias
// ============================================================================

/// Application result type
pub type AppResult<T> = std::result::Result<T, AppError>;
