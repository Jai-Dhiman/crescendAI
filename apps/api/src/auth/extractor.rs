//! AuthUser extractor for Axum handlers.

use axum::extract::FromRequestParts;
use http::request::Parts;

use crate::error::ApiError;
use crate::state::AppState;
use crate::types::StudentId;

/// Authenticated user extracted from JWT in cookie or Bearer header.
///
/// Add to any handler signature to require authentication.
/// Use `Option<AuthUser>` for optional authentication.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub student_id: StudentId,
}

impl FromRequestParts<AppState> for AuthUser {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let token = extract_token_from_cookie(&parts.headers)
            .or_else(|| extract_token_from_bearer(&parts.headers))
            .ok_or(ApiError::Unauthorized)?;

        let claims = state.auth.verify_jwt(&token)?;
        Ok(Self {
            student_id: StudentId::from(claims.sub),
        })
    }
}

fn extract_token_from_cookie(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies
                .split(';')
                .find_map(|c| c.trim().strip_prefix("token=").map(|t| t.to_string()))
        })
}

fn extract_token_from_bearer(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer ").map(|t| t.to_string()))
}
