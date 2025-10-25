use axum::{
    extract::{Extension, State},
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    auth::JwtClaims,
    errors::AppError,
    models::{User, UserResponse},
    state::AppState,
};

/// Register a new user via Supabase Auth
pub async fn register(
    State(state): State<AppState>,
    Json(payload): Json<RegisterRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Call Supabase Auth API to create user
    let client = Client::new();
    let supabase_url = std::env::var("SUPABASE_URL")
        .map_err(|_| AppError::Internal("SUPABASE_URL not set".to_string()))?;

    let auth_response = client
        .post(format!("{}/auth/v1/signup", supabase_url))
        .header(
            "apikey",
            std::env::var("SUPABASE_ANON_KEY")
                .map_err(|_| AppError::Internal("SUPABASE_ANON_KEY not set".to_string()))?,
        )
        .json(&json!({
            "email": payload.email,
            "password": payload.password,
        }))
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("Supabase API call failed: {}", e)))?;

    if !auth_response.status().is_success() {
        let error_text = auth_response.text().await.unwrap_or_default();
        return Err(AppError::BadRequest(format!(
            "Failed to register user: {}",
            error_text
        )));
    }

    let auth_data: SupabaseAuthResponse = auth_response
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse Supabase response: {}", e)))?;

    // Insert user into our database
    let user = sqlx::query_as::<_, User>(
        r#"
        INSERT INTO users (id, email, role, full_name)
        VALUES ($1, $2, $3, $4)
        RETURNING id, email, role, full_name, created_at, updated_at
        "#,
    )
    .bind(uuid::Uuid::parse_str(&auth_data.user.id).map_err(|e| {
        AppError::Internal(format!("Invalid UUID from Supabase: {}", e))
    })?)
    .bind(&payload.email)
    .bind(payload.role)
    .bind(&payload.full_name)
    .fetch_one(&state.pool)
    .await
    .map_err(AppError::Database)?;

    let response = AuthResponse {
        access_token: auth_data.access_token,
        refresh_token: auth_data.refresh_token,
        expires_in: auth_data.expires_in,
        user: user.into(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Login with Supabase Auth
pub async fn login(
    State(state): State<AppState>,
    Json(payload): Json<LoginRequest>,
) -> Result<impl IntoResponse, AppError> {
    let client = Client::new();
    let supabase_url = std::env::var("SUPABASE_URL")
        .map_err(|_| AppError::Internal("SUPABASE_URL not set".to_string()))?;

    let auth_response = client
        .post(format!("{}/auth/v1/token?grant_type=password", supabase_url))
        .header(
            "apikey",
            std::env::var("SUPABASE_ANON_KEY")
                .map_err(|_| AppError::Internal("SUPABASE_ANON_KEY not set".to_string()))?,
        )
        .json(&json!({
            "email": payload.email,
            "password": payload.password,
        }))
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("Supabase API call failed: {}", e)))?;

    if !auth_response.status().is_success() {
        return Err(AppError::Unauthorized("Invalid credentials".to_string()));
    }

    let auth_data: SupabaseAuthResponse = auth_response
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse Supabase response: {}", e)))?;

    // Fetch user from our database
    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, email, role, full_name, created_at, updated_at
        FROM users
        WHERE email = $1
        "#,
    )
    .bind(&payload.email)
    .fetch_one(&state.pool)
    .await
    .map_err(AppError::Database)?;

    let response = AuthResponse {
        access_token: auth_data.access_token,
        refresh_token: auth_data.refresh_token,
        expires_in: auth_data.expires_in,
        user: user.into(),
    };

    Ok(Json(response))
}

/// Get current user info
pub async fn me(
    Extension(claims): Extension<JwtClaims>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let user_id = uuid::Uuid::parse_str(&claims.sub)
        .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, email, role, full_name, created_at, updated_at
        FROM users
        WHERE id = $1
        "#,
    )
    .bind(user_id)
    .fetch_one(&state.pool)
    .await
    .map_err(|_| AppError::NotFound("User not found".to_string()))?;

    Ok(Json(UserResponse::from(user)))
}

/// Refresh JWT token
pub async fn refresh(
    Json(payload): Json<RefreshRequest>,
) -> Result<impl IntoResponse, AppError> {
    let client = Client::new();
    let supabase_url = std::env::var("SUPABASE_URL")
        .map_err(|_| AppError::Internal("SUPABASE_URL not set".to_string()))?;

    let auth_response = client
        .post(format!(
            "{}/auth/v1/token?grant_type=refresh_token",
            supabase_url
        ))
        .header(
            "apikey",
            std::env::var("SUPABASE_ANON_KEY")
                .map_err(|_| AppError::Internal("SUPABASE_ANON_KEY not set".to_string()))?,
        )
        .json(&json!({
            "refresh_token": payload.refresh_token,
        }))
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("Supabase API call failed: {}", e)))?;

    if !auth_response.status().is_success() {
        return Err(AppError::Unauthorized(
            "Invalid or expired refresh token".to_string(),
        ));
    }

    let auth_data: SupabaseAuthResponse = auth_response
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse Supabase response: {}", e)))?;

    let response = RefreshResponse {
        access_token: auth_data.access_token,
        refresh_token: auth_data.refresh_token,
        expires_in: auth_data.expires_in,
    };

    Ok(Json(response))
}

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
    pub full_name: String,
    pub role: crate::models::UserRole,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct RefreshRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
    pub user: UserResponse,
}

#[derive(Debug, Serialize)]
pub struct RefreshResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
}

#[derive(Debug, Deserialize)]
struct SupabaseAuthResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
    user: SupabaseUser,
}

#[derive(Debug, Deserialize)]
struct SupabaseUser {
    id: String,
    email: String,
}
