//! Auth handlers -- Axum-signature handlers and legacy wrappers.
//!
//! New handlers use `State<AppState>` + `Json<T>` extractors.
//! Legacy functions keep the old `(&Env, &[u8])` signatures for server.rs
//! compatibility until Task 6 rewrites the router.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use wasm_bindgen::JsValue;
use worker::{console_error, console_log, Env};

use super::jwt;
use crate::auth::extractor::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;
use crate::types::StudentId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const JWT_EXPIRY_SECONDS: u64 = 30 * 24 * 60 * 60;
const APPLE_ISSUER: &str = "https://appleid.apple.com";

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppleAuthRequest {
    pub identity_token: String,
    pub user_id: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GoogleAuthRequest {
    pub credential: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthResponse {
    pub student_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub is_new_user: bool,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleTokenClaims {
    #[allow(dead_code)]
    iss: Option<String>,
    sub: Option<String>,
    aud: Option<String>,
    email: Option<String>,
    /// Google's tokeninfo returns this as a string "true"/"false".
    email_verified: Option<String>,
    name: Option<String>,
    #[allow(dead_code)]
    exp: Option<String>,
}

#[derive(serde::Deserialize)]
struct AppleTokenClaims {
    iss: Option<String>,
    sub: Option<String>,
    aud: Option<String>,
    exp: Option<u64>,
}

// =========================================================================
// NEW Axum-style handlers
// =========================================================================

/// POST /api/auth/apple -- validate Apple identity token, issue JWT.
#[worker::send]
pub async fn handle_apple(
    State(state): State<AppState>,
    Json(request): Json<AppleAuthRequest>,
) -> Result<impl IntoResponse> {
    let env = state.auth.env();

    // Build allowed audiences from env vars
    let bundle_id = env
        .var("APPLE_BUNDLE_ID")
        .map_or_else(|_| "ai.crescend.ios".to_string(), |v| v.to_string());

    let mut allowed_audiences = vec![bundle_id];
    if let Ok(web_services_id) = env.var("APPLE_WEB_SERVICES_ID") {
        allowed_audiences.push(web_services_id.to_string());
    }

    // Parse and validate Apple identity token claims
    let apple_claims = parse_apple_token_claims(&request.identity_token)?;
    validate_apple_claims(&apple_claims, &request.user_id, &allowed_audiences)?;

    let db = state.db.d1()?;

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = find_or_create_student(
        &db,
        "apple",
        &request.user_id,
        request.email.as_deref(),
        request.display_name.as_deref(),
        &now,
    )
    .await?;

    let token = state.auth.sign_jwt(&student_id)?;

    let response = AuthResponse {
        student_id,
        email: request.email,
        display_name,
        is_new_user,
    };

    console_log!("Auth successful for user (new={})", is_new_user);

    let cookie = build_auth_cookie(&token, JWT_EXPIRY_SECONDS, env);

    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::SET_COOKIE,
        cookie
            .parse()
            .map_err(|_| ApiError::Internal("cookie header".into()))?,
    );
    Ok((headers, Json(response)))
}

/// POST /api/auth/google -- validate Google ID token via tokeninfo, issue JWT.
#[worker::send]
pub async fn handle_google(
    State(state): State<AppState>,
    Json(request): Json<GoogleAuthRequest>,
) -> Result<impl IntoResponse> {
    let env = state.auth.env();

    // Validate GOOGLE_CLIENT_ID is configured
    let expected_client_id = env
        .var("GOOGLE_CLIENT_ID")
        .map(|v| v.to_string())
        .map_err(|_| {
            console_error!("GOOGLE_CLIENT_ID not configured");
            ApiError::Internal("GOOGLE_CLIENT_ID not configured".into())
        })?;

    if expected_client_id.is_empty() {
        console_error!("GOOGLE_CLIENT_ID is empty");
        return Err(ApiError::Internal("GOOGLE_CLIENT_ID is empty".into()));
    }

    // Verify Google ID token via Google's tokeninfo endpoint.
    let tokeninfo_url = format!(
        "https://oauth2.googleapis.com/tokeninfo?id_token={}",
        request.credential
    );
    let url: worker::Url = tokeninfo_url
        .parse()
        .map_err(|_| ApiError::Internal("invalid tokeninfo URL".into()))?;

    let mut resp = worker::Fetch::Url(url)
        .send()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Google tokeninfo: {e}")))?;

    if resp.status_code() != 200 {
        console_error!("Google tokeninfo returned status {}", resp.status_code());
        return Err(ApiError::Unauthorized);
    }

    let text = resp
        .text()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Google tokeninfo body: {e}")))?;

    let claims: GoogleTokenClaims = serde_json::from_str(&text).map_err(|e| {
        console_error!("Failed to parse Google tokeninfo response: {:?}", e);
        ApiError::Unauthorized
    })?;

    // Validate audience matches our client ID
    match &claims.aud {
        Some(aud) if aud == &expected_client_id => {}
        Some(aud) => {
            console_error!(
                "Google token audience '{}' does not match expected '{}'",
                aud,
                expected_client_id
            );
            return Err(ApiError::Unauthorized);
        }
        None => return Err(ApiError::Unauthorized),
    }

    // Validate email is verified
    if claims.email_verified.as_deref() != Some("true") {
        return Err(ApiError::Unauthorized);
    }

    let google_user_id = claims.sub.as_ref().ok_or(ApiError::Unauthorized)?.clone();

    let db = state.db.d1()?;

    let now_iso = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = find_or_create_student(
        &db,
        "google",
        &google_user_id,
        claims.email.as_deref(),
        claims.name.as_deref(),
        &now_iso,
    )
    .await?;

    let token = state.auth.sign_jwt(&student_id)?;

    let response = AuthResponse {
        student_id,
        email: claims.email,
        display_name,
        is_new_user,
    };

    console_log!("Google auth successful for user (new={})", is_new_user);

    let cookie = build_auth_cookie(&token, JWT_EXPIRY_SECONDS, env);

    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::SET_COOKIE,
        cookie
            .parse()
            .map_err(|_| ApiError::Internal("cookie header".into()))?,
    );
    Ok((headers, Json(response)))
}

/// GET /api/auth/me -- return current user info from JWT.
#[worker::send]
pub async fn handle_me(
    State(state): State<AppState>,
    auth: AuthUser,
) -> Result<Json<AuthResponse>> {
    let db = state.db.d1()?;
    let student_id = auth.student_id.as_str().to_string();

    let row = db
        .prepare("SELECT student_id, email, display_name FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(&student_id)])
        .map_err(|e| ApiError::Internal(format!("bind query: {e:?}")))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| ApiError::Internal(format!("query student: {e:?}")))?
        .ok_or(ApiError::Unauthorized)?;

    let response = AuthResponse {
        student_id: row
            .get("student_id")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
            .unwrap_or_default(),
        email: row
            .get("email")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string)),
        display_name: row
            .get("display_name")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string)),
        is_new_user: false,
    };

    Ok(Json(response))
}

/// POST /api/auth/signout -- clear the auth cookie.
#[allow(clippy::unused_async)]
pub async fn handle_signout(State(state): State<AppState>) -> impl IntoResponse {
    let cookie = build_clear_cookie(state.auth.env());
    let mut headers = http::HeaderMap::new();
    // This parse will not fail for the known cookie format.
    if let Ok(val) = cookie.parse() {
        headers.insert(http::header::SET_COOKIE, val);
    }
    (headers, Json(serde_json::json!({"ok": true})))
}

/// POST /api/auth/debug -- dev-only login bypassing Apple Sign In.
/// Returns 404 in production.
#[worker::send]
pub async fn handle_debug(State(state): State<AppState>) -> Result<impl IntoResponse> {
    let env = state.auth.env();

    let environment = env
        .var("ENVIRONMENT")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if environment == "production" {
        return Err(ApiError::NotFound("Not found".into()));
    }

    let db = state.db.d1()?;

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = find_or_create_student(
        &db,
        "debug",
        "debug-local-dev",
        Some("dev@localhost"),
        Some("Debug User"),
        &now,
    )
    .await?;

    let token = state.auth.sign_jwt(&student_id)?;

    let response_body = serde_json::json!({
        "studentId": student_id,
        "email": "dev@localhost",
        "displayName": display_name,
        "isNewUser": is_new_user,
        "token": token,
    });

    let cookie =
        format!("token={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={JWT_EXPIRY_SECONDS}");

    console_log!("Debug auth: student_id={}, new={}", student_id, is_new_user);

    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::SET_COOKIE,
        cookie
            .parse()
            .map_err(|_| ApiError::Internal("cookie header".into()))?,
    );
    Ok((headers, Json(response_body)))
}

// =========================================================================
// Legacy functions -- called by server.rs with old signatures.
// Will be removed when server.rs is rewritten in Task 6.
// =========================================================================

/// Legacy: extract and verify JWT from cookie or Authorization header.
/// Returns the `student_id` (subject claim) on success.
#[allow(clippy::expect_used)] // Response::builder() with valid status/headers is infallible
pub fn verify_auth(
    headers: &http::HeaderMap,
    env: &Env,
) -> std::result::Result<String, http::Response<axum::body::Body>> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let token = legacy_extract_token_from_cookie(headers)
        .or_else(|| legacy_extract_token_from_bearer(headers));

    let token = token.ok_or_else(|| {
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(
                r#"{"error":"Missing or invalid Authorization header"}"#,
            ))
            .expect("response builder")
    })?;

    let jwt_secret = env
        .secret("JWT_SECRET")
        .map(|s| s.to_string())
        .map_err(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .expect("response builder")
        })?;

    let claims = jwt::verify_compat(&token, jwt_secret.as_bytes()).map_err(|e| {
        console_error!("JWT verification failed: {}", e);
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(r#"{{"error":"{e}"}}"#)))
            .expect("response builder")
    })?;

    Ok(claims.sub)
}

/// Legacy alias for backward compatibility.
pub fn verify_auth_header(
    headers: &http::HeaderMap,
    env: &Env,
) -> std::result::Result<String, http::Response<axum::body::Body>> {
    verify_auth(headers, env)
}

/// Legacy: POST /api/auth/apple
#[allow(clippy::expect_used, clippy::too_many_lines)] // Response::builder() is infallible; legacy code
pub async fn handle_apple_auth_legacy(env: &Env, body: &[u8]) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let request: AppleAuthRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse auth request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .expect("response builder");
        }
    };

    let bundle_id = env
        .var("APPLE_BUNDLE_ID")
        .map_or_else(|_| "ai.crescend.ios".to_string(), |v| v.to_string());

    let mut allowed_audiences = vec![bundle_id];
    if let Ok(web_services_id) = env.var("APPLE_WEB_SERVICES_ID") {
        allowed_audiences.push(web_services_id.to_string());
    }

    let apple_claims = match parse_apple_token_claims(&request.identity_token) {
        Ok(c) => c,
        Err(e) => {
            console_error!("Failed to parse Apple token: {}", e);
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(format!(r#"{{"error":"{e}"}}"#)))
                .expect("response builder");
        }
    };

    if let Err(e) = validate_apple_claims(&apple_claims, &request.user_id, &allowed_audiences) {
        console_error!("Apple token validation failed: {}", e);
        return Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(r#"{{"error":"{e}"}}"#)))
            .expect("response builder");
    }

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .expect("response builder");
        }
    };

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = match find_or_create_student(
        &db,
        "apple",
        &request.user_id,
        request.email.as_deref(),
        request.display_name.as_deref(),
        &now,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            console_error!("Failed to find/create student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create user record"}"#))
                .expect("response builder");
        }
    };

    let jwt_secret = match env.secret("JWT_SECRET") {
        Ok(s) => s.to_string(),
        Err(e) => {
            console_error!("JWT_SECRET not configured: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .expect("response builder");
        }
    };

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now_epoch = (js_sys::Date::now() / 1000.0) as u64;
    let claims = jwt::Claims {
        sub: student_id.clone(),
        iat: now_epoch,
        exp: now_epoch + JWT_EXPIRY_SECONDS,
    };

    let token = match jwt::sign_compat(&claims, jwt_secret.as_bytes()) {
        Ok(t) => t,
        Err(e) => {
            console_error!("Failed to sign JWT: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to generate token"}"#))
                .expect("response builder");
        }
    };

    let response = AuthResponse {
        student_id,
        email: request.email,
        display_name,
        is_new_user,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!("Auth successful for user (new={})", is_new_user);

    let cookie = build_auth_cookie(&token, JWT_EXPIRY_SECONDS, env);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", cookie)
        .body(Body::from(json))
        .expect("response builder")
}

/// Legacy: POST /api/auth/google
#[allow(clippy::expect_used, clippy::too_many_lines)] // Response::builder() is infallible; legacy code
pub async fn handle_google_auth_legacy(env: &Env, body: &[u8]) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let request: GoogleAuthRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse Google auth request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .expect("response builder");
        }
    };

    let expected_client_id = if let Ok(v) = env.var("GOOGLE_CLIENT_ID") {
        let id = v.to_string();
        if id.is_empty() {
            console_error!("GOOGLE_CLIENT_ID is empty");
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .expect("response builder");
        }
        id
    } else {
        console_error!("GOOGLE_CLIENT_ID not configured");
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Server configuration error"}"#))
            .expect("response builder");
    };

    let tokeninfo_url = format!(
        "https://oauth2.googleapis.com/tokeninfo?id_token={}",
        request.credential
    );
    #[allow(clippy::unwrap_used)]
    let tokeninfo_resp = match worker::Fetch::Url(tokeninfo_url.parse().unwrap())
        .send()
        .await
    {
        Ok(mut resp) => {
            if resp.status_code() != 200 {
                console_error!("Google tokeninfo returned status {}", resp.status_code());
                return Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Invalid Google token"}"#))
                    .expect("response builder");
            }
            match resp.text().await {
                Ok(text) => text,
                Err(e) => {
                    console_error!("Failed to read Google tokeninfo response: {:?}", e);
                    return Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .header("Content-Type", "application/json")
                        .body(Body::from(r#"{"error":"Token verification failed"}"#))
                        .expect("response builder");
                }
            }
        }
        Err(e) => {
            console_error!("Failed to call Google tokeninfo: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Token verification failed"}"#))
                .expect("response builder");
        }
    };

    let claims: GoogleTokenClaims = match serde_json::from_str(&tokeninfo_resp) {
        Ok(c) => c,
        Err(e) => {
            console_error!("Failed to parse Google tokeninfo response: {:?}", e);
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid Google token"}"#))
                .expect("response builder");
        }
    };

    match &claims.aud {
        Some(aud) if aud == &expected_client_id => {}
        Some(aud) => {
            console_error!(
                "Google token audience '{}' does not match expected '{}'",
                aud,
                expected_client_id
            );
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid audience"}"#))
                .expect("response builder");
        }
        None => {
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Missing audience claim"}"#))
                .expect("response builder");
        }
    }

    if claims.email_verified.as_deref() != Some("true") {
        return Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Email not verified"}"#))
            .expect("response builder");
    }

    let google_user_id = match &claims.sub {
        Some(sub) => sub.clone(),
        None => {
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Missing subject claim"}"#))
                .expect("response builder");
        }
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .expect("response builder");
        }
    };

    let now_iso = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = match find_or_create_student(
        &db,
        "google",
        &google_user_id,
        claims.email.as_deref(),
        claims.name.as_deref(),
        &now_iso,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            console_error!("Failed to find/create student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create user record"}"#))
                .expect("response builder");
        }
    };

    let jwt_secret = match env.secret("JWT_SECRET") {
        Ok(s) => s.to_string(),
        Err(e) => {
            console_error!("JWT_SECRET not configured: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .expect("response builder");
        }
    };

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now_epoch = (js_sys::Date::now() / 1000.0) as u64;
    let jwt_claims = jwt::Claims {
        sub: student_id.clone(),
        iat: now_epoch,
        exp: now_epoch + JWT_EXPIRY_SECONDS,
    };

    let token = match jwt::sign_compat(&jwt_claims, jwt_secret.as_bytes()) {
        Ok(t) => t,
        Err(e) => {
            console_error!("Failed to sign JWT: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to generate token"}"#))
                .expect("response builder");
        }
    };

    let response = AuthResponse {
        student_id,
        email: claims.email,
        display_name,
        is_new_user,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!("Google auth successful for user (new={})", is_new_user);

    let cookie = build_auth_cookie(&token, JWT_EXPIRY_SECONDS, env);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", cookie)
        .body(Body::from(json))
        .expect("response builder")
}

/// Legacy: GET /api/auth/me
#[allow(clippy::expect_used)] // Response::builder() is infallible
pub async fn handle_auth_me_legacy(
    env: &Env,
    headers: &http::HeaderMap,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match verify_auth(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .expect("response builder");
        }
    };

    let row = match db
        .prepare("SELECT student_id, email, display_name FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(&student_id)])
        .map_err(|e| format!("{e:?}"))
    {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                return Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"User not found"}"#))
                    .expect("response builder");
            }
            Err(e) => {
                console_error!("Failed to query student: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Database query failed"}"#))
                    .expect("response builder");
            }
        },
        Err(e) => {
            console_error!("Failed to bind query: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database query failed"}"#))
                .expect("response builder");
        }
    };

    let response = AuthResponse {
        student_id: row
            .get("student_id")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
            .unwrap_or_default(),
        email: row
            .get("email")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string)),
        display_name: row
            .get("display_name")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string)),
        is_new_user: false,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .expect("response builder")
}

/// Legacy: POST /api/auth/debug
#[allow(clippy::expect_used, clippy::too_many_lines)] // Response::builder() is infallible; legacy code
pub async fn handle_debug_auth_legacy(env: &Env) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let environment = env
        .var("ENVIRONMENT")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if environment == "production" {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Not found"}"#))
            .expect("response builder");
    }

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .expect("response builder");
        }
    };

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let (student_id, display_name, is_new_user) = match find_or_create_student(
        &db,
        "debug",
        "debug-local-dev",
        Some("dev@localhost"),
        Some("Debug User"),
        &now,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            console_error!("Failed to find/create debug student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create debug user"}"#))
                .expect("response builder");
        }
    };

    let jwt_secret = match env.secret("JWT_SECRET") {
        Ok(s) => s.to_string(),
        Err(e) => {
            console_error!("JWT_SECRET not configured: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .expect("response builder");
        }
    };

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now_epoch = (js_sys::Date::now() / 1000.0) as u64;
    let claims = jwt::Claims {
        sub: student_id.clone(),
        iat: now_epoch,
        exp: now_epoch + JWT_EXPIRY_SECONDS,
    };

    let token = match jwt::sign_compat(&claims, jwt_secret.as_bytes()) {
        Ok(t) => t,
        Err(e) => {
            console_error!("Failed to sign JWT: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to generate token"}"#))
                .expect("response builder");
        }
    };

    let response_body = serde_json::json!({
        "student_id": student_id,
        "email": "dev@localhost",
        "display_name": display_name,
        "is_new_user": is_new_user,
        "token": token,
    });

    let cookie =
        format!("token={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={JWT_EXPIRY_SECONDS}");

    console_log!("Debug auth: student_id={}, new={}", student_id, is_new_user);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", cookie)
        .body(Body::from(response_body.to_string()))
        .expect("response builder")
}

/// Legacy: POST /api/auth/signout
#[allow(clippy::expect_used)] // Response::builder() is infallible
pub fn handle_signout_legacy(env: &Env) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", build_clear_cookie(env))
        .body(Body::from(r#"{"ok":true}"#))
        .expect("response builder")
}

// =========================================================================
// Shared helpers
// =========================================================================

fn parse_apple_token_claims(token: &str) -> Result<AppleTokenClaims> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(ApiError::BadRequest(
            "Invalid Apple identity token format".into(),
        ));
    }

    let payload = URL_SAFE_NO_PAD
        .decode(parts[1])
        .or_else(|_| {
            // Try with standard base64 padding
            let padded = match parts[1].len() % 4 {
                2 => format!("{}==", parts[1]),
                3 => format!("{}=", parts[1]),
                _ => parts[1].to_string(),
            };
            URL_SAFE_NO_PAD.decode(&padded)
        })
        .map_err(|e| ApiError::BadRequest(format!("Failed to decode token payload: {e}")))?;

    serde_json::from_slice(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Failed to parse token claims: {e}")))
}

fn validate_apple_claims(
    claims: &AppleTokenClaims,
    expected_user_id: &str,
    allowed_audiences: &[String],
) -> Result<()> {
    match &claims.iss {
        Some(iss) if iss == APPLE_ISSUER => {}
        Some(_iss) => {
            return Err(ApiError::Unauthorized);
        }
        None => return Err(ApiError::Unauthorized),
    }

    match &claims.aud {
        Some(aud) if allowed_audiences.iter().any(|a| a == aud) => {}
        Some(aud) => {
            console_error!("Token audience '{}' not in allowed audiences", aud);
            return Err(ApiError::Unauthorized);
        }
        None => return Err(ApiError::Unauthorized),
    }

    match &claims.sub {
        Some(sub) if sub == expected_user_id => {}
        Some(sub) => {
            console_error!(
                "Token subject '{}' does not match user_id '{}'",
                sub,
                expected_user_id
            );
            return Err(ApiError::Unauthorized);
        }
        None => return Err(ApiError::Unauthorized),
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let now = (js_sys::Date::now() / 1000.0) as u64;
    match claims.exp {
        Some(exp) if exp > now => {}
        Some(_) | None => return Err(ApiError::Unauthorized),
    }

    Ok(())
}

/// Look up a student by provider identity, or create a new one.
/// Returns (`student_id`, `display_name`, `is_new_user`).
async fn find_or_create_student(
    db: &worker::D1Database,
    provider: &str,
    provider_user_id: &str,
    email: Option<&str>,
    display_name: Option<&str>,
    now: &str,
) -> Result<(String, Option<String>, bool)> {
    // Check auth_identities for existing mapping
    let existing = db
        .prepare(
            "SELECT student_id FROM auth_identities WHERE provider = ?1 AND provider_user_id = ?2",
        )
        .bind(&[
            JsValue::from_str(provider),
            JsValue::from_str(provider_user_id),
        ])
        .map_err(|e| ApiError::Internal(format!("bind identity query: {e:?}")))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| ApiError::Internal(format!("query auth_identities: {e:?}")))?;

    if let Some(row) = existing {
        let student_id = row
            .get("student_id")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
            .ok_or_else(|| {
                ApiError::Internal("Missing student_id in auth_identities row".into())
            })?;

        // Update email on students table if provided
        if let Some(email) = email {
            db.prepare("UPDATE students SET email = ?1, updated_at = ?2 WHERE student_id = ?3")
                .bind(&[
                    JsValue::from_str(email),
                    JsValue::from_str(now),
                    JsValue::from_str(&student_id),
                ])
                .map_err(|e| ApiError::Internal(format!("bind email update: {e:?}")))?
                .run()
                .await
                .map_err(|e| ApiError::Internal(format!("update email: {e:?}")))?;
        }

        // Fetch display_name from students
        let student_row = db
            .prepare("SELECT display_name FROM students WHERE student_id = ?1")
            .bind(&[JsValue::from_str(&student_id)])
            .map_err(|e| ApiError::Internal(format!("bind student query: {e:?}")))?
            .first::<serde_json::Value>(None)
            .await
            .map_err(|e| ApiError::Internal(format!("query student: {e:?}")))?;

        let existing_display_name = student_row.and_then(|r| {
            r.get("display_name")
                .and_then(|v| v.as_str().map(std::string::ToString::to_string))
        });

        Ok((student_id, existing_display_name, false))
    } else {
        // Check if a student with the same email already exists (account linking)
        if let Some(email) = email {
            let existing_by_email = db
                .prepare("SELECT student_id, display_name FROM students WHERE email = ?1")
                .bind(&[JsValue::from_str(email)])
                .map_err(|e| ApiError::Internal(format!("bind email lookup: {e:?}")))?
                .first::<serde_json::Value>(None)
                .await
                .map_err(|e| ApiError::Internal(format!("query students by email: {e:?}")))?;

            if let Some(row) = existing_by_email {
                let student_id = row
                    .get("student_id")
                    .and_then(|v| v.as_str().map(std::string::ToString::to_string))
                    .ok_or_else(|| {
                        ApiError::Internal("Missing student_id in email lookup".into())
                    })?;
                let existing_display_name = row
                    .get("display_name")
                    .and_then(|v| v.as_str().map(std::string::ToString::to_string));

                // Link this provider identity to the existing student
                db.prepare(
                    "INSERT INTO auth_identities (student_id, provider, provider_user_id, created_at) VALUES (?1, ?2, ?3, ?4)",
                )
                .bind(&[
                    JsValue::from_str(&student_id),
                    JsValue::from_str(provider),
                    JsValue::from_str(provider_user_id),
                    JsValue::from_str(now),
                ])
                .map_err(|e| ApiError::Internal(format!("bind identity insert for linking: {e:?}")))?
                .run()
                .await
                .map_err(|e| ApiError::Internal(format!("link identity: {e:?}")))?;

                return Ok((student_id, existing_display_name, false));
            }
        }

        // New user -- generate UUID and insert atomically
        let student_id = StudentId::new().to_string();

        let student_stmt = db
            .prepare(
                "INSERT INTO students (student_id, email, display_name, baseline_session_count, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, 0, ?4, ?5)",
            )
            .bind(&[
                JsValue::from_str(&student_id),
                match email {
                    Some(e) => JsValue::from_str(e),
                    None => JsValue::NULL,
                },
                match display_name {
                    Some(name) => JsValue::from_str(name),
                    None => JsValue::NULL,
                },
                JsValue::from_str(now),
                JsValue::from_str(now),
            ])
            .map_err(|e| ApiError::Internal(format!("bind student insert: {e:?}")))?;

        let identity_stmt = db
            .prepare(
                "INSERT INTO auth_identities (student_id, provider, provider_user_id, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .bind(&[
                JsValue::from_str(&student_id),
                JsValue::from_str(provider),
                JsValue::from_str(provider_user_id),
                JsValue::from_str(now),
            ])
            .map_err(|e| ApiError::Internal(format!("bind identity insert: {e:?}")))?;

        // Batch executes both inserts atomically
        db.batch(vec![student_stmt, identity_stmt])
            .await
            .map_err(|e| ApiError::Internal(format!("create student: {e:?}")))?;

        Ok((
            student_id,
            display_name.map(std::string::ToString::to_string),
            true,
        ))
    }
}

fn build_auth_cookie(token: &str, max_age: u64, env: &Env) -> String {
    let cookie_domain = env
        .var("COOKIE_DOMAIN")
        .ok()
        .map(|v| v.to_string())
        .filter(|v| !v.is_empty());
    match cookie_domain {
        Some(domain) => format!(
            "token={token}; HttpOnly; Secure; SameSite=None; Path=/; Max-Age={max_age}; Domain={domain}"
        ),
        None => format!(
            "token={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={max_age}"
        ),
    }
}

fn build_clear_cookie(env: &Env) -> String {
    let cookie_domain = env
        .var("COOKIE_DOMAIN")
        .ok()
        .map(|v| v.to_string())
        .filter(|v| !v.is_empty());
    match cookie_domain {
        Some(domain) => {
            format!("token=; HttpOnly; Secure; SameSite=None; Path=/; Max-Age=0; Domain={domain}")
        }
        None => "token=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0".to_string(),
    }
}

fn legacy_extract_token_from_cookie(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies.split(';').find_map(|c| {
                c.trim()
                    .strip_prefix("token=")
                    .map(std::string::ToString::to_string)
            })
        })
}

fn legacy_extract_token_from_bearer(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| {
            auth.strip_prefix("Bearer ")
                .map(std::string::ToString::to_string)
        })
}
