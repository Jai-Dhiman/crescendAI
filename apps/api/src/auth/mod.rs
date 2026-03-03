pub mod jwt;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use worker::{console_log, Env};

use self::jwt::Claims;

const JWT_EXPIRY_SECONDS: u64 = 30 * 24 * 60 * 60; // 30 days
const APPLE_ISSUER: &str = "https://appleid.apple.com";

#[derive(serde::Deserialize)]
pub struct AppleAuthRequest {
    pub identity_token: String,
    pub user_id: String,
    pub email: Option<String>,
}

#[derive(serde::Serialize)]
pub struct AuthResponse {
    pub jwt: String,
    pub apple_user_id: String,
    pub email: Option<String>,
    pub is_new_user: bool,
}

#[derive(serde::Deserialize)]
struct AppleTokenClaims {
    iss: Option<String>,
    sub: Option<String>,
    aud: Option<String>,
    exp: Option<u64>,
}

fn parse_apple_token_claims(token: &str) -> Result<AppleTokenClaims, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("Invalid Apple identity token format".to_string());
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
        .map_err(|e| format!("Failed to decode token payload: {}", e))?;

    serde_json::from_slice(&payload)
        .map_err(|e| format!("Failed to parse token claims: {}", e))
}

fn validate_apple_claims(
    claims: &AppleTokenClaims,
    expected_user_id: &str,
    bundle_id: &str,
) -> Result<(), String> {
    // Verify issuer
    match &claims.iss {
        Some(iss) if iss == APPLE_ISSUER => {}
        Some(iss) => return Err(format!("Invalid issuer: {}", iss)),
        None => return Err("Missing issuer claim".to_string()),
    }

    // Verify audience matches our bundle ID
    match &claims.aud {
        Some(aud) if aud == bundle_id => {}
        Some(aud) => {
            console_log!("Token audience '{}' does not match bundle ID '{}'", aud, bundle_id);
            return Err("Invalid audience".to_string());
        }
        None => return Err("Missing audience claim".to_string()),
    }

    // Verify subject matches the user ID from the client
    match &claims.sub {
        Some(sub) if sub == expected_user_id => {}
        Some(sub) => {
            console_log!("Token subject '{}' does not match user_id '{}'", sub, expected_user_id);
            return Err("Token subject mismatch".to_string());
        }
        None => return Err("Missing subject claim".to_string()),
    }

    // Verify expiration
    let now = js_sys::Date::now() as u64 / 1000;
    match claims.exp {
        Some(exp) if exp > now => {}
        Some(_) => return Err("Apple identity token expired".to_string()),
        None => return Err("Missing expiration claim".to_string()),
    }

    Ok(())
}

pub async fn handle_apple_auth(
    env: &Env,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let request: AppleAuthRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse auth request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Parse and validate Apple identity token claims
    let bundle_id = env
        .var("APPLE_BUNDLE_ID")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "ai.crescend.ios".to_string());

    let apple_claims = match parse_apple_token_claims(&request.identity_token) {
        Ok(c) => c,
        Err(e) => {
            console_log!("Failed to parse Apple token: {}", e);
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(format!(r#"{{"error":"{}"}}"#, e)))
                .unwrap();
        }
    };

    if let Err(e) = validate_apple_claims(&apple_claims, &request.user_id, &bundle_id) {
        console_log!("Apple token validation failed: {}", e);
        return Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(r#"{{"error":"{}"}}"#, e)))
            .unwrap();
    }

    // Get D1 database
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_log!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    // Upsert student record in D1
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    let is_new_user = match upsert_student(&db, &request.user_id, request.email.as_deref(), &now).await {
        Ok(is_new) => is_new,
        Err(e) => {
            console_log!("Failed to upsert student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create user record"}"#))
                .unwrap();
        }
    };

    // Issue JWT
    let jwt_secret = match env.secret("JWT_SECRET") {
        Ok(s) => s.to_string(),
        Err(e) => {
            console_log!("JWT_SECRET not configured: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .unwrap();
        }
    };

    let now_epoch = js_sys::Date::now() as u64 / 1000;
    let claims = Claims {
        sub: request.user_id.clone(),
        iat: now_epoch,
        exp: now_epoch + JWT_EXPIRY_SECONDS,
    };

    let token = match jwt::sign(&claims, jwt_secret.as_bytes()) {
        Ok(t) => t,
        Err(e) => {
            console_log!("Failed to sign JWT: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to generate token"}"#))
                .unwrap();
        }
    };

    let response = AuthResponse {
        jwt: token,
        apple_user_id: request.user_id,
        email: request.email,
        is_new_user,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!("Auth successful for user (new={})", is_new_user);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn upsert_student(
    db: &worker::D1Database,
    apple_user_id: &str,
    email: Option<&str>,
    updated_at: &str,
) -> Result<bool, String> {
    // Check if student exists
    let check = db
        .prepare("SELECT apple_user_id FROM students WHERE apple_user_id = ?1")
        .bind(&[apple_user_id.into()])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| format!("Failed to query student: {:?}", e))?;

    if check.is_some() {
        // Update existing student (only update email if provided, keep existing data)
        if let Some(email) = email {
            db.prepare("UPDATE students SET email = ?1, updated_at = ?2 WHERE apple_user_id = ?3")
                .bind(&[email.into(), updated_at.into(), apple_user_id.into()])
                .map_err(|e| format!("Failed to bind update: {:?}", e))?
                .run()
                .await
                .map_err(|e| format!("Failed to update student: {:?}", e))?;
        }
        Ok(false)
    } else {
        // Insert new student
        db.prepare(
            "INSERT INTO students (apple_user_id, email, baseline_session_count, updated_at) VALUES (?1, ?2, 0, ?3)",
        )
        .bind(&[
            apple_user_id.into(),
            email.unwrap_or("").into(),
            updated_at.into(),
        ])
        .map_err(|e| format!("Failed to bind insert: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to insert student: {:?}", e))?;
        Ok(true)
    }
}

/// Extract and verify JWT from Authorization header.
/// Returns the apple_user_id (subject claim) on success.
pub fn verify_auth_header(
    headers: &http::HeaderMap,
    env: &Env,
) -> Result<String, http::Response<axum::body::Body>> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| {
            Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Missing or invalid Authorization header"}"#))
                .unwrap()
        })?;

    let jwt_secret = env
        .secret("JWT_SECRET")
        .map(|s| s.to_string())
        .map_err(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Server configuration error"}"#))
                .unwrap()
        })?;

    let claims = jwt::verify(token, jwt_secret.as_bytes()).map_err(|e| {
        console_log!("JWT verification failed: {}", e);
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(r#"{{"error":"{}"}}"#, e)))
            .unwrap()
    })?;

    Ok(claims.sub)
}
