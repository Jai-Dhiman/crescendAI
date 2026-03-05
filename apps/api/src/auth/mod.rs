pub mod jwt;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use wasm_bindgen::JsValue;
use worker::{console_log, Env};

use self::jwt::Claims;

const JWT_EXPIRY_SECONDS: u64 = 30 * 24 * 60 * 60; // 30 days
const APPLE_ISSUER: &str = "https://appleid.apple.com";

#[derive(serde::Deserialize)]
pub struct AppleAuthRequest {
    pub identity_token: String,
    pub user_id: String,
    pub email: Option<String>,
    pub display_name: Option<String>,
}

#[derive(serde::Serialize)]
pub struct AuthResponse {
    pub student_id: String,
    pub email: Option<String>,
    pub display_name: Option<String>,
    pub is_new_user: bool,
}

#[derive(serde::Deserialize)]
struct AppleTokenClaims {
    iss: Option<String>,
    sub: Option<String>,
    aud: Option<String>,
    exp: Option<u64>,
}

/// Generate a UUID v4 using getrandom (js feature for WASM).
fn generate_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    // Set version 4
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant 1
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
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
    allowed_audiences: &[String],
) -> Result<(), String> {
    // Verify issuer
    match &claims.iss {
        Some(iss) if iss == APPLE_ISSUER => {}
        Some(iss) => return Err(format!("Invalid issuer: {}", iss)),
        None => return Err("Missing issuer claim".to_string()),
    }

    // Verify audience matches one of our allowed audiences
    match &claims.aud {
        Some(aud) if allowed_audiences.iter().any(|a| a == aud) => {}
        Some(aud) => {
            console_log!("Token audience '{}' not in allowed audiences", aud);
            return Err("Invalid audience".to_string());
        }
        None => return Err("Missing audience claim".to_string()),
    }

    // Verify subject matches the user ID from the client
    match &claims.sub {
        Some(sub) if sub == expected_user_id => {}
        Some(sub) => {
            console_log!(
                "Token subject '{}' does not match user_id '{}'",
                sub,
                expected_user_id
            );
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

    // Build allowed audiences from env vars
    let bundle_id = env
        .var("APPLE_BUNDLE_ID")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "ai.crescend.ios".to_string());

    let mut allowed_audiences = vec![bundle_id];
    if let Ok(web_services_id) = env.var("APPLE_WEB_SERVICES_ID") {
        allowed_audiences.push(web_services_id.to_string());
    }

    // Parse and validate Apple identity token claims
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

    if let Err(e) = validate_apple_claims(&apple_claims, &request.user_id, &allowed_audiences) {
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

    // Find or create student via auth_identities
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
            console_log!("Failed to find/create student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create user record"}"#))
                .unwrap();
        }
    };

    // Issue JWT with student_id as subject
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
        sub: student_id.clone(),
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
        student_id,
        email: request.email,
        display_name,
        is_new_user,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!("Auth successful for user (new={})", is_new_user);

    let cookie = format!(
        "token={}; HttpOnly; Secure; SameSite=None; Path=/; Max-Age={}",
        token, JWT_EXPIRY_SECONDS
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", cookie)
        .body(Body::from(json))
        .unwrap()
}

/// Look up a student by provider identity, or create a new one.
/// Returns (student_id, display_name, is_new_user).
async fn find_or_create_student(
    db: &worker::D1Database,
    provider: &str,
    provider_user_id: &str,
    email: Option<&str>,
    display_name: Option<&str>,
    now: &str,
) -> Result<(String, Option<String>, bool), String> {
    // Check auth_identities for existing mapping
    let existing = db
        .prepare("SELECT student_id FROM auth_identities WHERE provider = ?1 AND provider_user_id = ?2")
        .bind(&[JsValue::from_str(provider), JsValue::from_str(provider_user_id)])
        .map_err(|e| format!("Failed to bind identity query: {:?}", e))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| format!("Failed to query auth_identities: {:?}", e))?;

    if let Some(row) = existing {
        // Existing user -- get student_id
        let student_id = row
            .get("student_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| "Missing student_id in auth_identities row".to_string())?;

        // Update email on students table if provided
        if let Some(email) = email {
            db.prepare("UPDATE students SET email = ?1, updated_at = ?2 WHERE student_id = ?3")
                .bind(&[
                    JsValue::from_str(email),
                    JsValue::from_str(now),
                    JsValue::from_str(&student_id),
                ])
                .map_err(|e| format!("Failed to bind email update: {:?}", e))?
                .run()
                .await
                .map_err(|e| format!("Failed to update email: {:?}", e))?;
        }

        // Fetch display_name from students
        let student_row = db
            .prepare("SELECT display_name FROM students WHERE student_id = ?1")
            .bind(&[JsValue::from_str(&student_id)])
            .map_err(|e| format!("Failed to bind student query: {:?}", e))?
            .first::<serde_json::Value>(None)
            .await
            .map_err(|e| format!("Failed to query student: {:?}", e))?;

        let existing_display_name = student_row
            .and_then(|r| r.get("display_name").and_then(|v| v.as_str().map(|s| s.to_string())));

        Ok((student_id, existing_display_name, false))
    } else {
        // New user -- generate UUID and insert atomically
        let student_id = generate_uuid_v4();

        let student_stmt = db.prepare(
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
        .map_err(|e| format!("Failed to bind student insert: {:?}", e))?;

        let identity_stmt = db.prepare(
            "INSERT INTO auth_identities (student_id, provider, provider_user_id, created_at) \
             VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(&[
            JsValue::from_str(&student_id),
            JsValue::from_str(provider),
            JsValue::from_str(provider_user_id),
            JsValue::from_str(now),
        ])
        .map_err(|e| format!("Failed to bind identity insert: {:?}", e))?;

        // Batch executes both inserts atomically
        db.batch(vec![student_stmt, identity_stmt])
            .await
            .map_err(|e| format!("Failed to create student: {:?}", e))?;

        Ok((student_id, display_name.map(|s| s.to_string()), true))
    }
}

/// Extract and verify JWT from cookie or Authorization header.
/// Returns the student_id (subject claim) on success.
pub fn verify_auth(
    headers: &http::HeaderMap,
    env: &Env,
) -> Result<String, http::Response<axum::body::Body>> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let token = extract_token_from_cookie(headers)
        .or_else(|| extract_token_from_bearer(headers));

    let token = token.ok_or_else(|| {
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(
                r#"{"error":"Missing or invalid Authorization header"}"#,
            ))
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

    let claims = jwt::verify(&token, jwt_secret.as_bytes()).map_err(|e| {
        console_log!("JWT verification failed: {}", e);
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("Content-Type", "application/json")
            .body(Body::from(format!(r#"{{"error":"{}"}}"#, e)))
            .unwrap()
    })?;

    Ok(claims.sub)
}

/// Alias for backward compatibility.
pub fn verify_auth_header(
    headers: &http::HeaderMap,
    env: &Env,
) -> Result<String, http::Response<axum::body::Body>> {
    verify_auth(headers, env)
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

/// Handle GET /api/auth/me -- return current user info from JWT.
pub async fn handle_auth_me(
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
            console_log!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database connection failed"}"#))
                .unwrap();
        }
    };

    let row = match db
        .prepare("SELECT student_id, email, display_name FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(&student_id)])
        .map_err(|e| format!("{:?}", e))
        .and_then(|stmt| Ok(stmt))
    {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                return Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"User not found"}"#))
                    .unwrap();
            }
            Err(e) => {
                console_log!("Failed to query student: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Database query failed"}"#))
                    .unwrap();
            }
        },
        Err(e) => {
            console_log!("Failed to bind query: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database query failed"}"#))
                .unwrap();
        }
    };

    let response = AuthResponse {
        student_id: row
            .get("student_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default(),
        email: row
            .get("email")
            .and_then(|v| v.as_str().map(|s| s.to_string())),
        display_name: row
            .get("display_name")
            .and_then(|v| v.as_str().map(|s| s.to_string())),
        is_new_user: false,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// Handle POST /api/auth/debug -- dev-only login that bypasses Apple Sign In.
/// Only works when ENVIRONMENT is not "production".
pub async fn handle_debug_auth(
    env: &Env,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Block in production
    let environment = env
        .var("ENVIRONMENT")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if environment == "production" {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Not found"}"#))
            .unwrap();
    }

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
            console_log!("Failed to find/create debug student: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create debug user"}"#))
                .unwrap();
        }
    };

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
        sub: student_id.clone(),
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

    let response_body = serde_json::json!({
        "student_id": student_id,
        "email": "dev@localhost",
        "display_name": display_name,
        "is_new_user": is_new_user,
        "token": token,
    });

    let cookie = format!(
        "token={}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}",
        token, JWT_EXPIRY_SECONDS
    );

    console_log!("Debug auth: student_id={}, new={}", student_id, is_new_user);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Set-Cookie", cookie)
        .body(Body::from(response_body.to_string()))
        .unwrap()
}

/// Handle POST /api/auth/signout -- clear the auth cookie.
pub fn handle_signout() -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header(
            "Set-Cookie",
            "token=; HttpOnly; Secure; SameSite=None; Path=/; Max-Age=0",
        )
        .body(Body::from(r#"{"ok":true}"#))
        .unwrap()
}
