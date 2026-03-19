use js_sys;
use wasm_bindgen::JsValue;
use worker::{console_error, Env};

/// Validate email: non-empty local@domain.tld, no whitespace, 3-254 chars.
fn is_valid_email(email: &str) -> bool {
    if email.len() < 3 || email.len() > 254 {
        return false;
    }
    let parts: Vec<&str> = email.splitn(2, '@').collect();
    if parts.len() != 2 {
        return false;
    }
    let local = parts[0];
    let domain = parts[1];
    if local.is_empty() || domain.is_empty() {
        return false;
    }
    if email.contains(char::is_whitespace) {
        return false;
    }
    // Domain must contain a dot with non-empty TLD
    match domain.rfind('.') {
        Some(dot_pos) => dot_pos > 0 && dot_pos < domain.len() - 1,
        None => false,
    }
}

pub async fn handle_waitlist(
    env: &Env,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            return http::Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Invalid JSON"}"#))
                .unwrap();
        }
    };

    // Honeypot: if "name" field is non-empty, silently accept
    if let Some(name) = parsed.get("name").and_then(|v| v.as_str()) {
        if !name.is_empty() {
            return http::Response::builder()
                .status(http::StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"ok":true}"#))
                .unwrap();
        }
    }

    let email = match parsed.get("email").and_then(|v| v.as_str()) {
        Some(e) => e.trim().to_lowercase(),
        None => {
            return http::Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Invalid email"}"#))
                .unwrap();
        }
    };

    if !is_valid_email(&email) {
        return http::Response::builder()
            .status(http::StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Invalid email"}"#))
            .unwrap();
    }

    // Truncate context to 500 chars (char-boundary safe)
    let context: Option<String> = parsed
        .get("context")
        .and_then(|v| v.as_str())
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.chars().count() > 500 {
                trimmed.chars().take(500).collect::<String>()
            } else {
                trimmed.to_string()
            }
        })
        .filter(|s| !s.is_empty());

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return http::Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
                .unwrap();
        }
    };

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    let context_val = match &context {
        Some(c) => JsValue::from_str(c),
        None => JsValue::NULL,
    };

    let stmt = match db
        .prepare("INSERT OR IGNORE INTO waitlist (email, context, source, created_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(&email),
            context_val,
            JsValue::from_str("web"),
            JsValue::from_str(&now),
        ]) {
        Ok(stmt) => stmt,
        Err(e) => {
            console_error!("Waitlist bind failed: {:?}", e);
            return http::Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
                .unwrap();
        }
    };

    if let Err(e) = stmt.run().await {
        console_error!("Waitlist insert failed: {:?}", e);
        return http::Response::builder()
            .status(http::StatusCode::INTERNAL_SERVER_ERROR)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
            .unwrap();
    }

    http::Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(r#"{"ok":true}"#))
        .unwrap()
}
