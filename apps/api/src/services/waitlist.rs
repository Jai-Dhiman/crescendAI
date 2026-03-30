use axum::extract::{Json, State};
use wasm_bindgen::JsValue;
use worker::console_error;

use crate::error::{ApiError, Result};
use crate::state::AppState;

/// Validate email: non-empty local@domain.tld, no whitespace, 3-254 chars.
fn is_valid_email(email: &str) -> bool {
    if email.len() < 3 || email.len() > 254 {
        return false;
    }
    let parts: Vec<&str> = email.splitn(2, '@').collect();
    let [local, domain] = parts.as_slice() else {
        return false;
    };
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

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitlistRequest {
    pub email: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct WaitlistResponse {
    pub ok: bool,
}

#[worker::send]
pub async fn handle_waitlist(
    State(state): State<AppState>,
    Json(request): Json<WaitlistRequest>,
) -> Result<Json<WaitlistResponse>> {
    // Honeypot: if "name" field is non-empty, silently accept
    if let Some(ref name) = request.name {
        if !name.is_empty() {
            return Ok(Json(WaitlistResponse { ok: true }));
        }
    }

    let email = request.email.trim().to_lowercase();
    if email.is_empty() || !is_valid_email(&email) {
        return Err(ApiError::BadRequest("Invalid email".into()));
    }

    // Truncate context to 500 chars (char-boundary safe)
    let context: Option<String> = request
        .context
        .as_deref()
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.chars().count() > 500 {
                trimmed.chars().take(500).collect::<String>()
            } else {
                trimmed.to_string()
            }
        })
        .filter(|s| !s.is_empty());

    let db = state.db.d1()?;

    let now = crate::types::now_iso();

    let context_val = match &context {
        Some(c) => JsValue::from_str(c),
        None => JsValue::NULL,
    };

    let stmt = db
        .prepare("INSERT OR IGNORE INTO waitlist (email, context, source, created_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(&email),
            context_val,
            JsValue::from_str("web"),
            JsValue::from_str(&now),
        ])
        .map_err(|e| {
            console_error!("Waitlist bind failed: {:?}", e);
            ApiError::Internal("Database error".into())
        })?;

    stmt.run().await.map_err(|e| {
        console_error!("Waitlist insert failed: {:?}", e);
        ApiError::Internal("Database error".into())
    })?;

    Ok(Json(WaitlistResponse { ok: true }))
}
