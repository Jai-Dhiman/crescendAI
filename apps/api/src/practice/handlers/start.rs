use axum::extract::{Json, State};
use wasm_bindgen::JsValue;
use worker::Env;

use crate::auth::extractor::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;
use crate::types::{ConversationId, SessionId};

/// Request body for POST /api/practice/start.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StartRequest {
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// Response body for POST /api/practice/start.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartResponse {
    pub session_id: SessionId,
    pub conversation_id: ConversationId,
}

#[worker::send]
pub async fn handle_start(
    State(state): State<AppState>,
    auth: AuthUser,
    body: axum::body::Bytes,
) -> Result<Json<StartResponse>> {
    let student_id = auth.student_id;
    let env = state.practice.env();

    // Pre-warm HF inference endpoint (fire-and-forget)
    prewarm_hf_endpoint(env);

    let session_id = SessionId::new();
    let now = crate::types::now_iso();

    // Parse optional conversation_id from request body
    let conversation_id = if body.is_empty() {
        None
    } else {
        serde_json::from_slice::<StartRequest>(&body)
            .ok()
            .and_then(|r| r.conversation_id)
    };

    // Get D1 database binding
    let db = state.db.d1()?;

    // If no conversation_id provided, create a new conversation
    let conversation_id: ConversationId = if let Some(id) = conversation_id {
        ConversationId::from(id)
    } else {
        let new_id = ConversationId::new();
        db.prepare("INSERT INTO conversations (id, student_id, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)")
            .bind(&[
                JsValue::from_str(new_id.as_str()),
                JsValue::from_str(student_id.as_str()),
                JsValue::from_str(&now),
                JsValue::from_str(&now),
            ])
            .map_err(|e| {
                worker::console_error!("Failed to bind conversation insert: {:?}", e);
                ApiError::Internal("Failed to create conversation".into())
            })?
            .run()
            .await
            .map_err(|e| {
                worker::console_error!("Failed to insert conversation: {:?}", e);
                ApiError::Internal("Failed to create conversation".into())
            })?;
        new_id
    };

    // Insert session row linked to conversation
    db.prepare("INSERT INTO sessions (id, student_id, started_at, conversation_id) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(session_id.as_str()),
            JsValue::from_str(student_id.as_str()),
            JsValue::from_str(&now),
            JsValue::from_str(conversation_id.as_str()),
        ])
        .map_err(|e| {
            worker::console_error!("Failed to bind session insert: {:?}", e);
            ApiError::Internal("Failed to create session".into())
        })?
        .run()
        .await
        .map_err(|e| {
            worker::console_error!("Failed to insert session: {:?}", e);
            ApiError::Internal("Failed to create session".into())
        })?;

    // Insert a session_start message into the conversation
    let msg_id = crate::types::generate_uuid_v4();
    let msg_result = db
        .prepare("INSERT INTO messages (id, conversation_id, role, content, created_at, message_type, session_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
        .bind(&[
            JsValue::from_str(&msg_id),
            JsValue::from_str(conversation_id.as_str()),
            JsValue::from_str("assistant"),
            JsValue::from_str("Practice session started"),
            JsValue::from_str(&now),
            JsValue::from_str("session_start"),
            JsValue::from_str(session_id.as_str()),
        ]);
    match msg_result {
        Ok(stmt) => {
            if let Err(e) = stmt.run().await {
                worker::console_error!("Failed to insert session_start message: {:?}", e);
                // Non-fatal: session and conversation already created
            }
        }
        Err(e) => {
            worker::console_error!("Failed to bind session_start message: {:?}", e);
        }
    }

    Ok(Json(StartResponse {
        session_id,
        conversation_id,
    }))
}

/// Fire-and-forget ping to HF endpoint to trigger scale-up from zero.
/// Does not block the response -- the endpoint wakes in the background.
fn prewarm_hf_endpoint(env: &Env) {
    let endpoint = match env.var("HF_INFERENCE_ENDPOINT") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };
    let token = match env.secret("HF_TOKEN") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };

    // Spawn a non-awaited fetch -- we don't care about the response
    wasm_bindgen_futures::spawn_local(async move {
        let headers = worker::Headers::new();
        let _ = headers.set("Authorization", &format!("Bearer {token}"));
        let _ = headers.set("Content-Type", "application/json");

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        // Minimal JSON body -- will get 400 but triggers the scale-up
        init.with_body(Some(JsValue::from_str("{}")));

        let Ok(req) = worker::Request::new_with_init(&endpoint, &init) else {
            return;
        };
        let result = worker::Fetch::Request(req).send().await;
        match result {
            Ok(resp) => {
                let status = resp.status_code();
                if status == 503 {
                    worker::console_log!("HF pre-warm: endpoint is cold (503), wake-up triggered");
                } else {
                    worker::console_log!("HF pre-warm: endpoint responded with {}", status);
                }
            }
            Err(e) => {
                worker::console_error!("HF pre-warm failed: {:?}", e);
            }
        }
    });
}
