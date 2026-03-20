use wasm_bindgen::JsValue;
use worker::Env;

pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Pre-warm HF inference endpoint (fire-and-forget)
    // This wakes the endpoint from scale-to-zero so it's ready by the time
    // the first 15s audio chunk arrives.
    prewarm_hf_endpoint(env);

    let session_id = crate::services::ask::generate_uuid();
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    // Parse optional conversation_id from request body
    let conversation_id = if body.is_empty() {
        None
    } else {
        serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.get("conversationId")?.as_str().map(|s| s.to_string()))
    };

    // Get D1 database binding
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            worker::console_error!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Internal server error"}"#))
                .unwrap();
        }
    };

    // If no conversation_id provided, create a new conversation
    let conversation_id = match conversation_id {
        Some(id) => id,
        None => {
            let new_id = crate::services::ask::generate_uuid();
            let conv_result = db
                .prepare("INSERT INTO conversations (id, student_id, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)")
                .bind(&[
                    JsValue::from_str(&new_id),
                    JsValue::from_str(&student_id),
                    JsValue::from_str(&now),
                    JsValue::from_str(&now),
                ]);
            match conv_result {
                Ok(stmt) => {
                    if let Err(e) = stmt.run().await {
                        worker::console_error!("Failed to insert conversation: {:?}", e);
                        return Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .header("Content-Type", "application/json")
                            .body(Body::from(r#"{"error":"Failed to create conversation"}"#))
                            .unwrap();
                    }
                }
                Err(e) => {
                    worker::console_error!("Failed to bind conversation insert: {:?}", e);
                    return Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .header("Content-Type", "application/json")
                        .body(Body::from(r#"{"error":"Failed to create conversation"}"#))
                        .unwrap();
                }
            }
            new_id
        }
    };

    // Insert session row linked to conversation
    let session_result = db
        .prepare("INSERT INTO sessions (id, student_id, started_at, conversation_id) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(&session_id),
            JsValue::from_str(&student_id),
            JsValue::from_str(&now),
            JsValue::from_str(&conversation_id),
        ]);
    match session_result {
        Ok(stmt) => {
            if let Err(e) = stmt.run().await {
                worker::console_error!("Failed to insert session: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Failed to create session"}"#))
                    .unwrap();
            }
        }
        Err(e) => {
            worker::console_error!("Failed to bind session insert: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to create session"}"#))
                .unwrap();
        }
    }

    // Insert a session_start message into the conversation
    let msg_id = crate::services::ask::generate_uuid();
    let msg_result = db
        .prepare("INSERT INTO messages (id, conversation_id, role, content, created_at, message_type, session_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
        .bind(&[
            JsValue::from_str(&msg_id),
            JsValue::from_str(&conversation_id),
            JsValue::from_str("assistant"),
            JsValue::from_str("Practice session started"),
            JsValue::from_str(&now),
            JsValue::from_str("session_start"),
            JsValue::from_str(&session_id),
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

    let resp = serde_json::json!({
        "sessionId": session_id,
        "conversationId": conversation_id,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}

/// Fire-and-forget ping to HF endpoint to trigger scale-up from zero.
/// Does not block the response — the endpoint wakes in the background.
fn prewarm_hf_endpoint(env: &Env) {
    let endpoint = match env.var("HF_INFERENCE_ENDPOINT") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };
    let token = match env.secret("HF_TOKEN") {
        Ok(v) => v.to_string(),
        Err(_) => return,
    };

    // Spawn a non-awaited fetch — we don't care about the response
    wasm_bindgen_futures::spawn_local(async move {
        let headers = worker::Headers::new();
        let _ = headers.set("Authorization", &format!("Bearer {}", token));
        let _ = headers.set("Content-Type", "application/json");

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        // Minimal JSON body — will get 400 but triggers the scale-up
        init.with_body(Some(JsValue::from_str("{}")));

        let req = match worker::Request::new_with_init(&endpoint, &init) {
            Ok(r) => r,
            Err(_) => return,
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
                worker::console_log!("HF pre-warm failed: {:?}", e);
            }
        }
    });
}
