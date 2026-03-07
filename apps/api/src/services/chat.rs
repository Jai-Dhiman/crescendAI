//! Handlers for conversation CRUD and streaming chat.

use wasm_bindgen::JsValue;
use worker::{console_log, Env};

// --- Request / Response types ---

#[derive(serde::Deserialize)]
pub struct ChatRequest {
    pub conversation_id: Option<String>,
    pub message: String,
}

#[derive(serde::Serialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: Option<String>,
    pub updated_at: String,
}

#[derive(serde::Serialize)]
pub struct ConversationDetail {
    pub id: String,
    pub title: Option<String>,
    pub created_at: String,
}

#[derive(serde::Serialize)]
pub struct MessageRow {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(serde::Serialize)]
pub struct ConversationWithMessages {
    pub conversation: ConversationDetail,
    pub messages: Vec<MessageRow>,
}

#[derive(serde::Serialize)]
pub struct ConversationList {
    pub conversations: Vec<ConversationSummary>,
}

// --- Handlers ---

/// GET /api/conversations -- list conversations for sidebar
pub async fn handle_list_conversations(
    env: &Env,
    headers: &http::HeaderMap,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
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

    let stmt = match db
        .prepare("SELECT id, title, updated_at FROM conversations WHERE student_id = ?1 ORDER BY updated_at DESC")
        .bind(&[JsValue::from_str(&student_id)])
    {
        Ok(s) => s,
        Err(e) => {
            console_log!("Failed to bind list query: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query preparation failed"}"#))
                .unwrap();
        }
    };

    let results = match stmt.all().await {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to query conversations: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query failed"}"#))
                .unwrap();
        }
    };

    let rows: Vec<serde_json::Value> = results.results().unwrap_or_default();
    let conversations: Vec<ConversationSummary> = rows
        .iter()
        .filter_map(|row| {
            Some(ConversationSummary {
                id: row.get("id")?.as_str()?.to_string(),
                title: row.get("title").and_then(|v| v.as_str()).map(|s| s.to_string()),
                updated_at: row.get("updated_at")?.as_str()?.to_string(),
            })
        })
        .collect();

    let response = ConversationList { conversations };
    let json = serde_json::to_string(&response)
        .unwrap_or_else(|_| r#"{"conversations":[]}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// GET /api/conversations/:id -- load conversation with messages
pub async fn handle_get_conversation(
    env: &Env,
    headers: &http::HeaderMap,
    conversation_id: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
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

    // Fetch conversation (verify ownership)
    let conv_row: Option<serde_json::Value> = match db
        .prepare("SELECT id, title, created_at FROM conversations WHERE id = ?1 AND student_id = ?2")
        .bind(&[JsValue::from_str(conversation_id), JsValue::from_str(&student_id)])
    {
        Ok(s) => match s.first(None).await {
            Ok(r) => r,
            Err(e) => {
                console_log!("Failed to query conversation: {:?}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Query failed"}"#))
                    .unwrap();
            }
        },
        Err(e) => {
            console_log!("Failed to bind conversation query: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query preparation failed"}"#))
                .unwrap();
        }
    };

    let conv_row = match conv_row {
        Some(r) => r,
        None => {
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Conversation not found"}"#))
                .unwrap();
        }
    };

    let conversation = ConversationDetail {
        id: conv_row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        title: conv_row.get("title").and_then(|v| v.as_str()).map(|s| s.to_string()),
        created_at: conv_row.get("created_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    };

    // Fetch messages
    let msg_stmt = match db
        .prepare("SELECT id, role, content, created_at FROM messages WHERE conversation_id = ?1 ORDER BY created_at ASC")
        .bind(&[JsValue::from_str(conversation_id)])
    {
        Ok(s) => s,
        Err(e) => {
            console_log!("Failed to bind messages query: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query preparation failed"}"#))
                .unwrap();
        }
    };

    let msg_results = match msg_stmt.all().await {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to query messages: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Query failed"}"#))
                .unwrap();
        }
    };

    let msg_rows: Vec<serde_json::Value> = msg_results.results().unwrap_or_default();
    let messages: Vec<MessageRow> = msg_rows
        .iter()
        .filter_map(|row| {
            Some(MessageRow {
                id: row.get("id")?.as_str()?.to_string(),
                role: row.get("role")?.as_str()?.to_string(),
                content: row.get("content")?.as_str()?.to_string(),
                created_at: row.get("created_at")?.as_str()?.to_string(),
            })
        })
        .collect();

    let response = ConversationWithMessages { conversation, messages };
    let json = serde_json::to_string(&response)
        .unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// DELETE /api/conversations/:id -- delete conversation and its messages
pub async fn handle_delete_conversation(
    env: &Env,
    headers: &http::HeaderMap,
    conversation_id: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
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

    // Delete messages first (FK), then conversation (verify ownership)
    if let Ok(stmt) = db
        .prepare("DELETE FROM messages WHERE conversation_id = ?1")
        .bind(&[JsValue::from_str(conversation_id)])
    {
        let _ = stmt.run().await;
    }

    if let Ok(stmt) = db
        .prepare("DELETE FROM conversations WHERE id = ?1 AND student_id = ?2")
        .bind(&[JsValue::from_str(conversation_id), JsValue::from_str(&student_id)])
    {
        let _ = stmt.run().await;
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .unwrap()
}
