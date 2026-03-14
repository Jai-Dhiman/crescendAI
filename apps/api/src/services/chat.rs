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

// --- Streaming chat handler ---

use crate::services::llm;
use crate::services::prompts;
use futures_util::StreamExt;

fn format_sse(data: &serde_json::Value) -> Vec<u8> {
    format!("event: message\ndata: {}\n\n", data).into_bytes()
}

fn error_worker_response(status: u16, msg: &str) -> worker::Response {
    let body = format!(r#"{{"error":"{}"}}"#, msg);
    let mut resp = worker::Response::from_bytes(body.into_bytes()).unwrap();
    resp = resp.with_status(status);
    let _ = resp.headers_mut().set("Content-Type", "application/json");
    resp
}

/// POST /api/chat -- streaming chat with the piano teacher.
///
/// True token-by-token streaming: reads the Anthropic SSE stream chunk by chunk
/// and forwards each text delta to the client immediately.
pub async fn handle_chat_stream(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> worker::Response {
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => {
            let status = err_response.status().as_u16();
            return error_worker_response(status, "Authentication failed");
        }
    };

    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse chat request: {:?}", e);
            return error_worker_response(400, "Invalid request body");
        }
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_log!("D1 binding failed: {:?}", e);
            return error_worker_response(500, "Database connection failed");
        }
    };

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    // Create or validate conversation
    let conversation_id = match &request.conversation_id {
        Some(id) => {
            let exists = verify_conversation_ownership(&db, id, &student_id).await;
            if !exists {
                return error_worker_response(404, "Conversation not found");
            }
            if let Ok(stmt) = db
                .prepare("UPDATE conversations SET updated_at = ?1 WHERE id = ?2")
                .bind(&[JsValue::from_str(&now), JsValue::from_str(id)])
            {
                let _ = stmt.run().await;
            }
            id.clone()
        }
        None => {
            let id = crate::services::ask::generate_uuid();
            if let Err(e) = create_conversation(&db, &id, &student_id, &now).await {
                console_log!("Failed to create conversation: {}", e);
                return error_worker_response(500, "Failed to create conversation");
            }
            id
        }
    };

    // Store user message
    let user_msg_id = crate::services::ask::generate_uuid();
    if let Err(e) = store_message(&db, &user_msg_id, &conversation_id, "user", &request.message, &now).await {
        console_log!("Failed to store user message: {}", e);
    }

    // Fetch conversation history, student context, and memory
    let history = fetch_messages(&db, &conversation_id).await.unwrap_or_default();
    let student_row = fetch_student_row(&db, &student_id).await;
    let today = &now[..10.min(now.len())];
    let memory_ctx = crate::services::memory::build_memory_context(env, &student_id, None, today, Some(&request.message)).await;
    let memory_patterns = crate::services::memory::format_chat_memory_patterns(&memory_ctx);
    let recent_obs = crate::services::memory::format_chat_recent_observations(&memory_ctx);
    let student_facts = crate::services::memory::format_student_reported_context(&memory_ctx);
    let student_context = student_row.as_ref().and_then(|row| {
        prompts::build_chat_student_context(row, &memory_patterns, &recent_obs, &student_facts)
    });

    // Build messages array for Anthropic
    let mut llm_messages: Vec<llm::LlmMessage> = Vec::new();

    if let Some(ctx) = student_context {
        llm_messages.push(llm::LlmMessage {
            role: "user".to_string(),
            content: ctx,
        });
        llm_messages.push(llm::LlmMessage {
            role: "assistant".to_string(),
            content: "Understood, I'll keep this context in mind.".to_string(),
        });
    }

    for msg in &history {
        llm_messages.push(llm::LlmMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        });
    }

    // Call Anthropic streaming API
    let mut anthropic_response = match llm::call_anthropic_stream(
        env,
        prompts::CHAT_SYSTEM,
        llm_messages,
        4096,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            console_log!("Anthropic stream call failed: {}", e);
            let assistant_msg_id = crate::services::ask::generate_uuid();
            let fallback = "I'm having trouble responding right now. Could you try again?";
            let _ = store_message(&db, &assistant_msg_id, &conversation_id, "assistant", fallback, &now).await;

            let mut sse = Vec::new();
            sse.extend_from_slice(&format_sse(&serde_json::json!({"type": "start", "conversation_id": conversation_id, "message_id": assistant_msg_id})));
            sse.extend_from_slice(&format_sse(&serde_json::json!({"type": "delta", "text": fallback})));
            sse.extend_from_slice(&format_sse(&serde_json::json!({"type": "done", "message_id": assistant_msg_id})));

            let mut resp = worker::Response::from_bytes(sse).unwrap();
            resp = resp.with_status(200);
            let _ = resp.headers_mut().set("Content-Type", "text/event-stream");
            let _ = resp.headers_mut().set("Cache-Control", "no-cache");
            return resp;
        }
    };

    // Get the Anthropic response as a byte stream
    let byte_stream = match anthropic_response.stream() {
        Ok(s) => s,
        Err(e) => {
            console_log!("Failed to get Anthropic stream: {:?}", e);
            return error_worker_response(500, "Stream read failed");
        }
    };

    // Set up a channel to feed our SSE events to the response stream
    let (tx, rx) = futures_channel::mpsc::unbounded::<Result<Vec<u8>, worker::Error>>();

    let assistant_msg_id = crate::services::ask::generate_uuid();
    let is_first_exchange = history.len() <= 1;
    let user_message = request.message.clone();

    // Send the start event immediately
    let _ = tx.unbounded_send(Ok(format_sse(&serde_json::json!({
        "type": "start",
        "conversation_id": conversation_id,
        "message_id": assistant_msg_id
    }))));

    // Spawn background task to consume Anthropic stream and forward deltas
    let env_clone = env.clone();
    let student_id_clone = student_id.clone();
    let user_message_clone = user_message.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let mut line_buffer = String::new();
        let mut full_text = String::new();
        let mut stream = byte_stream;

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    console_log!("Stream chunk error: {:?}", e);
                    break;
                }
            };

            let text = String::from_utf8_lossy(&chunk);
            line_buffer.push_str(&text);

            // Process complete lines
            while let Some(newline_pos) = line_buffer.find('\n') {
                let line = line_buffer[..newline_pos].trim_end().to_string();
                line_buffer = line_buffer[newline_pos + 1..].to_string();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                        if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                            if let Some(delta_text) = event
                                .get("delta")
                                .and_then(|d| d.get("text"))
                                .and_then(|t| t.as_str())
                            {
                                full_text.push_str(delta_text);
                                let _ = tx.unbounded_send(Ok(format_sse(
                                    &serde_json::json!({"type": "delta", "text": delta_text}),
                                )));
                            }
                        }
                    }
                }
            }
        }

        if full_text.is_empty() {
            full_text = "I'm having trouble responding right now. Could you try again?".to_string();
            let _ = tx.unbounded_send(Ok(format_sse(
                &serde_json::json!({"type": "delta", "text": &full_text}),
            )));
        }

        // Send done event
        let _ = tx.unbounded_send(Ok(format_sse(
            &serde_json::json!({"type": "done", "message_id": assistant_msg_id}),
        )));

        // Store assistant message in D1 (before closing stream -- Workers
        // runtime may terminate outbound fetches after the response ends)
        let assistant_now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
        if let Err(e) = store_message(&db, &assistant_msg_id, &conversation_id, "assistant", &full_text, &assistant_now).await {
            console_log!("Failed to store assistant message: {}", e);
        }

        // Generate title if first exchange
        if is_first_exchange {
            generate_title(&env_clone, &db, &conversation_id, &user_message, &full_text).await;
        }

        // Extract and store chat memory facts
        if let Err(e) = crate::services::memory::extract_and_store_chat_facts(
            &env_clone, &student_id_clone, &user_message_clone, &full_text,
        ).await {
            console_log!("Chat memory extraction failed (non-fatal): {}", e);
        }

        // Close the channel last -- Workers runtime may kill outbound
        // fetches once the response stream ends
        drop(tx);
    });

    // Return streaming response immediately
    let mut resp = worker::Response::from_stream(rx).unwrap();
    resp = resp.with_status(200);
    let _ = resp.headers_mut().set("Content-Type", "text/event-stream");
    let _ = resp.headers_mut().set("Cache-Control", "no-cache");
    resp
}

// --- Helper functions ---

async fn verify_conversation_ownership(db: &worker::D1Database, id: &str, student_id: &str) -> bool {
    match db
        .prepare("SELECT id FROM conversations WHERE id = ?1 AND student_id = ?2")
        .bind(&[JsValue::from_str(id), JsValue::from_str(student_id)])
    {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(Some(_)) => true,
            _ => false,
        },
        Err(_) => false,
    }
}

async fn create_conversation(
    db: &worker::D1Database,
    id: &str,
    student_id: &str,
    now: &str,
) -> Result<(), String> {
    db.prepare("INSERT INTO conversations (id, student_id, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(id),
            JsValue::from_str(student_id),
            JsValue::from_str(now),
            JsValue::from_str(now),
        ])
        .map_err(|e| format!("Failed to bind create conversation: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to insert conversation: {:?}", e))?;
    Ok(())
}

async fn store_message(
    db: &worker::D1Database,
    id: &str,
    conversation_id: &str,
    role: &str,
    content: &str,
    now: &str,
) -> Result<(), String> {
    db.prepare("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)")
        .bind(&[
            JsValue::from_str(id),
            JsValue::from_str(conversation_id),
            JsValue::from_str(role),
            JsValue::from_str(content),
            JsValue::from_str(now),
        ])
        .map_err(|e| format!("Failed to bind insert message: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to insert message: {:?}", e))?;
    Ok(())
}

async fn fetch_messages(db: &worker::D1Database, conversation_id: &str) -> Result<Vec<MessageRow>, String> {
    let stmt = db
        .prepare("SELECT id, role, content, created_at FROM messages WHERE conversation_id = ?1 ORDER BY created_at ASC")
        .bind(&[JsValue::from_str(conversation_id)])
        .map_err(|e| format!("Failed to bind fetch messages: {:?}", e))?;

    let results = stmt.all().await.map_err(|e| format!("Failed to query messages: {:?}", e))?;
    let rows: Vec<serde_json::Value> = results.results().unwrap_or_default();

    Ok(rows
        .iter()
        .filter_map(|row| {
            Some(MessageRow {
                id: row.get("id")?.as_str()?.to_string(),
                role: row.get("role")?.as_str()?.to_string(),
                content: row.get("content")?.as_str()?.to_string(),
                created_at: row.get("created_at")?.as_str()?.to_string(),
            })
        })
        .collect())
}

async fn fetch_student_row(db: &worker::D1Database, student_id: &str) -> Option<serde_json::Value> {
    db.prepare("SELECT inferred_level, explicit_goals, baseline_dynamics, baseline_timing, baseline_pedaling, baseline_articulation, baseline_phrasing, baseline_interpretation FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .ok()?
        .first(None)
        .await
        .ok()?
}

async fn generate_title(env: &Env, db: &worker::D1Database, conversation_id: &str, user_msg: &str, assistant_msg: &str) {
    let prompt = prompts::build_title_prompt(user_msg, assistant_msg);
    match llm::call_anthropic(env, "Generate a short title.", &prompt, 30).await {
        Ok(title) => {
            let title = title.trim().trim_matches('"').to_string();
            if let Ok(stmt) = db
                .prepare("UPDATE conversations SET title = ?1 WHERE id = ?2")
                .bind(&[JsValue::from_str(&title), JsValue::from_str(conversation_id)])
            {
                let _ = stmt.run().await;
            }
        }
        Err(e) => {
            console_log!("Title generation failed: {}", e);
        }
    }
}
