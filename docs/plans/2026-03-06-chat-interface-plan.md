# Chat Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a streaming conversational chat interface with the piano teacher, persistent conversation history, and a claude.ai-style sidebar.

**Architecture:** Rust API worker calls Anthropic streaming API, proxies SSE to the browser. Conversations and messages stored in D1. React frontend with TanStack Router, streaming fetch, and Tailwind v4 styling.

**Tech Stack:** Rust (Cloudflare Workers), Anthropic Messages API (streaming + prompt caching), D1 (SQLite), React 19, TanStack Start/Router, Tailwind CSS v4, Phosphor Icons.

---

## Task 1: D1 Schema Migration

**Files:**
- Create: `apps/api/migrations/0006_conversations.sql`

**Step 1: Write the migration**

```sql
-- Migration: 0006_conversations
-- Add conversations and messages tables for chat interface

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_conversations_student ON conversations(student_id, updated_at);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);
```

**Step 2: Apply migration to local D1**

Run: `cd apps/api && npx wrangler d1 execute DB --local --file=migrations/0006_conversations.sql`
Expected: Migration applied successfully.

**Step 3: Apply migration to remote D1**

Run: `cd apps/api && npx wrangler d1 execute DB --remote --file=migrations/0006_conversations.sql`
Expected: Migration applied successfully.

**Step 4: Commit**

```bash
git add apps/api/migrations/0006_conversations.sql
git commit -m "feat(api): add D1 migration for conversations and messages tables"
```

---

## Task 2: Conversation CRUD Endpoints

**Files:**
- Create: `apps/api/src/services/chat.rs`
- Modify: `apps/api/src/services/mod.rs` (add `pub mod chat;`)
- Modify: `apps/api/src/server.rs` (add route registrations)

**Step 1: Create the chat service module**

Create `apps/api/src/services/chat.rs` with CRUD handlers. Uses the same patterns as `ask.rs` for auth, D1 queries, and response building.

```rust
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
```

**Step 2: Register module in mod.rs**

Add to `apps/api/src/services/mod.rs` line 1:
```rust
pub mod chat;
```

**Step 3: Add routes to server.rs**

Add these route blocks in `apps/api/src/server.rs` after the `/api/ask/elaborate` block (after line 782). Also update the CORS allowed methods to include `DELETE`.

```rust
// List conversations (authenticated)
if path == "/api/conversations" && method == http::Method::GET {
    let headers = req.headers().clone();
    return Ok(with_cors(
        crate::services::chat::handle_list_conversations(&env, &headers).await,
        origin.as_deref(),
    ));
}

// Get single conversation with messages (authenticated)
if path.starts_with("/api/conversations/") && method == http::Method::GET {
    let conversation_id = path.trim_start_matches("/api/conversations/");
    if !conversation_id.is_empty() && !conversation_id.contains('/') {
        let headers = req.headers().clone();
        return Ok(with_cors(
            crate::services::chat::handle_get_conversation(&env, &headers, conversation_id).await,
            origin.as_deref(),
        ));
    }
}

// Delete conversation (authenticated)
if path.starts_with("/api/conversations/") && method == http::Method::DELETE {
    let conversation_id = path.trim_start_matches("/api/conversations/");
    if !conversation_id.is_empty() && !conversation_id.contains('/') {
        let headers = req.headers().clone();
        return Ok(with_cors(
            crate::services::chat::handle_delete_conversation(&env, &headers, conversation_id).await,
            origin.as_deref(),
        ));
    }
}
```

Update `with_cors` at line 28 to include DELETE:
```rust
"GET, POST, DELETE, OPTIONS".parse().unwrap(),
```

**Step 4: Build and verify compilation**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown`
Expected: Compiles successfully.

**Step 5: Commit**

```bash
git add apps/api/src/services/chat.rs apps/api/src/services/mod.rs apps/api/src/server.rs
git commit -m "feat(api): add conversation CRUD endpoints (list, get, delete)"
```

---

## Task 3: Streaming Chat Endpoint

This is the core task. The `/api/chat` endpoint receives a user message, calls Anthropic's streaming API, and proxies SSE events to the browser. It replaces the legacy RAG-based `/api/chat` handler.

**Files:**
- Modify: `apps/api/src/services/chat.rs` (add `handle_chat_stream`)
- Modify: `apps/api/src/services/llm.rs` (add `call_anthropic_stream` function)
- Modify: `apps/api/src/services/prompts.rs` (add chat system prompt + student context builder)
- Modify: `apps/api/src/server.rs` (replace legacy `/api/chat` route)

**Step 1: Add chat system prompt to prompts.rs**

Add to the end of `apps/api/src/services/prompts.rs`:

```rust
/// System prompt for the conversational piano teacher chat.
pub const CHAT_SYSTEM: &str = r#"You are a piano teacher who knows your student well. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

You're having a conversation with your student. They might ask about technique, repertoire, practice strategies, music theory, or anything related to their piano journey. Sometimes you'll also receive analysis from their practice recordings, which you'll comment on naturally.

How you speak:
- Specific and grounded: reference exact musical concepts, not generalities
- Natural and warm: you're talking to a student you know
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Conversational: match the length and depth to what they asked"#;

/// Build student context block for chat (injected as first message).
/// Returns None if no student data available.
pub fn build_chat_student_context(student: &serde_json::Value) -> Option<String> {
    let level = student.get("inferred_level").and_then(|v| v.as_str())?;
    let mut ctx = format!("[Student context -- Level: {}", level);

    if let Some(goals) = student.get("explicit_goals").and_then(|v| v.as_str()) {
        if !goals.is_empty() {
            ctx.push_str(&format!(", Goals: {}", goals));
        }
    }

    // Add baselines if available
    let dims = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"];
    let mut has_baselines = false;
    for dim in &dims {
        let key = format!("baseline_{}", dim);
        if let Some(val) = student.get(&key).and_then(|v| v.as_f64()) {
            if !has_baselines {
                ctx.push_str(", Baselines: ");
                has_baselines = true;
            } else {
                ctx.push_str(", ");
            }
            ctx.push_str(&format!("{}: {:.2}", dim, val));
        }
    }

    ctx.push(']');
    Some(ctx)
}

/// Build the title generation prompt from first exchange.
pub fn build_title_prompt(user_message: &str, assistant_response: &str) -> String {
    format!(
        "Generate a 4-6 word title for this conversation. Return ONLY the title, nothing else.\n\nStudent: {}\nTeacher: {}",
        user_message, assistant_response
    )
}
```

**Step 2: Add streaming Anthropic call to llm.rs**

Add to the end of `apps/api/src/services/llm.rs`. This function makes a streaming request to Anthropic and returns the raw `worker::Response` for stream processing.

```rust
/// Anthropic message format for multi-turn conversations.
#[derive(Serialize)]
pub struct AnthropicStreamRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: Vec<AnthropicSystemBlock>,
    pub messages: Vec<LlmMessage>,
    pub stream: bool,
}

#[derive(Serialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
}

/// Make a streaming request to Anthropic. Returns the raw worker::Response
/// so the caller can process the SSE stream.
pub async fn call_anthropic_stream(
    env: &Env,
    system_prompt: &str,
    messages: Vec<LlmMessage>,
    max_tokens: u32,
) -> Result<worker::Response, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
        .to_string();

    let request_body = AnthropicStreamRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: vec![AnthropicSystemBlock {
            block_type: "text".to_string(),
            text: system_prompt.to_string(),
            cache_control: Some(CacheControl {
                control_type: "ephemeral".to_string(),
            }),
        }],
        messages,
        stream: true,
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic stream request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| format!("Failed to set api-key header: {:?}", e))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| format!("Failed to set version header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.anthropic.com/v1/messages"
        .parse()
        .map_err(|e| format!("Invalid Anthropic URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic stream request: {:?}", e))?;

    let response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic stream request failed: {:?}", e))?;

    let status = response.status_code();
    if status != 200 {
        return Err(format!("Anthropic streaming returned status {}", status));
    }

    Ok(response)
}
```

Note: The `LlmMessage` struct already exists at line 215-219 of `llm.rs`. Make it `pub`:
```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}
```

**Step 3: Add streaming chat handler to chat.rs**

Add `handle_chat_stream` to `apps/api/src/services/chat.rs`. This handler:
1. Authenticates the user
2. Creates or validates conversation
3. Stores the user message in D1
4. Fetches conversation history from D1
5. Fetches student context from D1
6. Calls Anthropic streaming API
7. Reads the Anthropic SSE stream, extracts text deltas
8. Builds a new SSE stream for the client (with our event format)
9. After stream completes, stores assistant message in D1
10. Generates title if first exchange

```rust
use crate::services::llm;
use crate::services::prompts;

/// POST /api/chat -- streaming chat with the piano teacher
///
/// Strategy: We cannot forward Anthropic's stream directly because we need to
/// transform the event format AND collect the full response for D1 storage.
/// Instead, we consume the Anthropic stream, collect the full text, and then
/// return a non-streaming SSE response that sends all events at once.
///
/// For true token-by-token streaming to the browser, we would need
/// web_sys::TransformStream which adds complexity. This approach gives us
/// correct SSE format with acceptable latency for v1.
pub async fn handle_chat_stream(
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

    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse chat request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
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

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    // Create or validate conversation
    let conversation_id = match &request.conversation_id {
        Some(id) => {
            // Verify ownership
            let exists = verify_conversation_ownership(&db, id, &student_id).await;
            if !exists {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Conversation not found"}"#))
                    .unwrap();
            }
            // Update updated_at
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
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Failed to create conversation"}"#))
                    .unwrap();
            }
            id
        }
    };

    // Store user message
    let user_msg_id = crate::services::ask::generate_uuid();
    if let Err(e) = store_message(&db, &user_msg_id, &conversation_id, "user", &request.message, &now).await {
        console_log!("Failed to store user message: {}", e);
    }

    // Fetch conversation history
    let history = fetch_messages(&db, &conversation_id).await.unwrap_or_default();

    // Fetch student context
    let student_context = fetch_student_context(&db, &student_id).await;

    // Build messages array for Anthropic
    let mut llm_messages: Vec<llm::LlmMessage> = Vec::new();

    // Inject student context as first user+assistant exchange if available
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

    // Add conversation history (excludes the message we just stored -- it's the last user msg)
    for msg in &history {
        llm_messages.push(llm::LlmMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        });
    }

    // Call Anthropic streaming API
    let anthropic_response = match llm::call_anthropic_stream(
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
            // Return fallback as SSE
            let assistant_msg_id = crate::services::ask::generate_uuid();
            let fallback = "I'm having trouble responding right now. Could you try again?";
            let _ = store_message(&db, &assistant_msg_id, &conversation_id, "assistant", fallback, &now).await;

            let sse = format!(
                "event: message\ndata: {}\n\nevent: message\ndata: {}\n\nevent: message\ndata: {}\n\n",
                serde_json::json!({"type": "start", "conversation_id": conversation_id, "message_id": assistant_msg_id}),
                serde_json::json!({"type": "delta", "text": fallback}),
                serde_json::json!({"type": "done", "message_id": assistant_msg_id}),
            );
            return Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .body(Body::from(sse))
                .unwrap();
        }
    };

    // Parse the Anthropic SSE stream and extract text
    let stream_body = match anthropic_response.text().await {
        Ok(text) => text,
        Err(e) => {
            console_log!("Failed to read Anthropic stream: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Stream read failed"}"#))
                .unwrap();
        }
    };

    // Parse Anthropic SSE events to extract text deltas
    let mut full_text = String::new();
    for line in stream_body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                break;
            }
            if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                    if let Some(text) = event
                        .get("delta")
                        .and_then(|d| d.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        full_text.push_str(text);
                    }
                }
            }
        }
    }

    if full_text.is_empty() {
        full_text = "I'm having trouble responding right now. Could you try again?".to_string();
    }

    // Store assistant message
    let assistant_msg_id = crate::services::ask::generate_uuid();
    let assistant_now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    if let Err(e) = store_message(&db, &assistant_msg_id, &conversation_id, "assistant", &full_text, &assistant_now).await {
        console_log!("Failed to store assistant message: {}", e);
    }

    // Generate title if this is the first exchange (2 messages: user + assistant)
    let is_first_exchange = history.len() <= 1; // history includes the user msg we just stored
    if is_first_exchange {
        generate_title(env, &db, &conversation_id, &request.message, &full_text).await;
    }

    // Build SSE response with our event format
    let mut sse = String::new();
    sse.push_str(&format!(
        "event: message\ndata: {}\n\n",
        serde_json::json!({"type": "start", "conversation_id": conversation_id, "message_id": assistant_msg_id})
    ));

    // Send full text as a single delta (v1 -- true streaming in v2)
    sse.push_str(&format!(
        "event: message\ndata: {}\n\n",
        serde_json::json!({"type": "delta", "text": full_text})
    ));

    sse.push_str(&format!(
        "event: message\ndata: {}\n\n",
        serde_json::json!({"type": "done", "message_id": assistant_msg_id})
    ));

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::from(sse))
        .unwrap()
}

// --- Helper functions ---

async fn verify_conversation_ownership(db: &worker::D1Database, id: &str, student_id: &str) -> bool {
    let result: Option<serde_json::Value> = db
        .prepare("SELECT id FROM conversations WHERE id = ?1 AND student_id = ?2")
        .bind(&[JsValue::from_str(id), JsValue::from_str(student_id)])
        .ok()
        .and_then(|s| futures::executor::block_on(s.first(None)).ok())
        .flatten();
    // Note: block_on won't work in WASM. Use the async pattern below instead.
    result.is_some()
}

// Replace verify_conversation_ownership with this async version:
async fn verify_conversation_ownership(db: &worker::D1Database, id: &str, student_id: &str) -> bool {
    match db
        .prepare("SELECT id FROM conversations WHERE id = ?1 AND student_id = ?2")
        .bind(&[JsValue::from_str(id), JsValue::from_str(student_id)])
    {
        Ok(stmt) => match stmt.first(None).await {
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

async fn fetch_student_context(db: &worker::D1Database, student_id: &str) -> Option<String> {
    let row: serde_json::Value = db
        .prepare("SELECT inferred_level, explicit_goals, baseline_dynamics, baseline_timing, baseline_pedaling, baseline_articulation, baseline_phrasing, baseline_interpretation FROM students WHERE apple_user_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .ok()?
        .first(None)
        .await
        .ok()??;

    prompts::build_chat_student_context(&row)
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
```

**Important:** There's a duplicate `verify_conversation_ownership` in the code above -- only use the second (async) version. The first one with `block_on` is shown to illustrate the problem; delete it during implementation.

**Step 4: Make `generate_uuid` public in ask.rs**

In `apps/api/src/services/ask.rs`, find `fn generate_uuid()` (around line 590) and make it `pub`:
```rust
pub fn generate_uuid() -> String {
```

**Step 5: Replace legacy /api/chat route in server.rs**

In `apps/api/src/server.rs`, replace lines 796-805 (the legacy `/api/chat` handler) with:

```rust
// Chat endpoint -- streaming teacher conversation (authenticated)
if path == "/api/chat" && method == http::Method::POST {
    let headers = req.headers().clone();
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();
    return Ok(with_cors(
        crate::services::chat::handle_chat_stream(&env, &headers, &body).await,
        origin.as_deref(),
    ));
}
```

**Step 6: Build and verify compilation**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown`
Expected: Compiles successfully.

**Step 7: Commit**

```bash
git add apps/api/src/services/chat.rs apps/api/src/services/llm.rs apps/api/src/services/prompts.rs apps/api/src/services/ask.rs apps/api/src/server.rs
git commit -m "feat(api): add streaming chat endpoint with Anthropic + prompt caching"
```

**Note on streaming:** This v1 implementation consumes the full Anthropic response, then returns it as SSE. The browser receives the complete response in SSE format (start → delta → done). True token-by-token streaming (piping Anthropic SSE through a TransformStream to the browser) requires `web_sys::TransformStream` in WASM and is a v2 enhancement. The SSE format is correct either way, so the frontend code won't need to change when we upgrade to true streaming.

---

## Task 4: Frontend API Client

**Files:**
- Modify: `apps/web/src/lib/api.ts` (add chat methods + streaming support)

**Step 1: Add chat API methods and streaming support**

Add these types and methods to `apps/web/src/lib/api.ts`:

```typescript
// --- Chat types ---

export interface ConversationSummary {
  id: string
  title: string | null
  updated_at: string
}

export interface MessageRow {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ConversationWithMessages {
  conversation: {
    id: string
    title: string | null
    created_at: string
  }
  messages: MessageRow[]
}

export interface ChatStreamEvent {
  type: 'start' | 'delta' | 'done'
  conversation_id?: string
  message_id?: string
  text?: string
}

// Add to the api object:

export const api = {
  auth: {
    // ... existing auth methods unchanged ...
  },

  chat: {
    async send(
      message: string,
      conversationId: string | null,
      onEvent: (event: ChatStreamEvent) => void,
    ): Promise<void> {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          message,
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText }))
        throw new ApiError(response.status, body.error ?? response.statusText)
      }

      if (!response.body) throw new Error('Response body is empty')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event: ChatStreamEvent = JSON.parse(line.slice(6))
                onEvent(event)
              } catch {
                // Skip unparseable lines
              }
            }
          }
        }
      } finally {
        reader.releaseLock()
      }
    },

    list(): Promise<{ conversations: ConversationSummary[] }> {
      return request('/api/conversations')
    },

    get(conversationId: string): Promise<ConversationWithMessages> {
      return request(`/api/conversations/${conversationId}`)
    },

    async delete(conversationId: string): Promise<void> {
      const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (!response.ok && response.status !== 204) {
        throw new ApiError(response.status, 'Failed to delete conversation')
      }
    },
  },
}
```

**Step 2: Commit**

```bash
git add apps/web/src/lib/api.ts
git commit -m "feat(web): add chat API client with SSE streaming support"
```

---

## Task 5: Chat UI Components

**Files:**
- Create: `apps/web/src/components/ChatMessages.tsx`
- Create: `apps/web/src/components/ChatInput.tsx`

**Step 1: Create ChatMessages component**

```typescript
// apps/web/src/components/ChatMessages.tsx
import { useEffect, useRef } from 'react'
import type { MessageRow } from '../lib/api'

interface ChatMessagesProps {
  messages: MessageRow[]
  streamingContent: string | null
}

export function ChatMessages({ messages, streamingContent }: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  if (messages.length === 0 && !streamingContent) {
    return null
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-8">
      <div className="max-w-2xl mx-auto space-y-6">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
        ))}
        {streamingContent !== null && (
          <MessageBubble role="assistant" content={streamingContent} />
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function MessageBubble({ role, content }: { role: string; content: string }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-surface border border-border rounded-2xl px-5 py-3 max-w-[80%]">
          <p className="text-body-md text-cream whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%]">
        <p className="text-body-md text-cream whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  )
}
```

**Step 2: Create ChatInput component**

```typescript
// apps/web/src/components/ChatInput.tsx
import { useState, useRef, useEffect } from 'react'
import { PaperPlaneRight } from '@phosphor-icons/react'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled: boolean
  placeholder?: string
}

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleSend() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  return (
    <div className="border-t border-border px-6 py-4">
      <div className="max-w-2xl mx-auto flex items-end gap-3">
        <div className="flex-1 bg-surface border border-border rounded-2xl flex items-end">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder ?? 'Message your teacher...'}
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent px-5 py-3 text-body-md text-cream placeholder:text-text-tertiary outline-none resize-none"
          />
        </div>
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="shrink-0 w-10 h-10 flex items-center justify-center rounded-full bg-cream text-espresso hover:brightness-110 transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <PaperPlaneRight size={18} weight="fill" />
        </button>
      </div>
    </div>
  )
}
```

**Step 3: Commit**

```bash
git add apps/web/src/components/ChatMessages.tsx apps/web/src/components/ChatInput.tsx
git commit -m "feat(web): add ChatMessages and ChatInput components"
```

---

## Task 6: App Route + Sidebar Integration

This is the main integration task. Refactor `app.tsx` to be the chat interface with working sidebar.

**Files:**
- Modify: `apps/web/src/routes/app.tsx` (full rewrite of AppPage component)
- Modify: `apps/web/src/routes/__root.tsx` (update path check for `/app` subroutes)

**Step 1: Update __root.tsx path check**

In `apps/web/src/routes/__root.tsx`, the `isAppShell` check at line 38-39 already handles `/app` prefixed paths:
```typescript
const isAppShell = pathname === '/signin' || pathname.startsWith('/app')
```
This already covers `/app/c/:id` -- no change needed.

**Step 2: Rewrite app.tsx**

Replace the full content of `apps/web/src/routes/app.tsx`. The new version:
- Keeps sidebar with working "New Chat" and conversation list
- Adds chat message display with streaming
- Adds input bar at bottom
- Handles empty state (new conversation) vs active conversation
- URL updates to `/app/c/:id` when conversation starts (using `window.history.replaceState` to avoid full navigation)

```typescript
import { useEffect, useRef, useState, useCallback } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import {
  SidebarSimple,
  PlusCircle,
  ChatCircle,
  Trash,
} from '@phosphor-icons/react'
import { useAuth } from '../lib/auth'
import { api } from '../lib/api'
import type { ConversationSummary, MessageRow, ChatStreamEvent } from '../lib/api'
import { ChatMessages } from '../components/ChatMessages'
import { ChatInput } from '../components/ChatInput'

export const Route = createFileRoute('/app')({
  component: AppPage,
})

function AppPage() {
  const { user, isLoading, isAuthenticated, signOut } = useAuth()
  const navigate = useNavigate()
  const [showProfile, setShowProfile] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)

  // Chat state
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<MessageRow[]>([])
  const [streamingContent, setStreamingContent] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Redirect if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate({ to: '/signin' })
    }
  }, [isLoading, isAuthenticated, navigate])

  // Load conversations on mount
  useEffect(() => {
    if (!isAuthenticated) return
    api.chat.list().then(({ conversations }) => {
      setConversations(conversations)
    }).catch((e) => {
      console.error('Failed to load conversations:', e)
    })
  }, [isAuthenticated])

  // Load conversation from URL on mount
  useEffect(() => {
    const path = window.location.pathname
    const match = path.match(/^\/app\/c\/(.+)$/)
    if (match) {
      loadConversation(match[1])
    }
  }, [])

  // Click outside to close profile dropdown
  useEffect(() => {
    if (!showProfile) return
    function handleClick(e: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setShowProfile(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showProfile])

  async function handleSignOut() {
    await signOut()
    navigate({ to: '/' })
  }

  const loadConversation = useCallback(async (id: string) => {
    try {
      const data = await api.chat.get(id)
      setActiveConversationId(id)
      setMessages(data.messages)
      window.history.replaceState(null, '', `/app/c/${id}`)
    } catch (e) {
      console.error('Failed to load conversation:', e)
    }
  }, [])

  function handleNewChat() {
    setActiveConversationId(null)
    setMessages([])
    setStreamingContent(null)
    window.history.replaceState(null, '', '/app')
  }

  async function handleDeleteConversation(id: string) {
    try {
      await api.chat.delete(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeConversationId === id) {
        handleNewChat()
      }
    } catch (e) {
      console.error('Failed to delete conversation:', e)
    }
  }

  async function handleSend(message: string) {
    if (isStreaming) return

    // Optimistically add user message
    const tempUserMsg: MessageRow = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])
    setStreamingContent('')
    setIsStreaming(true)

    try {
      let newConversationId: string | null = null

      await api.chat.send(message, activeConversationId, (event: ChatStreamEvent) => {
        switch (event.type) {
          case 'start':
            if (event.conversation_id && !activeConversationId) {
              newConversationId = event.conversation_id
              setActiveConversationId(event.conversation_id)
              window.history.replaceState(null, '', `/app/c/${event.conversation_id}`)
            }
            break
          case 'delta':
            if (event.text) {
              setStreamingContent((prev) => (prev ?? '') + event.text)
            }
            break
          case 'done':
            setStreamingContent((current) => {
              if (current !== null) {
                const assistantMsg: MessageRow = {
                  id: event.message_id ?? `msg-${Date.now()}`,
                  role: 'assistant',
                  content: current,
                  created_at: new Date().toISOString(),
                }
                setMessages((prev) => [...prev, assistantMsg])
              }
              return null
            })
            setIsStreaming(false)
            break
        }
      })

      // Refresh conversation list (to get new/updated titles)
      const { conversations: updated } = await api.chat.list()
      setConversations(updated)
    } catch (e) {
      console.error('Chat send failed:', e)
      setStreamingContent(null)
      setIsStreaming(false)
    }
  }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <p className="text-text-secondary text-body-md">Loading...</p>
      </div>
    )
  }

  // Time-aware greeting
  const hour = new Date().getHours()
  let greeting = 'Good morning'
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon'
  else if (hour >= 17) greeting = 'Good evening'

  const hasMessages = messages.length > 0 || streamingContent !== null

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`shrink-0 border-r border-border flex flex-col py-4 transition-all duration-200 ${
          sidebarOpen ? 'w-56' : 'w-12'
        }`}
      >
        <div className="flex flex-col items-center">
          <SidebarButton
            icon={<SidebarSimple size={20} />}
            label={sidebarOpen ? 'Collapse' : 'Expand'}
            expanded={sidebarOpen}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          />
          <div className="mt-2 w-full">
            <SidebarButton
              icon={<PlusCircle size={20} />}
              label="New Chat"
              expanded={sidebarOpen}
              onClick={handleNewChat}
            />
          </div>
        </div>

        {/* Conversation list */}
        {sidebarOpen && (
          <div className="mt-4 flex-1 overflow-y-auto px-2">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer text-body-sm transition ${
                  conv.id === activeConversationId
                    ? 'bg-surface text-cream'
                    : 'text-text-secondary hover:text-cream hover:bg-surface'
                }`}
                onClick={() => loadConversation(conv.id)}
                onKeyDown={(e) => e.key === 'Enter' && loadConversation(conv.id)}
                role="button"
                tabIndex={0}
              >
                <ChatCircle size={16} className="shrink-0" />
                <span className="flex-1 truncate">
                  {conv.title ?? 'New conversation'}
                </span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteConversation(conv.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 shrink-0 text-text-tertiary hover:text-cream transition"
                  aria-label="Delete conversation"
                >
                  <Trash size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </aside>

      {/* Main content */}
      <div className="flex-1 relative flex flex-col">
        {/* Profile button */}
        <div ref={profileRef} className="absolute top-4 right-4 z-20">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className="w-8 h-8 bg-surface border border-border rounded-full flex items-center justify-center text-body-sm text-cream font-medium hover:bg-surface-2 transition"
          >
            {user?.display_name?.charAt(0).toUpperCase() ?? user?.email?.charAt(0).toUpperCase() ?? '?'}
          </button>

          {showProfile && (
            <div className="absolute right-0 top-10 bg-surface border border-border rounded-lg py-1 min-w-[140px]">
              <button
                type="button"
                onClick={handleSignOut}
                className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition rounded-lg"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>

        {/* Empty state or chat messages */}
        {!hasMessages ? (
          <div className="flex-1 flex flex-col justify-start pt-[28vh] px-6">
            <div className="w-full max-w-2xl mx-auto text-center">
              <h1 className="font-display text-display-md text-cream">
                {greeting}.
              </h1>
            </div>
          </div>
        ) : (
          <ChatMessages messages={messages} streamingContent={streamingContent} />
        )}

        {/* Input bar */}
        <ChatInput
          onSend={handleSend}
          disabled={isStreaming}
          placeholder={hasMessages ? 'Message your teacher...' : 'What are you practicing today?'}
        />
      </div>
    </div>
  )
}

function SidebarButton({
  icon,
  label,
  expanded = false,
  onClick,
}: {
  icon: React.ReactNode
  label: string
  expanded?: boolean
  onClick?: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center text-text-secondary hover:text-cream hover:bg-surface transition group relative rounded-lg ${
        expanded ? 'w-[calc(100%-16px)] mx-2 px-3 h-10 gap-3' : 'w-10 h-10 justify-center mx-auto'
      }`}
      aria-label={label}
    >
      <span className="shrink-0">{icon}</span>
      {expanded && (
        <span className="text-body-sm whitespace-nowrap">{label}</span>
      )}
      {!expanded && (
        <span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 rounded text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
          {label}
        </span>
      )}
    </button>
  )
}
```

**Step 3: Verify the dev server runs**

Run: `cd apps/web && bun run dev`
Expected: Dev server starts on port 3000, `/app` loads without errors.

**Step 4: Commit**

```bash
git add apps/web/src/routes/app.tsx apps/web/src/routes/__root.tsx
git commit -m "feat(web): integrate chat UI with streaming, sidebar, and conversation management"
```

---

## Task 7: Deploy and Verify

**Step 1: Deploy API worker**

Run: `cd apps/api && npx wrangler deploy`
Expected: Deploys successfully to api.crescend.ai.

**Step 2: Deploy web app**

Run: `cd apps/web && bun run deploy`
Expected: Deploys successfully to crescend.ai.

**Step 3: Manual verification**

1. Navigate to crescend.ai, sign in
2. Type a message and send -- should see response appear
3. URL should update to `/app/c/:id`
4. Sidebar should show the new conversation with auto-generated title
5. Click "New Chat" -- empty state returns
6. Click existing conversation in sidebar -- loads history
7. Delete a conversation -- removes from sidebar

**Step 4: Commit any fixes and tag**

```bash
git add -A
git commit -m "fix: deployment adjustments for chat interface"
```

---

## Task 8: Install Missing Frontend Dependency

Check if `@phosphor-icons/react` already has `PaperPlaneRight` and `Trash` icons. These are standard Phosphor icons and should be available. If any import fails:

Run: `cd apps/web && bun add @phosphor-icons/react@latest`

This task may be a no-op if the current version already includes these icons.

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | D1 migration | `migrations/0006_conversations.sql` |
| 2 | CRUD endpoints | `services/chat.rs`, `services/mod.rs`, `server.rs` |
| 3 | Streaming chat | `services/chat.rs`, `services/llm.rs`, `services/prompts.rs`, `server.rs` |
| 4 | Frontend API client | `lib/api.ts` |
| 5 | Chat components | `components/ChatMessages.tsx`, `components/ChatInput.tsx` |
| 6 | App route + sidebar | `routes/app.tsx` |
| 7 | Deploy + verify | Deploy both services |
| 8 | Dependencies check | `package.json` (if needed) |
