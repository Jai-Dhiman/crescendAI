// CrescendAI Server - Chat Message Database Queries

use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use super::{DbError, DbResult, current_timestamp_ms, generate_id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    pub role: String, // "user", "assistant", "system", "tool"
    pub content: String,
    pub tool_calls: Option<String>, // JSON array of tool calls
    pub tool_call_id: Option<String>, // For tool response messages
    pub metadata: Option<String>, // JSON object for additional metadata
    pub created_at: i64,
}

// Insert a new chat message
pub async fn insert_message(
    env: &Env,
    session_id: &str,
    role: &str,
    content: &str,
    tool_calls: Option<&str>,
    tool_call_id: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<ChatMessage> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    // Validate role
    if !["user", "assistant", "system", "tool"].contains(&role) {
        return Err(DbError::InvalidInput(format!("Invalid role: {}", role)));
    }

    let message_id = generate_id();
    let now = current_timestamp_ms();

    let stmt = db.prepare("
        INSERT INTO chat_messages (id, session_id, role, content, tool_calls, tool_call_id, metadata, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(&message_id),
            JsValue::from_str(session_id),
            JsValue::from_str(role),
            JsValue::from_str(content),
            tool_calls.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            tool_call_id.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            metadata.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            JsValue::from_f64(now as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert message: {}", e)))?;

    Ok(ChatMessage {
        id: message_id,
        session_id: session_id.to_string(),
        role: role.to_string(),
        content: content.to_string(),
        tool_calls: tool_calls.map(|s| s.to_string()),
        tool_call_id: tool_call_id.map(|s| s.to_string()),
        metadata: metadata.map(|s| s.to_string()),
        created_at: now,
    })
}

// Get all messages for a chat session
pub async fn get_messages_by_session(
    env: &Env,
    session_id: &str,
) -> DbResult<Vec<ChatMessage>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, session_id, role, content, tool_calls, tool_call_id, metadata, created_at
        FROM chat_messages
        WHERE session_id = ?1
        ORDER BY created_at ASC
    ");

    let query = stmt
        .bind(&[JsValue::from_str(session_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query messages: {}", e)))?;

    result.results::<ChatMessage>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize messages: {}", e)))
}

// Get paginated messages for a chat session
pub async fn get_messages_paginated(
    env: &Env,
    session_id: &str,
    limit: u32,
    offset: u32,
) -> DbResult<Vec<ChatMessage>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, session_id, role, content, tool_calls, tool_call_id, metadata, created_at
        FROM chat_messages
        WHERE session_id = ?1
        ORDER BY created_at ASC
        LIMIT ?2 OFFSET ?3
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(session_id),
            JsValue::from_f64(limit as f64),
            JsValue::from_f64(offset as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query messages: {}", e)))?;

    result.results::<ChatMessage>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize messages: {}", e)))
}

// Get the last N messages for a session (useful for context window)
pub async fn get_last_n_messages(
    env: &Env,
    session_id: &str,
    n: u32,
) -> DbResult<Vec<ChatMessage>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, session_id, role, content, tool_calls, tool_call_id, metadata, created_at
        FROM chat_messages
        WHERE session_id = ?1
        ORDER BY created_at DESC
        LIMIT ?2
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(session_id),
            JsValue::from_f64(n as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query messages: {}", e)))?;

    let mut messages = result.results::<ChatMessage>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize messages: {}", e)))?;

    // Reverse to get chronological order
    messages.reverse();

    Ok(messages)
}

// Count messages in a session
pub async fn count_messages(env: &Env, session_id: &str) -> DbResult<u32> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT COUNT(*) as count
        FROM chat_messages
        WHERE session_id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(session_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    #[derive(Deserialize)]
    struct CountResult {
        count: u32,
    }

    let result = query.first::<CountResult>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to count messages: {}", e)))?;

    Ok(result.map(|r| r.count).unwrap_or(0))
}
