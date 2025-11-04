// CrescendAI Server - Chat Session Database Queries

use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use super::{DbError, DbResult, current_timestamp_ms, generate_id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub user_id: String,
    pub recording_id: Option<String>,
    pub title: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

// Create a new chat session
pub async fn create_session(
    env: &Env,
    user_id: &str,
    recording_id: Option<&str>,
    title: Option<&str>,
) -> DbResult<ChatSession> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let session_id = generate_id();
    let now = current_timestamp_ms();

    let stmt = db.prepare("
        INSERT INTO chat_sessions (id, user_id, recording_id, title, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(&session_id),
            JsValue::from_str(user_id),
            recording_id.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            title.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            JsValue::from_f64(now as f64),
            JsValue::from_f64(now as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert session: {}", e)))?;

    Ok(ChatSession {
        id: session_id,
        user_id: user_id.to_string(),
        recording_id: recording_id.map(|s| s.to_string()),
        title: title.map(|s| s.to_string()),
        created_at: now,
        updated_at: now,
    })
}

// Get a chat session by ID
pub async fn get_session(env: &Env, session_id: &str) -> DbResult<ChatSession> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, user_id, recording_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(session_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.first::<ChatSession>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query session: {}", e)))?;

    result.ok_or_else(|| DbError::NotFound(format!("Session not found: {}", session_id)))
}

// List chat sessions for a user
pub async fn list_sessions_by_user(
    env: &Env,
    user_id: &str,
    limit: Option<u32>,
) -> DbResult<Vec<ChatSession>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let limit_val = limit.unwrap_or(50);

    let stmt = db.prepare("
        SELECT id, user_id, recording_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE user_id = ?1
        ORDER BY created_at DESC
        LIMIT ?2
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(user_id),
            JsValue::from_f64(limit_val as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query sessions: {}", e)))?;

    result.results::<ChatSession>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize sessions: {}", e)))
}

// Delete a chat session (and cascade delete messages)
pub async fn delete_session(env: &Env, session_id: &str) -> DbResult<()> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("DELETE FROM chat_sessions WHERE id = ?1");

    let query = stmt
        .bind(&[JsValue::from_str(session_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to delete session: {}", e)))?;

    Ok(())
}

// Update session title
pub async fn update_session_title(
    env: &Env,
    session_id: &str,
    title: &str,
) -> DbResult<()> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let now = current_timestamp_ms();

    let stmt = db.prepare("
        UPDATE chat_sessions
        SET title = ?1, updated_at = ?2
        WHERE id = ?3
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(title),
            JsValue::from_f64(now as f64),
            JsValue::from_str(session_id),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to update session: {}", e)))?;

    Ok(())
}
