// CrescendAI Server - Recording Database Queries

use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use super::{DbError, DbResult, current_timestamp_ms, generate_id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    pub id: String,
    pub user_id: String,
    pub filename: String,
    pub file_size: i64,
    pub duration: Option<f64>, // Duration in seconds
    pub mime_type: String,
    pub r2_key: String,
    pub status: String, // "uploaded", "processing", "analyzed", "failed"
    pub created_at: i64,
    pub updated_at: i64,
}

// Insert a new recording
pub async fn insert_recording(
    env: &Env,
    user_id: &str,
    filename: &str,
    file_size: i64,
    duration: Option<f64>,
    mime_type: &str,
    r2_key: &str,
) -> DbResult<Recording> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let recording_id = generate_id();
    let now = current_timestamp_ms();

    let stmt = db.prepare("
        INSERT INTO recordings (id, user_id, filename, file_size, duration, mime_type, r2_key, status, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(&recording_id),
            JsValue::from_str(user_id),
            JsValue::from_str(filename),
            JsValue::from_f64(file_size as f64),
            duration.map(|d| JsValue::from_f64(d)).unwrap_or(JsValue::NULL),
            JsValue::from_str(mime_type),
            JsValue::from_str(r2_key),
            JsValue::from_str("uploaded"),
            JsValue::from_f64(now as f64),
            JsValue::from_f64(now as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert recording: {}", e)))?;

    Ok(Recording {
        id: recording_id,
        user_id: user_id.to_string(),
        filename: filename.to_string(),
        file_size,
        duration,
        mime_type: mime_type.to_string(),
        r2_key: r2_key.to_string(),
        status: "uploaded".to_string(),
        created_at: now,
        updated_at: now,
    })
}

// Get a recording by ID
pub async fn get_recording(env: &Env, recording_id: &str) -> DbResult<Recording> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, user_id, filename, file_size, duration, mime_type, r2_key, status, created_at, updated_at
        FROM recordings
        WHERE id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(recording_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.first::<Recording>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query recording: {}", e)))?;

    result.ok_or_else(|| DbError::NotFound(format!("Recording not found: {}", recording_id)))
}

// List recordings for a user
pub async fn list_recordings_by_user(
    env: &Env,
    user_id: &str,
    limit: Option<u32>,
    offset: Option<u32>,
) -> DbResult<Vec<Recording>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let limit_val = limit.unwrap_or(50);
    let offset_val = offset.unwrap_or(0);

    let stmt = db.prepare("
        SELECT id, user_id, filename, file_size, duration, mime_type, r2_key, status, created_at, updated_at
        FROM recordings
        WHERE user_id = ?1
        ORDER BY created_at DESC
        LIMIT ?2 OFFSET ?3
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(user_id),
            JsValue::from_f64(limit_val as f64),
            JsValue::from_f64(offset_val as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query recordings: {}", e)))?;

    result.results::<Recording>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize recordings: {}", e)))
}

// Update recording status
pub async fn update_recording_status(
    env: &Env,
    recording_id: &str,
    status: &str,
) -> DbResult<()> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    // Validate status
    if !["uploaded", "processing", "analyzed", "failed"].contains(&status) {
        return Err(DbError::InvalidInput(format!("Invalid status: {}", status)));
    }

    let now = current_timestamp_ms();

    let stmt = db.prepare("
        UPDATE recordings
        SET status = ?1, updated_at = ?2
        WHERE id = ?3
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(status),
            JsValue::from_f64(now as f64),
            JsValue::from_str(recording_id),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to update recording status: {}", e)))?;

    Ok(())
}

// Update recording duration
pub async fn update_recording_duration(
    env: &Env,
    recording_id: &str,
    duration: f64,
) -> DbResult<()> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let now = current_timestamp_ms();

    let stmt = db.prepare("
        UPDATE recordings
        SET duration = ?1, updated_at = ?2
        WHERE id = ?3
    ");

    let query = stmt
        .bind(&[
            JsValue::from_f64(duration),
            JsValue::from_f64(now as f64),
            JsValue::from_str(recording_id),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to update recording duration: {}", e)))?;

    Ok(())
}

// Count recordings for a user
pub async fn count_recordings_by_user(env: &Env, user_id: &str) -> DbResult<u32> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT COUNT(*) as count
        FROM recordings
        WHERE user_id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(user_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    #[derive(Deserialize)]
    struct CountResult {
        count: u32,
    }

    let result = query.first::<CountResult>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to count recordings: {}", e)))?;

    Ok(result.map(|r| r.count).unwrap_or(0))
}
