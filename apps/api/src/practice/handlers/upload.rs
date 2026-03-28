use axum::extract::{Json, Query, State};
use worker::console_log;

use crate::auth::extractor::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;

/// Query params for POST /api/practice/chunk.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkParams {
    pub session_id: Option<String>,
    pub chunk_index: Option<String>,
}

/// Response body for POST /api/practice/chunk.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkResponse {
    pub r2_key: String,
    pub session_id: String,
    pub chunk_index: String,
}

/// Handle POST /api/practice/chunk?sessionId=X&chunkIndex=N
/// Body: raw audio bytes (WebM/Opus)
#[worker::send]
pub async fn handle_upload_chunk(
    State(state): State<AppState>,
    auth: AuthUser,
    Query(params): Query<ChunkParams>,
    body: axum::body::Bytes,
) -> Result<Json<ChunkResponse>> {
    let _student_id = auth.student_id;

    let session_id = params
        .session_id
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("Missing sessionId".into()))?;

    let chunk_index = params.chunk_index.unwrap_or_else(|| "0".to_string());

    if body.is_empty() {
        return Err(ApiError::BadRequest("Empty body".into()));
    }

    let r2_key = format!("sessions/{}/chunks/{}.webm", session_id, chunk_index);

    let bucket = state.practice.chunks_bucket()?;

    bucket
        .put(&r2_key, body.to_vec())
        .execute()
        .await
        .map_err(|e| {
            console_log!("R2 put failed: {:?}", e);
            ApiError::Internal("Failed to store chunk".into())
        })?;

    Ok(Json(ChunkResponse {
        r2_key,
        session_id,
        chunk_index,
    }))
}
