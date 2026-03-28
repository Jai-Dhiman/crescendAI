//! Score library endpoints: piece catalog (D1) and score data (R2).
//!
//! These are public endpoints (no auth required) serving the score catalog.

use axum::extract::{Json, Path, Query, State};
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use wasm_bindgen::JsValue;
use worker::console_error;

use crate::error::{ApiError, Result};
use crate::state::AppState;

/// Query params for GET /api/scores
#[derive(serde::Deserialize)]
pub struct ListPiecesParams {
    pub composer: Option<String>,
}

/// GET /api/scores/:piece_id -- lookup a single piece from D1.
#[worker::send]
pub async fn handle_get_piece(
    State(state): State<AppState>,
    Path(piece_id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let db = state.db.d1()?;

    let results = db
        .prepare(
            "SELECT piece_id, composer, title, key_signature, time_signature, \
             tempo_bpm, bar_count, duration_seconds, note_count, \
             pitch_range_low, pitch_range_high, has_time_sig_changes, \
             has_tempo_changes, source, created_at \
             FROM pieces WHERE piece_id = ?1",
        )
        .bind(&[JsValue::from_str(&piece_id)])
        .map_err(|e| {
            console_error!("D1 bind failed for piece {}: {:?}", piece_id, e);
            ApiError::Internal("Query failed".into())
        })?
        .all()
        .await
        .map_err(|e| {
            console_error!("D1 query failed for piece {}: {:?}", piece_id, e);
            ApiError::Internal("Query failed".into())
        })?;

    let rows: Vec<serde_json::Value> = results.results().map_err(|e| {
        console_error!("D1 results parse failed: {:?}", e);
        ApiError::Internal("Failed to parse results".into())
    })?;

    if rows.is_empty() {
        return Err(ApiError::NotFound("Piece not found".into()));
    }

    Ok(Json(rows[0].clone()))
}

/// GET /api/scores/:piece_id/data -- fetch pre-built score JSON from R2.
///
/// Returns raw bytes with long-lived cache headers, so we use `impl IntoResponse`
/// instead of `Result<Json<...>>`.
#[worker::send]
pub async fn handle_get_piece_data(
    State(state): State<AppState>,
    Path(piece_id): Path<String>,
) -> std::result::Result<Response, ApiError> {
    let bucket = state.practice.scores_bucket()?;

    let key = format!("scores/v1/{}.json", piece_id);

    match bucket.get(&key).execute().await {
        Ok(Some(obj)) => {
            let bytes = obj
                .body()
                .ok_or_else(|| ApiError::Internal("R2 object has no body".into()))?
                .bytes()
                .await
                .map_err(|e| {
                    console_error!("R2 body read failed for {}: {:?}", key, e);
                    ApiError::Internal("Failed to read score data".into())
                })?;

            Ok((
                StatusCode::OK,
                [
                    ("content-type", "application/json"),
                    ("cache-control", "public, max-age=31536000, immutable"),
                ],
                bytes,
            )
                .into_response())
        }
        Ok(None) => Err(ApiError::NotFound("Score data not found".into())),
        Err(e) => {
            console_error!("R2 get failed for {}: {:?}", key, e);
            Ok((
                StatusCode::SERVICE_UNAVAILABLE,
                [
                    ("content-type", "application/json"),
                    ("retry-after", "5"),
                ],
                serde_json::json!({"error": "Score storage temporarily unavailable"}).to_string(),
            )
                .into_response())
        }
    }
}

/// GET /api/scores?composer=X -- list pieces, optionally filtered by composer.
#[worker::send]
pub async fn handle_list_pieces(
    State(state): State<AppState>,
    Query(params): Query<ListPiecesParams>,
) -> Result<Json<serde_json::Value>> {
    let db = state.db.d1()?;

    let results = match params.composer.as_deref() {
        Some(c) => {
            db.prepare(
                "SELECT piece_id, composer, title, key_signature, time_signature, \
                 tempo_bpm, bar_count, duration_seconds, note_count, \
                 pitch_range_low, pitch_range_high, has_time_sig_changes, \
                 has_tempo_changes, source, created_at \
                 FROM pieces WHERE composer = ?1 ORDER BY title",
            )
            .bind(&[JsValue::from_str(c)])
            .map_err(|e| {
                console_error!("D1 bind failed: {:?}", e);
                ApiError::Internal("Query failed".into())
            })?
            .all()
            .await
            .map_err(|e| {
                console_error!("D1 query failed for composer {}: {:?}", c, e);
                ApiError::Internal("Query failed".into())
            })?
        }
        None => {
            db.prepare(
                "SELECT piece_id, composer, title, key_signature, time_signature, \
                 tempo_bpm, bar_count, duration_seconds, note_count, \
                 pitch_range_low, pitch_range_high, has_time_sig_changes, \
                 has_tempo_changes, source, created_at \
                 FROM pieces ORDER BY composer, title",
            )
            .bind(&[])
            .map_err(|e| {
                console_error!("D1 bind failed: {:?}", e);
                ApiError::Internal("Query failed".into())
            })?
            .all()
            .await
            .map_err(|e| {
                console_error!("D1 query failed for list: {:?}", e);
                ApiError::Internal("Query failed".into())
            })?
        }
    };

    let rows: Vec<serde_json::Value> = results.results().map_err(|e| {
        console_error!("D1 results parse failed: {:?}", e);
        ApiError::Internal("Failed to parse results".into())
    })?;

    Ok(Json(serde_json::json!({ "pieces": rows })))
}
