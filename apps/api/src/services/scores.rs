//! Score library endpoints: piece catalog (D1) and score data (R2).
//!
//! These are public endpoints (no auth required) serving the score catalog.

use wasm_bindgen::JsValue;
use worker::{console_error, Env};

/// Build a JSON error response.
fn json_error(status: http::StatusCode, message: &str) -> http::Response<axum::body::Body> {
    let body = serde_json::json!({ "error": message });
    http::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

/// GET /api/scores/:piece_id -- lookup a single piece from D1.
pub async fn handle_get_piece(
    env: &Env,
    piece_id: &str,
) -> http::Response<axum::body::Body> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Database unavailable",
            );
        }
    };

    let results = match db
        .prepare(
            "SELECT piece_id, composer, title, key_signature, time_signature, \
             tempo_bpm, bar_count, duration_seconds, note_count, \
             pitch_range_low, pitch_range_high, has_time_sig_changes, \
             has_tempo_changes, source, created_at \
             FROM pieces WHERE piece_id = ?1",
        )
        .bind(&[JsValue::from_str(piece_id)])
    {
        Ok(stmt) => match stmt.all().await {
            Ok(r) => r,
            Err(e) => {
                console_error!("D1 query failed for piece {}: {:?}", piece_id, e);
                return json_error(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Query failed",
                );
            }
        },
        Err(e) => {
            console_error!("D1 bind failed for piece {}: {:?}", piece_id, e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Query failed",
            );
        }
    };

    let rows: Vec<serde_json::Value> = match results.results() {
        Ok(r) => r,
        Err(e) => {
            console_error!("D1 results parse failed: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to parse results",
            );
        }
    };

    if rows.is_empty() {
        return json_error(http::StatusCode::NOT_FOUND, "Piece not found");
    }

    http::Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_string(&rows[0]).unwrap(),
        ))
        .unwrap()
}

/// GET /api/scores/:piece_id/data -- fetch pre-built score JSON from R2.
pub async fn handle_get_piece_data(
    env: &Env,
    piece_id: &str,
) -> http::Response<axum::body::Body> {
    let bucket = match env.bucket("SCORES") {
        Ok(b) => b,
        Err(e) => {
            console_error!("Failed to get SCORES R2 bucket: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Storage unavailable",
            );
        }
    };

    let key = format!("scores/v1/{}.json", piece_id);

    match bucket.get(&key).execute().await {
        Ok(Some(obj)) => {
            let bytes = match obj.body().unwrap().bytes().await {
                Ok(b) => b,
                Err(e) => {
                    console_error!("R2 body read failed for {}: {:?}", key, e);
                    return json_error(
                        http::StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to read score data",
                    );
                }
            };

            http::Response::builder()
                .status(http::StatusCode::OK)
                .header("Content-Type", "application/json")
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(bytes))
                .unwrap()
        }
        Ok(None) => json_error(http::StatusCode::NOT_FOUND, "Score data not found"),
        Err(e) => {
            console_error!("R2 get failed for {}: {:?}", key, e);
            http::Response::builder()
                .status(http::StatusCode::SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .header("Retry-After", "5")
                .body(axum::body::Body::from(
                    serde_json::to_string(&serde_json::json!({
                        "error": "Score storage temporarily unavailable"
                    }))
                    .unwrap(),
                ))
                .unwrap()
        }
    }
}

/// GET /api/scores?composer=X -- list pieces, optionally filtered by composer.
pub async fn handle_list_pieces(
    env: &Env,
    composer: Option<&str>,
) -> http::Response<axum::body::Body> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Database unavailable",
            );
        }
    };

    let results = match composer {
        Some(c) => {
            let stmt = db.prepare(
                "SELECT piece_id, composer, title, key_signature, time_signature, \
                 tempo_bpm, bar_count, duration_seconds, note_count, \
                 pitch_range_low, pitch_range_high, has_time_sig_changes, \
                 has_tempo_changes, source, created_at \
                 FROM pieces WHERE composer = ?1 ORDER BY title",
            );
            match stmt.bind(&[JsValue::from_str(c)]) {
                Ok(s) => match s.all().await {
                    Ok(r) => r,
                    Err(e) => {
                        console_error!("D1 query failed for composer {}: {:?}", c, e);
                        return json_error(
                            http::StatusCode::INTERNAL_SERVER_ERROR,
                            "Query failed",
                        );
                    }
                },
                Err(e) => {
                    console_error!("D1 bind failed: {:?}", e);
                    return json_error(
                        http::StatusCode::INTERNAL_SERVER_ERROR,
                        "Query failed",
                    );
                }
            }
        }
        None => {
            let stmt = db.prepare(
                "SELECT piece_id, composer, title, key_signature, time_signature, \
                 tempo_bpm, bar_count, duration_seconds, note_count, \
                 pitch_range_low, pitch_range_high, has_time_sig_changes, \
                 has_tempo_changes, source, created_at \
                 FROM pieces ORDER BY composer, title",
            );
            match stmt.bind(&[]) {
                Ok(s) => match s.all().await {
                    Ok(r) => r,
                    Err(e) => {
                        console_error!("D1 query failed for list: {:?}", e);
                        return json_error(
                            http::StatusCode::INTERNAL_SERVER_ERROR,
                            "Query failed",
                        );
                    }
                },
                Err(e) => {
                    console_error!("D1 bind failed: {:?}", e);
                    return json_error(
                        http::StatusCode::INTERNAL_SERVER_ERROR,
                        "Query failed",
                    );
                }
            }
        }
    };

    let rows: Vec<serde_json::Value> = match results.results() {
        Ok(r) => r,
        Err(e) => {
            console_error!("D1 results parse failed: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to parse results",
            );
        }
    };

    let body = serde_json::json!({ "pieces": rows });
    http::Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}
