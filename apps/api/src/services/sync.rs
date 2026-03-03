use wasm_bindgen::JsValue;
use worker::{console_log, D1Database, Env};

#[derive(serde::Deserialize)]
pub struct SyncRequest {
    pub student: StudentDelta,
    pub new_sessions: Vec<SessionDelta>,
    pub last_sync_timestamp: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct StudentDelta {
    pub inferred_level: Option<String>,
    pub baseline_dynamics: Option<f64>,
    pub baseline_timing: Option<f64>,
    pub baseline_pedaling: Option<f64>,
    pub baseline_articulation: Option<f64>,
    pub baseline_phrasing: Option<f64>,
    pub baseline_interpretation: Option<f64>,
    pub baseline_session_count: Option<i32>,
    pub explicit_goals: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct SessionDelta {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub avg_dynamics: Option<f64>,
    pub avg_timing: Option<f64>,
    pub avg_pedaling: Option<f64>,
    pub avg_articulation: Option<f64>,
    pub avg_phrasing: Option<f64>,
    pub avg_interpretation: Option<f64>,
    pub observations_json: Option<String>,
    pub chunks_summary_json: Option<String>,
}

#[derive(serde::Serialize)]
pub struct SyncResponse {
    pub sync_timestamp: String,
    pub exercise_updates: Vec<serde_json::Value>,
}

/// Convert an Option<f64> to a JsValue (f64 or null).
fn opt_f64(v: Option<f64>) -> JsValue {
    match v {
        Some(f) => JsValue::from_f64(f),
        None => JsValue::NULL,
    }
}

/// Convert an Option<i32> to a JsValue (i32 or null).
fn opt_i32(v: Option<i32>) -> JsValue {
    match v {
        Some(i) => JsValue::from(i),
        None => JsValue::NULL,
    }
}

/// Convert an Option<&str> to a JsValue (string or null).
fn opt_str(v: Option<&str>) -> JsValue {
    match v {
        Some(s) => JsValue::from_str(s),
        None => JsValue::NULL,
    }
}

pub async fn handle_sync(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Verify auth
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: SyncRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse sync request: {:?}", e);
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

    // Upsert student baselines
    if let Err(e) = upsert_student_baselines(&db, &student_id, &request.student).await {
        console_log!("Failed to upsert student baselines: {}", e);
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Failed to update student data"}"#))
            .unwrap();
    }

    // Insert new sessions
    for session in &request.new_sessions {
        if let Err(e) = insert_session(&db, &student_id, session).await {
            console_log!("Failed to insert session {}: {}", session.id, e);
        }
    }

    let sync_timestamp = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    // Exercise updates (Slice 7 will populate this)
    let exercise_updates = Vec::new();

    let response = SyncResponse {
        sync_timestamp,
        exercise_updates,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());

    console_log!(
        "Sync complete for student: {} sessions synced",
        request.new_sessions.len()
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn upsert_student_baselines(
    db: &D1Database,
    apple_user_id: &str,
    delta: &StudentDelta,
) -> Result<(), String> {
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "UPDATE students SET \
         inferred_level = COALESCE(?1, inferred_level), \
         baseline_dynamics = COALESCE(?2, baseline_dynamics), \
         baseline_timing = COALESCE(?3, baseline_timing), \
         baseline_pedaling = COALESCE(?4, baseline_pedaling), \
         baseline_articulation = COALESCE(?5, baseline_articulation), \
         baseline_phrasing = COALESCE(?6, baseline_phrasing), \
         baseline_interpretation = COALESCE(?7, baseline_interpretation), \
         baseline_session_count = COALESCE(?8, baseline_session_count), \
         explicit_goals = COALESCE(?9, explicit_goals), \
         updated_at = ?10 \
         WHERE apple_user_id = ?11",
    )
    .bind(&[
        opt_str(delta.inferred_level.as_deref()),
        opt_f64(delta.baseline_dynamics),
        opt_f64(delta.baseline_timing),
        opt_f64(delta.baseline_pedaling),
        opt_f64(delta.baseline_articulation),
        opt_f64(delta.baseline_phrasing),
        opt_f64(delta.baseline_interpretation),
        opt_i32(delta.baseline_session_count),
        opt_str(delta.explicit_goals.as_deref()),
        JsValue::from_str(&now),
        JsValue::from_str(apple_user_id),
    ])
    .map_err(|e| format!("Failed to bind update: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update student: {:?}", e))?;

    Ok(())
}

async fn insert_session(
    db: &D1Database,
    student_id: &str,
    session: &SessionDelta,
) -> Result<(), String> {
    db.prepare(
        "INSERT OR IGNORE INTO sessions \
         (id, student_id, started_at, ended_at, avg_dynamics, avg_timing, avg_pedaling, \
          avg_articulation, avg_phrasing, avg_interpretation, observations_json, chunks_summary_json) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
    )
    .bind(&[
        JsValue::from_str(&session.id),
        JsValue::from_str(student_id),
        JsValue::from_str(&session.started_at),
        opt_str(session.ended_at.as_deref()),
        opt_f64(session.avg_dynamics),
        opt_f64(session.avg_timing),
        opt_f64(session.avg_pedaling),
        opt_f64(session.avg_articulation),
        opt_f64(session.avg_phrasing),
        opt_f64(session.avg_interpretation),
        opt_str(session.observations_json.as_deref()),
        opt_str(session.chunks_summary_json.as_deref()),
    ])
    .map_err(|e| format!("Failed to bind insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert session: {:?}", e))?;

    Ok(())
}
