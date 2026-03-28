use axum::extract::{Json, State};
use wasm_bindgen::JsValue;
use worker::{console_error, console_log, D1Database};

use crate::auth::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SyncRequest {
    pub student: StudentDelta,
    pub new_sessions: Vec<SessionDelta>,
    #[serde(default)]
    pub last_sync_timestamp: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudentDelta {
    #[serde(default)]
    pub inferred_level: Option<String>,
    #[serde(default)]
    pub baseline_dynamics: Option<f64>,
    #[serde(default)]
    pub baseline_timing: Option<f64>,
    #[serde(default)]
    pub baseline_pedaling: Option<f64>,
    #[serde(default)]
    pub baseline_articulation: Option<f64>,
    #[serde(default)]
    pub baseline_phrasing: Option<f64>,
    #[serde(default)]
    pub baseline_interpretation: Option<f64>,
    #[serde(default)]
    pub baseline_session_count: Option<i32>,
    #[serde(default)]
    pub explicit_goals: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDelta {
    pub id: String,
    pub started_at: String,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub avg_dynamics: Option<f64>,
    #[serde(default)]
    pub avg_timing: Option<f64>,
    #[serde(default)]
    pub avg_pedaling: Option<f64>,
    #[serde(default)]
    pub avg_articulation: Option<f64>,
    #[serde(default)]
    pub avg_phrasing: Option<f64>,
    #[serde(default)]
    pub avg_interpretation: Option<f64>,
    #[serde(default)]
    pub observations_json: Option<String>,
    #[serde(default)]
    pub chunks_summary_json: Option<String>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncResponse {
    pub sync_timestamp: String,
    pub exercise_updates: Vec<serde_json::Value>,
}

/// Convert an Option<f64> to a `JsValue` (f64 or null).
fn opt_f64(v: Option<f64>) -> JsValue {
    match v {
        Some(f) => JsValue::from_f64(f),
        None => JsValue::NULL,
    }
}

/// Convert an Option<i32> to a `JsValue` (i32 or null).
fn opt_i32(v: Option<i32>) -> JsValue {
    match v {
        Some(i) => JsValue::from(i),
        None => JsValue::NULL,
    }
}

/// Convert an Option<&str> to a `JsValue` (string or null).
fn opt_str(v: Option<&str>) -> JsValue {
    match v {
        Some(s) => JsValue::from_str(s),
        None => JsValue::NULL,
    }
}

#[worker::send]
pub async fn handle_sync(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<SyncRequest>,
) -> Result<Json<SyncResponse>> {
    let student_id = auth.student_id.as_str().to_string();
    let db = state.db.d1()?;

    // Upsert student baselines
    upsert_student_baselines(&db, &student_id, &request.student)
        .await
        .map_err(|e| {
            console_error!("Failed to upsert student baselines: {}", e);
            ApiError::Internal("Failed to update student data".into())
        })?;

    // Insert new sessions
    for session in &request.new_sessions {
        if let Err(e) = insert_session(&db, &student_id, session).await {
            console_error!("Failed to insert session {}: {}", session.id, e);
        }
    }

    // Memory synthesis: check if we should synthesize facts from accumulated observations
    let env = state.db.env();
    match crate::services::memory::should_synthesize(env, &student_id).await {
        Ok(true) => {
            console_log!("Triggering memory synthesis for student {}", student_id);
            if let Err(e) = crate::services::memory::run_synthesis(env, &student_id).await {
                console_error!("Memory synthesis failed (non-fatal): {}", e);
            }
        }
        Ok(false) => {
            console_log!("Synthesis not needed for student {}", student_id);
        }
        Err(e) => {
            console_error!("Failed to check synthesis eligibility: {}", e);
        }
    }

    let sync_timestamp = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    // Exercise updates (Slice 7 will populate this)
    let exercise_updates = Vec::new();

    console_log!(
        "Sync complete for student: {} sessions synced",
        request.new_sessions.len()
    );

    Ok(Json(SyncResponse {
        sync_timestamp,
        exercise_updates,
    }))
}

async fn upsert_student_baselines(
    db: &D1Database,
    student_id: &str,
    delta: &StudentDelta,
) -> std::result::Result<(), String> {
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
         WHERE student_id = ?11",
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
        JsValue::from_str(student_id),
    ])
    .map_err(|e| format!("Failed to bind update: {e:?}"))?
    .run()
    .await
    .map_err(|e| format!("Failed to update student: {e:?}"))?;

    Ok(())
}

async fn insert_session(
    db: &D1Database,
    student_id: &str,
    session: &SessionDelta,
) -> std::result::Result<(), String> {
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
    .map_err(|e| format!("Failed to bind insert: {e:?}"))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert session: {e:?}"))?;

    Ok(())
}
