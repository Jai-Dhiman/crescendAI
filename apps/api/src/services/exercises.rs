use axum::extract::{Json, Query, State};
use http::StatusCode;
use wasm_bindgen::JsValue;
use worker::console_error;

use crate::auth::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;

// Response types

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Exercise {
    pub id: String,
    pub title: String,
    pub description: String,
    pub instructions: String,
    pub difficulty: String,
    pub category: String,
    pub repertoire_tags: Option<String>,
    pub source: String,
    pub dimensions: Vec<String>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StudentExercise {
    pub id: String,
    pub student_id: String,
    pub exercise_id: String,
    pub session_id: Option<String>,
    pub assigned_at: String,
    pub completed: bool,
    pub response: Option<String>,
    pub times_assigned: i64,
}

// Request types

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssignRequest {
    pub exercise_id: String,
    pub session_id: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompleteRequest {
    pub student_exercise_id: String,
    pub response: Option<String>,
    pub dimension_before_json: Option<String>,
    pub dimension_after_json: Option<String>,
    pub notes: Option<String>,
}

// Query param types

#[derive(serde::Deserialize)]
pub struct ExerciseQueryParams {
    pub dimension: Option<String>,
    pub level: Option<String>,
    pub repertoire: Option<String>,
}

fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// GET /api/exercises
#[worker::send]
pub async fn handle_exercises(
    State(state): State<AppState>,
    auth: AuthUser,
    Query(params): Query<ExerciseQueryParams>,
) -> Result<Json<serde_json::Value>> {
    let student_id = auth.student_id.as_str().to_string();
    let db = state.db.d1()?;

    // Build dynamic SQL with optional filters.
    // Exclude exercises already assigned (not yet completed) to this student.
    // Prefer exercises matching the repertoire tag, then random order.
    let mut conditions = vec!["se.id IS NULL".to_string()];
    let mut bind_values: Vec<String> = Vec::new();
    let mut param_idx = 1usize;

    // Always bind student_id for the LEFT JOIN exclusion subquery
    bind_values.push(student_id);
    param_idx += 1;

    if let Some(ref dim) = params.dimension {
        conditions.push(format!("ed.dimension = ?{}", param_idx));
        bind_values.push(dim.clone());
        param_idx += 1;
    }

    if let Some(ref lvl) = params.level {
        conditions.push(format!("e.difficulty = ?{}", param_idx));
        bind_values.push(lvl.clone());
        param_idx += 1;
    }

    let order_clause = if let Some(ref rep) = params.repertoire {
        let rep_pattern = format!("%{}%", rep);
        let score_expr = format!(
            "CASE WHEN e.repertoire_tags LIKE ?{} THEN 0 ELSE 1 END",
            param_idx
        );
        bind_values.push(rep_pattern);
        format!("{}, RANDOM()", score_expr)
    } else {
        "RANDOM()".to_string()
    };

    let where_clause = conditions.join(" AND ");

    let sql = format!(
        "SELECT DISTINCT e.id, e.title, e.description, e.instructions, \
         e.difficulty, e.category, e.repertoire_tags, e.source \
         FROM exercises e \
         JOIN exercise_dimensions ed ON ed.exercise_id = e.id \
         LEFT JOIN student_exercises se ON se.exercise_id = e.id \
             AND se.student_id = ?1 AND se.completed = 0 \
         WHERE {} \
         ORDER BY {} \
         LIMIT 3",
        where_clause, order_clause
    );

    let js_binds: Vec<JsValue> = bind_values
        .iter()
        .map(|v| JsValue::from_str(v))
        .collect();

    let stmt = db.prepare(&sql).bind(&js_binds).map_err(|e| {
        console_error!("Failed to bind exercises query: {:?}", e);
        ApiError::Internal("Query preparation failed".into())
    })?;

    let rows = stmt.all().await.map_err(|e| {
        console_error!("Exercises query failed: {:?}", e);
        ApiError::Internal("Query failed".into())
    })?;

    let exercise_rows: Vec<serde_json::Value> = rows.results().map_err(|e| {
        console_error!("Failed to deserialize exercise rows: {:?}", e);
        ApiError::Internal("Failed to read results".into())
    })?;

    let mut exercises: Vec<Exercise> = Vec::new();

    for row in exercise_rows {
        let exercise_id = match row.get("id").and_then(|v| v.as_str().map(|s| s.to_string())) {
            Some(id) => id,
            None => continue,
        };

        // Fetch dimensions for this exercise
        let dim_stmt = match db
            .prepare("SELECT dimension FROM exercise_dimensions WHERE exercise_id = ?1")
            .bind(&[JsValue::from_str(&exercise_id)])
        {
            Ok(s) => s,
            Err(e) => {
                console_error!(
                    "Failed to bind dimension query for {}: {:?}",
                    exercise_id,
                    e
                );
                continue;
            }
        };

        let dim_rows = match dim_stmt.all().await {
            Ok(r) => r,
            Err(e) => {
                console_error!("Dimension query failed for {}: {:?}", exercise_id, e);
                continue;
            }
        };

        let dimensions: Vec<String> = match dim_rows.results::<serde_json::Value>() {
            Ok(rows) => rows
                .into_iter()
                .filter_map(|r| {
                    r.get("dimension")
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                })
                .collect(),
            Err(_) => vec![],
        };

        exercises.push(Exercise {
            id: exercise_id,
            title: row
                .get("title")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            description: row
                .get("description")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            instructions: row
                .get("instructions")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            difficulty: row
                .get("difficulty")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            category: row
                .get("category")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            repertoire_tags: row
                .get("repertoire_tags")
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            source: row
                .get("source")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default(),
            dimensions,
        });
    }

    Ok(Json(serde_json::json!({ "exercises": exercises })))
}

/// POST /api/exercises/assign
#[worker::send]
pub async fn handle_assign_exercise(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<AssignRequest>,
) -> Result<(StatusCode, Json<StudentExercise>)> {
    let student_id = auth.student_id.as_str().to_string();
    let db = state.db.d1()?;

    // Query MAX(times_assigned) for this (student, exercise) pair
    let max_row = db
        .prepare(
            "SELECT COALESCE(MAX(times_assigned), 0) as max_times \
             FROM student_exercises \
             WHERE student_id = ?1 AND exercise_id = ?2",
        )
        .bind(&[
            JsValue::from_str(&student_id),
            JsValue::from_str(&request.exercise_id),
        ])
        .map_err(|e| {
            console_error!("Failed to bind times_assigned query: {:?}", e);
            ApiError::Internal("Query preparation failed".into())
        })?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| {
            console_error!("Failed to query times_assigned: {:?}", e);
            ApiError::Internal("Query failed".into())
        })?;

    let times_assigned = max_row
        .as_ref()
        .and_then(|r| r.get("max_times"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        + 1;

    let id = format!("se-{}", generate_uuid());
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let session_id_js = match &request.session_id {
        Some(sid) => JsValue::from_str(sid),
        None => JsValue::NULL,
    };

    db.prepare(
        "INSERT INTO student_exercises \
         (id, student_id, exercise_id, session_id, assigned_at, completed, times_assigned) \
         VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
    )
    .bind(&[
        JsValue::from_str(&id),
        JsValue::from_str(&student_id),
        JsValue::from_str(&request.exercise_id),
        session_id_js,
        JsValue::from_str(&now),
        JsValue::from_f64(times_assigned as f64),
    ])
    .map_err(|e| {
        console_error!("Failed to bind insert statement: {:?}", e);
        ApiError::Internal("Insert preparation failed".into())
    })?
    .run()
    .await
    .map_err(|e| {
        console_error!("Failed to insert student_exercise: {:?}", e);
        ApiError::Internal("Failed to assign exercise".into())
    })?;

    Ok((
        StatusCode::CREATED,
        Json(StudentExercise {
            id,
            student_id,
            exercise_id: request.exercise_id,
            session_id: request.session_id,
            assigned_at: now,
            completed: false,
            response: None,
            times_assigned,
        }),
    ))
}

/// POST /api/exercises/complete
#[worker::send]
pub async fn handle_complete_exercise(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<CompleteRequest>,
) -> Result<Json<StudentExercise>> {
    let student_id = auth.student_id.as_str().to_string();
    let db = state.db.d1()?;

    // Verify the record exists and belongs to this student
    let row = db
        .prepare(
            "SELECT id, student_id, exercise_id, session_id, assigned_at, \
             completed, response, times_assigned \
             FROM student_exercises WHERE id = ?1",
        )
        .bind(&[JsValue::from_str(&request.student_exercise_id)])
        .map_err(|e| {
            console_error!("Failed to bind existence query: {:?}", e);
            ApiError::Internal("Query preparation failed".into())
        })?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| {
            console_error!(
                "Failed to query student_exercise {}: {:?}",
                request.student_exercise_id,
                e
            );
            ApiError::Internal("Query failed".into())
        })?
        .ok_or_else(|| ApiError::NotFound("Exercise record not found".into()))?;

    let record_student_id = row
        .get("student_id")
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default();

    if record_student_id != student_id {
        return Err(ApiError::Forbidden);
    }

    let response_js = match &request.response {
        Some(r) => JsValue::from_str(r),
        None => JsValue::NULL,
    };
    let dim_before_js = match &request.dimension_before_json {
        Some(d) => JsValue::from_str(d),
        None => JsValue::NULL,
    };
    let dim_after_js = match &request.dimension_after_json {
        Some(d) => JsValue::from_str(d),
        None => JsValue::NULL,
    };
    let notes_js = match &request.notes {
        Some(n) => JsValue::from_str(n),
        None => JsValue::NULL,
    };

    db.prepare(
        "UPDATE student_exercises \
         SET completed = 1, response = ?1, \
         dimension_before_json = ?2, dimension_after_json = ?3, notes = ?4 \
         WHERE id = ?5",
    )
    .bind(&[
        response_js,
        dim_before_js,
        dim_after_js,
        notes_js,
        JsValue::from_str(&request.student_exercise_id),
    ])
    .map_err(|e| {
        console_error!("Failed to bind update statement: {:?}", e);
        ApiError::Internal("Update preparation failed".into())
    })?
    .run()
    .await
    .map_err(|e| {
        console_error!(
            "Failed to update student_exercise {}: {:?}",
            request.student_exercise_id,
            e
        );
        ApiError::Internal("Failed to complete exercise".into())
    })?;

    let times_assigned = row
        .get("times_assigned")
        .and_then(|v| v.as_i64())
        .unwrap_or(1);

    Ok(Json(StudentExercise {
        id: request.student_exercise_id,
        student_id,
        exercise_id: row
            .get("exercise_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default(),
        session_id: row
            .get("session_id")
            .and_then(|v| v.as_str().map(|s| s.to_string())),
        assigned_at: row
            .get("assigned_at")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default(),
        completed: true,
        response: request.response,
        times_assigned,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Request deserialization tests ---

    #[test]
    fn test_assign_request_full() {
        let json = r#"{"exerciseId":"ex-123","sessionId":"sess-456"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-123");
        assert_eq!(req.session_id.as_deref(), Some("sess-456"));
    }

    #[test]
    fn test_assign_request_no_session() {
        let json = r#"{"exerciseId":"ex-789"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-789");
        assert!(req.session_id.is_none());
    }

    #[test]
    fn test_complete_request_full() {
        let json = r#"{
            "studentExerciseId": "se-abc",
            "response": "Done well",
            "dimensionBeforeJson": "{\"dynamics\":0.6}",
            "dimensionAfterJson": "{\"dynamics\":0.8}",
            "notes": "Felt much better"
        }"#;
        let req: CompleteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.student_exercise_id, "se-abc");
        assert_eq!(req.response.as_deref(), Some("Done well"));
        assert_eq!(
            req.dimension_before_json.as_deref(),
            Some("{\"dynamics\":0.6}")
        );
        assert_eq!(
            req.dimension_after_json.as_deref(),
            Some("{\"dynamics\":0.8}")
        );
        assert_eq!(req.notes.as_deref(), Some("Felt much better"));
    }

    #[test]
    fn test_complete_request_minimal() {
        let json = r#"{"studentExerciseId":"se-xyz"}"#;
        let req: CompleteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.student_exercise_id, "se-xyz");
        assert!(req.response.is_none());
        assert!(req.dimension_before_json.is_none());
        assert!(req.dimension_after_json.is_none());
        assert!(req.notes.is_none());
    }
}
