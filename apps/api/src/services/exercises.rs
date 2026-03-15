use wasm_bindgen::JsValue;
use worker::{console_error, Env};

// Response types

#[derive(serde::Serialize)]
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
pub struct AssignRequest {
    pub exercise_id: String,
    pub session_id: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct CompleteRequest {
    pub student_exercise_id: String,
    pub response: Option<String>,
    pub dimension_before_json: Option<String>,
    pub dimension_after_json: Option<String>,
    pub notes: Option<String>,
}

// Query param parser

pub struct ExerciseQueryParams {
    pub dimension: Option<String>,
    pub level: Option<String>,
    pub repertoire: Option<String>,
}

pub fn parse_exercise_query_params(query: &str) -> ExerciseQueryParams {
    let mut dimension = None;
    let mut level = None;
    let mut repertoire = None;

    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut parts = pair.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();
        if value.is_empty() {
            continue;
        }
        match key {
            "dimension" => dimension = Some(value.to_string()),
            "level" => level = Some(value.to_string()),
            "repertoire" => repertoire = Some(value.to_string()),
            _ => {}
        }
    }

    ExerciseQueryParams {
        dimension,
        level,
        repertoire,
    }
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

fn json_error(status: http::StatusCode, message: &str) -> http::Response<axum::body::Body> {
    let body = serde_json::json!({ "error": message });
    http::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_string(&body).unwrap(),
        ))
        .unwrap()
}

/// GET /api/exercises
pub async fn handle_exercises(
    env: &Env,
    headers: &http::HeaderMap,
    query_string: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::Response;

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let params = parse_exercise_query_params(query_string);

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

    // Build dynamic SQL with optional filters.
    // Exclude exercises already assigned (not yet completed) to this student.
    // Prefer exercises matching the repertoire tag, then random order.
    let mut conditions = vec![
        "se.id IS NULL".to_string(),
    ];
    let mut bind_values: Vec<String> = Vec::new();
    let mut param_idx = 1usize;

    // Always bind student_id for the LEFT JOIN exclusion subquery
    bind_values.push(student_id.clone());
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

    // Ensure the bind array is non-empty (it always has at least ?1 = student_id)
    let stmt = match db.prepare(&sql).bind(&js_binds) {
        Ok(s) => s,
        Err(e) => {
            console_error!("Failed to bind exercises query: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Query preparation failed",
            );
        }
    };

    let rows = match stmt.all().await {
        Ok(r) => r,
        Err(e) => {
            console_error!("Exercises query failed: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Query failed",
            );
        }
    };

    let exercise_rows: Vec<serde_json::Value> = match rows.results() {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to deserialize exercise rows: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to read results",
            );
        }
    };

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
                .filter_map(|r| r.get("dimension").and_then(|v| v.as_str().map(|s| s.to_string())))
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

    let json = serde_json::json!({ "exercises": exercises });
    Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&json).unwrap()))
        .unwrap()
}

/// POST /api/exercises/assign
pub async fn handle_assign_exercise(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::Response;

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: AssignRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse assign request: {:?}", e);
            return json_error(http::StatusCode::BAD_REQUEST, "Invalid request body");
        }
    };

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

    // Query MAX(times_assigned) for this (student, exercise) pair
    let max_row = match db
        .prepare(
            "SELECT COALESCE(MAX(times_assigned), 0) as max_times \
             FROM student_exercises \
             WHERE student_id = ?1 AND exercise_id = ?2",
        )
        .bind(&[
            JsValue::from_str(&student_id),
            JsValue::from_str(&request.exercise_id),
        ]) {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(row) => row,
            Err(e) => {
                console_error!("Failed to query times_assigned: {:?}", e);
                return json_error(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Query failed",
                );
            }
        },
        Err(e) => {
            console_error!("Failed to bind times_assigned query: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Query preparation failed",
            );
        }
    };

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

    let insert_stmt = match db
        .prepare(
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
        ]) {
        Ok(s) => s,
        Err(e) => {
            console_error!("Failed to bind insert statement: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Insert preparation failed",
            );
        }
    };

    if let Err(e) = insert_stmt.run().await {
        console_error!("Failed to insert student_exercise: {:?}", e);
        return json_error(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to assign exercise",
        );
    }

    let student_exercise = StudentExercise {
        id,
        student_id,
        exercise_id: request.exercise_id,
        session_id: request.session_id,
        assigned_at: now,
        completed: false,
        response: None,
        times_assigned,
    };

    let json = serde_json::to_string(&student_exercise).unwrap_or_else(|_| "{}".to_string());
    Response::builder()
        .status(http::StatusCode::CREATED)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// POST /api/exercises/complete
pub async fn handle_complete_exercise(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::Response;

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: CompleteRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse complete request: {:?}", e);
            return json_error(http::StatusCode::BAD_REQUEST, "Invalid request body");
        }
    };

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

    // Verify the record exists and belongs to this student
    let existing = match db
        .prepare(
            "SELECT id, student_id, exercise_id, session_id, assigned_at, \
             completed, response, times_assigned \
             FROM student_exercises WHERE id = ?1",
        )
        .bind(&[JsValue::from_str(&request.student_exercise_id)])
    {
        Ok(stmt) => match stmt.first::<serde_json::Value>(None).await {
            Ok(row) => row,
            Err(e) => {
                console_error!(
                    "Failed to query student_exercise {}: {:?}",
                    request.student_exercise_id,
                    e
                );
                return json_error(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Query failed",
                );
            }
        },
        Err(e) => {
            console_error!("Failed to bind existence query: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Query preparation failed",
            );
        }
    };

    let row = match existing {
        Some(r) => r,
        None => {
            return json_error(http::StatusCode::NOT_FOUND, "Exercise record not found");
        }
    };

    let record_student_id = row
        .get("student_id")
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default();

    if record_student_id != student_id {
        return json_error(http::StatusCode::FORBIDDEN, "Access denied");
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

    let update_stmt = match db
        .prepare(
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
        ]) {
        Ok(s) => s,
        Err(e) => {
            console_error!("Failed to bind update statement: {:?}", e);
            return json_error(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "Update preparation failed",
            );
        }
    };

    if let Err(e) = update_stmt.run().await {
        console_error!(
            "Failed to update student_exercise {}: {:?}",
            request.student_exercise_id,
            e
        );
        return json_error(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to complete exercise",
        );
    }

    let times_assigned = row
        .get("times_assigned")
        .and_then(|v| v.as_i64())
        .unwrap_or(1);

    let student_exercise = StudentExercise {
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
    };

    let json = serde_json::to_string(&student_exercise).unwrap_or_else(|_| "{}".to_string());
    Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_exercise_query_params tests ---

    #[test]
    fn test_parse_empty_query() {
        let params = parse_exercise_query_params("");
        assert!(params.dimension.is_none());
        assert!(params.level.is_none());
        assert!(params.repertoire.is_none());
    }

    #[test]
    fn test_parse_dimension_only() {
        let params = parse_exercise_query_params("dimension=dynamics");
        assert_eq!(params.dimension.as_deref(), Some("dynamics"));
        assert!(params.level.is_none());
        assert!(params.repertoire.is_none());
    }

    #[test]
    fn test_parse_all_params() {
        let params =
            parse_exercise_query_params("dimension=timing&level=intermediate&repertoire=chopin");
        assert_eq!(params.dimension.as_deref(), Some("timing"));
        assert_eq!(params.level.as_deref(), Some("intermediate"));
        assert_eq!(params.repertoire.as_deref(), Some("chopin"));
    }

    #[test]
    fn test_parse_ignores_empty_values() {
        let params = parse_exercise_query_params("dimension=&level=beginner");
        assert!(params.dimension.is_none());
        assert_eq!(params.level.as_deref(), Some("beginner"));
    }

    #[test]
    fn test_parse_unknown_keys_ignored() {
        let params = parse_exercise_query_params("foo=bar&dimension=pedaling");
        assert_eq!(params.dimension.as_deref(), Some("pedaling"));
        assert!(params.level.is_none());
        assert!(params.repertoire.is_none());
    }

    // --- Request deserialization tests ---

    #[test]
    fn test_assign_request_full() {
        let json = r#"{"exercise_id":"ex-123","session_id":"sess-456"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-123");
        assert_eq!(req.session_id.as_deref(), Some("sess-456"));
    }

    #[test]
    fn test_assign_request_no_session() {
        let json = r#"{"exercise_id":"ex-789"}"#;
        let req: AssignRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.exercise_id, "ex-789");
        assert!(req.session_id.is_none());
    }

    #[test]
    fn test_complete_request_full() {
        let json = r#"{
            "student_exercise_id": "se-abc",
            "response": "Done well",
            "dimension_before_json": "{\"dynamics\":0.6}",
            "dimension_after_json": "{\"dynamics\":0.8}",
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
        let json = r#"{"student_exercise_id":"se-xyz"}"#;
        let req: CompleteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.student_exercise_id, "se-xyz");
        assert!(req.response.is_none());
        assert!(req.dimension_before_json.is_none());
        assert!(req.dimension_after_json.is_none());
        assert!(req.notes.is_none());
    }
}
