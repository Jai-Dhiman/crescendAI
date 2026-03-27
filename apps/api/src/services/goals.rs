use wasm_bindgen::JsValue;
use worker::{console_log, Env};

#[derive(serde::Deserialize)]
pub struct ExtractGoalsRequest {
    pub message: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ExtractedGoals {
    pub pieces: Vec<String>,
    pub focus_areas: Vec<String>,
    pub deadlines: Vec<GoalDeadline>,
    pub raw_text: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct GoalDeadline {
    pub description: String,
    pub date: Option<String>,
}


pub async fn handle_extract_goals(
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

    let request: ExtractGoalsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse goals request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Extract goals via LLM
    let extracted = match extract_goals_with_llm(env, &request.message).await {
        Ok(goals) => goals,
        Err(e) => {
            console_log!("Goal extraction failed: {}", e);
            let error_json = serde_json::json!({"error": format!("Goal extraction failed: {}", e)});
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(error_json.to_string()))
                .unwrap();
        }
    };

    // Merge into student's explicit_goals in D1
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

    if let Err(e) = merge_goals(&db, &student_id, &extracted).await {
        console_log!("Failed to merge goals: {}", e);
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Failed to save goals"}"#))
            .unwrap();
    }

    // Store extracted goals as student-reported facts in synthesized_facts
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];

    for piece in &extracted.pieces {
        let fact_id = crate::services::memory::generate_fact_id();
        let piece_ctx = serde_json::json!({"title": piece}).to_string();
        let fact_text = format!("Working on {}", piece);
        if let Ok(stmt) = db
            .prepare(
                "INSERT OR IGNORE INTO synthesized_facts \
                 (id, student_id, fact_text, fact_type, dimension, piece_context, \
                  valid_at, confidence, evidence, source_type, created_at) \
                 VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .bind(&[
                JsValue::from_str(&fact_id),
                JsValue::from_str(&student_id),
                JsValue::from_str(&fact_text),
                JsValue::from_str("arc"),
                JsValue::from_str(&piece_ctx),
                JsValue::from_str(today),
                JsValue::from_str("high"),
                JsValue::from_str("[]"),
                JsValue::from_str("student_reported"),
                JsValue::from_str(&now),
            ])
        {
            let _ = stmt.run().await;
        }
    }

    for deadline in &extracted.deadlines {
        let fact_id = crate::services::memory::generate_fact_id();
        let invalid_at = deadline.date.as_deref();
        if let Ok(stmt) = db
            .prepare(
                "INSERT OR IGNORE INTO synthesized_facts \
                 (id, student_id, fact_text, fact_type, dimension, valid_at, invalid_at, \
                  confidence, evidence, source_type, created_at) \
                 VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .bind(&[
                JsValue::from_str(&fact_id),
                JsValue::from_str(&student_id),
                JsValue::from_str(&deadline.description),
                JsValue::from_str("arc"),
                JsValue::from_str(today),
                match invalid_at {
                    Some(d) => JsValue::from_str(d),
                    None => JsValue::NULL,
                },
                JsValue::from_str("high"),
                JsValue::from_str("[]"),
                JsValue::from_str("student_reported"),
                JsValue::from_str(&now),
            ])
        {
            let _ = stmt.run().await;
        }
    }

    let json = serde_json::to_string(&extracted).unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn extract_goals_with_llm(env: &Env, message: &str) -> Result<ExtractedGoals, String> {
    let prompt = format!(
        r#"Extract structured practice goals from this pianist's message. Return ONLY valid JSON with no other text.

Message: "{}"

Return this exact JSON structure:
{{
  "pieces": ["list of piece names mentioned"],
  "focus_areas": ["list of musical dimensions or techniques to focus on, e.g. pedaling, dynamics, articulation"],
  "deadlines": [{{"description": "what the deadline is for", "date": "YYYY-MM-DD or null if not specific"}}],
  "raw_text": "the original message"
}}

If a field has no matches, use an empty array. Always include raw_text."#,
        message
    );

    let response = crate::services::llm::call_workers_ai(
        env,
        crate::services::llm::WORKERS_AI_CHEAP_MODEL,
        "You extract structured data from pianist messages. Return only valid JSON.",
        &prompt,
        0.1,
        500,
    )
    .await?;

    // Parse the LLM's JSON response
    let extracted: ExtractedGoals = serde_json::from_str(&response).map_err(|e| {
        console_log!("LLM returned invalid JSON: {}", response);
        format!("Failed to parse extracted goals: {}", e)
    })?;

    Ok(extracted)
}

async fn merge_goals(
    db: &worker::D1Database,
    student_id: &str,
    new_goals: &ExtractedGoals,
) -> Result<(), String> {
    // Fetch existing goals
    let existing_row = db
        .prepare("SELECT explicit_goals FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| format!("Failed to query student: {:?}", e))?;

    let mut merged = if let Some(row) = existing_row {
        let goals_str = row
            .get("explicit_goals")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        if goals_str.is_empty() {
            ExplicitGoals::default()
        } else {
            serde_json::from_str(&goals_str).unwrap_or_default()
        }
    } else {
        ExplicitGoals::default()
    };

    // Merge new goals (append, dedup)
    for piece in &new_goals.pieces {
        if !merged.pieces.contains(piece) {
            merged.pieces.push(piece.clone());
        }
    }
    for area in &new_goals.focus_areas {
        if !merged.focus_areas.contains(area) {
            merged.focus_areas.push(area.clone());
        }
    }
    for deadline in &new_goals.deadlines {
        merged.deadlines.push(deadline.clone());
    }

    let merged_json = serde_json::to_string(&merged)
        .map_err(|e| format!("Failed to serialize merged goals: {}", e))?;

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare("UPDATE students SET explicit_goals = ?1, updated_at = ?2 WHERE student_id = ?3")
        .bind(&[
            JsValue::from_str(&merged_json),
            JsValue::from_str(&now),
            JsValue::from_str(student_id),
        ])
        .map_err(|e| format!("Failed to bind update: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to update goals: {:?}", e))?;

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ExplicitGoals {
    pieces: Vec<String>,
    focus_areas: Vec<String>,
    deadlines: Vec<GoalDeadline>,
}
