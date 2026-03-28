use axum::extract::{Json, State};
use wasm_bindgen::JsValue;
use worker::console_log;

use crate::auth::AuthUser;
use crate::error::{ApiError, Result};
use crate::state::AppState;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExtractGoalsRequest {
    pub message: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExtractedGoals {
    pub pieces: Vec<String>,
    pub focus_areas: Vec<String>,
    pub deadlines: Vec<GoalDeadline>,
    pub raw_text: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GoalDeadline {
    pub description: String,
    pub date: Option<String>,
}

#[worker::send]
pub async fn handle_extract_goals(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<ExtractGoalsRequest>,
) -> Result<Json<ExtractedGoals>> {
    let student_id = auth.student_id.as_str().to_string();

    // Extract goals via LLM
    let extracted = extract_goals_with_llm(state.db.env(), &request.message)
        .await
        .map_err(|e| {
            console_log!("Goal extraction failed: {}", e);
            ApiError::Internal(format!("Goal extraction failed: {e}"))
        })?;

    // Merge into student's explicit_goals in D1
    let db = state.db.d1()?;

    merge_goals(&db, &student_id, &extracted)
        .await
        .map_err(|e| {
            console_log!("Failed to merge goals: {}", e);
            ApiError::Internal("Failed to save goals".into())
        })?;

    // Store extracted goals as student-reported facts in synthesized_facts
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];

    for piece in &extracted.pieces {
        let fact_id = crate::services::memory::generate_fact_id();
        let piece_ctx = serde_json::json!({"title": piece}).to_string();
        let fact_text = format!("Working on {piece}");
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

    Ok(Json(extracted))
}

async fn extract_goals_with_llm(
    env: &worker::Env,
    message: &str,
) -> std::result::Result<ExtractedGoals, String> {
    let prompt = format!(
        r#"Extract structured practice goals from this pianist's message. Return ONLY valid JSON with no other text.

Message: "{message}"

Return this exact JSON structure:
{{
  "pieces": ["list of piece names mentioned"],
  "focus_areas": ["list of musical dimensions or techniques to focus on, e.g. pedaling, dynamics, articulation"],
  "deadlines": [{{"description": "what the deadline is for", "date": "YYYY-MM-DD or null if not specific"}}],
  "raw_text": "the original message"
}}

If a field has no matches, use an empty array. Always include raw_text."#
    );

    let response = crate::services::llm::call_workers_ai(
        env,
        crate::services::llm::WORKERS_AI_CHEAP_MODEL,
        "You extract structured data from pianist messages. Return only valid JSON.",
        &prompt,
        0.1,
        500,
    )
    .await
    .map_err(|e| e.to_string())?;

    // Parse the LLM's JSON response
    let extracted: ExtractedGoals = serde_json::from_str(&response).map_err(|e| {
        console_log!("LLM returned invalid JSON: {}", response);
        format!("Failed to parse extracted goals: {e}")
    })?;

    Ok(extracted)
}

async fn merge_goals(
    db: &worker::D1Database,
    student_id: &str,
    new_goals: &ExtractedGoals,
) -> std::result::Result<(), String> {
    // Fetch existing goals
    let existing_row = db
        .prepare("SELECT explicit_goals FROM students WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {e:?}"))?
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| format!("Failed to query student: {e:?}"))?;

    let mut merged = if let Some(row) = existing_row {
        let goals_str = row
            .get("explicit_goals")
            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
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
        .map_err(|e| format!("Failed to serialize merged goals: {e}"))?;

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
        .map_err(|e| format!("Failed to bind update: {e:?}"))?
        .run()
        .await
        .map_err(|e| format!("Failed to update goals: {e:?}"))?;

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ExplicitGoals {
    pieces: Vec<String>,
    focus_areas: Vec<String>,
    deadlines: Vec<GoalDeadline>,
}
