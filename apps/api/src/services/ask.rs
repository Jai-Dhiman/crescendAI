//! Handler for POST /api/ask -- the two-stage teacher pipeline.
//!
//! Stage 1: Groq (Llama 70B) subagent analyzes the teaching moment.
//! Stage 2: Anthropic (Sonnet 4.6) teacher generates the observation.

use axum::extract::{Json, State};
use wasm_bindgen::JsValue;
use worker::{console_error, Env};

use crate::auth::AuthUser;
use crate::error::{ApiError, Result};
use crate::services::{llm, prompts};
use crate::state::AppState;
use crate::types::{SessionId, StudentId};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskRequest {
    pub teaching_moment: serde_json::Value,
    pub student: serde_json::Value,
    pub session: serde_json::Value,
    #[serde(default)]
    pub piece_context: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AskResponse {
    pub observation: String,
    pub observation_id: String,
    pub dimension: String,
    pub framing: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_fallback: bool,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ElaborateRequest {
    pub observation_id: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ElaborateResponse {
    pub elaboration: String,
    pub observation_id: String,
}

/// Input for the core LLM pipeline (used by both HTTP handler and DO).
#[derive(Debug)]
pub struct AskInnerRequest {
    pub teaching_moment: serde_json::Value,
    pub student_id: StudentId,
    pub session_id: SessionId,
    pub piece_context: Option<serde_json::Value>,
}

/// Output from the core LLM pipeline (no D1 side effects).
#[derive(Debug)]
pub struct AskInnerResponse {
    pub observation_text: String,
    pub dimension: String,
    pub framing: String,
    pub reasoning_trace: String,
    pub is_fallback: bool,
    /// Optional exercise artifact from teacher tool use.
    pub components_json: Option<String>,
}

/// Dimension descriptions for fallback templates.
fn dimension_description(dimension: &str) -> &str {
    match dimension {
        "dynamics" => "dynamic range and volume control",
        "timing" => "rhythmic accuracy and tempo consistency",
        "pedaling" => "pedal clarity and harmonic changes",
        "articulation" => "note clarity and touch",
        "phrasing" => "musical phrasing and melodic shaping",
        "interpretation" => "musical expression and stylistic choices",
        _ => "that aspect of your playing",
    }
}

/// Generate a template fallback observation when LLM calls fail.
fn fallback_observation(dimension: &str) -> String {
    format!(
        "I noticed your {} could use some attention in that last section. \
         Try recording yourself and listening back -- sometimes it's hard to hear {} while you're playing.",
        dimension,
        dimension_description(dimension),
    )
}

/// Core two-stage LLM pipeline. No D1 persistence, no HTTP.
/// Called by both the HTTP handler (`handle_ask`) and the DO (`PracticeSession`).
pub async fn handle_ask_inner(env: &Env, req: &AskInnerRequest) -> AskInnerResponse {
    let dimension = req
        .teaching_moment
        .get("dimension")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Build memory context
    let piece_title = req
        .piece_context
        .as_ref()
        .and_then(|pc| pc.get("title"))
        .and_then(|v| v.as_str());

    let now = crate::types::now_iso();
    let today = &now[..10.min(now.len())];
    let memory_ctx = crate::services::memory::build_memory_context(
        env,
        &req.student_id,
        piece_title,
        today,
        piece_title,
    )
    .await;
    let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

    let recent_observations: Vec<prompts::ObservationRow> = memory_ctx
        .recent_observations
        .iter()
        .map(|obs| prompts::ObservationRow {
            dimension: obs.dimension.clone(),
            observation_text: obs.observation_text.clone(),
            framing: obs.framing.clone(),
            created_at: obs.created_at.clone(),
        })
        .collect();

    // Build student/session context for subagent prompt
    let student_json = serde_json::json!({
        "level": "intermediate",
        "goals": "",
    });
    let session_json = serde_json::json!({
        "id": req.session_id,
    });

    // Stage 1: Subagent (Groq)
    let subagent_user_prompt = prompts::build_subagent_user_prompt(
        &req.teaching_moment,
        &student_json,
        &session_json,
        &req.piece_context,
        &recent_observations,
        &memory_text,
    );

    let subagent_result = llm::call_groq(
        env,
        prompts::SUBAGENT_SYSTEM,
        &subagent_user_prompt,
        0.3,
        800,
    )
    .await;

    let (subagent_output, framing) = match subagent_result {
        Ok(output) => {
            let framing = extract_framing(&output).unwrap_or_else(|| "correction".to_string());
            (output, framing)
        }
        Err(e) => {
            console_error!("Subagent failed: {}", e);
            return AskInnerResponse {
                observation_text: fallback_observation(&dimension),
                dimension,
                framing: "correction".to_string(),
                reasoning_trace: "{}".to_string(),
                is_fallback: true,
                components_json: None,
            };
        }
    };

    let (subagent_json, subagent_narrative) = split_subagent_output(&subagent_output);

    // Stage 2: Teacher (Anthropic) with tool use
    let catalog = lookup_catalog_exercises(env, &dimension).await;

    let teacher_user_prompt = if catalog.is_empty() {
        prompts::build_teacher_user_prompt(&subagent_json, &subagent_narrative, "intermediate", "")
    } else {
        prompts::build_teacher_user_prompt_with_catalog(
            &subagent_json,
            &subagent_narrative,
            "intermediate",
            "",
            &catalog,
        )
    };

    let tools = vec![prompts::exercise_tool_definition()];

    let teacher_result = llm::call_anthropic_with_tools(
        env,
        prompts::TEACHER_SYSTEM,
        &teacher_user_prompt,
        500,
        Some(tools),
    )
    .await;

    let (observation_text, is_fallback, components_json) = match teacher_result {
        Ok(result) => {
            let text = post_process_observation(&result.text);

            // Process any exercise tool calls
            let components = if let Some(tc) = result.tool_calls.first() {
                if tc.name == "create_exercise" {
                    match process_exercise_tool_call(env, &tc.input).await {
                        Ok(json) => Some(json),
                        Err(e) => {
                            console_error!("Exercise tool call processing failed: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            (text, false, components)
        }
        Err(e) => {
            console_error!("Teacher LLM failed: {}", e);
            (fallback_observation(&dimension), true, None)
        }
    };

    let reasoning_trace = serde_json::json!({
        "subagent_output": subagent_output,
    })
    .to_string();

    AskInnerResponse {
        observation_text,
        dimension,
        framing,
        reasoning_trace,
        is_fallback,
        components_json,
    }
}

/// Handle POST /api/ask
#[worker::send]
pub async fn handle_ask(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<AskRequest>,
) -> Result<Json<AskResponse>> {
    let student_id = auth.student_id;
    let env = state.db.env();

    let dimension_score = request
        .teaching_moment
        .get("dimension_score")
        .and_then(serde_json::Value::as_f64);

    let dimension = request
        .teaching_moment
        .get("dimension")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let student_baseline = request
        .student
        .get("baselines")
        .and_then(|b| b.get(&dimension))
        .and_then(serde_json::Value::as_f64);

    // Call the core LLM pipeline
    let inner_req = AskInnerRequest {
        teaching_moment: request.teaching_moment.clone(),
        student_id: student_id.clone(),
        session_id: SessionId::from(
            request
                .session
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        ),
        piece_context: request.piece_context.clone(),
    };

    let inner_resp = handle_ask_inner(env, &inner_req).await;

    let observation_text = inner_resp.observation_text;
    let framing = inner_resp.framing;
    let is_fallback = inner_resp.is_fallback;
    // inner_resp.components_json unused: HTTP path does not use components yet (observations go through DO WebSocket)

    // Generate observation ID
    let observation_id = crate::types::generate_uuid_v4();

    // Build reasoning trace
    let reasoning_trace = serde_json::json!({
        "reasoning_trace": inner_resp.reasoning_trace,
    });

    // Store observation in D1
    let session_id = request
        .session
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let chunk_index = request
        .teaching_moment
        .get("chunk_index")
        .and_then(serde_json::Value::as_i64);
    let piece_context_str = request
        .piece_context
        .as_ref()
        .and_then(|v| serde_json::to_string(v).ok());

    if let Err(e) = store_observation(
        env,
        &observation_id,
        &student_id,
        session_id,
        chunk_index,
        &dimension,
        &observation_text,
        &reasoning_trace.to_string(),
        &framing,
        dimension_score,
        student_baseline,
        piece_context_str.as_deref(),
        is_fallback,
    )
    .await
    {
        console_error!("Failed to store observation: {}", e);
        // Non-fatal: still return the observation
    }

    // Store teaching approach record
    let approach_id = crate::types::generate_uuid_v4();
    let approach_summary = format!("{framing} on {dimension}");
    if let Err(e) = crate::services::memory::store_teaching_approach(
        env,
        &approach_id,
        &student_id,
        &observation_id,
        &dimension,
        &framing,
        &approach_summary,
    )
    .await
    {
        console_error!("Failed to store teaching approach: {}", e);
    }

    // Increment observation count for synthesis tracking
    if let Err(e) = crate::services::memory::increment_observation_count(env, &student_id).await {
        console_error!("Failed to increment observation count: {}", e);
    }

    Ok(Json(AskResponse {
        observation: observation_text,
        observation_id,
        dimension,
        framing,
        is_fallback,
    }))
}

/// Handle POST /api/ask/elaborate
#[worker::send]
pub async fn handle_elaborate(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(request): Json<ElaborateRequest>,
) -> Result<Json<ElaborateResponse>> {
    let student_id = auth.student_id;
    let env = state.db.env();

    // Fetch observation from D1 (verify student ownership)
    let (observation_text, reasoning_trace) =
        fetch_observation(env, &request.observation_id, &student_id)
            .await
            .map_err(|e| ApiError::NotFound(format!("Observation not found: {e}")))?;

    // Fetch memory context for richer elaboration
    let elab_now = crate::types::now_iso();
    let elab_today = &elab_now[..10.min(elab_now.len())];
    let memory_ctx =
        crate::services::memory::build_memory_context(env, &student_id, None, elab_today, None)
            .await;
    let memory_text = crate::services::memory::format_chat_memory_patterns(&memory_ctx);

    // Call teacher LLM for elaboration
    let elaboration_prompt =
        prompts::build_elaboration_prompt(&observation_text, &reasoning_trace, &memory_text);

    let elaboration = llm::call_anthropic(env, prompts::TEACHER_SYSTEM, &elaboration_prompt, 500)
        .await
        .map(|text| post_process_observation(&text))
        .map_err(|e| {
            console_error!("Elaboration LLM failed: {}", e);
            ApiError::ExternalService("Unable to generate elaboration".into())
        })?;

    // Store elaboration (verify student ownership)
    if let Err(e) = store_elaboration(env, &request.observation_id, &student_id, &elaboration).await
    {
        console_error!("Failed to store elaboration: {}", e);
    }

    // Mark teaching approach as engaged (verify student ownership)
    if let Err(e) =
        crate::services::memory::mark_approach_engaged(env, &request.observation_id, &student_id)
            .await
    {
        console_error!("Failed to mark approach engaged: {}", e);
    }

    Ok(Json(ElaborateResponse {
        elaboration,
        observation_id: request.observation_id,
    }))
}

// --- Helper functions ---

#[allow(clippy::too_many_arguments)]
async fn store_observation(
    env: &Env,
    id: &str,
    student_id: &StudentId,
    session_id: &str,
    chunk_index: Option<i64>,
    dimension: &str,
    observation_text: &str,
    reasoning_trace: &str,
    framing: &str,
    dimension_score: Option<f64>,
    student_baseline: Option<f64>,
    piece_context: Option<&str>,
    is_fallback: bool,
) -> Result<()> {
    let db = env
        .d1("DB")
        .map_err(|e| ApiError::Internal(format!("D1 binding failed: {e:?}")))?;
    let now = crate::types::now_iso();

    db.prepare(
        "INSERT INTO observations (id, student_id, session_id, chunk_index, dimension, \
         observation_text, reasoning_trace, framing, dimension_score, student_baseline, \
         piece_context, is_fallback, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )
    .bind(&[
        JsValue::from_str(id),
        JsValue::from_str(student_id.as_str()),
        JsValue::from_str(session_id),
        match chunk_index {
            Some(i) => JsValue::from_f64(i as f64),
            None => JsValue::NULL,
        },
        JsValue::from_str(dimension),
        JsValue::from_str(observation_text),
        JsValue::from_str(reasoning_trace),
        JsValue::from_str(framing),
        match dimension_score {
            Some(f) => JsValue::from_f64(f),
            None => JsValue::NULL,
        },
        match student_baseline {
            Some(f) => JsValue::from_f64(f),
            None => JsValue::NULL,
        },
        match piece_context {
            Some(s) => JsValue::from_str(s),
            None => JsValue::NULL,
        },
        JsValue::from_bool(is_fallback),
        JsValue::from_str(&now),
    ])
    .map_err(|e| ApiError::Internal(format!("Failed to bind insert: {e:?}")))?
    .run()
    .await
    .map_err(|e| ApiError::Internal(format!("Failed to insert observation: {e:?}")))?;

    Ok(())
}

async fn store_elaboration(
    env: &Env,
    observation_id: &str,
    student_id: &StudentId,
    elaboration: &str,
) -> Result<()> {
    let db = env
        .d1("DB")
        .map_err(|e| ApiError::Internal(format!("D1 binding failed: {e:?}")))?;

    db.prepare(
        "UPDATE observations SET elaboration_text = ?1 WHERE id = ?2 AND student_id = ?3",
    )
    .bind(&[
        JsValue::from_str(elaboration),
        JsValue::from_str(observation_id),
        JsValue::from_str(student_id.as_str()),
    ])
    .map_err(|e| ApiError::Internal(format!("Failed to bind update: {e:?}")))?
    .run()
    .await
    .map_err(|e| ApiError::Internal(format!("Failed to update elaboration: {e:?}")))?;

    Ok(())
}

async fn fetch_observation(
    env: &Env,
    observation_id: &str,
    student_id: &StudentId,
) -> Result<(String, String)> {
    let db = env
        .d1("DB")
        .map_err(|e| ApiError::Internal(format!("D1 binding failed: {e:?}")))?;

    let row: Option<serde_json::Value> = db
        .prepare(
            "SELECT observation_text, reasoning_trace FROM observations \
             WHERE id = ?1 AND student_id = ?2",
        )
        .bind(&[
            JsValue::from_str(observation_id),
            JsValue::from_str(student_id.as_str()),
        ])
        .map_err(|e| ApiError::Internal(format!("Failed to bind query: {e:?}")))?
        .first(None)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to query observation: {e:?}")))?;

    match row {
        Some(r) => {
            let text = r
                .get("observation_text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let trace = r
                .get("reasoning_trace")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok((text, trace))
        }
        None => Err(ApiError::NotFound("Observation not found".to_string())),
    }
}

/// Extract the framing field from subagent JSON output.
fn extract_framing(subagent_output: &str) -> Option<String> {
    let json_str = extract_json_block(subagent_output)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    parsed
        .get("framing")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)
}

/// Split subagent output into JSON block and narrative.
fn split_subagent_output(output: &str) -> (String, String) {
    if let Some(json_str) = extract_json_block(output) {
        // Everything after the JSON block is the narrative
        let json_end = output.rfind('}').unwrap_or(0) + 1;
        // Also skip past ``` if present
        let narrative_start = output[json_end..]
            .find(|c: char| c.is_alphabetic())
            .map_or(json_end, |i| json_end + i);
        let narrative = output[narrative_start..].trim().to_string();
        (json_str, narrative)
    } else {
        // No JSON found, treat entire output as narrative
        ("{}".to_string(), output.trim().to_string())
    }
}

/// Extract a JSON object from text that may contain markdown code fences.
fn extract_json_block(text: &str) -> Option<String> {
    // Try to find JSON within ```json ... ``` fences
    if let Some(start) = text.find("```json") {
        let json_start = start + 7;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim().to_string());
        }
    }
    // Try to find JSON within ``` ... ``` fences
    if let Some(start) = text.find("```\n{") {
        let json_start = start + 4;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim().to_string());
        }
    }
    // Try to find a raw JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            let candidate = &text[start..=end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// Strip markdown formatting and validate length.
fn post_process_observation(text: &str) -> String {
    let mut cleaned = text.trim().to_string();
    // Strip leading/trailing quotes
    if cleaned.starts_with('"') && cleaned.ends_with('"') {
        cleaned = cleaned[1..cleaned.len() - 1].to_string();
    }
    // Strip markdown bold/italic
    cleaned = cleaned.replace("**", "").replace("__", "");
    // Truncate if too long (500 char limit from design)
    if cleaned.len() > 500 {
        if let Some(last_period) = cleaned[..500].rfind('.') {
            cleaned = cleaned[..=last_period].to_string();
        } else {
            cleaned.truncate(500);
        }
    }
    cleaned
}


/// Look up catalog exercises matching a dimension.
/// Returns up to 5 matching exercises as (id, title, description, difficulty).
async fn lookup_catalog_exercises(
    env: &Env,
    dimension: &str,
) -> Vec<(String, String, String, String)> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed for catalog lookup: {:?}", e);
            return Vec::new();
        }
    };

    let query = match db
        .prepare(
            "SELECT e.id, e.title, e.description, e.difficulty \
             FROM exercises e \
             JOIN exercise_dimensions ed ON e.id = ed.exercise_id \
             WHERE ed.dimension = ?1 AND e.source = 'curated' \
             LIMIT 5",
        )
        .bind(&[JsValue::from_str(dimension)])
    {
        Ok(q) => q,
        Err(e) => {
            console_error!("Catalog query bind failed: {:?}", e);
            return Vec::new();
        }
    };

    match query.all().await {
        Ok(result) => {
            let rows: Vec<serde_json::Value> = result.results().unwrap_or_default();
            rows.iter()
                .filter_map(|row| {
                    let id = row.get("id")?.as_str()?.to_string();
                    let title = row.get("title")?.as_str()?.to_string();
                    let desc = row.get("description")?.as_str()?.to_string();
                    let diff = row.get("difficulty")?.as_str()?.to_string();
                    Some((id, title, desc, diff))
                })
                .collect()
        }
        Err(e) => {
            console_error!("Catalog query failed: {:?}", e);
            Vec::new()
        }
    }
}

/// Persist a teacher-generated exercise to D1 and return the `exercise_id`.
async fn persist_generated_exercise(
    env: &Env,
    title: &str,
    instruction: &str,
    focus_dimension: &str,
    source_passage: &str,
    target_skill: &str,
) -> Result<String> {
    let db = env
        .d1("DB")
        .map_err(|e| ApiError::Internal(format!("D1 binding failed: {e:?}")))?;
    let exercise_id = crate::types::generate_uuid_v4();
    let now = crate::types::now_iso();

    db.prepare(
        "INSERT INTO exercises (id, title, description, instructions, difficulty, category, source, created_at) \
         VALUES (?1, ?2, ?3, ?4, 'intermediate', 'generated', 'teacher_llm', ?5)",
    )
    .bind(&[
        JsValue::from_str(&exercise_id),
        JsValue::from_str(title),
        JsValue::from_str(&format!("{target_skill} -- {source_passage}")),
        JsValue::from_str(instruction),
        JsValue::from_str(&now),
    ])
    .map_err(|e| ApiError::Internal(format!("Failed to bind exercise insert: {e:?}")))?
    .run()
    .await
    .map_err(|e| ApiError::Internal(format!("Failed to insert exercise: {e:?}")))?;

    // Link to dimension
    let _ = db
        .prepare("INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES (?1, ?2)")
        .bind(&[
            JsValue::from_str(&exercise_id),
            JsValue::from_str(focus_dimension),
        ])
        .map_err(|e| ApiError::Internal(format!("Failed to bind dimension insert: {e:?}")))?
        .run()
        .await;

    Ok(exercise_id)
}

/// Process a `create_exercise` tool call: validate, persist each exercise, return components JSON.
async fn process_exercise_tool_call(env: &Env, input: &serde_json::Value) -> Result<String> {
    const VALID_DIMS: &[&str] = &[
        "dynamics",
        "timing",
        "pedaling",
        "articulation",
        "phrasing",
        "interpretation",
    ];

    let source_passage = input
        .get("source_passage")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("Missing source_passage".into()))?;
    let target_skill = input
        .get("target_skill")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("Missing target_skill".into()))?;
    let exercises = input
        .get("exercises")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ApiError::BadRequest("Missing exercises array".into()))?;

    let mut processed_exercises = Vec::new();

    for ex in exercises {
        let title = ex
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Practice Drill");
        let instruction = ex.get("instruction").and_then(|v| v.as_str()).unwrap_or("");
        let raw_dim = ex
            .get("focus_dimension")
            .and_then(|v| v.as_str())
            .unwrap_or("dynamics");
        let focus_dim = if VALID_DIMS.contains(&raw_dim) {
            raw_dim
        } else {
            "dynamics"
        };
        let hands = ex.get("hands").and_then(|v| v.as_str());

        // Check if this references a catalog exercise by ID
        let exercise_id = if let Some(id) = ex.get("exercise_id").and_then(|v| v.as_str()) {
            id.to_string()
        } else {
            // Generate and persist new exercise
            persist_generated_exercise(
                env,
                title,
                instruction,
                focus_dim,
                source_passage,
                target_skill,
            )
            .await?
        };

        let mut ex_json = serde_json::json!({
            "title": title,
            "instruction": instruction,
            "focusDimension": focus_dim,
            "exerciseId": exercise_id,
        });
        if let Some(h) = hands {
            ex_json["hands"] = serde_json::json!(h);
        }
        processed_exercises.push(ex_json);
    }

    let component = serde_json::json!([{
        "type": "exercise_set",
        "config": {
            "sourcePassage": source_passage,
            "targetSkill": target_skill,
            "exercises": processed_exercises,
        }
    }]);

    serde_json::to_string(&component)
        .map_err(|e| ApiError::Internal(format!("Failed to serialize components: {e:?}")))
}
