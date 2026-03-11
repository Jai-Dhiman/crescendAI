//! Handler for POST /api/ask -- the two-stage teacher pipeline.
//!
//! Stage 1: Groq (Llama 70B) subagent analyzes the teaching moment.
//! Stage 2: Anthropic (Sonnet 4.6) teacher generates the observation.

use wasm_bindgen::JsValue;
use worker::{console_error, console_log, Env};

use crate::services::{llm, prompts};

#[derive(serde::Deserialize)]
pub struct AskRequest {
    pub teaching_moment: serde_json::Value,
    pub student: serde_json::Value,
    pub session: serde_json::Value,
    pub piece_context: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
pub struct AskResponse {
    pub observation: String,
    pub observation_id: String,
    pub dimension: String,
    pub framing: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_fallback: bool,
}

#[derive(serde::Deserialize)]
pub struct ElaborateRequest {
    pub observation_id: String,
}

#[derive(serde::Serialize)]
pub struct ElaborateResponse {
    pub elaboration: String,
    pub observation_id: String,
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

/// Handle POST /api/ask
pub async fn handle_ask(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: AskRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse ask request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    let dimension = request
        .teaching_moment
        .get("dimension")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let dimension_score = request
        .teaching_moment
        .get("dimension_score")
        .and_then(|v| v.as_f64());

    let student_baseline = request
        .student
        .get("baselines")
        .and_then(|b| b.get(&dimension))
        .and_then(|v| v.as_f64());

    // Build memory context (replaces simple query_recent_observations)
    let piece_title = request
        .piece_context
        .as_ref()
        .and_then(|pc| pc.get("title"))
        .and_then(|v| v.as_str());

    let now_ask = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    let today_ask = &now_ask[..10.min(now_ask.len())];
    let memory_ctx = crate::services::memory::build_memory_context(
        env,
        &student_id,
        piece_title,
        today_ask,
    ).await;

    let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

    // Convert recent observations for backward compat with prompt builder
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

    // Stage 1: Subagent (Groq)
    let subagent_user_prompt = prompts::build_subagent_user_prompt(
        &request.teaching_moment,
        &request.student,
        &request.session,
        &request.piece_context,
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
            // Fallback: skip subagent, go directly to fallback template
            return build_fallback_response(
                env,
                &student_id,
                &request,
                &dimension,
                dimension_score,
                student_baseline,
            )
            .await;
        }
    };

    // Parse subagent JSON and narrative
    let (subagent_json, subagent_narrative) = split_subagent_output(&subagent_output);

    // Stage 2: Teacher (Anthropic)
    let student_level = request
        .student
        .get("level")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let student_goals = request
        .student
        .get("goals")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let teacher_user_prompt = prompts::build_teacher_user_prompt(
        &subagent_json,
        &subagent_narrative,
        student_level,
        student_goals,
    );

    let teacher_result = llm::call_anthropic(
        env,
        prompts::TEACHER_SYSTEM,
        &teacher_user_prompt,
        300,
    )
    .await;

    let (observation_text, is_fallback) = match teacher_result {
        Ok(text) => {
            let cleaned = post_process_observation(&text);
            (cleaned, false)
        }
        Err(e) => {
            console_error!("Teacher LLM failed: {}", e);
            (fallback_observation(&dimension), true)
        }
    };

    // Generate observation ID
    let observation_id = generate_uuid();

    // Build reasoning trace
    let reasoning_trace = serde_json::json!({
        "subagent_output": subagent_output,
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
        .and_then(|v| v.as_i64());
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
    let approach_id = generate_uuid();
    let approach_summary = format!("{} on {}", framing, dimension);
    if let Err(e) = crate::services::memory::store_teaching_approach(
        env,
        &approach_id,
        &student_id,
        &observation_id,
        &dimension,
        &framing,
        &approach_summary,
    ).await {
        console_error!("Failed to store teaching approach: {}", e);
    }

    // Increment observation count for synthesis tracking
    if let Err(e) = crate::services::memory::increment_observation_count(env, &student_id).await {
        console_error!("Failed to increment observation count: {}", e);
    }

    let response = AskResponse {
        observation: observation_text,
        observation_id,
        dimension,
        framing,
        is_fallback,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// Handle POST /api/ask/elaborate
pub async fn handle_elaborate(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: ElaborateRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to parse elaborate request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Fetch observation from D1
    let (observation_text, reasoning_trace) =
        match fetch_observation(env, &request.observation_id).await {
            Ok(obs) => obs,
            Err(e) => {
                console_error!("Failed to fetch observation: {}", e);
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Observation not found"}"#))
                    .unwrap();
            }
        };

    // Fetch memory context for richer elaboration
    let elab_now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    let elab_today = &elab_now[..10.min(elab_now.len())];
    let memory_ctx = crate::services::memory::build_memory_context(env, &student_id, None, elab_today).await;
    let memory_text = crate::services::memory::format_chat_memory_patterns(&memory_ctx);

    // Call teacher LLM for elaboration
    let elaboration_prompt =
        prompts::build_elaboration_prompt(&observation_text, &reasoning_trace, &memory_text);

    let elaboration = match llm::call_anthropic(
        env,
        prompts::TEACHER_SYSTEM,
        &elaboration_prompt,
        500,
    )
    .await
    {
        Ok(text) => post_process_observation(&text),
        Err(e) => {
            console_error!("Elaboration LLM failed: {}", e);
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Unable to generate elaboration"}"#))
                .unwrap();
        }
    };

    // Store elaboration
    if let Err(e) = store_elaboration(env, &request.observation_id, &elaboration).await {
        console_error!("Failed to store elaboration: {}", e);
    }

    // Mark teaching approach as engaged
    if let Err(e) = crate::services::memory::mark_approach_engaged(env, &request.observation_id).await {
        console_error!("Failed to mark approach engaged: {}", e);
    }

    let response = ElaborateResponse {
        elaboration,
        observation_id: request.observation_id,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

// --- Helper functions ---

#[allow(clippy::too_many_arguments)]
async fn store_observation(
    env: &Env,
    id: &str,
    student_id: &str,
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
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "INSERT INTO observations (id, student_id, session_id, chunk_index, dimension, \
         observation_text, reasoning_trace, framing, dimension_score, student_baseline, \
         piece_context, is_fallback, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )
    .bind(&[
        JsValue::from_str(id),
        JsValue::from_str(student_id),
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
    .map_err(|e| format!("Failed to bind insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert observation: {:?}", e))?;

    Ok(())
}

async fn store_elaboration(
    env: &Env,
    observation_id: &str,
    elaboration: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare("UPDATE observations SET elaboration_text = ?1 WHERE id = ?2")
        .bind(&[
            JsValue::from_str(elaboration),
            JsValue::from_str(observation_id),
        ])
        .map_err(|e| format!("Failed to bind update: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to update elaboration: {:?}", e))?;

    Ok(())
}

async fn fetch_observation(
    env: &Env,
    observation_id: &str,
) -> Result<(String, String), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let row: Option<serde_json::Value> = db
        .prepare("SELECT observation_text, reasoning_trace FROM observations WHERE id = ?1")
        .bind(&[JsValue::from_str(observation_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query observation: {:?}", e))?;

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
        None => Err("Observation not found".to_string()),
    }
}

/// Extract the framing field from subagent JSON output.
fn extract_framing(subagent_output: &str) -> Option<String> {
    let json_str = extract_json_block(subagent_output)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    parsed
        .get("framing")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Split subagent output into JSON block and narrative.
fn split_subagent_output(output: &str) -> (String, String) {
    if let Some(json_str) = extract_json_block(output) {
        // Everything after the JSON block is the narrative
        let json_end = output.rfind('}').unwrap_or(0) + 1;
        // Also skip past ``` if present
        let narrative_start = output[json_end..]
            .find(|c: char| c.is_alphabetic())
            .map(|i| json_end + i)
            .unwrap_or(json_end);
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

/// Generate a UUID v4 (same pattern as auth module).
pub fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// Build a fallback response when the subagent fails.
async fn build_fallback_response(
    env: &Env,
    student_id: &str,
    request: &AskRequest,
    dimension: &str,
    dimension_score: Option<f64>,
    student_baseline: Option<f64>,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let observation_text = fallback_observation(dimension);
    let observation_id = generate_uuid();
    let session_id = request
        .session
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let chunk_index = request
        .teaching_moment
        .get("chunk_index")
        .and_then(|v| v.as_i64());

    let _ = store_observation(
        env,
        &observation_id,
        student_id,
        session_id,
        chunk_index,
        dimension,
        &observation_text,
        "{}",
        "correction",
        dimension_score,
        student_baseline,
        None,
        true,
    )
    .await;

    let response = AskResponse {
        observation: observation_text,
        observation_id,
        dimension: dimension.to_string(),
        framing: "correction".to_string(),
        is_fallback: true,
    };

    let json = serde_json::to_string(&response)
        .unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}
