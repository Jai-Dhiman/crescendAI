use axum::extract::{Json, Query, State};
use wasm_bindgen::JsValue;
use worker::{console_error, console_log, Env};

use super::accumulator::{AccumulatedMoment, SessionAccumulator};
use super::error::PracticeError;
use crate::auth::extractor::AuthUser;
use crate::error::{ApiError, Result as ApiResult};
use crate::services::prompts::SESSION_SYNTHESIS_SYSTEM;
use crate::services::teaching_moments::StudentBaselines;
use crate::state::AppState;
use crate::types::{ConversationId, SessionId, StudentId};

/// Input context for synthesis: identifiers + optional enrichment signals.
pub struct SynthesisContext {
    pub session_id: SessionId,
    pub student_id: StudentId,
    pub conversation_id: ConversationId,
    pub baselines: Option<StudentBaselines>,
    pub piece_context: Option<serde_json::Value>,
    pub student_memory: Option<String>,
    pub total_chunks: usize,
    pub session_duration_ms: u64,
}

/// Output of the synthesis LLM call.
pub struct SynthesisResult {
    pub text: String,
    pub is_fallback: bool,
}

/// Build the structured JSON prompt that is passed as the user message to the
/// synthesis LLM call.
pub fn build_synthesis_prompt(
    acc: &SessionAccumulator,
    ctx: &SynthesisContext,
) -> serde_json::Value {
    let duration_min = (ctx.session_duration_ms as f64 / 60_000.0 * 10.0).round() / 10.0;

    let practice_pattern = build_practice_pattern(acc, ctx.session_duration_ms);
    let top_moments: Vec<serde_json::Value> = acc
        .top_moments(None)
        .iter()
        .map(|m| moment_to_json(m))
        .collect();

    let mut obj = serde_json::json!({
        "session_duration_minutes": duration_min,
        "chunks_processed": ctx.total_chunks,
        "practice_pattern": practice_pattern,
        "top_moments": top_moments,
    });

    // Baselines
    if let Some(ref b) = ctx.baselines {
        obj["baselines"] = serde_json::json!({
            "dynamics": b.dynamics,
            "timing": b.timing,
            "pedaling": b.pedaling,
            "articulation": b.articulation,
            "phrasing": b.phrasing,
            "interpretation": b.interpretation,
        });
    } else {
        obj["baselines"] = serde_json::Value::Null;
    }

    // Optional piece context
    if let Some(ref piece) = ctx.piece_context {
        obj["piece"] = piece.clone();
    }

    // Optional student memory
    if let Some(ref memory) = ctx.student_memory {
        obj["student_memory"] = serde_json::Value::String(memory.clone());
    }

    // Drilling progress (if any drilling occurred)
    if !acc.drilling_records.is_empty() {
        let dims = [
            "dynamics",
            "timing",
            "pedaling",
            "articulation",
            "phrasing",
            "interpretation",
        ];
        let drilling_progress: Vec<serde_json::Value> = acc
            .drilling_records
            .iter()
            .map(|dr| {
                let first: std::collections::HashMap<&str, f64> = dims
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (*d, (dr.first_scores[i] * 1000.0).round() / 1000.0))
                    .collect();
                let last: std::collections::HashMap<&str, f64> = dims
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (*d, (dr.final_scores[i] * 1000.0).round() / 1000.0))
                    .collect();
                let mut entry = serde_json::json!({
                    "repetitions": dr.repetition_count,
                    "first_scores": first,
                    "final_scores": last,
                });
                if let Some((start, end)) = dr.bar_range {
                    entry["bar_range"] = serde_json::json!([start, end]);
                }
                entry
            })
            .collect();
        obj["drilling_progress"] = serde_json::Value::Array(drilling_progress);
    }

    obj
}

/// Build the `practice_pattern` array from mode transitions.
fn build_practice_pattern(acc: &SessionAccumulator, session_duration_ms: u64) -> serde_json::Value {
    if acc.mode_transitions.is_empty() {
        return serde_json::Value::Array(vec![]);
    }

    let mut entries: Vec<serde_json::Value> = Vec::new();

    for (i, tr) in acc.mode_transitions.iter().enumerate() {
        let end_ts = if i + 1 < acc.mode_transitions.len() {
            acc.mode_transitions[i + 1].timestamp_ms
        } else {
            session_duration_ms
        };

        let duration_min =
            ((end_ts.saturating_sub(tr.timestamp_ms)) as f64 / 60_000.0 * 10.0).round() / 10.0;

        let mode_name = format!("{:?}", tr.to).to_lowercase();
        let mut entry = serde_json::json!({
            "mode": mode_name,
            "duration_min": duration_min,
        });

        // If transitioning into drilling, include bar range and repetitions from
        // a drilling record if one matches by chunk index.
        if tr.to == super::practice_mode::PracticeMode::Drilling {
            if let Some(dr) = acc
                .drilling_records
                .iter()
                .find(|dr| dr.started_at_chunk == tr.chunk_index)
            {
                if let Some((start, end)) = dr.bar_range {
                    entry["bar_range"] = serde_json::json!([start, end]);
                }
                entry["repetitions"] = serde_json::Value::Number(dr.repetition_count.into());
            }
        }

        entries.push(entry);
    }

    serde_json::Value::Array(entries)
}

/// Serialize a single `AccumulatedMoment` to JSON for the synthesis prompt.
fn moment_to_json(m: &AccumulatedMoment) -> serde_json::Value {
    let deviation_rounded = (m.deviation * 1000.0).round() / 1000.0;
    let mut obj = serde_json::json!({
        "dimension": m.dimension,
        "deviation": deviation_rounded,
        "is_positive": m.is_positive,
        "reasoning": m.reasoning,
    });
    if let Some((start, end)) = m.bar_range {
        obj["bar_range"] = serde_json::json!([start, end]);
    }
    obj
}

/// Call the Anthropic API to synthesize a session summary.
///
/// On any error, returns a fallback message with `is_fallback=true`.
#[allow(clippy::items_after_statements)] // response structs are scoped near their usage
pub async fn call_synthesis_llm(env: &Env, prompt_context: &serde_json::Value) -> SynthesisResult {
    let fallback = SynthesisResult {
        text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
        is_fallback: true,
    };

    let api_key = match env.secret("ANTHROPIC_API_KEY") {
        Ok(s) => s.to_string(),
        Err(e) => {
            console_error!("[synthesis] ANTHROPIC_API_KEY not configured: {:?}", e);
            return fallback;
        }
    };

    let model = env.var("ANTHROPIC_MODEL").map_or_else(
        |_| "claude-sonnet-4-20250514".to_string(),
        |v| v.to_string(),
    );

    let user_content = match serde_json::to_string_pretty(prompt_context) {
        Ok(s) => s,
        Err(e) => {
            console_error!("[synthesis] Failed to serialize prompt context: {:?}", e);
            return fallback;
        }
    };

    let request_body = serde_json::json!({
        "model": model,
        "max_tokens": 1024,
        "system": SESSION_SYNTHESIS_SYSTEM,
        "messages": [{"role": "user", "content": user_content}]
    });

    let body_str = match serde_json::to_string(&request_body) {
        Ok(s) => s,
        Err(e) => {
            console_error!("[synthesis] Failed to serialize request body: {:?}", e);
            return fallback;
        }
    };

    let headers = worker::Headers::new();
    if headers.set("Content-Type", "application/json").is_err()
        || headers.set("x-api-key", &api_key).is_err()
        || headers.set("anthropic-version", "2023-06-01").is_err()
    {
        console_error!("[synthesis] Failed to set request headers");
        return fallback;
    }

    let mut init = worker::RequestInit::new();
    init.with_method(worker::Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_str.into()));

    let request =
        match worker::Request::new_with_init("https://api.anthropic.com/v1/messages", &init) {
            Ok(r) => r,
            Err(e) => {
                console_error!("[synthesis] Failed to create request: {:?}", e);
                return fallback;
            }
        };

    let t0 = js_sys::Date::now();
    let mut response = match worker::Fetch::Request(request).send().await {
        Ok(r) => r,
        Err(e) => {
            console_error!("[synthesis] HTTP request failed: {:?}", e);
            return fallback;
        }
    };
    let latency_ms = js_sys::Date::now() - t0;

    let status = response.status_code();
    let response_text = match response.text().await {
        Ok(t) => t,
        Err(e) => {
            console_error!("[synthesis] Failed to read response body: {:?}", e);
            return fallback;
        }
    };

    if status != 200 {
        console_error!(
            "[synthesis] Anthropic returned status {}: {}",
            status,
            crate::truncate_str(&response_text, 200)
        );
        return fallback;
    }

    #[derive(serde::Deserialize)]
    struct AnthropicResponse {
        content: Vec<AnthropicContent>,
    }
    #[derive(serde::Deserialize)]
    struct AnthropicContent {
        text: String,
    }

    let parsed: AnthropicResponse = match serde_json::from_str(&response_text) {
        Ok(p) => p,
        Err(e) => {
            console_error!("[synthesis] Failed to parse Anthropic response: {:?}", e);
            return fallback;
        }
    };

    let text = if let Some(c) = parsed.content.into_iter().next() {
        c.text
    } else {
        console_error!("[synthesis] No content in Anthropic response");
        return fallback;
    };

    console_log!(
        "[synthesis] LLM call complete in {:.0}ms ({} chars)",
        latency_ms,
        text.len()
    );

    SynthesisResult {
        text,
        is_fallback: false,
    }
}

/// Persist the synthesis message to D1.
///
/// Inserts into messages with role='assistant', `message_type`='synthesis'.
/// Returns the new message id.
pub async fn persist_synthesis_message(
    env: &Env,
    conversation_id: &ConversationId,
    session_id: &SessionId,
    synthesis_text: &str,
) -> Result<String, PracticeError> {
    let db = env
        .d1("DB")
        .map_err(|e| PracticeError::Storage(format!("D1 binding: {e:?}")))?;

    let msg_id = crate::services::ask::generate_uuid();
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )
    .bind(&[
        JsValue::from_str(&msg_id),
        JsValue::from_str(conversation_id.as_str()),
        JsValue::from_str("assistant"),
        JsValue::from_str(synthesis_text),
        JsValue::from_str("synthesis"),
        JsValue::from_str(session_id.as_str()),
        JsValue::from_str(&now),
    ])
    .map_err(|e| PracticeError::Storage(format!("bind INSERT message: {e:?}")))?
    .run()
    .await
    .map_err(|e| PracticeError::Storage(format!("INSERT message: {e:?}")))?;

    Ok(msg_id)
}

/// Persist accumulated teaching moments to the observations table.
///
/// Uses INSERT OR IGNORE to avoid duplicate rows if called more than once.
pub async fn persist_accumulated_moments(
    env: &Env,
    student_id: &StudentId,
    session_id: &SessionId,
    moments: &[AccumulatedMoment],
) -> Result<(), PracticeError> {
    let db = env
        .d1("DB")
        .map_err(|e| PracticeError::Storage(format!("D1 binding: {e:?}")))?;

    for m in moments {
        let obs_id = crate::services::ask::generate_uuid();
        let framing = if m.is_positive {
            "recognition"
        } else {
            "correction"
        };
        let now = js_sys::Date::new_0()
            .to_iso_string()
            .as_string()
            .unwrap_or_default();

        // reasoning is stored as observation_text; llm_analysis as reasoning_trace if present
        let reasoning_trace = m
            .llm_analysis
            .clone()
            .unwrap_or_else(|| m.reasoning.clone());

        db.prepare(
            "INSERT OR IGNORE INTO observations \
             (id, student_id, session_id, dimension, observation_text, reasoning_trace, \
              framing, dimension_score, student_baseline, is_fallback, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )
        .bind(&[
            JsValue::from_str(&obs_id),
            JsValue::from_str(student_id.as_str()),
            JsValue::from_str(session_id.as_str()),
            JsValue::from_str(&m.dimension),
            JsValue::from_str(&m.reasoning),
            JsValue::from_str(&reasoning_trace),
            JsValue::from_str(framing),
            JsValue::from_f64(m.score),
            JsValue::from_f64(m.baseline),
            JsValue::from_f64(0.0),
            JsValue::from_str(&now),
        ])
        .map_err(|e| PracticeError::Storage(format!("bind INSERT observation: {e:?}")))?
        .run()
        .await
        .map_err(|e| PracticeError::Storage(format!("INSERT observation: {e:?}")))?;
    }

    Ok(())
}

/// Load student baselines from D1 observations table.
///
/// Falls back to `SCALER_MEAN` defaults for any dimension with no data.
pub async fn load_baselines_from_d1(
    env: &Env,
    student_id: &StudentId,
) -> Result<crate::services::teaching_moments::StudentBaselines, PracticeError> {
    let defaults = crate::services::stop::SCALER_MEAN;

    let db = env
        .d1("DB")
        .map_err(|e| PracticeError::Storage(format!("D1 binding: {e:?}")))?;

    let stmt = db
        .prepare(
            "SELECT dimension, AVG(dimension_score) as avg_score \
             FROM observations WHERE student_id = ?1 \
             AND created_at > datetime('now', '-30 days') \
             GROUP BY dimension",
        )
        .bind(&[JsValue::from_str(student_id.as_str())])
        .map_err(|e| PracticeError::Storage(format!("baselines bind: {e:?}")))?;

    let rows = stmt
        .all()
        .await
        .map_err(|e| PracticeError::Storage(format!("baselines query: {e:?}")))?;

    let results: Vec<serde_json::Value> = rows.results().unwrap_or_default();
    let mut dim_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for row in &results {
        if let (Some(dim), Some(avg)) = (
            row.get("dimension").and_then(|v| v.as_str()),
            row.get("avg_score").and_then(serde_json::Value::as_f64),
        ) {
            dim_map.insert(dim.to_string(), avg);
        }
    }

    Ok(crate::services::teaching_moments::StudentBaselines {
        dynamics: dim_map.get("dynamics").copied().unwrap_or(defaults[0]),
        timing: dim_map.get("timing").copied().unwrap_or(defaults[1]),
        pedaling: dim_map.get("pedaling").copied().unwrap_or(defaults[2]),
        articulation: dim_map.get("articulation").copied().unwrap_or(defaults[3]),
        phrasing: dim_map.get("phrasing").copied().unwrap_or(defaults[4]),
        interpretation: dim_map
            .get("interpretation")
            .copied()
            .unwrap_or(defaults[5]),
    })
}

/// Clear the `needs_synthesis` flag for a session after successful synthesis.
pub async fn clear_needs_synthesis(env: &Env, session_id: &SessionId) -> Result<(), PracticeError> {
    let db = env
        .d1("DB")
        .map_err(|e| PracticeError::Storage(format!("D1 binding: {e:?}")))?;

    db.prepare("UPDATE sessions SET needs_synthesis = 0 WHERE id = ?1")
        .bind(&[JsValue::from_str(session_id.as_str())])
        .map_err(|e| PracticeError::Storage(format!("clear_needs_synthesis bind: {e:?}")))?
        .run()
        .await
        .map_err(|e| PracticeError::Storage(format!("clear_needs_synthesis UPDATE: {e:?}")))?;

    Ok(())
}

/// Query params for GET /api/practice/needs-synthesis.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeedsSynthesisParams {
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// Response for GET /api/practice/needs-synthesis.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeedsSynthesisResponse {
    pub session_ids: Vec<String>,
}

/// GET /api/practice/needs-synthesis?conversation_id=...
///
/// Returns session IDs for a conversation that have `needs_synthesis=1`.
#[worker::send]
pub async fn handle_check_needs_synthesis(
    State(state): State<AppState>,
    auth: AuthUser,
    Query(params): Query<NeedsSynthesisParams>,
) -> ApiResult<Json<NeedsSynthesisResponse>> {
    let student_id = auth.student_id.to_string();
    let conversation_id = params
        .conversation_id
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("conversation_id is required".into()))?;

    let db = state.db.d1()?;

    let stmt = db
        .prepare(
            "SELECT id FROM sessions WHERE conversation_id = ?1 AND student_id = ?2 AND needs_synthesis = 1",
        )
        .bind(&[
            JsValue::from_str(&conversation_id),
            JsValue::from_str(&student_id),
        ])
        .map_err(|e| {
            console_error!("[synthesis] needs-synthesis bind failed: {:?}", e);
            ApiError::Internal("Query preparation failed".into())
        })?;

    let results = stmt.all().await.map_err(|e| {
        console_error!("[synthesis] needs-synthesis query failed: {:?}", e);
        ApiError::Internal("Query failed".into())
    })?;

    let rows: Vec<serde_json::Value> = results.results().unwrap_or_default();
    let session_ids: Vec<String> = rows
        .iter()
        .filter_map(|row| {
            row.get("id")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string)
        })
        .collect();

    Ok(Json(NeedsSynthesisResponse { session_ids }))
}

/// Request body for POST /api/practice/synthesize.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeferredSynthesisRequest {
    pub session_id: String,
}

/// Response for POST /api/practice/synthesize.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredSynthesisResponse {
    pub status: String,
    pub session_id: String,
    pub is_fallback: bool,
}

/// POST /api/practice/synthesize
///
/// Performs deferred synthesis for a session that has `needs_synthesis=1`.
#[worker::send]
pub async fn handle_deferred_synthesis(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(body): Json<DeferredSynthesisRequest>,
) -> ApiResult<Json<DeferredSynthesisResponse>> {
    let student_id_str = auth.student_id.to_string();

    if body.session_id.is_empty() {
        return Err(ApiError::BadRequest("session_id is required".into()));
    }
    let session_id = body.session_id;
    let session_id_typed = SessionId::from(session_id.clone());

    let db = state.db.d1()?;
    let env = state.practice.env();

    // Load session row
    let stmt = db
        .prepare(
            "SELECT student_id, conversation_id, accumulator_json, needs_synthesis FROM sessions WHERE id = ?1",
        )
        .bind(&[JsValue::from_str(&session_id)])
        .map_err(|e| {
            console_error!("[synthesis] deferred-synthesis bind failed: {:?}", e);
            ApiError::Internal("Query preparation failed".into())
        })?;

    let row = stmt
        .first::<serde_json::Value>(None)
        .await
        .map_err(|e| {
            console_error!("[synthesis] deferred-synthesis query failed: {:?}", e);
            ApiError::Internal("Query failed".into())
        })?
        .ok_or_else(|| ApiError::NotFound("Session not found".into()))?;

    // Verify ownership
    let row_student_id = row.get("student_id").and_then(|v| v.as_str()).unwrap_or("");
    if row_student_id != student_id_str {
        return Err(ApiError::Forbidden);
    }

    let needs = row
        .get("needs_synthesis")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    if needs != 1 {
        return Ok(Json(DeferredSynthesisResponse {
            status: "no_synthesis_needed".into(),
            session_id,
            is_fallback: false,
        }));
    }

    let conversation_id = row
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let accumulator_json = match row.get("accumulator_json").and_then(|v| v.as_str()) {
        Some(j) if !j.is_empty() => j.to_string(),
        _ => {
            console_error!(
                "[synthesis] accumulator_json is null for session {}",
                session_id
            );
            // Clear flag to avoid repeated attempts
            if let Err(e) = clear_needs_synthesis(env, &session_id_typed).await {
                console_error!("[synthesis] Failed to clear needs_synthesis: {}", e);
            }
            return Err(ApiError::BadRequest(
                "No accumulator data available for this session".into(),
            ));
        }
    };

    // Deserialize accumulator
    let acc: super::accumulator::SessionAccumulator = match serde_json::from_str(&accumulator_json)
    {
        Ok(a) => a,
        Err(e) => {
            console_error!(
                "[synthesis] Failed to deserialize accumulator for session {}: {:?}",
                session_id,
                e
            );
            // Clear flag -- schema mismatch, can't recover
            if let Err(ce) = clear_needs_synthesis(env, &session_id_typed).await {
                console_error!("[synthesis] Failed to clear needs_synthesis: {}", ce);
            }
            return Err(ApiError::Internal(
                "Accumulator deserialization failed".into(),
            ));
        }
    };

    // Wrap string IDs into newtypes for SynthesisContext
    let student_id = StudentId::from(student_id_str);
    let conversation_id_typed = ConversationId::from(conversation_id.clone());

    // Load baselines
    let baselines = match load_baselines_from_d1(env, &student_id).await {
        Ok(b) => Some(b),
        Err(e) => {
            console_error!("[synthesis] Failed to load baselines: {}", e);
            None
        }
    };

    let total_chunks = acc.timeline.len();
    let session_duration_ms = acc.timeline.last().map_or(0, |e| e.timestamp_ms + 15_000);

    let ctx = SynthesisContext {
        session_id: session_id_typed.clone(),
        student_id: student_id.clone(),
        conversation_id: conversation_id_typed.clone(),
        baselines,
        piece_context: None,
        student_memory: None,
        total_chunks,
        session_duration_ms,
    };

    let prompt_context = build_synthesis_prompt(&acc, &ctx);
    let result = call_synthesis_llm(env, &prompt_context).await;

    // Persist synthesis message
    persist_synthesis_message(env, &conversation_id_typed, &session_id_typed, &result.text)
        .await
        .map_err(|e| {
            console_error!("[synthesis] Failed to persist synthesis message: {}", e);
            ApiError::Internal("Failed to persist synthesis".into())
        })?;

    // Persist accumulated moments
    let moments: Vec<AccumulatedMoment> = acc.top_moments(None).into_iter().cloned().collect();
    if let Err(e) = persist_accumulated_moments(env, &student_id, &session_id_typed, &moments).await
    {
        console_error!("[synthesis] Failed to persist accumulated moments: {}", e);
        // Non-fatal -- synthesis message already saved
    }

    // Clear the deferred flag
    if let Err(e) = clear_needs_synthesis(env, &session_id_typed).await {
        console_error!("[synthesis] Failed to clear needs_synthesis flag: {}", e);
        // Non-fatal -- synthesis already persisted
    }

    Ok(Json(DeferredSynthesisResponse {
        status: "synthesized".into(),
        session_id,
        is_fallback: result.is_fallback,
    }))
}

#[cfg(test)]
mod tests {
    use super::accumulator::{DrillingRecord, ModeTransitionRecord, SessionAccumulator};
    use super::practice_mode::PracticeMode;
    use super::*;

    fn make_moment(chunk: usize, dim: &str, deviation: f64, positive: bool) -> AccumulatedMoment {
        AccumulatedMoment {
            chunk_index: chunk,
            dimension: dim.to_string(),
            score: 0.5 + deviation,
            baseline: 0.5,
            deviation,
            is_positive: positive,
            reasoning: format!("{} at chunk {}", dim, chunk),
            bar_range: None,
            analysis_tier: 1,
            timestamp_ms: (chunk as u64) * 15_000,
            llm_analysis: None,
        }
    }

    #[test]
    fn test_build_synthesis_prompt_minimal() {
        let acc = SessionAccumulator::default();
        let ctx = SynthesisContext {
            session_id: "sess-1".to_string(),
            student_id: "student-1".to_string(),
            conversation_id: "conv-1".to_string(),
            baselines: None,
            piece_context: None,
            student_memory: None,
            total_chunks: 0,
            session_duration_ms: 0,
        };

        let prompt = build_synthesis_prompt(&acc, &ctx);

        assert!(prompt.is_object(), "prompt must be a JSON object");
        assert!(
            prompt.get("session_duration_minutes").is_some(),
            "must have session_duration_minutes"
        );
        assert!(
            prompt.get("chunks_processed").is_some(),
            "must have chunks_processed"
        );
        assert!(
            prompt.get("practice_pattern").is_some(),
            "must have practice_pattern"
        );
        assert!(prompt.get("top_moments").is_some(), "must have top_moments");
        assert_eq!(
            prompt["baselines"],
            serde_json::Value::Null,
            "baselines should be null when not provided"
        );
        assert!(
            prompt.get("piece").is_none(),
            "piece should be absent when not provided"
        );
        assert!(
            prompt.get("student_memory").is_none(),
            "student_memory should be absent when not provided"
        );

        // top_moments should be an empty array
        let moments = prompt["top_moments"]
            .as_array()
            .expect("top_moments must be array");
        assert!(
            moments.is_empty(),
            "top_moments should be empty for empty accumulator"
        );

        // practice_pattern should be an empty array
        let pattern = prompt["practice_pattern"]
            .as_array()
            .expect("practice_pattern must be array");
        assert!(
            pattern.is_empty(),
            "practice_pattern should be empty for empty accumulator"
        );
    }

    #[test]
    fn test_build_synthesis_prompt_with_data() {
        let mut acc = SessionAccumulator::default();

        // Add a mode transition
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Running,
            chunk_index: 2,
            timestamp_ms: 30_000,
            dwell_ms: 30_000,
        });

        // Add a drilling record
        acc.accumulate_drilling_record(DrillingRecord {
            bar_range: Some((5, 8)),
            repetition_count: 4,
            first_scores: [0.4, 0.5, 0.3, 0.6, 0.5, 0.4],
            final_scores: [0.6, 0.7, 0.5, 0.7, 0.6, 0.6],
            started_at_chunk: 5,
            ended_at_chunk: 9,
        });

        // Add teaching moments
        acc.accumulate_moment(make_moment(0, "dynamics", 0.3, false));
        acc.accumulate_moment(make_moment(1, "timing", -0.25, false));
        acc.accumulate_moment(make_moment(3, "phrasing", 0.4, true));

        let baselines = StudentBaselines {
            dynamics: 0.55,
            timing: 0.48,
            pedaling: 0.46,
            articulation: 0.54,
            phrasing: 0.52,
            interpretation: 0.50,
        };

        let piece_context = serde_json::json!({
            "title": "Prelude in C",
            "composer": "Bach"
        });

        let ctx = SynthesisContext {
            session_id: "sess-2".to_string(),
            student_id: "student-2".to_string(),
            conversation_id: "conv-2".to_string(),
            baselines: Some(baselines),
            piece_context: Some(piece_context),
            student_memory: Some("Student has been working on dynamics control.".to_string()),
            total_chunks: 10,
            session_duration_ms: 150_000,
        };

        let prompt = build_synthesis_prompt(&acc, &ctx);

        // Duration should be 2.5 minutes
        let duration = prompt["session_duration_minutes"]
            .as_f64()
            .expect("duration must be f64");
        assert!(
            (duration - 2.5).abs() < 0.01,
            "duration should be 2.5 minutes, got {}",
            duration
        );

        // chunks_processed should be 10
        assert_eq!(prompt["chunks_processed"], 10);

        // Baselines should be present
        let baselines_val = &prompt["baselines"];
        assert!(baselines_val.is_object(), "baselines should be an object");
        assert!((baselines_val["dynamics"].as_f64().unwrap() - 0.55).abs() < 0.001);

        // Piece should be present
        assert_eq!(prompt["piece"]["title"], "Prelude in C");
        assert_eq!(prompt["piece"]["composer"], "Bach");

        // Student memory should be present
        assert_eq!(
            prompt["student_memory"].as_str().unwrap(),
            "Student has been working on dynamics control."
        );

        // top_moments should be non-empty
        let moments = prompt["top_moments"]
            .as_array()
            .expect("top_moments must be array");
        assert!(!moments.is_empty(), "top_moments should not be empty");

        // Each moment must have required fields
        for m in moments {
            assert!(m.get("dimension").is_some());
            assert!(m.get("deviation").is_some());
            assert!(m.get("is_positive").is_some());
            assert!(m.get("reasoning").is_some());
        }

        // drilling_progress should be present
        let drilling = prompt
            .get("drilling_progress")
            .expect("drilling_progress should be present");
        let drilling_arr = drilling
            .as_array()
            .expect("drilling_progress must be array");
        assert!(
            !drilling_arr.is_empty(),
            "drilling_progress should not be empty"
        );
        assert_eq!(drilling_arr[0]["repetitions"], 4);

        // practice_pattern should have the transition we added
        let pattern = prompt["practice_pattern"]
            .as_array()
            .expect("practice_pattern must be array");
        assert!(!pattern.is_empty(), "practice_pattern should not be empty");
        assert_eq!(pattern[0]["mode"].as_str().unwrap(), "running");
    }
}
