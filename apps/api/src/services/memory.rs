//! Student memory system: synthesized facts, teaching approaches, retrieval, and synthesis.
//!
//! See docs/plans/2026-03-06-memory-system-design.md for the full design.

use wasm_bindgen::JsValue;
use worker::{console_log, Env};

/// A synthesized fact from the event clock.
pub struct SynthesizedFact {
    pub id: String,
    pub fact_text: String,
    pub fact_type: String,
    pub dimension: Option<String>,
    pub piece_context: Option<String>,
    pub valid_at: String,
    pub trend: Option<String>,
    pub confidence: String,
    pub source_type: String,
}

/// A recent observation with engagement data.
pub struct RecentObservationWithEngagement {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub created_at: String,
    pub engaged: bool,
}

/// The assembled context map for the subagent.
pub struct StudentMemoryContext {
    pub active_facts: Vec<SynthesizedFact>,
    pub recent_observations: Vec<RecentObservationWithEngagement>,
    pub piece_facts: Vec<SynthesizedFact>,
}

/// Query active synthesized facts for a student.
pub async fn query_active_facts(
    env: &Env,
    student_id: &str,
) -> Result<Vec<SynthesizedFact>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT id, fact_text, fact_type, dimension, piece_context, valid_at, \
             trend, confidence, source_type \
             FROM synthesized_facts \
             WHERE student_id = ?1 AND invalid_at IS NULL AND expired_at IS NULL \
             ORDER BY fact_type, valid_at DESC \
             LIMIT 15",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query facts: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| SynthesizedFact {
            id: row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_text: row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_type: row.get("fact_type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            dimension: row.get("dimension").and_then(|v| v.as_str()).map(|s| s.to_string()),
            piece_context: row.get("piece_context").and_then(|v| v.as_str()).map(|s| s.to_string()),
            valid_at: row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            trend: row.get("trend").and_then(|v| v.as_str()).map(|s| s.to_string()),
            confidence: row.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium").to_string(),
            source_type: row.get("source_type").and_then(|v| v.as_str()).unwrap_or("synthesized").to_string(),
        })
        .collect())
}

/// Query recent observations with engagement data from teaching_approaches.
pub async fn query_recent_observations_with_engagement(
    env: &Env,
    student_id: &str,
) -> Result<Vec<RecentObservationWithEngagement>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT o.dimension, o.observation_text, o.framing, o.created_at, \
             COALESCE(ta.engaged, 0) AS engaged \
             FROM observations o \
             LEFT JOIN teaching_approaches ta ON ta.observation_id = o.id \
             WHERE o.student_id = ?1 \
             ORDER BY o.created_at DESC \
             LIMIT 5",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query observations: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| RecentObservationWithEngagement {
            dimension: row.get("dimension").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            observation_text: row.get("observation_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            framing: row.get("framing").and_then(|v| v.as_str()).unwrap_or("correction").to_string(),
            created_at: row.get("created_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            engaged: row.get("engaged").and_then(|v| v.as_i64()).unwrap_or(0) == 1,
        })
        .collect())
}

/// Query piece-specific facts for a student and piece title.
pub async fn query_piece_facts(
    env: &Env,
    student_id: &str,
    piece_title: &str,
) -> Result<Vec<SynthesizedFact>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT id, fact_text, fact_type, dimension, piece_context, valid_at, \
             trend, confidence, source_type \
             FROM synthesized_facts \
             WHERE student_id = ?1 \
             AND piece_context IS NOT NULL \
             AND json_extract(piece_context, '$.title') = ?2 \
             AND invalid_at IS NULL AND expired_at IS NULL",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(piece_title),
        ])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query piece facts: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| SynthesizedFact {
            id: row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_text: row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_type: row.get("fact_type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            dimension: row.get("dimension").and_then(|v| v.as_str()).map(|s| s.to_string()),
            piece_context: row.get("piece_context").and_then(|v| v.as_str()).map(|s| s.to_string()),
            valid_at: row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            trend: row.get("trend").and_then(|v| v.as_str()).map(|s| s.to_string()),
            confidence: row.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium").to_string(),
            source_type: row.get("source_type").and_then(|v| v.as_str()).unwrap_or("synthesized").to_string(),
        })
        .collect())
}

/// Build the full memory context for the subagent.
/// Runs queries 1, 2, and optionally 4 (piece-specific).
pub async fn build_memory_context(
    env: &Env,
    student_id: &str,
    piece_title: Option<&str>,
) -> StudentMemoryContext {
    let active_facts = match query_active_facts(env, student_id).await {
        Ok(facts) => facts,
        Err(e) => {
            console_log!("Failed to query active facts: {}", e);
            vec![]
        }
    };

    let recent_observations = match query_recent_observations_with_engagement(env, student_id).await {
        Ok(obs) => obs,
        Err(e) => {
            console_log!("Failed to query recent observations: {}", e);
            vec![]
        }
    };

    let piece_facts = if let Some(title) = piece_title {
        match query_piece_facts(env, student_id, title).await {
            Ok(facts) => {
                // Deduplicate against active_facts
                let active_ids: std::collections::HashSet<&str> =
                    active_facts.iter().map(|f| f.id.as_str()).collect();
                facts
                    .into_iter()
                    .filter(|f| !active_ids.contains(f.id.as_str()))
                    .collect()
            }
            Err(e) => {
                console_log!("Failed to query piece facts: {}", e);
                vec![]
            }
        }
    } else {
        vec![]
    };

    StudentMemoryContext {
        active_facts,
        recent_observations,
        piece_facts,
    }
}

/// Format the memory context as plain text for the subagent prompt.
pub fn format_memory_context(ctx: &StudentMemoryContext) -> String {
    if ctx.active_facts.is_empty() && ctx.recent_observations.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(1500);
    out.push_str("## Student Memory\n\n");

    if !ctx.active_facts.is_empty() {
        out.push_str("### Active Patterns\n");
        for fact in &ctx.active_facts {
            let dim_label = fact
                .dimension
                .as_deref()
                .map(|d| format!("{}/", d))
                .unwrap_or_default();
            let trend_label = fact
                .trend
                .as_deref()
                .map(|t| format!(", {}", t))
                .unwrap_or_default();
            out.push_str(&format!(
                "- [{}{}{}, {} confidence] {} (since {})\n",
                fact.fact_type, dim_label, trend_label, fact.confidence,
                fact.fact_text, fact.valid_at,
            ));
        }
        out.push('\n');
    }

    if !ctx.recent_observations.is_empty() {
        out.push_str("### Recent Feedback\n");
        for obs in &ctx.recent_observations {
            let engaged_label = if obs.engaged { ", student asked for elaboration" } else { "" };
            out.push_str(&format!(
                "- [{}] {}: \"{}\" (framing: {}{})\n",
                obs.created_at, obs.dimension, obs.observation_text, obs.framing, engaged_label,
            ));
        }
        out.push('\n');
    }

    if !ctx.piece_facts.is_empty() {
        out.push_str("### Current Piece History\n");
        for fact in &ctx.piece_facts {
            out.push_str(&format!("- {} (since {})\n", fact.fact_text, fact.valid_at));
        }
        out.push('\n');
    }

    out
}

/// Store a teaching approach record after an /api/ask call.
pub async fn store_teaching_approach(
    env: &Env,
    id: &str,
    student_id: &str,
    observation_id: &str,
    dimension: &str,
    framing: &str,
    approach_summary: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "INSERT INTO teaching_approaches \
         (id, student_id, observation_id, dimension, framing, approach_summary, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )
    .bind(&[
        JsValue::from_str(id),
        JsValue::from_str(student_id),
        JsValue::from_str(observation_id),
        JsValue::from_str(dimension),
        JsValue::from_str(framing),
        JsValue::from_str(approach_summary),
        JsValue::from_str(&now),
    ])
    .map_err(|e| format!("Failed to bind insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert teaching approach: {:?}", e))?;

    Ok(())
}

/// Mark a teaching approach as engaged (student tapped "tell me more").
pub async fn mark_approach_engaged(
    env: &Env,
    observation_id: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare("UPDATE teaching_approaches SET engaged = 1 WHERE observation_id = ?1")
        .bind(&[JsValue::from_str(observation_id)])
        .map_err(|e| format!("Failed to bind update: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to update engagement: {:?}", e))?;

    Ok(())
}

/// Increment total_observations in student_memory_meta.
pub async fn increment_observation_count(
    env: &Env,
    student_id: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, total_observations) VALUES (?1, 1) \
         ON CONFLICT(student_id) DO UPDATE SET total_observations = total_observations + 1",
    )
    .bind(&[JsValue::from_str(student_id)])
    .map_err(|e| format!("Failed to bind upsert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update observation count: {:?}", e))?;

    Ok(())
}

/// Public wrapper for UUID generation (used by goals.rs).
pub fn generate_fact_id() -> String {
    generate_uuid()
}

/// Generate a UUID v4.
fn generate_uuid() -> String {
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
