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
    pub student_facts: Vec<SynthesizedFact>,
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
             AND source_type != 'student_reported' \
             ORDER BY fact_type, valid_at DESC \
             LIMIT 12",
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

/// Query student-reported facts (chat-derived personal info).
pub async fn query_student_reported_facts(
    env: &Env,
    student_id: &str,
    today: &str,
) -> Result<Vec<SynthesizedFact>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT id, fact_text, fact_type, dimension, piece_context, valid_at, \
             trend, confidence, source_type \
             FROM synthesized_facts \
             WHERE student_id = ?1 \
             AND source_type = 'student_reported' \
             AND (invalid_at IS NULL OR invalid_at > ?2) \
             AND expired_at IS NULL \
             ORDER BY created_at DESC \
             LIMIT 10",
        )
        .bind(&[JsValue::from_str(student_id), JsValue::from_str(today)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query student reported facts: {:?}", e))?;

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
            source_type: row.get("source_type").and_then(|v| v.as_str()).unwrap_or("student_reported").to_string(),
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
///
/// When `query` is provided, uses hybrid search to find the most relevant
/// student-reported facts instead of the generic LIMIT 10 query.
pub async fn build_memory_context(
    env: &Env,
    student_id: &str,
    piece_title: Option<&str>,
    today: &str,
    query: Option<&str>,
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

    let student_facts = if let Some(q) = query {
        // Use hybrid search to find relevant student-reported facts
        match search_relevant_facts(env, student_id, q, 15, true).await {
            Ok(result) => result.facts.into_iter().map(|f| SynthesizedFact {
                id: f.id,
                fact_text: f.fact_text,
                fact_type: "student_reported".to_string(),
                dimension: Some(f.category),
                piece_context: None,
                valid_at: f.date,
                trend: None,
                confidence: "high".to_string(),
                source_type: "student_reported".to_string(),
            }).collect(),
            Err(e) => {
                console_log!("Search failed, falling back to generic query: {}", e);
                query_student_reported_facts(env, student_id, today).await.unwrap_or_default()
            }
        }
    } else {
        match query_student_reported_facts(env, student_id, today).await {
            Ok(facts) => facts,
            Err(e) => {
                console_log!("Failed to query student reported facts: {}", e);
                vec![]
            }
        }
    };

    StudentMemoryContext {
        active_facts,
        recent_observations,
        piece_facts,
        student_facts,
    }
}

/// Format the memory context as plain text for the subagent prompt.
pub fn format_memory_context(ctx: &StudentMemoryContext) -> String {
    if ctx.active_facts.is_empty() && ctx.recent_observations.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(1500);

    if !ctx.active_facts.is_empty() {
        out.push_str("Active patterns:\n");
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
        out.push_str("Recent feedback:\n");
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
        out.push_str("Current piece history:\n");
        for fact in &ctx.piece_facts {
            out.push_str(&format!("- {} (since {})\n", fact.fact_text, fact.valid_at));
        }
        out.push('\n');
    }

    out
}

/// Format memory patterns for chat context (concise, no metadata).
pub fn format_chat_memory_patterns(ctx: &StudentMemoryContext) -> String {
    if ctx.active_facts.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for fact in &ctx.active_facts {
        let trend_label = fact.trend.as_deref().unwrap_or("stable");
        out.push_str(&format!("- {} ({})\n", fact.fact_text, trend_label));
    }
    out
}

/// Format recent observations for chat context.
pub fn format_chat_recent_observations(ctx: &StudentMemoryContext) -> String {
    if ctx.recent_observations.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for obs in &ctx.recent_observations {
        out.push_str(&format!(
            "- [{}] {}: \"{}\"\n",
            obs.created_at, obs.dimension, obs.observation_text,
        ));
    }
    out
}

/// Format student-reported facts for chat context (simple natural-language bullets).
pub fn format_student_reported_context(ctx: &StudentMemoryContext) -> String {
    if ctx.student_facts.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for fact in &ctx.student_facts {
        let category = fact.dimension.as_deref().unwrap_or("general");
        out.push_str(&format!("- [{}] {}\n", category, fact.fact_text));
    }
    out
}

/// Extract and store personal facts from a chat exchange (fire-and-forget).
pub async fn extract_and_store_chat_facts(
    env: &Env,
    student_id: &str,
    user_message: &str,
    assistant_response: &str,
) -> Result<(), String> {
    use crate::services::{llm, prompts};

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];

    // 1. Query existing student_reported facts
    let existing_facts = query_student_reported_facts(env, student_id, today).await?;

    // 2. Build extraction prompt
    let user_prompt = prompts::build_chat_extraction_prompt(
        user_message,
        assistant_response,
        &existing_facts,
        today,
    );

    // 3. Call Workers AI (cheap, background task)
    let output = llm::call_workers_ai(
        env,
        prompts::CHAT_EXTRACTION_SYSTEM,
        &user_prompt,
        0.0,
        500,
    )
    .await?;

    // 4. Parse JSON
    let extraction_json = extract_synthesis_json(&output)?;

    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    // 5. Process ADD operations
    if let Some(adds) = extraction_json.get("add").and_then(|v| v.as_array()) {
        for fact in adds {
            let fact_text = fact.get("fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let category = fact.get("category").and_then(|v| v.as_str()).unwrap_or("general");
            let invalid_at = fact.get("invalid_at").and_then(|v| v.as_str());

            if fact_text.is_empty() {
                continue;
            }

            let fact_id = generate_fact_id();
            let _ = db
                .prepare(
                    "INSERT OR IGNORE INTO synthesized_facts \
                     (id, student_id, fact_text, fact_type, dimension, valid_at, invalid_at, \
                      confidence, evidence, source_type, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                )
                .bind(&[
                    JsValue::from_str(&fact_id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(fact_text),
                    JsValue::from_str("student_reported"),
                    JsValue::from_str(category),
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
                .map_err(|e| format!("Failed to bind fact insert: {:?}", e))?
                .run()
                .await;
        }
    }

    // 6. Process UPDATE operations (invalidate old, insert new)
    if let Some(updates) = extraction_json.get("update").and_then(|v| v.as_array()) {
        for update in updates {
            let existing_id = update.get("existing_fact_id").and_then(|v| v.as_str()).unwrap_or("");
            let new_text = update.get("new_fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let category = update.get("category").and_then(|v| v.as_str()).unwrap_or("general");
            let invalid_at = update.get("invalid_at").and_then(|v| v.as_str());

            if existing_id.is_empty() || new_text.is_empty() {
                continue;
            }

            // Invalidate old fact
            let _ = db
                .prepare(
                    "UPDATE synthesized_facts SET invalid_at = ?1, expired_at = ?2 \
                     WHERE id = ?3 AND student_id = ?4",
                )
                .bind(&[
                    JsValue::from_str(today),
                    JsValue::from_str(&now),
                    JsValue::from_str(existing_id),
                    JsValue::from_str(student_id),
                ])
                .map_err(|e| format!("Failed to bind invalidation: {:?}", e))?
                .run()
                .await;

            // Insert replacement fact
            let fact_id = generate_fact_id();
            let _ = db
                .prepare(
                    "INSERT OR IGNORE INTO synthesized_facts \
                     (id, student_id, fact_text, fact_type, dimension, valid_at, invalid_at, \
                      confidence, evidence, source_type, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                )
                .bind(&[
                    JsValue::from_str(&fact_id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(new_text),
                    JsValue::from_str("student_reported"),
                    JsValue::from_str(category),
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
                .map_err(|e| format!("Failed to bind replacement insert: {:?}", e))?
                .run()
                .await;
        }
    }

    let add_count = extraction_json.get("add").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    let update_count = extraction_json.get("update").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    if add_count > 0 || update_count > 0 {
        console_log!(
            "Chat memory extraction for student {}: {} added, {} updated",
            student_id, add_count, update_count
        );
    }

    Ok(())
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

/// Increment observation count by N (batched version for DO session finalization).
pub async fn increment_observation_count_by(
    env: &Env,
    student_id: &str,
    count: usize,
) -> Result<(), String> {
    if count == 0 {
        return Ok(());
    }
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, total_observations) VALUES (?1, ?2) \
         ON CONFLICT(student_id) DO UPDATE SET total_observations = total_observations + ?2",
    )
    .bind(&[
        JsValue::from_str(student_id),
        JsValue::from_f64(count as f64),
    ])
    .map_err(|e| format!("Failed to bind upsert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update observation count: {:?}", e))?;

    Ok(())
}

/// Check if synthesis should run for this student.
/// Returns true if >= 3 new observations since last synthesis,
/// or any new observations and last synthesis was > 7 days ago.
pub async fn should_synthesize(
    env: &Env,
    student_id: &str,
) -> Result<bool, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let meta: Option<serde_json::Value> = db
        .prepare("SELECT last_synthesis_at, total_observations FROM student_memory_meta WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query meta: {:?}", e))?;

    let last_synthesis = meta
        .as_ref()
        .and_then(|m| m.get("last_synthesis_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if last_synthesis.is_empty() {
        // Never synthesized -- check if there are at least 3 observations
        let total = meta
            .as_ref()
            .and_then(|m| m.get("total_observations"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        return Ok(total >= 3);
    }

    // Count observations since last synthesis
    let count_result: Option<serde_json::Value> = db
        .prepare(
            "SELECT COUNT(*) as cnt FROM observations \
             WHERE student_id = ?1 AND created_at > ?2",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind count query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to count observations: {:?}", e))?;

    let new_count = count_result
        .as_ref()
        .and_then(|r| r.get("cnt"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    if new_count >= 3 {
        return Ok(true);
    }

    // Check if last synthesis was > 7 days ago and there are any new observations
    if new_count > 0 {
        let last_ms = js_sys::Date::parse(last_synthesis);
        let now_ms = js_sys::Date::now();
        let days_diff = (now_ms - last_ms) / (1000.0 * 60.0 * 60.0 * 24.0);
        if days_diff >= 7.0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Result of a synthesis run, for observability.
pub struct SynthesisResult {
    pub new_facts: usize,
    pub invalidated: usize,
    pub unchanged: usize,
    pub observations_processed: usize,
}

/// Run the synthesis pipeline for a student.
pub async fn run_synthesis(
    env: &Env,
    student_id: &str,
) -> Result<SynthesisResult, String> {
    use crate::services::llm;
    use crate::services::prompts;

    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    // 1. Get current active facts
    let active_facts = query_active_facts(env, student_id).await?;

    // 2. Get last synthesis timestamp
    let meta: Option<serde_json::Value> = db
        .prepare("SELECT last_synthesis_at FROM student_memory_meta WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind meta query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query meta: {:?}", e))?;

    let last_synthesis = meta
        .as_ref()
        .and_then(|m| m.get("last_synthesis_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("1970-01-01T00:00:00.000Z");

    // 3. Get new observations since last synthesis
    let obs_results = db
        .prepare(
            "SELECT id, dimension, observation_text, framing, dimension_score, \
             student_baseline, reasoning_trace, piece_context, created_at \
             FROM observations \
             WHERE student_id = ?1 AND created_at > ?2 \
             ORDER BY created_at ASC",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind obs query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query new observations: {:?}", e))?;

    let new_observations: Vec<serde_json::Value> = obs_results
        .results()
        .map_err(|e| format!("Failed to get obs results: {:?}", e))?;

    if new_observations.is_empty() {
        return Ok(SynthesisResult {
            new_facts: 0,
            invalidated: 0,
            unchanged: active_facts.len(),
            observations_processed: 0,
        });
    }

    // 4. Get teaching approaches since last synthesis
    let ta_results = db
        .prepare(
            "SELECT dimension, framing, approach_summary, engaged \
             FROM teaching_approaches \
             WHERE student_id = ?1 AND created_at > ?2",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind ta query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query teaching approaches: {:?}", e))?;

    let teaching_approaches: Vec<serde_json::Value> = ta_results
        .results()
        .map_err(|e| format!("Failed to get ta results: {:?}", e))?;

    // 5. Get student baselines
    let baselines: serde_json::Value = db
        .prepare(
            "SELECT baseline_dynamics, baseline_timing, baseline_pedaling, \
             baseline_articulation, baseline_phrasing, baseline_interpretation \
             FROM students WHERE student_id = ?1",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind baselines query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query baselines: {:?}", e))?
        .unwrap_or(serde_json::json!({}));

    // 6. Build synthesis prompt and call Groq
    let user_prompt = prompts::build_synthesis_prompt(
        &active_facts,
        &new_observations,
        &teaching_approaches,
        &baselines,
    );

    let synthesis_output = llm::call_groq(
        env,
        prompts::SYNTHESIS_SYSTEM,
        &user_prompt,
        0.1,
        800,
    )
    .await?;

    // 7. Parse synthesis output
    let synthesis_json = extract_synthesis_json(&synthesis_output)?;

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];
    let active_facts_count = active_facts.len();
    let observations_count = new_observations.len();

    let mut invalidated_count = 0usize;
    // 8. Apply invalidations
    if let Some(invalidated) = synthesis_json.get("invalidated_facts").and_then(|v| v.as_array()) {
        for inv in invalidated {
            let fact_id = inv.get("fact_id").and_then(|v| v.as_str()).unwrap_or("");
            let invalid_at = inv.get("invalid_at").and_then(|v| v.as_str()).unwrap_or(today);
            if !fact_id.is_empty() {
                let _ = db
                    .prepare(
                        "UPDATE synthesized_facts SET invalid_at = ?1, expired_at = ?2 \
                         WHERE id = ?3 AND student_id = ?4",
                    )
                    .bind(&[
                        JsValue::from_str(invalid_at),
                        JsValue::from_str(&now),
                        JsValue::from_str(fact_id),
                        JsValue::from_str(student_id),
                    ])
                    .map_err(|e| format!("Failed to bind invalidation: {:?}", e))?
                    .run()
                    .await;
                invalidated_count += 1;
            }
        }
    }

    let mut new_facts_count = 0usize;
    // 9. Insert new facts
    if let Some(new_facts) = synthesis_json.get("new_facts").and_then(|v| v.as_array()) {
        for fact in new_facts {
            let fact_id = generate_uuid();
            let fact_text = fact.get("fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let fact_type = fact.get("fact_type").and_then(|v| v.as_str()).unwrap_or("dimension");
            let dimension = fact.get("dimension").and_then(|v| v.as_str());
            let piece_ctx = fact.get("piece_context").map(|v| v.to_string());
            let trend = fact.get("trend").and_then(|v| v.as_str());
            let confidence = fact.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium");
            let evidence = fact.get("evidence").map(|v| v.to_string()).unwrap_or_else(|| "[]".to_string());
            let valid_at = today;

            if fact_text.is_empty() {
                continue;
            }

            let _ = db
                .prepare(
                    "INSERT INTO synthesized_facts \
                     (id, student_id, fact_text, fact_type, dimension, piece_context, \
                      valid_at, trend, confidence, evidence, source_type, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .bind(&[
                    JsValue::from_str(&fact_id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(fact_text),
                    JsValue::from_str(fact_type),
                    match dimension {
                        Some(d) => JsValue::from_str(d),
                        None => JsValue::NULL,
                    },
                    match piece_ctx.as_deref() {
                        Some(pc) if pc != "null" => JsValue::from_str(pc),
                        _ => JsValue::NULL,
                    },
                    JsValue::from_str(valid_at),
                    match trend {
                        Some(t) => JsValue::from_str(t),
                        None => JsValue::NULL,
                    },
                    JsValue::from_str(confidence),
                    JsValue::from_str(&evidence),
                    JsValue::from_str("synthesized"),
                    JsValue::from_str(&now),
                ])
                .map_err(|e| format!("Failed to bind fact insert: {:?}", e))?
                .run()
                .await;
            new_facts_count += 1;
        }
    }

    // 10. Update meta
    let fact_count_result: Option<serde_json::Value> = db
        .prepare(
            "SELECT COUNT(*) as cnt FROM synthesized_facts \
             WHERE student_id = ?1 AND invalid_at IS NULL AND expired_at IS NULL",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind fact count: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to count facts: {:?}", e))?;

    let total_facts = fact_count_result
        .as_ref()
        .and_then(|r| r.get("cnt"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, last_synthesis_at, total_facts) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(student_id) DO UPDATE SET last_synthesis_at = ?2, total_facts = ?3",
    )
    .bind(&[
        JsValue::from_str(student_id),
        JsValue::from_str(&now),
        JsValue::from_f64(total_facts as f64),
    ])
    .map_err(|e| format!("Failed to bind meta update: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update meta: {:?}", e))?;

    console_log!(
        "Synthesis complete for student {}: {} new, {} invalidated, {} observations",
        student_id, new_facts_count, invalidated_count, observations_count
    );

    Ok(SynthesisResult {
        new_facts: new_facts_count,
        invalidated: invalidated_count,
        unchanged: active_facts_count.saturating_sub(invalidated_count),
        observations_processed: observations_count,
    })
}

/// POST /api/memory/extract-chat -- eval endpoint for chat memory extraction.
/// Accepts existing facts as input, calls Groq with production prompt, returns extraction JSON.
/// Does NOT write to D1 -- the caller manages state.
pub async fn handle_extract_chat(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: ExtractChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse extract-chat request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Convert ExistingFactInput items to SynthesizedFact structs
    let synthesized_facts: Vec<SynthesizedFact> = request
        .existing_facts
        .iter()
        .map(|f| SynthesizedFact {
            id: f.id.clone(),
            fact_text: f.fact_text.clone(),
            fact_type: "student_reported".to_string(),
            dimension: Some(f.category.clone()),
            piece_context: None,
            valid_at: String::new(),
            trend: None,
            confidence: "high".to_string(),
            source_type: "student_reported".to_string(),
        })
        .collect();

    // Build extraction prompt
    let user_prompt = crate::services::prompts::build_chat_extraction_prompt(
        &request.user_message,
        &request.assistant_response,
        &synthesized_facts,
        &request.today,
    );

    // Call Workers AI (cheap, background task)
    let output = match crate::services::llm::call_workers_ai(
        env,
        crate::services::prompts::CHAT_EXTRACTION_SYSTEM,
        &user_prompt,
        0.0,
        500,
    )
    .await
    {
        Ok(output) => output,
        Err(e) => {
            console_log!("Workers AI call failed for extract-chat: {}", e);
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("LLM call failed: {}", e)}).to_string(),
                ))
                .unwrap();
        }
    };

    // Parse JSON from LLM output
    let extraction_json = match extract_synthesis_json(&output) {
        Ok(json) => json,
        Err(e) => {
            console_log!("Failed to parse extraction JSON: {}", e);
            return Response::builder()
                .status(StatusCode::UNPROCESSABLE_ENTITY)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("Failed to parse LLM output: {}", e)}).to_string(),
                ))
                .unwrap();
        }
    };

    // Return parsed JSON
    let json = serde_json::to_string(&extraction_json)
        .unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

#[derive(serde::Deserialize)]
pub struct ExtractChatRequest {
    pub user_message: String,
    pub assistant_response: String,
    pub existing_facts: Vec<ExistingFactInput>,
    pub today: String,
}

#[derive(serde::Deserialize)]
pub struct ExistingFactInput {
    pub id: String,
    pub fact_text: String,
    pub category: String,
}

// ---------------------------------------------------------------------------
// POST /api/memory/store-facts -- store benchmark facts in D1
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct StoreFactsRequest {
    pub student_id: String,
    pub facts: Vec<StoredFactInput>,
}

#[derive(serde::Deserialize)]
pub struct StoredFactInput {
    pub fact_text: String,
    pub category: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub date: String,
}

/// POST /api/memory/store-facts -- persist eval/benchmark facts into D1.
pub async fn handle_store_facts(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: StoreFactsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse store-facts request: {:?}", e);
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
                .body(Body::from(r#"{"error":"Database unavailable"}"#))
                .unwrap();
        }
    };

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let mut stored = 0u32;
    for fact in &request.facts {
        if fact.fact_text.is_empty() {
            continue;
        }
        let fact_id = generate_uuid();
        let valid_at = if fact.date.is_empty() { &now[..10.min(now.len())] } else { &fact.date };
        let entities_json = serde_json::to_string(&fact.entities).unwrap_or_else(|_| "[]".to_string());

        let result = db
            .prepare(
                "INSERT OR IGNORE INTO synthesized_facts \
                 (id, student_id, fact_text, fact_type, dimension, valid_at, \
                  confidence, evidence, source_type, entities, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )
            .bind(&[
                JsValue::from_str(&fact_id),
                JsValue::from_str(&request.student_id),
                JsValue::from_str(&fact.fact_text),
                JsValue::from_str("student_reported"),
                JsValue::from_str(&fact.category),
                JsValue::from_str(valid_at),
                JsValue::from_str("high"),
                JsValue::from_str("[]"),
                JsValue::from_str("benchmark"),
                JsValue::from_str(&entities_json),
                JsValue::from_str(&now),
            ]);

        match result {
            Ok(stmt) => {
                if let Err(e) = stmt.run().await {
                    console_log!("Failed to insert fact: {:?}", e);
                } else {
                    stored += 1;
                }
            }
            Err(e) => {
                console_log!("Failed to bind fact insert: {:?}", e);
            }
        }
    }

    let resp = serde_json::json!({ "stored": stored });
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(resp.to_string()))
        .unwrap()
}

// ---------------------------------------------------------------------------
// POST /api/memory/search -- hybrid retrieval (entity + keyword + dimension + recency)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct SearchFactsRequest {
    pub query: String,
    pub student_id: String,
    #[serde(default = "default_max_facts")]
    pub max_facts: usize,
}

fn default_max_facts() -> usize { 50 }

#[derive(serde::Serialize)]
pub struct SearchFactResult {
    pub id: String,
    pub fact_text: String,
    pub category: String,
    pub date: String,
    pub entities: Vec<String>,
    pub score: f64,
}

/// Aggregated search results from hybrid retrieval.
pub struct SearchResult {
    pub facts: Vec<SearchFactResult>,
    pub total_facts: usize,
    pub query_entities: Vec<String>,
    pub is_temporal: bool,
    pub is_adversarial: bool,
}

/// Detect if query is temporal (asks about when, time, duration).
fn is_temporal_query(query_lower: &str) -> bool {
    let temporal_keywords = [
        "when", "how long", "how many years", "how many months",
        "how many days", "how many weeks", "since when", "last time",
        "first time", "recently", "ago", "before", "after",
        "date", "year", "month", "started", "began", "ended",
        "how old", "birthday", "anniversary",
    ];
    temporal_keywords.iter().any(|kw| query_lower.contains(kw))
}

/// Detect if query is adversarial/unanswerable (asks about non-existent info).
fn is_likely_adversarial(query_lower: &str) -> bool {
    let adversarial_patterns = [
        "never mentioned", "not mentioned", "didn't say", "didn't mention",
        "did not say", "did not mention", "never said", "never told",
    ];
    adversarial_patterns.iter().any(|p| query_lower.contains(p))
}


const ENTITY_EXTRACTION_PROMPT: &str = "\
Extract ALL named entities and key topics from this query. Include:
- People's names (first names, full names, nicknames)
- Places, organizations, institutions
- Activities, hobbies, sports, subjects
- Key topics being asked about
- For possessive chains like \"X's Y's Z\", extract EACH entity separately

Return ONLY a JSON array of strings. No explanation.

Examples:
Query: \"What are Caroline's hobbies?\"
[\"Caroline\", \"hobbies\"]

Query: \"Where did John and Mary go on vacation?\"
[\"John\", \"Mary\", \"vacation\"]

Query: \"What does the user think about climate change?\"
[\"climate change\"]

Query: \"Who is Sarah's brother and where does he work?\"
[\"Sarah\", \"brother\", \"work\"]

Query: \"When did they move to Portland?\"
[\"Portland\"]

Query: \"What happened at the birthday party?\"
[\"birthday party\"]

Query: \"Has Caroline's opinion about yoga changed over time?\"
[\"Caroline\", \"yoga\", \"opinion\"]

Query: ";

/// Reusable hybrid search: fetch facts, score, and rank.
///
/// Used by both the `/api/memory/search` endpoint and `build_memory_context`.
/// When `active_only` is true, only non-invalidated/non-expired facts are searched.
pub async fn search_relevant_facts(
    env: &Env,
    student_id: &str,
    query: &str,
    max_facts: usize,
    active_only: bool,
) -> Result<SearchResult, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let sql = if active_only {
        "SELECT id, fact_text, fact_type, dimension, valid_at, invalid_at, \
         source_type, entities \
         FROM synthesized_facts \
         WHERE student_id = ?1 AND invalid_at IS NULL AND expired_at IS NULL \
         ORDER BY valid_at DESC"
    } else {
        "SELECT id, fact_text, fact_type, dimension, valid_at, invalid_at, \
         source_type, entities \
         FROM synthesized_facts \
         WHERE student_id = ?1 \
         ORDER BY valid_at DESC"
    };

    let all_facts: Vec<serde_json::Value> = db
        .prepare(sql)
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("bind failed: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("query failed: {:?}", e))?
        .results()
        .unwrap_or_default();

    let total_facts = all_facts.len();

    // Extract entities from query via Groq
    let query_entities = extract_query_entities(env, query).await;

    // Analyze query characteristics
    let query_lower = query.to_lowercase();
    let query_keywords = extract_keywords(&query_lower);
    let query_dimension = detect_dimension(&query_lower);
    let temporal = is_temporal_query(&query_lower);
    let adversarial = is_likely_adversarial(&query_lower);

    // Score each fact with hybrid signals
    let mut scored: Vec<(f64, &serde_json::Value)> = all_facts
        .iter()
        .map(|row| {
            let fact_text = row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let dimension = row.get("dimension").and_then(|v| v.as_str());
            let invalid_at = row.get("invalid_at").and_then(|v| v.as_str());
            let valid_at = row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("");
            let entities_str = row.get("entities").and_then(|v| v.as_str()).unwrap_or("[]");

            let fact_entities: Vec<String> = serde_json::from_str(entities_str).unwrap_or_default();
            let fact_lower = fact_text.to_lowercase();

            // 1. Entity match score (2.0 per matching entity)
            let entity_score = if !query_entities.is_empty() {
                let matches = query_entities.iter().filter(|qe| {
                    let qe_lower = qe.to_lowercase();
                    fact_entities.iter().any(|fe| {
                        let fe_lower = fe.to_lowercase();
                        fe_lower == qe_lower
                            || fe_lower.contains(&qe_lower)
                            || qe_lower.contains(&fe_lower)
                    }) || fact_lower.contains(&qe_lower)
                }).count();
                (matches as f64) * 2.0
            } else {
                0.0
            };

            // 2. Keyword score (1.0 per hit, not normalized by length)
            let keyword_hits = keyword_hit_count(&query_keywords, fact_text);
            let keyword_score = keyword_hits as f64;

            // 3. Multi-keyword bonus (1.5 if 2+ keywords match)
            let multi_kw_bonus = if keyword_hits >= 2 { 1.5 } else { 0.0 };

            // 4. Dimension match score (2.0 weight)
            let dimension_score = match (&query_dimension, dimension) {
                (Some(qd), Some(fd)) if qd == fd => 2.0,
                _ => 0.0,
            };

            // 5. Active fact bonus
            let active_bonus = if invalid_at.is_none() { 0.5 } else { 0.0 };

            // 6. Temporal boost
            let temporal_score = if temporal && !valid_at.is_empty() {
                let has_date_in_text = fact_lower.contains("202")
                    || fact_lower.contains("january") || fact_lower.contains("february")
                    || fact_lower.contains("march") || fact_lower.contains("april")
                    || fact_lower.contains("may") || fact_lower.contains("june")
                    || fact_lower.contains("july") || fact_lower.contains("august")
                    || fact_lower.contains("september") || fact_lower.contains("october")
                    || fact_lower.contains("november") || fact_lower.contains("december");
                if has_date_in_text { 1.5 } else { 0.5 }
            } else {
                0.0
            };

            // 7. Length bonus (0.3 for facts >10 words -- more specific facts are more useful)
            let word_count = fact_text.split_whitespace().count();
            let length_bonus = if word_count > 10 { 0.3 } else { 0.0 };

            let total = entity_score + keyword_score + multi_kw_bonus + dimension_score
                + active_bonus + temporal_score + length_bonus;
            (total, row)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let results: Vec<SearchFactResult> = scored
        .iter()
        .filter(|(score, _)| *score > 0.0)
        .take(max_facts)
        .map(|(score, row)| {
            let entities_str = row.get("entities").and_then(|v| v.as_str()).unwrap_or("[]");
            let entities: Vec<String> = serde_json::from_str(entities_str).unwrap_or_default();
            SearchFactResult {
                id: row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                fact_text: row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                category: row.get("dimension").and_then(|v| v.as_str()).unwrap_or("general").to_string(),
                date: row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                entities,
                score: *score,
            }
        })
        .collect();

    Ok(SearchResult {
        facts: results,
        total_facts,
        query_entities,
        is_temporal: temporal,
        is_adversarial: adversarial,
    })
}

/// POST /api/memory/search -- hybrid retrieval for memory facts.
pub async fn handle_search_facts(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: SearchFactsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse search request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Delegate to reusable search (active_only=false for benchmark compatibility)
    let search = match search_relevant_facts(
        env, &request.student_id, &request.query, request.max_facts, false,
    ).await {
        Ok(s) => s,
        Err(e) => {
            console_log!("Search failed: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Search failed"}"#))
                .unwrap();
        }
    };

    let max_score = search.facts.first().map(|r| r.score).unwrap_or(0.0);
    let avg_score = if search.facts.is_empty() {
        0.0
    } else {
        search.facts.iter().map(|r| r.score).sum::<f64>() / search.facts.len() as f64
    };

    let resp = serde_json::json!({
        "facts": search.facts,
        "total_facts": search.total_facts,
        "max_score": max_score,
        "avg_score": avg_score,
        "query_entities": search.query_entities,
        "is_temporal": search.is_temporal,
        "is_adversarial": search.is_adversarial,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(resp.to_string()))
        .unwrap()
}

/// Extract entities from a query using Workers AI (cheap, background).
async fn extract_query_entities(env: &Env, query: &str) -> Vec<String> {
    let prompt = format!("{}\"{}\"\n", ENTITY_EXTRACTION_PROMPT, query);

    let output = match crate::services::llm::call_workers_ai(
        env,
        "You extract entities from text. Return only a JSON array of strings.",
        &prompt,
        0.0,
        100,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            console_log!("Entity extraction failed: {}", e);
            return vec![];
        }
    };

    // Parse JSON array from output
    let trimmed = output.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    serde_json::from_str::<Vec<String>>(json_str).unwrap_or_default()
}

/// Extract keywords from a query (lowercase, stopwords removed).
fn extract_keywords(query_lower: &str) -> Vec<String> {
    let stopwords: std::collections::HashSet<&str> = [
        "a", "an", "the", "is", "was", "were", "are", "do", "does", "did",
        "what", "when", "where", "who", "how", "which", "that", "this",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "and",
        "or", "not", "but", "if", "about", "has", "had", "have", "be", "been",
        "it", "its", "they", "their", "them", "she", "her", "he", "his",
    ].iter().copied().collect();

    query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 1 && !stopwords.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Detect if the query mentions a specific musical dimension.
fn detect_dimension(query_lower: &str) -> Option<String> {
    let dims = [
        ("dynamics", "dynamics"),
        ("timing", "timing"),
        ("pedaling", "pedaling"),
        ("pedal", "pedaling"),
        ("articulation", "articulation"),
        ("phrasing", "phrasing"),
        ("interpretation", "interpretation"),
    ];
    for (keyword, dim) in &dims {
        if query_lower.contains(keyword) {
            return Some(dim.to_string());
        }
    }
    None
}

/// Count keyword hits in a fact (1.0 per hit, no length normalization).
fn keyword_hit_count(keywords: &[String], fact_text: &str) -> usize {
    if keywords.is_empty() {
        return 0;
    }
    let fact_lower = fact_text.to_lowercase();
    keywords.iter().filter(|kw| fact_lower.contains(kw.as_str())).count()
}

// ---------------------------------------------------------------------------
// POST /api/memory/clear-benchmark -- clear benchmark facts for a student
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct ClearBenchmarkRequest {
    pub student_id: String,
}

/// POST /api/memory/clear-benchmark -- remove all benchmark-sourced facts for a student.
pub async fn handle_clear_benchmark(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let request: ClearBenchmarkRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse clear-benchmark request: {:?}", e);
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
                .body(Body::from(r#"{"error":"Database unavailable"}"#))
                .unwrap();
        }
    };

    let result = db
        .prepare("DELETE FROM synthesized_facts WHERE student_id = ?1 AND source_type = 'benchmark'")
        .bind(&[JsValue::from_str(&request.student_id)]);

    match result {
        Ok(stmt) => {
            if let Err(e) = stmt.run().await {
                console_log!("Failed to clear benchmark facts: {:?}", e);
            }
        }
        Err(e) => {
            console_log!("Failed to bind delete: {:?}", e);
        }
    }

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"ok":true}"#))
        .unwrap()
}

// ---------------------------------------------------------------------------
// POST /api/memory/synthesize -- trigger synthesis for a student
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct SynthesizeRequest {
    pub student_id: String,
}

/// POST /api/memory/synthesize -- manually trigger synthesis.
pub async fn handle_synthesize(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: SynthesizeRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse synthesize request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Check if synthesis is needed
    let should = match should_synthesize(env, &request.student_id).await {
        Ok(s) => s,
        Err(e) => {
            console_log!("Synthesis check failed: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("Synthesis check failed: {}", e)}).to_string(),
                ))
                .unwrap();
        }
    };

    if !should {
        let resp = serde_json::json!({"skipped": true, "reason": "Not enough new observations"});
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(resp.to_string()))
            .unwrap();
    }

    // Run synthesis
    match run_synthesis(env, &request.student_id).await {
        Ok(result) => {
            let resp = serde_json::json!({
                "new_facts": result.new_facts,
                "invalidated": result.invalidated,
                "unchanged": result.unchanged,
                "observations_processed": result.observations_processed,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Body::from(resp.to_string()))
                .unwrap()
        }
        Err(e) => {
            console_log!("Synthesis failed: {}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("Synthesis failed: {}", e)}).to_string(),
                ))
                .unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/memory/seed-observations -- dev-only observation seeding for tests
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct SeedObservationsRequest {
    pub student_id: String,
    pub observations: Vec<SeedObservation>,
}

#[derive(serde::Deserialize)]
pub struct SeedObservation {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub dimension_score: f64,
    pub student_baseline: f64,
    #[serde(default)]
    pub reasoning_trace: String,
}

/// POST /api/memory/seed-observations -- insert test observations directly into D1.
/// Dev-only: returns 404 in production.
pub async fn handle_seed_observations(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Block in production
    let environment = env
        .var("ENVIRONMENT")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if environment == "production" {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Not found"}"#))
            .unwrap();
    }

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: SeedObservationsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse seed-observations request: {:?}", e);
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
                .body(Body::from(r#"{"error":"Database unavailable"}"#))
                .unwrap();
        }
    };

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let mut seeded = 0u32;
    let session_id = format!("seed-{}", &generate_uuid()[..8]);

    for obs in &request.observations {
        let obs_id = generate_uuid();
        let trace = if obs.reasoning_trace.is_empty() {
            "{}".to_string()
        } else {
            obs.reasoning_trace.clone()
        };

        let result = db
            .prepare(
                "INSERT INTO observations (id, student_id, session_id, dimension, \
                 observation_text, reasoning_trace, framing, dimension_score, \
                 student_baseline, is_fallback, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )
            .bind(&[
                JsValue::from_str(&obs_id),
                JsValue::from_str(&request.student_id),
                JsValue::from_str(&session_id),
                JsValue::from_str(&obs.dimension),
                JsValue::from_str(&obs.observation_text),
                JsValue::from_str(&trace),
                JsValue::from_str(&obs.framing),
                JsValue::from_f64(obs.dimension_score),
                JsValue::from_f64(obs.student_baseline),
                JsValue::from_bool(false),
                JsValue::from_str(&now),
            ]);

        match result {
            Ok(stmt) => {
                if let Err(e) = stmt.run().await {
                    console_log!("Failed to insert seeded observation: {:?}", e);
                } else {
                    seeded += 1;
                }
            }
            Err(e) => {
                console_log!("Failed to bind seeded observation: {:?}", e);
            }
        }
    }

    // Update meta counter so should_synthesize() works correctly
    if seeded > 0 {
        if let Err(e) = increment_observation_count_by(env, &request.student_id, seeded as usize).await {
            console_log!("Failed to update observation count after seeding: {}", e);
        }
    }

    let resp = serde_json::json!({"seeded": seeded});
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(resp.to_string()))
        .unwrap()
}

/// Extract JSON from synthesis LLM output (may be wrapped in code fences).
fn extract_synthesis_json(output: &str) -> Result<serde_json::Value, String> {
    // Try to find JSON within ```json ... ``` fences
    let json_str = if let Some(start) = output.find("```json") {
        let json_start = start + 7;
        if let Some(end) = output[json_start..].find("```") {
            // Happy path: both opening and closing fences present
            output[json_start..json_start + end].trim()
        } else {
            // Opening fence but no closing fence (LLM output truncated).
            // Strip the fence prefix and find JSON in the remainder.
            let remainder = &output[json_start..];
            if let Some(brace) = remainder.find('{') {
                if let Some(end_brace) = remainder.rfind('}') {
                    &remainder[brace..=end_brace]
                } else {
                    remainder.trim()
                }
            } else {
                remainder.trim()
            }
        }
    } else if let Some(start) = output.find('{') {
        if let Some(end) = output.rfind('}') {
            &output[start..=end]
        } else {
            output.trim()
        }
    } else {
        output.trim()
    };

    serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse synthesis JSON: {:?} - raw: {}", e, &json_str[..200.min(json_str.len())]))
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
