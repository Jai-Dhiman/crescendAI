use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use super::session::{ObservationRecord, PracticeSession};
use crate::services::stop::SCALER_MEAN;
use crate::services::teaching_moments::StudentBaselines;

impl PracticeSession {
    // --- Baselines ---

    pub(crate) async fn load_baselines(&self, student_id: &str) -> StudentBaselines {
        let defaults = SCALER_MEAN;

        let db = match self.env.d1("DB") {
            Ok(db) => db,
            Err(e) => {
                console_error!("D1 binding failed for baselines: {:?}", e);
                return Self::baselines_from_defaults(&defaults);
            }
        };

        let stmt = match db
            .prepare(
                "SELECT dimension, AVG(dimension_score) as avg_score \
                 FROM observations WHERE student_id = ?1 \
                 AND created_at > datetime('now', '-30 days') \
                 GROUP BY dimension",
            )
            .bind(&[JsValue::from_str(student_id)])
        {
            Ok(s) => s,
            Err(e) => {
                console_error!("Baselines bind failed: {:?}", e);
                return Self::baselines_from_defaults(&defaults);
            }
        };

        let rows = match stmt.all().await {
            Ok(r) => r,
            Err(e) => {
                console_error!("Baselines query failed: {:?}", e);
                return Self::baselines_from_defaults(&defaults);
            }
        };

        let results: Vec<serde_json::Value> = rows.results().unwrap_or_default();
        let mut dim_map: HashMap<String, f64> = HashMap::new();
        for row in &results {
            if let (Some(dim), Some(avg)) = (
                row.get("dimension").and_then(|v| v.as_str()),
                row.get("avg_score").and_then(|v| v.as_f64()),
            ) {
                dim_map.insert(dim.to_string(), avg);
            }
        }

        StudentBaselines {
            dynamics: dim_map.get("dynamics").copied().unwrap_or(defaults[0]),
            timing: dim_map.get("timing").copied().unwrap_or(defaults[1]),
            pedaling: dim_map.get("pedaling").copied().unwrap_or(defaults[2]),
            articulation: dim_map.get("articulation").copied().unwrap_or(defaults[3]),
            phrasing: dim_map.get("phrasing").copied().unwrap_or(defaults[4]),
            interpretation: dim_map
                .get("interpretation")
                .copied()
                .unwrap_or(defaults[5]),
        }
    }

    fn baselines_from_defaults(defaults: &[f64; 6]) -> StudentBaselines {
        StudentBaselines {
            dynamics: defaults[0],
            timing: defaults[1],
            pedaling: defaults[2],
            articulation: defaults[3],
            phrasing: defaults[4],
            interpretation: defaults[5],
        }
    }

    // --- Session summary ---

    /// Generate an LLM session summary using Anthropic.
    /// Returns the summary text, or None if generation fails or should be skipped.
    async fn generate_session_summary(
        &self,
        observations: &[ObservationRecord],
        student_id: &str,
    ) -> Option<String> {
        // Skip for eval sessions
        if self.inner.borrow().is_eval_session {
            return None;
        }

        // 1. Load memory context (D1 queries for cross-session facts)
        let now = js_sys::Date::new_0()
            .to_iso_string()
            .as_string()
            .unwrap_or_default();
        let today = &now[..10.min(now.len())];
        let memory_ctx =
            crate::services::memory::build_memory_context(&self.env, student_id, None, today, None)
                .await;
        let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

        // 2. Build piece context string
        let piece_context = {
            let s = self.inner.borrow();
            s.score_context
                .as_ref()
                .map(|ctx| format!("{} - {}", ctx.composer, ctx.title))
        };

        // 3. Build observation tuples for the prompt
        let obs_tuples: Vec<(String, String, String, f64, f64)> = observations
            .iter()
            .map(|o| {
                (
                    o.dimension.clone(),
                    o.text.clone(),
                    o.framing.clone(),
                    o.score,
                    o.baseline,
                )
            })
            .collect();

        // 3b. Collect chunk scores for context when no observations exist
        let chunk_scores: Vec<([f64; 6], usize)> = {
            let s = self.inner.borrow();
            s.scored_chunks
                .iter()
                .map(|c| {
                    // Count notes from the HF response (approximate via chunk index)
                    (c.scores, 0usize) // note_count not tracked in ScoredChunk, use 0
                })
                .collect()
        };

        // 4. Build prompt
        let chunk_scores_ref = if observations.is_empty() && !chunk_scores.is_empty() {
            Some(chunk_scores.as_slice())
        } else {
            None
        };
        let user_prompt = crate::services::prompts::build_session_summary_prompt(
            &obs_tuples,
            &memory_text,
            piece_context.as_deref(),
            chunk_scores_ref,
        );

        // 5. Call Anthropic (non-streaming, 300 max tokens)
        // No explicit timeout needed: CF Workers enforces a 30s subrequest limit,
        // and the 300 max_tokens cap keeps generation fast (~1-2s typical).
        match crate::services::llm::call_anthropic(
            &self.env,
            crate::services::prompts::SESSION_SUMMARY_SYSTEM,
            &user_prompt,
            300,
        )
        .await
        {
            Ok(text) if !text.is_empty() => {
                console_log!("Session summary generated for {}", student_id);
                Some(text)
            }
            Ok(_) => None,
            Err(e) => {
                console_error!("Session summary generation failed: {}", e);
                None
            }
        }
    }

    // --- Session synthesis ---

    pub(crate) async fn run_synthesis_and_persist(&self, ws: &WebSocket) {
        self.ensure_session_state().await;

        // Idempotency guard: persisted to DO storage, survives eviction
        if let Ok(Some(true)) = self
            .state
            .storage()
            .get::<bool>("synthesis_completed")
            .await
        {
            console_log!("Synthesis already completed, skipping duplicate");
            self.inner.borrow_mut().synthesis_completed = true;
            return;
        }

        let (acc, ctx) = {
            let s = self.inner.borrow();
            if !s.accumulator.has_teaching_content()
                && s.accumulator.timeline.iter().all(|t| !t.has_audio)
            {
                console_log!("No teaching content and no audio detected, skipping synthesis");
                return;
            }

            let session_duration_ms = s
                .accumulator
                .timeline
                .last()
                .map(|t| t.timestamp_ms)
                .unwrap_or(0)
                .saturating_sub(
                    s.accumulator
                        .timeline
                        .first()
                        .map(|t| t.timestamp_ms)
                        .unwrap_or(0),
                );

            let ctx = crate::practice::synthesis::SynthesisContext {
                session_id: s.session_id.clone(),
                student_id: s.student_id.clone(),
                conversation_id: s.conversation_id.clone().unwrap_or_default(),
                baselines: s.baselines.clone(),
                piece_context: s.score_context.as_ref().map(|sc| {
                    serde_json::json!({
                        "composer": sc.composer,
                        "title": sc.title,
                        "piece_id": sc.piece_id,
                    })
                }),
                student_memory: None,
                total_chunks: s.scored_chunks.len(),
                session_duration_ms,
            };
            (s.accumulator.clone(), ctx)
        };

        if ctx.conversation_id.is_empty() {
            console_error!("No conversation_id at synthesis time -- cannot persist");
            return;
        }

        console_log!(
            "Starting synthesis: moments={}, transitions={}, drilling={}",
            acc.teaching_moments.len(),
            acc.mode_transitions.len(),
            acc.drilling_records.len()
        );

        let prompt = crate::practice::synthesis::build_synthesis_prompt(&acc, &ctx);
        let result = crate::practice::synthesis::call_synthesis_llm(&self.env, &prompt).await;

        // Send to client via WS
        let synthesis_event = serde_json::json!({
            "type": "synthesis",
            "text": result.text,
            "is_fallback": result.is_fallback,
        });
        let _ = ws.send_with_str(&synthesis_event.to_string());

        // Persist synthesis message to D1
        match crate::practice::synthesis::persist_synthesis_message(
            &self.env,
            &ctx.conversation_id,
            &ctx.session_id,
            &result.text,
        )
        .await
        {
            Ok(msg_id) => console_log!("Synthesis message persisted: {}", msg_id),
            Err(e) => console_error!("Failed to persist synthesis message: {}", e),
        }

        // Persist accumulated moments to observations table
        if let Err(e) = crate::practice::synthesis::persist_accumulated_moments(
            &self.env,
            &ctx.student_id,
            &ctx.session_id,
            &acc.teaching_moments,
        )
        .await
        {
            console_error!("Failed to persist accumulated moments: {}", e);
        }

        self.inner.borrow_mut().synthesis_completed = true;
        let _ = self.state.storage().put("synthesis_completed", &true).await;
    }

    // --- Session finalization ---

    pub(crate) async fn finalize_session(&self, ws: Option<&WebSocket>) {
        self.ensure_session_state().await;

        // Idempotency guard: persisted to DO storage, survives eviction.
        // Put-before-work: even if we crash mid-finalization, we won't re-enter.
        if let Ok(Some(true)) = self.state.storage().get::<bool>("finalized").await {
            console_log!("Session already finalized, skipping duplicate finalize_session call");
            for ws in self.state.get_websockets() {
                let _ = ws.close(Some(1000), Some(String::from("Session ended")));
            }
            return;
        }
        let _ = self.state.storage().put("finalized", &true).await;

        let is_eval = self.inner.borrow().is_eval_session;

        if is_eval {
            // Eval path: preserve old observation-based flow
            let (observations, session_id, student_id, inference_failures, total_chunks) = {
                let s = self.inner.borrow();
                (
                    s.observations.clone(),
                    s.session_id.clone(),
                    s.student_id.clone(),
                    s.inference_failures,
                    s.scored_chunks.len() + s.inference_failures,
                )
            };

            if !observations.is_empty() {
                if let Err(e) = self
                    .persist_observations(&student_id, &session_id, &observations)
                    .await
                {
                    console_error!("Failed to persist observations: {}", e);
                }
            }

            // Send session_summary for eval
            if let Some(ws) = ws {
                let obs_json: Vec<serde_json::Value> = observations.iter()
                    .map(|o| serde_json::json!({"text": o.text, "dimension": o.dimension, "framing": o.framing}))
                    .collect();
                let summary_text = self
                    .generate_session_summary(&observations, &student_id)
                    .await
                    .unwrap_or_default();
                let summary = serde_json::json!({
                    "type": "session_summary",
                    "observations": obs_json,
                    "summary": summary_text,
                    "inference_failures": inference_failures,
                    "total_chunks": total_chunks,
                });
                let _ = ws.send_with_str(&summary.to_string());
            }
        } else {
            // Production path: accumulator-based
            let (moment_count, session_id, student_id) = {
                let s = self.inner.borrow();
                (
                    s.accumulator.teaching_moments.len(),
                    s.session_id.clone(),
                    s.student_id.clone(),
                )
            };

            // Update observation count for memory synthesis
            if moment_count > 0 {
                if let Err(e) = crate::services::memory::increment_observation_count_by(
                    &self.env,
                    &student_id,
                    moment_count,
                )
                .await
                {
                    console_error!("Failed to increment observation count: {}", e);
                }

                match crate::services::memory::should_synthesize(&self.env, &student_id).await {
                    Ok(true) => {
                        match crate::services::memory::run_synthesis(&self.env, &student_id).await {
                            Ok(result) => console_log!(
                                "Memory synthesis for {}: {} new, {} invalidated, {} unchanged",
                                student_id,
                                result.new_facts,
                                result.invalidated,
                                result.unchanged
                            ),
                            Err(e) => {
                                console_error!("Memory synthesis failed for {}: {}", student_id, e)
                            }
                        }
                    }
                    Ok(false) => {}
                    Err(e) => console_error!("Memory synthesis check failed: {}", e),
                }
            }

            // Safety net: if synthesis didn't happen (disconnect, alarm timeout),
            // persist the accumulator to D1 for deferred recovery
            {
                let (needs_deferred, acc, sid, conv) = {
                    let s = self.inner.borrow();
                    (
                        !s.synthesis_completed && s.accumulator.has_teaching_content(),
                        s.accumulator.clone(),
                        s.session_id.clone(),
                        s.conversation_id.clone(),
                    )
                };
                if needs_deferred {
                    if conv.is_some() {
                        let acc_json = serde_json::to_string(&acc).unwrap_or_default();
                        if let Ok(db) = self.env.d1("DB") {
                            if let Ok(stmt) = db.prepare(
                                "UPDATE sessions SET accumulator_json = ?1, needs_synthesis = 1 WHERE id = ?2"
                            ).bind(&[
                                JsValue::from_str(&acc_json),
                                JsValue::from_str(&sid),
                            ]) {
                                if let Err(e) = stmt.run().await {
                                    console_error!("Failed to persist accumulator for deferred synthesis: {:?}", e);
                                } else {
                                    console_log!("Accumulator persisted for deferred synthesis: session={}", sid);
                                }
                            }
                        }
                    }
                }
            }

            // Persist session_end message (deterministic ID for idempotency)
            let conv_id = self.inner.borrow().conversation_id.clone();
            if let Some(ref conv_id) = conv_id {
                if let Ok(db) = self.env.d1("DB") {
                    let end_msg_id = format!("session_end_{}", session_id);
                    let now = js_sys::Date::new_0()
                        .to_iso_string()
                        .as_string()
                        .unwrap_or_default();
                    if let Ok(q) = db.prepare(
                        "INSERT OR IGNORE INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
                         VALUES (?1, ?2, 'assistant', 'Recording ended', 'session_end', ?3, ?4)"
                    ).bind(&[
                        JsValue::from_str(&end_msg_id),
                        JsValue::from_str(conv_id),
                        JsValue::from_str(&session_id),
                        JsValue::from_str(&now),
                    ]) {
                        if let Err(e) = q.run().await {
                            console_error!("Failed to persist session_end message: {:?}", e);
                        }
                    }
                }
            }
        }

        // Close all WebSockets
        for ws in self.state.get_websockets() {
            let _ = ws.close(Some(1000), Some(String::from("Session ended")));
        }
    }

    async fn persist_observations(
        &self,
        student_id: &str,
        session_id: &str,
        observations: &[ObservationRecord],
    ) -> std::result::Result<(), String> {
        let db = self
            .env
            .d1("DB")
            .map_err(|e| format!("D1 binding: {:?}", e))?;
        let now = js_sys::Date::new_0()
            .to_iso_string()
            .as_string()
            .unwrap_or_default();

        for obs in observations {
            let stmt = db
                .prepare(
                    "INSERT OR IGNORE INTO observations (id, student_id, session_id, chunk_index, \
                     dimension, observation_text, reasoning_trace, framing, dimension_score, \
                     student_baseline, piece_context, is_fallback, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                )
                .bind(&[
                    JsValue::from_str(&obs.id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(session_id),
                    JsValue::from_f64(obs.chunk_index as f64),
                    JsValue::from_str(&obs.dimension),
                    JsValue::from_str(&obs.text),
                    JsValue::from_str(&obs.reasoning_trace),
                    JsValue::from_str(&obs.framing),
                    JsValue::from_f64(obs.score),
                    JsValue::from_f64(obs.baseline),
                    JsValue::NULL, // piece_context (deferred)
                    JsValue::from_bool(obs.is_fallback),
                    JsValue::from_str(&now),
                ]);

            match stmt {
                Ok(s) => {
                    if let Err(e) = s.run().await {
                        console_error!("Failed to insert observation {}: {:?}", obs.id, e);
                    }
                }
                Err(e) => {
                    console_error!("Failed to bind observation {}: {:?}", obs.id, e);
                }
            }
        }

        Ok(())
    }
}
