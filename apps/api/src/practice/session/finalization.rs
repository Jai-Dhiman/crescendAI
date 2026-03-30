use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::{console_error, console_log, wasm_bindgen, WebSocket};

use super::PracticeSession;
use crate::services::stop::SCALER_MEAN;
use crate::services::teaching_moments::StudentBaselines;

impl PracticeSession {
    // --- Baselines ---

    pub(crate) async fn load_baselines(
        &self,
        student_id: &crate::types::StudentId,
    ) -> StudentBaselines {
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
            .bind(&[JsValue::from_str(student_id.as_str())])
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
                row.get("avg_score").and_then(serde_json::Value::as_f64),
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

    // --- Session synthesis ---

    pub(crate) async fn run_synthesis_and_persist(&self, ws: Option<&WebSocket>) {
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
                .map_or(0, |t| t.timestamp_ms)
                .saturating_sub(s.accumulator.timeline.first().map_or(0, |t| t.timestamp_ms));

            let ctx = super::synthesis::SynthesisContext {
                session_id: crate::types::SessionId::from(s.session_id.clone()),
                student_id: crate::types::StudentId::from(s.student_id.clone()),
                conversation_id: crate::types::ConversationId::from(
                    s.conversation_id.clone().unwrap_or_default(),
                ),
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

        if ctx.conversation_id.as_str().is_empty() {
            console_error!("No conversation_id at synthesis time -- cannot persist");
            return;
        }

        console_log!(
            "Starting synthesis: moments={}, transitions={}, drilling={}",
            acc.teaching_moments.len(),
            acc.mode_transitions.len(),
            acc.drilling_records.len()
        );

        let prompt = super::synthesis::build_synthesis_prompt(&acc, &ctx);
        let result = super::synthesis::call_synthesis_llm(&self.env, &prompt).await;

        // Send to client via WS
        let mut synthesis_event = serde_json::json!({
            "type": "synthesis",
            "text": result.text,
            "isFallback": result.is_fallback,
        });

        // In dev mode, include accumulator snapshot for eval analysis
        let is_eval = self.inner.borrow().is_eval;
        if is_eval {
            let s = self.inner.borrow();
            synthesis_event["eval_context"] = serde_json::json!({
                "teaching_moments": &s.accumulator.teaching_moments,
                "mode_transitions": &s.accumulator.mode_transitions,
                "drilling_records": &s.accumulator.drilling_records,
                "timeline": &s.accumulator.timeline,
                "scored_chunks": &s.scored_chunks,
                "piece_identification": s.piece_identification.as_ref().map(|pi| serde_json::json!({
                    "piece_id": &pi.piece_id,
                    "confidence": pi.confidence,
                    "method": &pi.method,
                })),
                "baselines": &s.baselines,
                "piece_context": s.score_context.as_ref().map(|sc| serde_json::json!({
                    "composer": &sc.composer,
                    "title": &sc.title,
                    "piece_id": &sc.piece_id,
                })),
            });
        }

        if let Some(ws) = ws {
            let _ = ws.send_with_str(synthesis_event.to_string());
        }

        // Persist synthesis message to D1
        match super::synthesis::persist_synthesis_message(
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
        if let Err(e) = super::synthesis::persist_accumulated_moments(
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

    pub(crate) async fn finalize_session(&self, _ws: Option<&WebSocket>) {
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

        {
            let (moment_count, session_id, student_id) = {
                let s = self.inner.borrow();
                (
                    s.accumulator.teaching_moments.len(),
                    s.session_id.clone(),
                    crate::types::StudentId::from(s.student_id.clone()),
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
                                console_error!("Memory synthesis failed for {}: {}", student_id, e);
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
                if needs_deferred && conv.is_some() {
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

            // Persist session_end message (deterministic ID for idempotency)
            let conv_id = self.inner.borrow().conversation_id.clone();
            if let Some(ref conv_id) = conv_id {
                if let Ok(db) = self.env.d1("DB") {
                    let end_msg_id = format!("session_end_{session_id}");
                    let now = crate::types::now_iso();
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
}
