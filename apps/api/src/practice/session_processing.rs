use std::collections::HashMap;
use worker::*;

use crate::practice::accumulator::{AccumulatedMoment, ModeTransitionRecord, TimelineEvent};
use crate::practice::analysis;
use crate::practice::dims::DIMS_6;
use crate::practice::practice_mode::{
    pitch_bigrams_from_notes, ChunkSignal, ModeTransition, PracticeMode,
};
use crate::practice::score_follower::{PerfNote, PerfPedalEvent};
use crate::services::teaching_moments::{RecentObservation, ScoredChunk};

use super::session::{MuqResponse, PracticeSession, ALARM_DURATION_MS};

impl PracticeSession {
    pub(crate) async fn handle_chunk_ready(
        &self,
        ws: &WebSocket,
        index: usize,
        r2_key: &str,
    ) -> Result<()> {
        // 1. Fetch audio from R2
        let audio_bytes = match self.fetch_audio_from_r2(r2_key).await {
            Ok(bytes) => bytes,
            Err(e) => {
                console_error!("R2 fetch failed for {}: {}", r2_key, e);
                self.send_zeroed_chunk_processed(ws, index)?;
                return Ok(());
            }
        };

        // 2. Dispatch MuQ and AMT in parallel
        let context_audio = self.inner.borrow().previous_chunk_audio.clone();
        let (muq_result, amt_result) = futures_util::future::join(
            self.call_muq_endpoint(&audio_bytes),
            self.call_amt_endpoint(context_audio.as_deref(), &audio_bytes),
        )
        .await;

        // Store current chunk as context for the next chunk (NOT persisted to durable storage)
        self.inner.borrow_mut().previous_chunk_audio = Some(audio_bytes.to_vec());

        // Reload identity if DO was evicted during inference
        self.ensure_session_state().await;

        // 3. Process MuQ result
        let scores_array = match muq_result {
            Ok(muq) => {
                let (scores_array, scores_map) = self.process_muq_result(&muq);

                // Send chunk_processed immediately (UI needs scores)
                let scores_json = serde_json::json!({
                    "dynamics": scores_array[0],
                    "timing": scores_array[1],
                    "pedaling": scores_array[2],
                    "articulation": scores_array[3],
                    "phrasing": scores_array[4],
                    "interpretation": scores_array[5],
                });
                let response = serde_json::json!({
                    "type": "chunk_processed",
                    "index": index,
                    "scores": scores_json,
                });
                let _ = ws.send_with_str(&response.to_string());
                console_log!(
                    "chunk_scores[{}]: dyn={:.2} tim={:.2} ped={:.2} art={:.2} phr={:.2} int={:.2}",
                    index, scores_array[0], scores_array[1], scores_array[2],
                    scores_array[3], scores_array[4], scores_array[5]
                );

                // Update DimStats and store ScoredChunk
                {
                    let mut s = self.inner.borrow_mut();
                    s.dim_stats.update(&scores_map);
                    s.scored_chunks.push(ScoredChunk {
                        chunk_index: index,
                        scores: scores_array,
                    });
                }

                scores_array
            }
            Err(e) => {
                console_error!("MuQ inference failed for chunk {}: {}", index, e);
                self.inner.borrow_mut().inference_failures += 1;
                self.send_zeroed_chunk_processed(ws, index)?;
                return Ok(());
            }
        };

        // 4. Process AMT result (graceful degradation if AMT fails)
        let (perf_notes, perf_pedal) = match amt_result {
            Ok(amt) => {
                console_log!(
                    "AMT returned {} notes, {} pedal events for chunk {}",
                    amt.midi_notes.len(),
                    amt.pedal_events.len(),
                    index
                );
                (amt.midi_notes, amt.pedal_events)
            }
            Err(e) => {
                console_error!(
                    "AMT inference failed for chunk {} (proceeding with Tier 3): {}",
                    index,
                    e
                );
                (Vec::new(), Vec::new())
            }
        };

        // 4b. Accumulate notes and run piece identification (if not yet locked)
        if !perf_notes.is_empty() {
            {
                let mut s = self.inner.borrow_mut();
                s.accumulated_notes.extend(perf_notes.iter().cloned());
                // Cap to keep most recent notes (bounds memory, tail is most relevant)
                const MAX_ACCUMULATED_NOTES: usize = 800;
                if s.accumulated_notes.len() > MAX_ACCUMULATED_NOTES {
                    let drain_count = s.accumulated_notes.len() - MAX_ACCUMULATED_NOTES;
                    s.accumulated_notes.drain(..drain_count);
                }
            }
            self.try_identify_piece(ws).await;
        }

        // 5. Load baselines (one-time)
        {
            let needs_load = !self.inner.borrow().baselines_loaded;
            if needs_load {
                let student_id = self.inner.borrow().student_id.clone();
                let baselines = self.load_baselines(&student_id).await;
                let mut s = self.inner.borrow_mut();
                s.baselines = Some(baselines);
                s.baselines_loaded = true;
            }
        }

        // 6. Load score context (one-time, if piece_query is set)
        {
            let needs_load = {
                let s = self.inner.borrow();
                !s.score_context_loaded && s.piece_query.is_some()
            };
            if needs_load {
                let (query, student_id) = {
                    let s = self.inner.borrow();
                    (
                        s.piece_query.clone().unwrap_or_default(),
                        s.student_id.clone(),
                    )
                };
                let ctx =
                    crate::practice::score_context::resolve_piece(&self.env, &query, &student_id)
                        .await;
                let mut s = self.inner.borrow_mut();
                s.score_context = ctx;
                s.score_context_loaded = true;
            }
        }

        // 7. Run score following + analysis using AMT result
        let (chunk_analysis, chunk_bar_range) =
            self.process_amt_result(index, &perf_notes, &perf_pedal, &scores_array);

        // 8. Build ChunkSignal and update practice mode
        let perf_pitches: Vec<u8> = perf_notes.iter().map(|n| n.pitch).collect();
        let has_piece_match = self.inner.borrow().score_context.is_some();
        let chunk_signal = ChunkSignal {
            chunk_index: index,
            timestamp_ms: js_sys::Date::now() as u64,
            pitch_bigrams: pitch_bigrams_from_notes(&perf_pitches),
            bar_range: chunk_bar_range,
            has_piece_match,
            scores: scores_array,
        };

        let mode_transitions = self.inner.borrow_mut().mode_detector.update(&chunk_signal);

        // Broadcast mode changes over WebSocket
        for transition in &mode_transitions {
            let context = self.build_mode_context(transition);
            let msg = serde_json::json!({
                "type": "mode_change",
                "mode": transition.mode,
                "chunkIndex": transition.chunk_index,
                "context": context,
            });
            let _ = ws.send_with_str(&msg.to_string());
        }

        // 9. Accumulate signals silently (synthesis happens on end_session)
        {
            let timestamp_ms = js_sys::Date::now() as u64;
            let has_audio = !perf_notes.is_empty();

            // Timeline event
            self.inner
                .borrow_mut()
                .accumulator
                .accumulate_timeline_event(TimelineEvent {
                    chunk_index: index,
                    timestamp_ms,
                    has_audio,
                });

            // Mode transitions
            for transition in &mode_transitions {
                let from_mode = {
                    let s = self.inner.borrow();
                    s.accumulator
                        .mode_transitions
                        .last()
                        .map(|t| t.to)
                        .unwrap_or(PracticeMode::Warming)
                };
                let dwell_ms = {
                    let s = self.inner.borrow();
                    let mc = s.mode_detector.mode_context();
                    timestamp_ms.saturating_sub(mc.entered_at_ms)
                };
                self.inner
                    .borrow_mut()
                    .accumulator
                    .accumulate_mode_transition(ModeTransitionRecord {
                        from: from_mode,
                        to: transition.mode,
                        chunk_index: transition.chunk_index,
                        timestamp_ms,
                        dwell_ms,
                    });
            }

            // Teaching moment accumulation (every 2 chunks).
            // select_teaching_moment handles STOP classification internally and falls
            // back to positive moments when no chunks pass STOP -- no outer gate needed.
            let chunk_count = self.inner.borrow().scored_chunks.len();
            let should_attempt_moment = chunk_count >= 2 && chunk_count % 2 == 0;

            if should_attempt_moment {
                // Clone all state upfront to avoid holding Ref across borrow_mut.
                // (if-let temporaries have extended lifetimes in Rust)
                let baselines = self.inner.borrow().baselines.clone();
                if let Some(baselines) = baselines {
                    let (recent_obs, scored_chunks) = {
                        let s = self.inner.borrow();
                        let obs: Vec<RecentObservation> = s
                            .accumulator
                            .teaching_moments
                            .iter()
                            .rev()
                            .take(3)
                            .map(|m| RecentObservation {
                                dimension: m.dimension.clone(),
                            })
                            .collect();
                        let chunks = s.scored_chunks.clone();
                        (obs, chunks)
                    };

                    if let Some(moment) = crate::services::teaching_moments::select_teaching_moment(
                        &scored_chunks,
                        &baselines,
                        &recent_obs,
                    ) {
                        let bar_range = chunk_bar_range;
                        let tier = chunk_analysis.as_ref().map(|ca| ca.tier).unwrap_or(3);

                        self.inner
                            .borrow_mut()
                            .accumulator
                            .accumulate_moment(AccumulatedMoment {
                                chunk_index: moment.chunk_index,
                                dimension: moment.dimension.clone(),
                                score: moment.score,
                                baseline: moment.baseline,
                                deviation: moment.deviation,
                                is_positive: moment.is_positive,
                                reasoning: moment.reasoning.clone(),
                                bar_range,
                                analysis_tier: tier,
                                timestamp_ms,
                                llm_analysis: None,
                            });

                        // Send lightweight observation to client during recording
                        let obs_text = if moment.is_positive {
                            format!("Nice work on your {}.", moment.dimension)
                        } else {
                            format!("I'm noticing something in your {} -- let's talk after.", moment.dimension)
                        };
                        let framing = if moment.is_positive { "recognition" } else { "correction" };
                        let obs_msg = serde_json::json!({
                            "type": "observation",
                            "text": obs_text,
                            "dimension": moment.dimension,
                            "framing": framing,
                        });
                        let _ = ws.send_with_str(&obs_msg.to_string());
                    }
                }
            }

            // Drilling record on mode exit
            {
                let mut s = self.inner.borrow_mut();
                if let Some(dr) = s.mode_detector.take_completed_drilling(scores_array, index) {
                    s.accumulator.accumulate_drilling_record(dr);
                }
            }

            // Persist accumulator to DO storage
            let acc_json =
                serde_json::to_string(&self.inner.borrow().accumulator).unwrap_or_default();
            let _ = self.state.storage().put("accumulator", &acc_json).await;
        }

        // 10. Reset alarm
        let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;

        Ok(())
    }

    /// Extract 6-dim scores from MuQ response. Returns (scores_array, scores_map).
    pub(crate) fn process_muq_result(&self, muq: &MuqResponse) -> ([f64; 6], HashMap<String, f64>) {
        let scores_map: HashMap<String, f64> = DIMS_6
            .iter()
            .filter_map(|&dim| {
                muq.predictions
                    .get(dim)
                    .copied()
                    .map(|val| (dim.to_string(), val))
            })
            .collect();
        let scores_array: [f64; 6] = DIMS_6.map(|dim| scores_map.get(dim).copied().unwrap_or(0.0));
        (scores_array, scores_map)
    }

    /// Run score following and bar-aligned analysis from AMT notes.
    /// Returns (chunk_analysis, bar_range) -- both None if no perf notes.
    pub(crate) fn process_amt_result(
        &self,
        index: usize,
        perf_notes: &[PerfNote],
        perf_pedal: &[PerfPedalEvent],
        scores_array: &[f64; 6],
    ) -> (Option<analysis::ChunkAnalysis>, Option<(u32, u32)>) {
        // Extract what we need from state, then drop the borrow
        let (score_data_clone, follower_state_clone) = {
            let s = self.inner.borrow();
            (
                s.score_context.as_ref().map(|ctx| ctx.score.clone()),
                s.follower_state.clone(),
            )
        };

        if !perf_notes.is_empty() {
            if let Some(score_data) = score_data_clone {
                // Run alignment with a mutable copy, then store updated state
                let mut fs = follower_state_clone;
                let bar_map = crate::practice::score_follower::align_chunk(
                    index,
                    0.0,
                    perf_notes,
                    &score_data,
                    &mut fs,
                );
                // Store updated follower state
                self.inner.borrow_mut().follower_state = fs;

                if let Some(ref bm) = bar_map {
                    let bar_range = (bm.bar_start, bm.bar_end);
                    let score_ctx = self.inner.borrow().score_context.clone().unwrap();
                    let analysis_result = analysis::analyze_tier1(
                        bm,
                        perf_notes,
                        perf_pedal,
                        scores_array,
                        &score_ctx,
                    );
                    (Some(analysis_result), Some(bar_range))
                } else {
                    (
                        Some(analysis::analyze_tier2(
                            perf_notes,
                            perf_pedal,
                            scores_array,
                        )),
                        None,
                    )
                }
            } else {
                (
                    Some(analysis::analyze_tier2(
                        perf_notes,
                        perf_pedal,
                        scores_array,
                    )),
                    None,
                )
            }
        } else {
            // No perf notes -> Tier 3 (scores only)
            (None, None)
        }
    }

    /// Entry point for eval_chunk messages (combined MuQ+AMT response).
    pub(crate) async fn process_inference_result(
        &self,
        ws: &WebSocket,
        index: usize,
        hf_response: serde_json::Value,
    ) -> Result<()> {
        // Reload identity if DO was evicted during inference
        self.ensure_session_state().await;

        // Extract scores from predictions
        let predictions = hf_response.get("predictions").unwrap_or(&hf_response);
        let scores_map: HashMap<String, f64> = DIMS_6
            .iter()
            .filter_map(|&dim| {
                predictions
                    .get(dim)
                    .and_then(|v| v.as_f64())
                    .map(|val| (dim.to_string(), val))
            })
            .collect();
        let scores_array: [f64; 6] = DIMS_6.map(|dim| scores_map.get(dim).copied().unwrap_or(0.0));

        // Extract midi_notes and pedal_events
        let perf_notes: Vec<PerfNote> = hf_response
            .get("midi_notes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|n| {
                        Some(PerfNote {
                            pitch: n.get("pitch")?.as_u64()? as u8,
                            onset: n.get("onset")?.as_f64()?,
                            offset: n.get("offset")?.as_f64()?,
                            velocity: n.get("velocity")?.as_u64()? as u8,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let perf_pedal: Vec<PerfPedalEvent> = hf_response
            .get("pedal_events")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| {
                        Some(PerfPedalEvent {
                            time: e.get("time")?.as_f64()?,
                            value: e.get("value")?.as_u64()? as u8,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Send chunk_processed
        let scores_json = serde_json::json!({
            "dynamics": scores_array[0],
            "timing": scores_array[1],
            "pedaling": scores_array[2],
            "articulation": scores_array[3],
            "phrasing": scores_array[4],
            "interpretation": scores_array[5],
        });
        let response = serde_json::json!({
            "type": "chunk_processed",
            "index": index,
            "scores": scores_json,
        });
        let _ = ws.send_with_str(&response.to_string());

        // Update DimStats and store ScoredChunk
        {
            let mut s = self.inner.borrow_mut();
            s.dim_stats.update(&scores_map);
            s.scored_chunks.push(ScoredChunk {
                chunk_index: index,
                scores: scores_array,
            });
        }

        // Accumulate notes and run piece identification (if not yet locked)
        if !perf_notes.is_empty() {
            {
                let mut s = self.inner.borrow_mut();
                s.accumulated_notes.extend(perf_notes.iter().cloned());
                const MAX_ACCUMULATED_NOTES: usize = 800;
                if s.accumulated_notes.len() > MAX_ACCUMULATED_NOTES {
                    let drain_count = s.accumulated_notes.len() - MAX_ACCUMULATED_NOTES;
                    s.accumulated_notes.drain(..drain_count);
                }
            }
            self.try_identify_piece(ws).await;
        }

        // Load baselines (one-time)
        {
            let needs_load = !self.inner.borrow().baselines_loaded;
            if needs_load {
                let student_id = self.inner.borrow().student_id.clone();
                let baselines = self.load_baselines(&student_id).await;
                let mut s = self.inner.borrow_mut();
                s.baselines = Some(baselines);
                s.baselines_loaded = true;
            }
        }

        // Load score context (one-time)
        {
            let needs_load = {
                let s = self.inner.borrow();
                !s.score_context_loaded && s.piece_query.is_some()
            };
            if needs_load {
                let (query, student_id) = {
                    let s = self.inner.borrow();
                    (
                        s.piece_query.clone().unwrap_or_default(),
                        s.student_id.clone(),
                    )
                };
                let ctx =
                    crate::practice::score_context::resolve_piece(&self.env, &query, &student_id)
                        .await;
                let mut s = self.inner.borrow_mut();
                s.score_context = ctx;
                s.score_context_loaded = true;
            }
        }

        // Score following + analysis
        let (chunk_analysis, chunk_bar_range) =
            self.process_amt_result(index, &perf_notes, &perf_pedal, &scores_array);

        // Practice mode update
        let perf_pitches: Vec<u8> = perf_notes.iter().map(|n| n.pitch).collect();
        let has_piece_match = self.inner.borrow().score_context.is_some();
        let chunk_signal = ChunkSignal {
            chunk_index: index,
            timestamp_ms: js_sys::Date::now() as u64,
            pitch_bigrams: pitch_bigrams_from_notes(&perf_pitches),
            bar_range: chunk_bar_range,
            has_piece_match,
            scores: scores_array,
        };

        let mode_transitions = self.inner.borrow_mut().mode_detector.update(&chunk_signal);
        for transition in &mode_transitions {
            let context = self.build_mode_context(transition);
            let msg = serde_json::json!({
                "type": "mode_change",
                "mode": transition.mode,
                "chunkIndex": transition.chunk_index,
                "context": context,
            });
            let _ = ws.send_with_str(&msg.to_string());
        }

        // Accumulate signals silently (synthesis happens on end_session)
        {
            let timestamp_ms = js_sys::Date::now() as u64;
            let has_audio = !perf_notes.is_empty();

            // Timeline event
            self.inner
                .borrow_mut()
                .accumulator
                .accumulate_timeline_event(TimelineEvent {
                    chunk_index: index,
                    timestamp_ms,
                    has_audio,
                });

            // Mode transitions
            for transition in &mode_transitions {
                let from_mode = {
                    let s = self.inner.borrow();
                    s.accumulator
                        .mode_transitions
                        .last()
                        .map(|t| t.to)
                        .unwrap_or(PracticeMode::Warming)
                };
                let dwell_ms = {
                    let s = self.inner.borrow();
                    let mc = s.mode_detector.mode_context();
                    timestamp_ms.saturating_sub(mc.entered_at_ms)
                };
                self.inner
                    .borrow_mut()
                    .accumulator
                    .accumulate_mode_transition(ModeTransitionRecord {
                        from: from_mode,
                        to: transition.mode,
                        chunk_index: transition.chunk_index,
                        timestamp_ms,
                        dwell_ms,
                    });
            }

            // Teaching moment accumulation (every 2 chunks)
            let chunk_count = self.inner.borrow().scored_chunks.len();
            let should_attempt_moment = chunk_count >= 2 && chunk_count % 2 == 0;

            if should_attempt_moment {
                let baselines = self.inner.borrow().baselines.clone();
                if let Some(baselines) = baselines {
                    let (recent_obs, scored_chunks) = {
                        let s = self.inner.borrow();
                        let obs: Vec<RecentObservation> = s
                            .accumulator
                            .teaching_moments
                            .iter()
                            .rev()
                            .take(3)
                            .map(|m| RecentObservation {
                                dimension: m.dimension.clone(),
                            })
                            .collect();
                        let chunks = s.scored_chunks.clone();
                        (obs, chunks)
                    };

                    if let Some(moment) =
                        crate::services::teaching_moments::select_teaching_moment(
                            &scored_chunks,
                            &baselines,
                            &recent_obs,
                        )
                    {
                        let bar_range = chunk_bar_range;
                        let tier = chunk_analysis.as_ref().map(|ca| ca.tier).unwrap_or(3);

                        self.inner
                            .borrow_mut()
                            .accumulator
                            .accumulate_moment(AccumulatedMoment {
                                chunk_index: moment.chunk_index,
                                dimension: moment.dimension.clone(),
                                score: moment.score,
                                baseline: moment.baseline,
                                deviation: moment.deviation,
                                is_positive: moment.is_positive,
                                reasoning: moment.reasoning.clone(),
                                bar_range,
                                analysis_tier: tier,
                                timestamp_ms,
                                llm_analysis: None,
                            });

                        // Send lightweight observation to client during recording
                        let obs_text = if moment.is_positive {
                            format!("Nice work on your {}.", moment.dimension)
                        } else {
                            format!(
                                "I'm noticing something in your {} -- let's talk after.",
                                moment.dimension
                            )
                        };
                        let framing = if moment.is_positive {
                            "recognition"
                        } else {
                            "correction"
                        };
                        let obs_msg = serde_json::json!({
                            "type": "observation",
                            "text": obs_text,
                            "dimension": moment.dimension,
                            "framing": framing,
                        });
                        let _ = ws.send_with_str(&obs_msg.to_string());
                    }
                }
            }

            // Drilling record on mode exit
            {
                let mut s = self.inner.borrow_mut();
                if let Some(dr) = s.mode_detector.take_completed_drilling(scores_array, index) {
                    s.accumulator.accumulate_drilling_record(dr);
                }
            }

            // Persist accumulator to DO storage
            let acc_json =
                serde_json::to_string(&self.inner.borrow().accumulator).unwrap_or_default();
            let _ = self.state.storage().put("accumulator", &acc_json).await;
        }

        // Reset alarm
        let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;

        Ok(())
    }

    pub(crate) fn build_mode_context(&self, transition: &ModeTransition) -> serde_json::Value {
        let s = self.inner.borrow();
        match transition.mode {
            PracticeMode::Drilling => {
                let mut ctx = serde_json::json!({});
                if let Some(ref dp) = s.mode_detector.drilling_passage {
                    if let Some(br) = dp.bar_range {
                        ctx["bars"] = serde_json::json!([br.0, br.1]);
                    }
                    ctx["repetition"] = serde_json::json!(dp.repetition_count);
                }
                ctx
            }
            PracticeMode::Running => {
                let mut ctx = serde_json::json!({});
                if let Some(ref sc) = s.score_context {
                    ctx["piece"] = serde_json::json!(format!("{} - {}", sc.composer, sc.title));
                }
                ctx
            }
            _ => serde_json::json!({}),
        }
    }

    pub(crate) fn send_zeroed_chunk_processed(&self, ws: &WebSocket, index: usize) -> Result<()> {
        let response = serde_json::json!({
            "type": "chunk_processed",
            "index": index,
            "scores": {
                "dynamics": 0.0, "timing": 0.0, "pedaling": 0.0,
                "articulation": 0.0, "phrasing": 0.0, "interpretation": 0.0,
            },
        });
        // Swallow errors -- WS may be closed if session is ending
        let _ = ws.send_with_str(&response.to_string());
        Ok(())
    }

}
