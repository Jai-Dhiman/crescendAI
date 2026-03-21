use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use crate::practice::analysis;
use crate::practice::dims::DIMS_6;
use crate::practice::practice_mode::{
    ChunkSignal, ModeDetector, ModeTransition, ObservationPolicy, PracticeMode,
    pitch_bigrams_from_notes,
};
use crate::practice::score_context::ScoreContext;
use crate::practice::score_follower::{FollowerState, PerfNote, PerfPedalEvent};
use crate::practice::teaching_moment::DimStats;
use crate::services::stop;
use crate::services::teaching_moments::{
    RecentObservation, ScoredChunk, StudentBaselines,
};

/// JS setTimeout-based sleep for Cloudflare Workers WASM
async fn sleep_ms(ms: u64) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .expect("setTimeout not found on global");
        let set_timeout_fn: js_sys::Function = set_timeout.into();
        let _ = set_timeout_fn.call2(&JsValue::NULL, &resolve, &JsValue::from(ms as f64));
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

const ALARM_DURATION_MS: i64 = 30 * 60 * 1000; // 30 minutes
const HF_RETRY_DELAYS_MS: &[u64] = &[10_000, 20_000, 40_000]; // retry 503s with backoff (70s total)
const HF_RETRY_DELAYS_ENDING_MS: &[u64] = &[3_000, 5_000]; // shorter retries when session is ending

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ObservationRecord {
    pub id: String,
    pub text: String,
    pub dimension: String,
    pub framing: String,
    pub chunk_index: usize,
    pub score: f64,
    pub baseline: f64,
    pub reasoning_trace: String,
    pub is_fallback: bool,
    pub components_json: Option<String>,
}

struct SessionState {
    session_id: String,
    student_id: String,
    baselines: Option<StudentBaselines>,
    baselines_loaded: bool,
    scored_chunks: Vec<ScoredChunk>,
    observations: Vec<ObservationRecord>,
    inference_failures: usize,
    chunks_in_flight: usize,
    session_ending: bool,
    dim_stats: DimStats,
    last_observation_at: Option<u64>,
    piece_query: Option<String>,
    score_context: Option<ScoreContext>,
    score_context_loaded: bool,
    follower_state: FollowerState,
    is_eval_session: bool,
    mode_detector: ModeDetector,
    conversation_id: Option<String>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            student_id: String::new(),
            baselines: None,
            baselines_loaded: false,
            scored_chunks: Vec::new(),
            observations: Vec::new(),
            inference_failures: 0,
            chunks_in_flight: 0,
            session_ending: false,
            dim_stats: DimStats::default(),
            last_observation_at: None,
            piece_query: None,
            score_context: None,
            score_context_loaded: false,
            follower_state: FollowerState::default(),
            is_eval_session: false,
            mode_detector: ModeDetector::new(),
            conversation_id: None,
        }
    }
}

#[durable_object]
pub struct PracticeSession {
    state: State,
    env: Env,
    inner: RefCell<SessionState>,
}

impl DurableObject for PracticeSession {
    fn new(state: State, env: Env) -> Self {
        Self {
            state,
            env,
            inner: RefCell::new(SessionState::default()),
        }
    }

    async fn fetch(&self, req: Request) -> Result<Response> {
        let url = req.url()?;
        let path = url.path();
        let session_id = path.strip_prefix("/ws/").unwrap_or("").to_string();

        // Extract student_id from query param (set by server.rs after auth validation)
        let student_id = url.query_pairs()
            .find(|(k, _)| k == "student_id")
            .map(|(_, v)| v.to_string())
            .unwrap_or_default();

        let conversation_id = url.query_pairs()
            .find(|(k, _)| k == "conversation_id")
            .map(|(_, v)| v.to_string())
            .filter(|s| !s.is_empty());

        console_log!("DO fetch: session_id={}, student_id={}, conversation_id={:?}", session_id, student_id, conversation_id);

        // Store session info (on first connect only; reconnections keep existing state)
        {
            let mut s = self.inner.borrow_mut();
            if s.session_id.is_empty() {
                s.session_id = session_id.clone();
                s.student_id = student_id;
                s.conversation_id = conversation_id;
            }
        }

        // Close any existing WebSocket connections (reconnection case)
        let existing_sockets = self.state.get_websockets();
        for old_ws in existing_sockets {
            let _ = old_ws.close(Some(1000), Some(String::from("New connection replacing old one")));
        }

        // Accept WebSocket upgrade
        let pair = WebSocketPair::new()?;
        let server = pair.server;
        self.state.accept_web_socket(&server);

        // Set 30-minute alarm
        self.state.storage().set_alarm(ALARM_DURATION_MS).await?;

        // Send welcome
        let welcome = serde_json::json!({
            "type": "connected",
            "sessionId": session_id,
        });
        server.send_with_str(&welcome.to_string())?;

        Response::from_websocket(pair.client)
    }

    async fn websocket_message(&self, ws: WebSocket, msg: WebSocketIncomingMessage) -> Result<()> {
        let text = match msg {
            WebSocketIncomingMessage::String(s) => s,
            WebSocketIncomingMessage::Binary(_) => return Ok(()),
        };

        let parsed: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };

        let msg_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match msg_type {
            "chunk_ready" => {
                // Skip new chunks if session is ending
                if self.inner.borrow().session_ending {
                    return Ok(());
                }
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let r2_key = parsed.get("r2Key").and_then(|v| v.as_str()).unwrap_or("");

                // Track in-flight for deferred finalization
                self.inner.borrow_mut().chunks_in_flight += 1;
                let result = self.handle_chunk_ready(&ws, index, r2_key).await;

                // Decrement and check if we should finalize
                let should_finalize = {
                    let mut s = self.inner.borrow_mut();
                    s.chunks_in_flight -= 1;
                    s.session_ending && s.chunks_in_flight == 0
                };
                if should_finalize {
                    console_log!("Last in-flight chunk completed, finalizing session");
                    self.finalize_session(Some(&ws)).await;
                }

                result?;
            }
            "end_session" => {
                let in_flight = {
                    let mut s = self.inner.borrow_mut();
                    s.session_ending = true;
                    s.chunks_in_flight
                };
                if in_flight == 0 {
                    self.finalize_session(Some(&ws)).await;
                } else {
                    console_log!("end_session received, waiting for {} in-flight chunks", in_flight);
                    // finalize_session will be called by the last completing chunk handler
                }
            }
            "eval_chunk" => {
                // Only available in dev mode
                let is_dev = self.env.var("ENVIRONMENT")
                    .map(|v| v.to_string() == "development")
                    .unwrap_or(false);
                if !is_dev {
                    let _ = ws.send_with_str(r#"{"type":"error","message":"eval_chunk only available in dev"}"#);
                    return Ok(());
                }

                // Mark session as eval session on first eval_chunk
                self.inner.borrow_mut().is_eval_session = true;

                let chunk_index = parsed.get("chunk_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                // Build HF-response-shaped JSON from the eval payload
                let hf_response = serde_json::json!({
                    "predictions": parsed.get("predictions").cloned().unwrap_or_default(),
                    "midi_notes": parsed.get("midi_notes").cloned().unwrap_or_default(),
                    "pedal_events": parsed.get("pedal_events").cloned().unwrap_or_default(),
                });

                self.process_inference_result(&ws, chunk_index, hf_response).await?;
            }
            "set_piece" => {
                let query = parsed.get("query").and_then(|v| v.as_str()).unwrap_or("");
                if !query.is_empty() {
                    let mut s = self.inner.borrow_mut();
                    s.piece_query = Some(query.to_string());
                    s.score_context = None;
                    s.score_context_loaded = false;
                    s.follower_state = FollowerState::default();
                }
                let ack = serde_json::json!({"type": "piece_set", "query": query});
                let _ = ws.send_with_str(&ack.to_string());
            }
            _ => {}
        }

        Ok(())
    }

    async fn websocket_close(
        &self,
        _ws: WebSocket,
        _code: usize,
        _reason: String,
        _was_clean: bool,
    ) -> Result<()> {
        // Client disconnected -- mark session as ending
        let (has_chunks, in_flight) = {
            let mut s = self.inner.borrow_mut();
            s.session_ending = true;
            (!s.scored_chunks.is_empty() || s.inference_failures > 0, s.chunks_in_flight)
        };
        if has_chunks && in_flight == 0 {
            self.finalize_session(None).await;
        }
        // If in_flight > 0, last completing chunk will finalize
        Ok(())
    }

    async fn alarm(&self) -> Result<Response> {
        // Alarm fired: session timed out (30 min inactivity)
        let sockets = self.state.get_websockets();
        let ws = sockets.first();
        self.finalize_session(ws).await;
        Response::ok("alarm handled")
    }
}

// --- Pipeline methods ---

impl PracticeSession {
    async fn handle_chunk_ready(&self, ws: &WebSocket, index: usize, r2_key: &str) -> Result<()> {
        // 1. Fetch audio from R2
        let audio_bytes = match self.fetch_audio_from_r2(r2_key).await {
            Ok(bytes) => bytes,
            Err(e) => {
                console_error!("R2 fetch failed for {}: {}", r2_key, e);
                self.send_zeroed_chunk_processed(ws, index)?;
                return Ok(());
            }
        };

        // 2. Call HF inference (returns full response body)
        let hf_response = match self.call_hf_inference(&audio_bytes).await {
            Ok(resp) => resp,
            Err(e) => {
                console_error!("HF inference failed for chunk {}: {}", index, e);
                self.inner.borrow_mut().inference_failures += 1;
                self.send_zeroed_chunk_processed(ws, index)?;
                return Ok(());
            }
        };

        // 3-12. Process inference result through the pipeline
        self.process_inference_result(ws, index, hf_response).await
    }

    async fn process_inference_result(
        &self,
        ws: &WebSocket,
        index: usize,
        hf_response: serde_json::Value,
    ) -> Result<()> {
        // 3. Extract scores from predictions
        let predictions = hf_response.get("predictions").unwrap_or(&hf_response);
        let scores_map: HashMap<String, f64> = DIMS_6
            .iter()
            .filter_map(|&dim| {
                predictions.get(dim).and_then(|v| v.as_f64()).map(|val| (dim.to_string(), val))
            })
            .collect();
        let scores_array: [f64; 6] = DIMS_6.map(|dim| {
            scores_map.get(dim).copied().unwrap_or(0.0)
        });

        // 4. Extract midi_notes and pedal_events from HF response
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

        // 5. Send chunk_processed immediately (skip if session is ending)
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

        // 6. Update DimStats and store ScoredChunk
        {
            let mut s = self.inner.borrow_mut();
            s.dim_stats.update(&scores_map);
            s.scored_chunks.push(ScoredChunk {
                chunk_index: index,
                scores: scores_array,
            });
        }

        // 7. Load baselines (one-time)
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

        // 8. Load score context (one-time, if piece_query is set)
        {
            let needs_load = {
                let s = self.inner.borrow();
                !s.score_context_loaded && s.piece_query.is_some()
            };
            if needs_load {
                let (query, student_id) = {
                    let s = self.inner.borrow();
                    (s.piece_query.clone().unwrap_or_default(), s.student_id.clone())
                };
                let ctx = crate::practice::score_context::resolve_piece(
                    &self.env,
                    &query,
                    &student_id,
                ).await;
                let mut s = self.inner.borrow_mut();
                s.score_context = ctx;
                s.score_context_loaded = true;
            }
        }

        // 9. Run score following + analysis (also extract raw bar range for mode detector)
        let (chunk_analysis, chunk_bar_range): (Option<analysis::ChunkAnalysis>, Option<(u32, u32)>) = {
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
                        &perf_notes,
                        &score_data,
                        &mut fs,
                    );
                    // Store updated follower state
                    self.inner.borrow_mut().follower_state = fs;

                    if let Some(ref bm) = bar_map {
                        // Capture raw bar range before passing bm to analyze_tier1
                        let bar_range = (bm.bar_start, bm.bar_end);
                        // Tier 1: full bar-aligned analysis
                        let score_ctx = self.inner.borrow().score_context.clone().unwrap();
                        let analysis = analysis::analyze_tier1(
                            bm,
                            &perf_notes,
                            &perf_pedal,
                            &scores_array,
                            &score_ctx,
                        );
                        (Some(analysis), Some(bar_range))
                    } else {
                        // Score context exists but alignment failed -> Tier 2
                        (Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array)), None)
                    }
                } else {
                    // No score context but have perf notes -> Tier 2
                    (Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array)), None)
                }
            } else {
                // No perf notes -> Tier 3 (no analysis, current behavior)
                (None, None)
            }
        };

        // 9b. Build ChunkSignal and update practice mode
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

        // 10. Run STOP classifier on current chunk
        let stop_result = stop::classify(&scores_array);

        // 11. Check if we should generate an observation (mode-aware)
        let policy = self.inner.borrow().mode_detector.observation_policy();
        let (should_generate, gate_debug) = {
            let s = self.inner.borrow();
            let suppress = policy.suppress;
            let stop_triggered = stop_result.triggered;
            let baselines_loaded = s.baselines.is_some();
            let throttle_allows = self.mode_throttle_allows(&s, &policy);
            let mode = format!("{:?}", s.mode_detector.mode);
            let result = !suppress && stop_triggered && baselines_loaded && throttle_allows;
            (result, format!(
                "mode={}, suppress={}, stop={} (p={:.2}), baselines={}, throttle={}, chunks={}",
                mode, suppress, stop_triggered, stop_result.probability,
                baselines_loaded, throttle_allows, s.scored_chunks.len()
            ))
        };
        console_log!("Observation gate: {} -> generate={}", gate_debug, should_generate);

        if should_generate {
            self.generate_observation(ws, chunk_analysis.as_ref()).await;
        }

        // 12. Reset alarm
        let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;

        Ok(())
    }

    fn mode_throttle_allows(&self, s: &SessionState, policy: &ObservationPolicy) -> bool {
        match s.last_observation_at {
            None => true,
            Some(last) => {
                let now = js_sys::Date::now() as u64;
                now - last >= policy.min_interval_ms
            }
        }
    }

    fn build_mode_context(&self, transition: &ModeTransition) -> serde_json::Value {
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

    fn send_zeroed_chunk_processed(&self, ws: &WebSocket, index: usize) -> Result<()> {
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

    async fn generate_observation(&self, ws: &WebSocket, chunk_analysis: Option<&analysis::ChunkAnalysis>) {
        console_log!("generate_observation: starting LLM pipeline");
        let (scored_chunks, baselines, recent_obs, student_id, session_id) = {
            let s = self.inner.borrow();
            let recent: Vec<RecentObservation> = s.observations
                .iter()
                .rev()
                .take(3)
                .map(|o| RecentObservation { dimension: o.dimension.clone() })
                .collect();
            (
                s.scored_chunks.clone(),
                s.baselines.clone().unwrap(),
                recent,
                s.student_id.clone(),
                s.session_id.clone(),
            )
        };

        // Run teaching moment selection
        let moment = match crate::services::teaching_moments::select_teaching_moment(
            &scored_chunks,
            &baselines,
            &recent_obs,
        ) {
            Some(m) => m,
            None => return,
        };

        // Build teaching moment JSON for handle_ask_inner
        let tm_json = serde_json::json!({
            "dimension": moment.dimension,
            "dimension_score": moment.score,
            "chunk_index": moment.chunk_index,
            "deviation": moment.deviation,
            "stop_probability": moment.stop_probability,
            "is_positive": moment.is_positive,
            "reasoning": moment.reasoning,
        });

        // Build enriched piece_context from score context + analysis
        let piece_context = {
            let s = self.inner.borrow();
            let mut ctx = serde_json::Map::new();
            if let Some(score_ctx) = &s.score_context {
                ctx.insert("composer".into(), serde_json::json!(score_ctx.composer));
                ctx.insert("title".into(), serde_json::json!(score_ctx.title));
                ctx.insert("piece_id".into(), serde_json::json!(score_ctx.piece_id));
            }
            if let Some(ca) = chunk_analysis {
                if let Some(bar_range) = &ca.bar_range {
                    ctx.insert("bar_range".into(), serde_json::json!(bar_range));
                }
                ctx.insert("analysis_tier".into(), serde_json::json!(ca.tier));
                ctx.insert("musical_analysis".into(), serde_json::to_value(&ca.dimensions).unwrap_or_default());
            }
            if ctx.is_empty() { None } else { Some(serde_json::Value::Object(ctx)) }
        };

        // Inject drilling comparison context if in drilling mode
        let piece_context = {
            let mut pc = piece_context;
            let s = self.inner.borrow();
            if let Some(ref dp) = s.mode_detector.drilling_passage {
                let current_scores = s.scored_chunks.last().map(|c| c.scores).unwrap_or([0.0; 6]);
                let drilling_ctx = serde_json::json!({
                    "repetition_count": dp.repetition_count,
                    "first_attempt_scores": {
                        "dynamics": dp.first_scores[0],
                        "timing": dp.first_scores[1],
                        "pedaling": dp.first_scores[2],
                        "articulation": dp.first_scores[3],
                        "phrasing": dp.first_scores[4],
                        "interpretation": dp.first_scores[5],
                    },
                    "current_scores": {
                        "dynamics": current_scores[0],
                        "timing": current_scores[1],
                        "pedaling": current_scores[2],
                        "articulation": current_scores[3],
                        "phrasing": current_scores[4],
                        "interpretation": current_scores[5],
                    },
                    "bar_range": dp.bar_range,
                });
                match &mut pc {
                    Some(serde_json::Value::Object(ref mut ctx)) => {
                        ctx.insert("drilling_context".into(), drilling_ctx);
                    }
                    None => {
                        let mut ctx = serde_json::Map::new();
                        ctx.insert("drilling_context".into(), drilling_ctx);
                        pc = Some(serde_json::Value::Object(ctx));
                    }
                    _ => {}
                }
            }
            pc
        };

        let tm_json_clone = tm_json.clone();
        let inner_req = crate::services::ask::AskInnerRequest {
            teaching_moment: tm_json,
            student_id: student_id.clone(),
            session_id: session_id.clone(),
            piece_context,
        };

        let inner_resp = crate::services::ask::handle_ask_inner(&self.env, &inner_req).await;
        console_log!("generate_observation: LLM pipeline returned. text={}, fallback={}", &inner_resp.observation_text[..inner_resp.observation_text.len().min(80)], inner_resp.is_fallback);

        // Push observation to client
        let mut obs_event = serde_json::json!({
            "type": "observation",
            "text": inner_resp.observation_text,
            "dimension": inner_resp.dimension,
            "framing": inner_resp.framing,
        });
        if let Some(br) = chunk_analysis.and_then(|a| a.bar_range.as_ref()) {
            obs_event["barRange"] = serde_json::json!(br);
        }
        if let Some(ref components) = inner_resp.components_json {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(components) {
                obs_event["components"] = parsed;
            }
        }

        // For eval sessions, include the context the teacher LLM saw
        {
            let is_eval = self.inner.borrow().is_eval_session;
            if is_eval {
                let piece_name = self.inner.borrow().piece_query.clone();
                let baselines_json = serde_json::to_value(&baselines).unwrap_or_default();
                let recent_obs_json = serde_json::to_value(&recent_obs).unwrap_or_default();
                let analysis_json = chunk_analysis
                    .map(|ca| serde_json::to_value(ca).unwrap_or_default())
                    .unwrap_or_default();
                obs_event["eval_context"] = serde_json::json!({
                    "teaching_moment": &tm_json_clone,
                    "baselines": baselines_json,
                    "recent_observations": recent_obs_json,
                    "analysis_facts": analysis_json,
                    "piece_name": piece_name,
                });
            }
        }

        let _ = ws.send_with_str(&obs_event.to_string());

        // Generate observation ID and extract fields before borrowing
        let obs_id = crate::services::ask::generate_uuid();
        let observation_text = inner_resp.observation_text.clone();
        let obs_dimension = inner_resp.dimension.clone();
        let obs_framing = inner_resp.framing.clone();
        let components_json = inner_resp.components_json.clone();

        // Store in session state
        {
            let mut s = self.inner.borrow_mut();
            s.observations.push(ObservationRecord {
                id: obs_id.clone(),
                text: observation_text.clone(),
                dimension: obs_dimension.clone(),
                framing: obs_framing.clone(),
                chunk_index: moment.chunk_index,
                score: moment.score,
                baseline: moment.baseline,
                reasoning_trace: inner_resp.reasoning_trace.clone(),
                is_fallback: inner_resp.is_fallback,
                components_json: components_json.clone(),
            });
            s.last_observation_at = Some(js_sys::Date::now() as u64);
        }

        // Persist as message in conversation (awaited to prevent data loss)
        let conv_id = self.inner.borrow().conversation_id.clone();
        console_log!("Persisting observation message: conv_id={:?}, obs_id={}", conv_id, obs_id);
        if let Some(ref conv_id) = conv_id {
            let msg_id = crate::services::ask::generate_uuid();
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
            match self.env.d1("DB") {
                Ok(db) => {
                    let bind_result = db.prepare(
                        "INSERT INTO messages (id, conversation_id, role, content, message_type, \
                         dimension, framing, components_json, session_id, observation_id, created_at) \
                         VALUES (?1, ?2, 'assistant', ?3, 'observation', ?4, ?5, ?6, ?7, ?8, ?9)"
                    )
                    .bind(&[
                        JsValue::from_str(&msg_id),
                        JsValue::from_str(conv_id),
                        JsValue::from_str(&observation_text),
                        JsValue::from_str(&obs_dimension),
                        JsValue::from_str(&obs_framing),
                        match components_json.as_deref() {
                            Some(c) => JsValue::from_str(c),
                            None => JsValue::NULL,
                        },
                        JsValue::from_str(&session_id),
                        JsValue::from_str(&obs_id),
                        JsValue::from_str(&now),
                    ]);
                    match bind_result {
                        Ok(q) => {
                            match q.run().await {
                                Ok(_) => console_log!("Observation message persisted: msg_id={}", msg_id),
                                Err(e) => console_error!("Failed to persist observation message: {:?}", e),
                            }
                        }
                        Err(e) => console_error!("Failed to bind observation message insert: {:?}", e),
                    }
                }
                Err(e) => console_error!("D1 binding failed for observation message: {:?}", e),
            }
        } else {
            console_log!("No conversation_id, skipping observation message persist");
        }
    }

    // --- External service calls ---

    async fn fetch_audio_from_r2(&self, r2_key: &str) -> std::result::Result<Vec<u8>, String> {
        let bucket = self.env.bucket("CHUNKS")
            .map_err(|e| format!("R2 binding failed: {:?}", e))?;
        let object = bucket.get(r2_key).execute().await
            .map_err(|e| format!("R2 get failed: {:?}", e))?;
        let object = object.ok_or_else(|| format!("R2 object not found: {}", r2_key))?;
        let bytes = object.body()
            .ok_or_else(|| "R2 object has no body".to_string())?
            .bytes().await
            .map_err(|e| format!("R2 read failed: {:?}", e))?;
        Ok(bytes)
    }

    async fn call_hf_inference(
        &self,
        audio_bytes: &[u8],
    ) -> std::result::Result<serde_json::Value, String> {
        let endpoint = self.env.var("HF_INFERENCE_ENDPOINT")
            .map_err(|e| format!("HF_INFERENCE_ENDPOINT not set: {:?}", e))?
            .to_string();
        let token = self.env.secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        let mut last_err = String::new();

        // Use shorter retries if session is ending (user is waiting)
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        // Try once, then retry on 503 (HF cold start) with backoff
        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers.set("Content-Type", "audio/webm;codecs=opus").map_err(|e| format!("{:?}", e))?;
            headers.set("Authorization", &format!("Bearer {}", token)).map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from(js_sys::Uint8Array::from(audio_bytes))));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| format!("Failed to create request: {:?}", e))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("HF fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!("HF fetch failed (attempt {}), retrying in {}s: {}", attempt + 1, delay / 1000, last_err);
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("HF returned {}: {}", status, body);
                if attempt < HF_RETRY_DELAYS_MS.len() {
                    let delay = HF_RETRY_DELAYS_MS[attempt];
                    console_log!("HF {} (attempt {}), endpoint likely cold-starting, retrying in {}s", status, attempt + 1, delay / 1000);
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("HF returned {}: {}", status, body));
            }

            let body_text = response.text().await
                .map_err(|e| format!("HF response read failed: {:?}", e))?;
            console_log!("HF response body (first 500 chars): {}", &body_text[..body_text.len().min(500)]);

            let body: serde_json::Value = serde_json::from_str(&body_text)
                .map_err(|e| format!("HF response parse failed: {:?} - body: {}", e, &body_text[..body_text.len().min(200)]))?;

            // Validate that predictions contain all 6 dimensions
            let predictions = body.get("predictions").unwrap_or(&body);
            let dim_count = DIMS_6.iter().filter(|dim| predictions.get(*dim).and_then(|v| v.as_f64()).is_some()).count();
            if dim_count < 6 {
                console_error!("HF dim validation failed: dim_count={}, predictions keys={:?}", dim_count, predictions);
                return Err(format!("HF returned only {} dimensions", dim_count));
            }

            if attempt > 0 {
                console_log!("HF inference succeeded after {} retries", attempt);
            }

            return Ok(body);
        }

        Err(last_err)
    }

    // --- Baselines ---

    async fn load_baselines(&self, student_id: &str) -> StudentBaselines {
        let defaults = crate::services::stop::SCALER_MEAN;

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
            interpretation: dim_map.get("interpretation").copied().unwrap_or(defaults[5]),
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

        // Skip if no observations
        if observations.is_empty() {
            return None;
        }

        // 1. Load memory context (D1 queries for cross-session facts)
        let now = js_sys::Date::new_0()
            .to_iso_string()
            .as_string()
            .unwrap_or_default();
        let today = &now[..10.min(now.len())];
        let memory_ctx = crate::services::memory::build_memory_context(
            &self.env, student_id, None, today, None,
        ).await;
        let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

        // 2. Build piece context string
        let piece_context = {
            let s = self.inner.borrow();
            s.score_context.as_ref().map(|ctx| {
                format!("{} - {}", ctx.composer, ctx.title)
            })
        };

        // 3. Build observation tuples for the prompt
        let obs_tuples: Vec<(String, String, String, f64, f64)> = observations
            .iter()
            .map(|o| (
                o.dimension.clone(),
                o.text.clone(),
                o.framing.clone(),
                o.score,
                o.baseline,
            ))
            .collect();

        // 4. Build prompt
        let user_prompt = crate::services::prompts::build_session_summary_prompt(
            &obs_tuples,
            &memory_text,
            piece_context.as_deref(),
        );

        // 5. Call Anthropic (non-streaming, 300 max tokens)
        // No explicit timeout needed: CF Workers enforces a 30s subrequest limit,
        // and the 300 max_tokens cap keeps generation fast (~1-2s typical).
        match crate::services::llm::call_anthropic(
            &self.env,
            crate::services::prompts::SESSION_SUMMARY_SYSTEM,
            &user_prompt,
            300,
        ).await {
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

    // --- Session finalization ---

    async fn finalize_session(&self, ws: Option<&WebSocket>) {
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

        // 1. Persist observations to D1
        if !observations.is_empty() {
            if let Err(e) = self.persist_observations(&student_id, &session_id, &observations).await {
                console_error!("Failed to persist observations: {}", e);
            }
        }

        // 2. Update observation count and run synthesis
        if !observations.is_empty() {
            // Fix: DO-originated observations were not incrementing the meta counter.
            // This ensures should_synthesize() has an accurate count.
            if let Err(e) = crate::services::memory::increment_observation_count_by(
                &self.env, &student_id, observations.len()
            ).await {
                console_error!("Failed to increment observation count: {}", e);
            }

            // Run synthesis if enough observations have accumulated
            match crate::services::memory::should_synthesize(&self.env, &student_id).await {
                Ok(true) => {
                    match crate::services::memory::run_synthesis(&self.env, &student_id).await {
                        Ok(result) => {
                            console_log!(
                                "Synthesis for {}: {} new, {} invalidated, {} unchanged",
                                student_id, result.new_facts, result.invalidated, result.unchanged
                            );
                        }
                        Err(e) => {
                            console_error!("Synthesis failed for {}: {}", student_id, e);
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    console_error!("Synthesis check failed: {}", e);
                }
            }
        }

        // 3. Generate LLM summary + send session_summary via WebSocket
        if let Some(ws) = ws {
            let obs_json: Vec<serde_json::Value> = observations
                .iter()
                .map(|o| serde_json::json!({
                    "text": o.text,
                    "dimension": o.dimension,
                    "framing": o.framing,
                }))
                .collect();

            // Generate LLM summary (skips for eval sessions, empty sessions, or on failure)
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

            // Persist summary + session_end as messages in conversation
            let conv_id = self.inner.borrow().conversation_id.clone();
            if let Some(ref conv_id) = conv_id {
                let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

                if !summary_text.is_empty() {
                    if let Ok(db) = self.env.d1("DB") {
                        let msg_id = crate::services::ask::generate_uuid();
                        if let Ok(q) = db.prepare(
                            "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
                             VALUES (?1, ?2, 'assistant', ?3, 'summary', ?4, ?5)"
                        )
                        .bind(&[
                            JsValue::from_str(&msg_id),
                            JsValue::from_str(conv_id),
                            JsValue::from_str(&summary_text),
                            JsValue::from_str(&session_id),
                            JsValue::from_str(&now),
                        ]) {
                            if let Err(e) = q.run().await {
                                console_error!("Failed to persist summary message: {:?}", e);
                            }
                        }
                    }
                }

                // Insert session_end message
                if let Ok(db) = self.env.d1("DB") {
                    let end_msg_id = crate::services::ask::generate_uuid();
                    let end_now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
                    if let Ok(q) = db.prepare(
                        "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
                         VALUES (?1, ?2, 'assistant', 'Recording ended', 'session_end', ?3, ?4)"
                    )
                    .bind(&[
                        JsValue::from_str(&end_msg_id),
                        JsValue::from_str(conv_id),
                        JsValue::from_str(&session_id),
                        JsValue::from_str(&end_now),
                    ]) {
                        if let Err(e) = q.run().await {
                            console_error!("Failed to persist session_end message: {:?}", e);
                        }
                    }
                }
            }
        }

        // 4. Close all WebSockets
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
        let db = self.env.d1("DB").map_err(|e| format!("D1 binding: {:?}", e))?;
        let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

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
