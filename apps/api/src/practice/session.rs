use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use crate::practice::accumulator::{SessionAccumulator, AccumulatedMoment, ModeTransitionRecord, TimelineEvent};
use crate::practice::analysis;
use crate::practice::dims::DIMS_6;
use crate::practice::practice_mode::{
    ChunkSignal, ModeDetector, ModeTransition, ObservationPolicy, PracticeMode,
    pitch_bigrams_from_notes,
};
use crate::practice::piece_identify::{
    DTW_CONFIRM_THRESHOLD, NgramIndex, PieceIdentification, RerankFeatures,
};
use crate::practice::score_context::ScoreContext;
use crate::practice::score_follower::{FollowerState, PerfNote, PerfPedalEvent};
use crate::practice::teaching_moment::DimStats;
use crate::services::stop;
use crate::services::teaching_moments::{
    RecentObservation, ScoredChunk, StudentBaselines,
};

// --- Response types for split MuQ / AMT endpoints ---

#[derive(serde::Deserialize)]
struct MuqResponse {
    predictions: HashMap<String, f64>,
    #[allow(dead_code)]
    processing_time_ms: Option<u64>,
}

#[derive(serde::Deserialize)]
struct AmtResponse {
    midi_notes: Vec<PerfNote>,
    pedal_events: Vec<PerfPedalEvent>,
    #[allow(dead_code)]
    transcription_info: Option<TranscriptionInfo>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct TranscriptionInfo {
    note_count: Option<u32>,
    pitch_range: Option<Vec<u8>>,
    pedal_event_count: Option<u32>,
    transcription_time_ms: Option<u64>,
    context_duration_s: Option<f64>,
    chunk_duration_s: Option<f64>,
}

/// Base64-encode bytes using standard alphabet (for AMT endpoint JSON payloads).
fn base64_encode(bytes: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};
    STANDARD.encode(bytes)
}

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
    accumulator: SessionAccumulator,
    inference_failures: usize,
    chunks_in_flight: usize,
    session_ending: bool,
    synthesis_completed: bool,
    dim_stats: DimStats,
    last_observation_at: Option<u64>,
    piece_query: Option<String>,
    score_context: Option<ScoreContext>,
    score_context_loaded: bool,
    follower_state: FollowerState,
    is_eval_session: bool,
    mode_detector: ModeDetector,
    conversation_id: Option<String>,
    /// Encoded WebM bytes from previous chunk (NOT persisted to durable storage).
    /// Used to provide 30s context window for Aria-AMT.
    previous_chunk_audio: Option<Vec<u8>>,
    /// Accumulated AMT notes across chunks for piece fingerprinting.
    accumulated_notes: Vec<PerfNote>,
    /// Result of piece identification (fingerprint or text query).
    piece_identification: Option<PieceIdentification>,
    /// Whether piece identity is locked (confirmed via DTW or text query).
    piece_locked: bool,
    /// Running count of notes fed into identification attempts.
    identification_note_count: u32,
    /// Cached N-gram index from R2 (lazy-loaded on first identification attempt).
    ngram_index: Option<NgramIndex>,
    /// Cached rerank features from R2 (lazy-loaded on first identification attempt).
    rerank_features: Option<RerankFeatures>,
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
            accumulator: SessionAccumulator::default(),
            inference_failures: 0,
            chunks_in_flight: 0,
            session_ending: false,
            synthesis_completed: false,
            dim_stats: DimStats::default(),
            last_observation_at: None,
            piece_query: None,
            score_context: None,
            score_context_loaded: false,
            follower_state: FollowerState::default(),
            is_eval_session: false,
            mode_detector: ModeDetector::new(),
            conversation_id: None,
            previous_chunk_audio: None,
            accumulated_notes: Vec::new(),
            piece_identification: None,
            piece_locked: false,
            identification_note_count: 0,
            ngram_index: None,
            rerank_features: None,
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
                s.conversation_id = conversation_id.clone();
            }
        }

        // Persist identity to durable storage (survives DO eviction during long async ops)
        let storage = self.state.storage();
        let _ = storage.put("session_id", &session_id).await;
        let _ = storage.put("student_id", &self.inner.borrow().student_id).await;
        if let Some(ref conv_id) = conversation_id {
            let _ = storage.put("conversation_id", conv_id).await;
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
                    if !self.inner.borrow().is_eval_session {
                        self.run_synthesis_and_persist(&ws).await;
                    }
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
                    // Synthesize before finalizing (keeps WS open for synthesis push)
                    if !self.inner.borrow().is_eval_session {
                        self.run_synthesis_and_persist(&ws).await;
                    }
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
                    // Converge text query path with fingerprint path
                    s.piece_identification = Some(PieceIdentification {
                        piece_id: String::new(), // resolved during score_context load
                        confidence: 1.0,
                        method: "text_query".to_string(),
                    });
                    s.piece_locked = true;
                    s.accumulated_notes.clear();
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
    /// Reload session identity and accumulator from durable storage if in-memory
    /// state was lost (happens when the DO is evicted during long async operations
    /// like LLM calls).
    async fn ensure_session_state(&self) {
        let needs_reload = self.inner.borrow().session_id.is_empty();
        if !needs_reload {
            return;
        }

        console_log!("DO state was evicted, reloading identity from storage");
        let storage = self.state.storage();

        if let Ok(Some(sid)) = storage.get::<String>("session_id").await {
            let mut s = self.inner.borrow_mut();
            s.session_id = sid;
        }
        if let Ok(Some(uid)) = storage.get::<String>("student_id").await {
            let mut s = self.inner.borrow_mut();
            s.student_id = uid;
        }
        if let Ok(Some(cid)) = storage.get::<String>("conversation_id").await {
            let mut s = self.inner.borrow_mut();
            s.conversation_id = Some(cid);
        }

        // Reload accumulator from durable storage
        if let Ok(Some(acc_json)) = storage.get::<String>("accumulator").await {
            if let Ok(acc) = serde_json::from_str::<SessionAccumulator>(&acc_json) {
                let mut s = self.inner.borrow_mut();
                s.accumulator = acc;
                console_log!("DO accumulator reloaded: {} moments, {} transitions",
                    s.accumulator.teaching_moments.len(),
                    s.accumulator.mode_transitions.len());
            }
        }

        let state_debug = {
            let s = self.inner.borrow();
            format!("session_id={}, student_id={}, conv_id={:?}", s.session_id, s.student_id, s.conversation_id)
        };
        console_log!("DO identity reloaded: {}", state_debug);
    }

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

        // 2. Dispatch MuQ and AMT in parallel
        let context_audio = self.inner.borrow().previous_chunk_audio.clone();
        let (muq_result, amt_result) = futures_util::future::join(
            self.call_muq_endpoint(&audio_bytes),
            self.call_amt_endpoint(context_audio.as_deref(), &audio_bytes),
        ).await;

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
                console_log!("AMT returned {} notes, {} pedal events for chunk {}",
                    amt.midi_notes.len(), amt.pedal_events.len(), index);
                (amt.midi_notes, amt.pedal_events)
            }
            Err(e) => {
                console_error!("AMT inference failed for chunk {} (proceeding with Tier 3): {}", index, e);
                (Vec::new(), Vec::new())
            }
        };

        // 4b. Accumulate notes and run piece identification (if not yet locked)
        if !perf_notes.is_empty() {
            self.inner.borrow_mut().accumulated_notes.extend(perf_notes.iter().cloned());
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

        // 7. Run score following + analysis using AMT result
        let (chunk_analysis, chunk_bar_range) = self.process_amt_result(
            index, &perf_notes, &perf_pedal, &scores_array,
        );

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

        // 9. Accumulate or generate observation (eval uses old path, production accumulates)
        if self.inner.borrow().is_eval_session {
            // Eval sessions preserve the per-chunk observation path
            self.try_generate_observation(ws, chunk_analysis.as_ref(), &scores_array).await;
        } else {
            // Production: accumulate signals silently
            let timestamp_ms = js_sys::Date::now() as u64;
            let has_audio = !perf_notes.is_empty();

            // Timeline event
            self.inner.borrow_mut().accumulator.accumulate_timeline_event(TimelineEvent {
                chunk_index: index,
                timestamp_ms,
                has_audio,
            });

            // Mode transitions
            for transition in &mode_transitions {
                let from_mode = {
                    let s = self.inner.borrow();
                    s.accumulator.mode_transitions.last()
                        .map(|t| t.to)
                        .unwrap_or(PracticeMode::Warming)
                };
                let dwell_ms = {
                    let s = self.inner.borrow();
                    let mc = s.mode_detector.mode_context();
                    timestamp_ms.saturating_sub(mc.entered_at_ms)
                };
                self.inner.borrow_mut().accumulator.accumulate_mode_transition(ModeTransitionRecord {
                    from: from_mode,
                    to: transition.mode,
                    chunk_index: transition.chunk_index,
                    timestamp_ms,
                    dwell_ms,
                });
            }

            // Teaching moment accumulation (STOP-gated)
            let stop_result = stop::classify(&scores_array);
            if stop_result.triggered {
                if let Some(baselines) = self.inner.borrow().baselines.clone() {
                    let recent_obs: Vec<RecentObservation> = self.inner.borrow()
                        .accumulator.teaching_moments.iter().rev().take(3)
                        .map(|m| RecentObservation { dimension: m.dimension.clone() })
                        .collect();
                    let scored_chunks = self.inner.borrow().scored_chunks.clone();

                    if let Some(moment) = crate::services::teaching_moments::select_teaching_moment(
                        &scored_chunks, &baselines, &recent_obs,
                    ) {
                        let bar_range = chunk_bar_range;
                        let tier = chunk_analysis.as_ref().map(|ca| ca.tier).unwrap_or(3);

                        self.inner.borrow_mut().accumulator.accumulate_moment(AccumulatedMoment {
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
            let acc_json = serde_json::to_string(&self.inner.borrow().accumulator).unwrap_or_default();
            let _ = self.state.storage().put("accumulator", &acc_json).await;
        }

        // 10. Reset alarm
        let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;

        Ok(())
    }

    /// Extract 6-dim scores from MuQ response. Returns (scores_array, scores_map).
    fn process_muq_result(&self, muq: &MuqResponse) -> ([f64; 6], HashMap<String, f64>) {
        let scores_map: HashMap<String, f64> = DIMS_6
            .iter()
            .filter_map(|&dim| {
                muq.predictions.get(dim).copied().map(|val| (dim.to_string(), val))
            })
            .collect();
        let scores_array: [f64; 6] = DIMS_6.map(|dim| {
            scores_map.get(dim).copied().unwrap_or(0.0)
        });
        (scores_array, scores_map)
    }

    /// Run score following and bar-aligned analysis from AMT notes.
    /// Returns (chunk_analysis, bar_range) -- both None if no perf notes.
    fn process_amt_result(
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
                    (Some(analysis::analyze_tier2(perf_notes, perf_pedal, scores_array)), None)
                }
            } else {
                (Some(analysis::analyze_tier2(perf_notes, perf_pedal, scores_array)), None)
            }
        } else {
            // No perf notes -> Tier 3 (scores only)
            (None, None)
        }
    }

    /// Attempt piece identification from accumulated AMT notes.
    ///
    /// Runs N-gram recall + rerank (Stage 1+2), then DTW confirmation (Stage 3)
    /// against the top candidate's score data. If confirmed, locks in the piece and
    /// loads the full ScoreContext for subsequent score following.
    async fn try_identify_piece(&self, ws: &WebSocket) {
        // Check if already locked or no notes to identify
        let (piece_locked, note_count) = {
            let s = self.inner.borrow();
            (s.piece_locked, s.accumulated_notes.len())
        };

        if piece_locked {
            return;
        }

        // Too many notes without a match -> give up
        if note_count > 200 {
            let student_id = self.inner.borrow().student_id.clone();
            console_log!(
                "piece_id_attempt: notes={} exceeded 200, giving up on identification",
                note_count
            );
            self.inner.borrow_mut().piece_locked = true;
            // Log the failed attempt
            crate::practice::score_context::log_fingerprint_piece_request(
                &self.env,
                &student_id,
                "",
                0.0,
                "exhausted",
            ).await;
            return;
        }

        // Lazy-load N-gram index from R2 (cached in session state)
        {
            let needs_index = self.inner.borrow().ngram_index.is_none();
            if needs_index {
                match crate::practice::score_context::load_ngram_index(&self.env).await {
                    Ok(index) => {
                        self.inner.borrow_mut().ngram_index = Some(index);
                    }
                    Err(e) => {
                        console_error!("Failed to load N-gram index: {}", e);
                        return;
                    }
                }
            }
        }
        {
            let needs_features = self.inner.borrow().rerank_features.is_none();
            if needs_features {
                match crate::practice::score_context::load_rerank_features(&self.env).await {
                    Ok(features) => {
                        self.inner.borrow_mut().rerank_features = Some(features);
                    }
                    Err(e) => {
                        console_error!("Failed to load rerank features: {}", e);
                        return;
                    }
                }
            }
        }

        // Clone what we need before calling identify_piece (no borrows across await)
        let (notes, ngram_index, rerank_features) = {
            let s = self.inner.borrow();
            (
                s.accumulated_notes.clone(),
                s.ngram_index.clone().unwrap(),
                s.rerank_features.clone().unwrap(),
            )
        };

        // Stage 1+2: N-gram recall + rerank
        let identification = crate::practice::piece_identify::identify_piece(
            &notes,
            &ngram_index,
            &rerank_features,
        );

        let candidate = match identification {
            Some(id) => id,
            None => {
                console_log!(
                    "piece_id_attempt: notes={} candidates=0 top_piece=none top_score=0.000 locked=false",
                    notes.len()
                );
                self.inner.borrow_mut().identification_note_count = notes.len() as u32;
                return;
            }
        };

        console_log!(
            "piece_id_attempt: notes={} top_piece={} top_score={:.3} method={} locked=false",
            notes.len(),
            candidate.piece_id,
            candidate.confidence,
            candidate.method
        );

        // Stage 3: DTW confirmation -- load the candidate's score and align
        let score_data = match crate::practice::score_context::load_score(
            &self.env,
            &candidate.piece_id,
        ).await {
            Ok(s) => s,
            Err(e) => {
                console_error!(
                    "Failed to load score for DTW confirmation of {}: {}",
                    candidate.piece_id, e
                );
                return;
            }
        };

        let mut dtw_state = FollowerState::default();
        let bar_map = crate::practice::score_follower::align_chunk(
            0,
            0.0,
            &notes,
            &score_data,
            &mut dtw_state,
        );

        let dtw_cost = bar_map.as_ref().map(|bm| 1.0 / bm.confidence - 1.0);
        let dtw_confirmed = dtw_cost.map(|c| c < DTW_CONFIRM_THRESHOLD).unwrap_or(false);

        console_log!(
            "piece_id_dtw: piece={} cost={:.3} threshold={:.3} confirmed={}",
            candidate.piece_id,
            dtw_cost.unwrap_or(f64::MAX),
            DTW_CONFIRM_THRESHOLD,
            dtw_confirmed
        );

        if !dtw_confirmed {
            self.inner.borrow_mut().identification_note_count = notes.len() as u32;
            return;
        }

        // DTW confirmed -- lock in piece and load full ScoreContext
        let reference = crate::practice::score_context::load_reference(
            &self.env,
            &candidate.piece_id,
        ).await;

        // Look up composer/title from the score data
        let composer = score_data.composer.clone();
        let title = score_data.title.clone();

        {
            let mut s = self.inner.borrow_mut();
            s.piece_identification = Some(candidate.clone());
            s.piece_locked = true;
            s.score_context = Some(ScoreContext {
                piece_id: candidate.piece_id.clone(),
                composer: composer.clone(),
                title: title.clone(),
                score: score_data,
                reference,
                match_confidence: candidate.confidence,
            });
            s.score_context_loaded = true;
            s.follower_state = FollowerState::default();
        }

        console_log!(
            "piece_id_locked: piece={} ({} - {}) confidence={:.3} method={}",
            candidate.piece_id, composer, title, candidate.confidence, candidate.method
        );

        // Notify client
        let msg = serde_json::json!({
            "type": "piece_identified",
            "pieceId": candidate.piece_id,
            "composer": composer,
            "title": title,
            "confidence": candidate.confidence,
            "method": candidate.method,
        });
        let _ = ws.send_with_str(&msg.to_string());

        // Log to piece_requests for demand tracking
        let student_id = self.inner.borrow().student_id.clone();
        crate::practice::score_context::log_fingerprint_piece_request(
            &self.env,
            &student_id,
            &candidate.piece_id,
            candidate.confidence,
            &candidate.method,
        ).await;
    }

    /// Check observation gate (STOP + mode throttle) and generate if ready.
    async fn try_generate_observation(
        &self,
        ws: &WebSocket,
        chunk_analysis: Option<&analysis::ChunkAnalysis>,
        scores_array: &[f64; 6],
    ) {
        let stop_result = stop::classify(scores_array);

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
            self.generate_observation(ws, chunk_analysis).await;
        }
    }

    /// Legacy entry point for eval_chunk messages (combined MuQ+AMT response).
    async fn process_inference_result(
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
                predictions.get(dim).and_then(|v| v.as_f64()).map(|val| (dim.to_string(), val))
            })
            .collect();
        let scores_array: [f64; 6] = DIMS_6.map(|dim| {
            scores_map.get(dim).copied().unwrap_or(0.0)
        });

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

        // Score following + analysis
        let (chunk_analysis, chunk_bar_range) = self.process_amt_result(
            index, &perf_notes, &perf_pedal, &scores_array,
        );

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

        // Observation generation
        self.try_generate_observation(ws, chunk_analysis.as_ref(), &scores_array).await;

        // Reset alarm
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
        self.ensure_session_state().await;
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
        let session_state_debug = {
            let s = self.inner.borrow();
            format!("session_id={}, conv_id={:?}, obs_count={}", s.session_id, s.conversation_id, s.observations.len())
        };
        console_log!("Persisting observation message: state=[{}], obs_id={}", session_state_debug, obs_id);
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

    /// Call the MuQ-only endpoint. Sends raw WebM audio bytes, returns 6-dim predictions.
    async fn call_muq_endpoint(
        &self,
        audio_bytes: &[u8],
    ) -> std::result::Result<MuqResponse, String> {
        // MuQ endpoint uses the same transport as the existing HF inference endpoint:
        // POST raw audio bytes with Content-Type: audio/webm;codecs=opus
        let endpoint = self.env.var("HF_INFERENCE_ENDPOINT")
            .map_err(|e| format!("HF_INFERENCE_ENDPOINT not set: {:?}", e))?
            .to_string();
        let token = self.env.secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers.set("Content-Type", "audio/webm;codecs=opus").map_err(|e| format!("{:?}", e))?;
            headers.set("Authorization", &format!("Bearer {}", token)).map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from(js_sys::Uint8Array::from(audio_bytes))));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| format!("MuQ request creation failed: {:?}", e))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("MuQ fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!("MuQ fetch failed (attempt {}), retrying in {}s: {}", attempt + 1, delay / 1000, last_err);
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("MuQ returned {}: {}", status, body);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    console_log!("MuQ {} (attempt {}), retrying in {}s", status, attempt + 1, delay / 1000);
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("MuQ returned {}: {}", status, body));
            }

            let body_text = response.text().await
                .map_err(|e| format!("MuQ response read failed: {:?}", e))?;

            let muq: MuqResponse = serde_json::from_str(&body_text)
                .map_err(|e| format!("MuQ response parse failed: {:?} - body: {}", e, &body_text[..body_text.len().min(200)]))?;

            // Validate all 6 dimensions present
            let dim_count = DIMS_6.iter().filter(|dim| muq.predictions.contains_key(**dim)).count();
            if dim_count < 6 {
                return Err(format!("MuQ returned only {} dimensions", dim_count));
            }

            if attempt > 0 {
                console_log!("MuQ inference succeeded after {} retries", attempt);
            }

            return Ok(muq);
        }

        Err(last_err)
    }

    /// Call the Aria-AMT endpoint. Sends JSON with base64-encoded audio fields.
    /// Returns transcribed MIDI notes and pedal events.
    async fn call_amt_endpoint(
        &self,
        context_audio: Option<&[u8]>,
        chunk_audio: &[u8],
    ) -> std::result::Result<AmtResponse, String> {
        let endpoint = self.env.var("HF_AMT_ENDPOINT")
            .map_err(|e| format!("HF_AMT_ENDPOINT not set: {:?}", e))?
            .to_string();

        if endpoint.is_empty() {
            return Err("HF_AMT_ENDPOINT is empty (not yet deployed)".to_string());
        }

        let token = self.env.secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        // Build JSON payload with base64-encoded audio
        let chunk_b64 = base64_encode(chunk_audio);
        let context_b64 = context_audio.map(base64_encode);

        let payload = serde_json::json!({
            "chunk_audio": chunk_b64,
            "context_audio": context_b64,
        });
        let payload_str = payload.to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers.set("Content-Type", "application/json").map_err(|e| format!("{:?}", e))?;
            headers.set("Authorization", &format!("Bearer {}", token)).map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from_str(&payload_str)));

            let request = worker::Request::new_with_init(&endpoint, &init)
                .map_err(|e| format!("AMT request creation failed: {:?}", e))?;

            let mut response = match worker::Fetch::Request(request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("AMT fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!("AMT fetch failed (attempt {}), retrying in {}s: {}", attempt + 1, delay / 1000, last_err);
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("AMT returned {}: {}", status, body);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    console_log!("AMT {} (attempt {}), retrying in {}s", status, attempt + 1, delay / 1000);
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("AMT returned {}: {}", status, body));
            }

            let body_text = response.text().await
                .map_err(|e| format!("AMT response read failed: {:?}", e))?;

            let amt: AmtResponse = serde_json::from_str(&body_text)
                .map_err(|e| format!("AMT response parse failed: {:?} - body: {}", e, &body_text[..body_text.len().min(200)]))?;

            if attempt > 0 {
                console_log!("AMT inference succeeded after {} retries", attempt);
            }

            return Ok(amt);
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

        // 3b. Collect chunk scores for context when no observations exist
        let chunk_scores: Vec<([f64; 6], usize)> = {
            let s = self.inner.borrow();
            s.scored_chunks.iter().map(|c| {
                // Count notes from the HF response (approximate via chunk index)
                (c.scores, 0usize)  // note_count not tracked in ScoredChunk, use 0
            }).collect()
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

    // --- Session synthesis ---

    async fn run_synthesis_and_persist(&self, ws: &WebSocket) {
        self.ensure_session_state().await;

        let (acc, ctx) = {
            let s = self.inner.borrow();
            if !s.accumulator.has_teaching_content() && s.accumulator.timeline.iter().all(|t| !t.has_audio) {
                console_log!("No teaching content and no audio detected, skipping synthesis");
                return;
            }

            let session_duration_ms = s.accumulator.timeline.last()
                .map(|t| t.timestamp_ms)
                .unwrap_or(0)
                .saturating_sub(s.accumulator.timeline.first().map(|t| t.timestamp_ms).unwrap_or(0));

            let ctx = crate::practice::synthesis::SynthesisContext {
                session_id: s.session_id.clone(),
                student_id: s.student_id.clone(),
                conversation_id: s.conversation_id.clone().unwrap_or_default(),
                baselines: s.baselines.clone(),
                piece_context: s.score_context.as_ref().map(|sc| serde_json::json!({
                    "composer": sc.composer,
                    "title": sc.title,
                    "piece_id": sc.piece_id,
                })),
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

        console_log!("Starting synthesis: moments={}, transitions={}, drilling={}",
            acc.teaching_moments.len(), acc.mode_transitions.len(), acc.drilling_records.len());

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
            &self.env, &ctx.conversation_id, &ctx.session_id, &result.text
        ).await {
            Ok(msg_id) => console_log!("Synthesis message persisted: {}", msg_id),
            Err(e) => console_error!("Failed to persist synthesis message: {}", e),
        }

        // Persist accumulated moments to observations table
        if let Err(e) = crate::practice::synthesis::persist_accumulated_moments(
            &self.env, &ctx.student_id, &ctx.session_id, &acc.teaching_moments
        ).await {
            console_error!("Failed to persist accumulated moments: {}", e);
        }

        self.inner.borrow_mut().synthesis_completed = true;
    }

    // --- Session finalization ---

    async fn finalize_session(&self, ws: Option<&WebSocket>) {
        self.ensure_session_state().await;

        let is_eval = self.inner.borrow().is_eval_session;

        if is_eval {
            // Eval path: preserve old observation-based flow
            let (observations, session_id, student_id, inference_failures, total_chunks) = {
                let s = self.inner.borrow();
                (s.observations.clone(), s.session_id.clone(), s.student_id.clone(),
                 s.inference_failures, s.scored_chunks.len() + s.inference_failures)
            };

            if !observations.is_empty() {
                if let Err(e) = self.persist_observations(&student_id, &session_id, &observations).await {
                    console_error!("Failed to persist observations: {}", e);
                }
            }

            // Send session_summary for eval
            if let Some(ws) = ws {
                let obs_json: Vec<serde_json::Value> = observations.iter()
                    .map(|o| serde_json::json!({"text": o.text, "dimension": o.dimension, "framing": o.framing}))
                    .collect();
                let summary_text = self.generate_session_summary(&observations, &student_id).await.unwrap_or_default();
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
                (s.accumulator.teaching_moments.len(), s.session_id.clone(), s.student_id.clone())
            };

            // Update observation count for memory synthesis
            if moment_count > 0 {
                if let Err(e) = crate::services::memory::increment_observation_count_by(
                    &self.env, &student_id, moment_count
                ).await {
                    console_error!("Failed to increment observation count: {}", e);
                }

                match crate::services::memory::should_synthesize(&self.env, &student_id).await {
                    Ok(true) => {
                        match crate::services::memory::run_synthesis(&self.env, &student_id).await {
                            Ok(result) => console_log!("Memory synthesis for {}: {} new, {} invalidated, {} unchanged",
                                student_id, result.new_facts, result.invalidated, result.unchanged),
                            Err(e) => console_error!("Memory synthesis failed for {}: {}", student_id, e),
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
                    (!s.synthesis_completed && s.accumulator.has_teaching_content(),
                     s.accumulator.clone(), s.session_id.clone(), s.conversation_id.clone())
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

            // Persist session_end message
            let conv_id = self.inner.borrow().conversation_id.clone();
            if let Some(ref conv_id) = conv_id {
                if let Ok(db) = self.env.d1("DB") {
                    let end_msg_id = crate::services::ask::generate_uuid();
                    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
                    if let Ok(q) = db.prepare(
                        "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
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
