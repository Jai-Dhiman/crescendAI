use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use crate::practice::analysis;
use crate::practice::dims::DIMS_6;
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
const OBSERVATION_THROTTLE_MS: u64 = 180_000; // 3 minutes
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

        // Store session info (on first connect only; reconnections keep existing state)
        {
            let mut s = self.inner.borrow_mut();
            if s.session_id.is_empty() {
                s.session_id = session_id.clone();
                s.student_id = student_id;
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

        // 9. Run score following + analysis
        let chunk_analysis: Option<analysis::ChunkAnalysis> = {
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
                        // Tier 1: full bar-aligned analysis
                        let score_ctx = self.inner.borrow().score_context.clone().unwrap();
                        Some(analysis::analyze_tier1(
                            bm,
                            &perf_notes,
                            &perf_pedal,
                            &scores_array,
                            &score_ctx,
                        ))
                    } else {
                        // Score context exists but alignment failed -> Tier 2
                        Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array))
                    }
                } else {
                    // No score context but have perf notes -> Tier 2
                    Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array))
                }
            } else {
                // No perf notes -> Tier 3 (no analysis, current behavior)
                None
            }
        };

        // 10. Run STOP classifier on current chunk
        let stop_result = stop::classify(&scores_array);

        // 11. Check if we should generate an observation
        let should_generate = {
            let s = self.inner.borrow();
            stop_result.triggered
                && s.baselines.is_some()
                && self.throttle_allows(&s)
        };

        if should_generate {
            self.generate_observation(ws, chunk_analysis.as_ref()).await;
        }

        // 12. Reset alarm
        let _ = self.state.storage().set_alarm(ALARM_DURATION_MS).await;

        Ok(())
    }

    fn throttle_allows(&self, s: &SessionState) -> bool {
        match s.last_observation_at {
            None => true,
            Some(last) => {
                let now = js_sys::Date::now() as u64;
                now - last >= OBSERVATION_THROTTLE_MS
            }
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

        let inner_req = crate::services::ask::AskInnerRequest {
            teaching_moment: tm_json,
            student_id: student_id.clone(),
            session_id: session_id.clone(),
            piece_context,
        };

        let inner_resp = crate::services::ask::handle_ask_inner(&self.env, &inner_req).await;

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
        let _ = ws.send_with_str(&obs_event.to_string());

        // Store in session state
        {
            let mut s = self.inner.borrow_mut();
            s.observations.push(ObservationRecord {
                id: crate::services::ask::generate_uuid(),
                text: inner_resp.observation_text,
                dimension: inner_resp.dimension,
                framing: inner_resp.framing,
                chunk_index: moment.chunk_index,
                score: moment.score,
                baseline: moment.baseline,
                reasoning_trace: inner_resp.reasoning_trace,
                is_fallback: inner_resp.is_fallback,
            });
            s.last_observation_at = Some(js_sys::Date::now() as u64);
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
            headers.set("Content-Type", "application/octet-stream").map_err(|e| format!("{:?}", e))?;
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

            let body: serde_json::Value = response.json().await
                .map_err(|e| format!("HF response parse failed: {:?}", e))?;

            // Validate that predictions contain all 6 dimensions
            let predictions = body.get("predictions").unwrap_or(&body);
            let dim_count = DIMS_6.iter().filter(|dim| predictions.get(*dim).and_then(|v| v.as_f64()).is_some()).count();
            if dim_count < 6 {
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

        // 2. Send session_summary via WebSocket
        if let Some(ws) = ws {
            let obs_json: Vec<serde_json::Value> = observations
                .iter()
                .map(|o| serde_json::json!({
                    "text": o.text,
                    "dimension": o.dimension,
                    "framing": o.framing,
                }))
                .collect();

            let summary = serde_json::json!({
                "type": "session_summary",
                "observations": obs_json,
                "summary": "",
                "inference_failures": inference_failures,
                "total_chunks": total_chunks,
            });
            let _ = ws.send_with_str(&summary.to_string());
        }

        // 3. Close all WebSockets
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
                    "INSERT INTO observations (id, student_id, session_id, chunk_index, \
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
