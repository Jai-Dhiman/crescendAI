pub mod accumulator;
pub mod error;
pub mod finalization;
pub mod inference;
pub mod practice_mode;
pub mod processing;
pub mod synthesis;

use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use self::accumulator::SessionAccumulator;
use self::practice_mode::ModeDetector;
use crate::practice::analysis::piece_identify::{NgramIndex, PieceIdentification, RerankFeatures};
use crate::practice::analysis::score_context::ScoreContext;
use crate::practice::analysis::score_follower::{FollowerState, PerfNote, PerfPedalEvent};
use crate::practice::teaching_moment::DimStats;
use crate::services::teaching_moments::{ScoredChunk, StudentBaselines};

// --- Response types for split MuQ / AMT endpoints ---

#[derive(serde::Deserialize)]
pub(crate) struct MuqResponse {
    pub(crate) predictions: HashMap<String, f64>,
    #[allow(dead_code)]
    processing_time_ms: Option<u64>,
}

#[derive(serde::Deserialize)]
pub(crate) struct AmtResponse {
    pub(crate) midi_notes: Vec<PerfNote>,
    pub(crate) pedal_events: Vec<PerfPedalEvent>,
    #[allow(dead_code)]
    pub(crate) transcription_info: Option<TranscriptionInfo>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct TranscriptionInfo {
    note_count: Option<u32>,
    pitch_range: Option<Vec<u8>>,
    pedal_event_count: Option<u32>,
    transcription_time_ms: Option<u64>,
    context_duration_s: Option<f64>,
    chunk_duration_s: Option<f64>,
}

/// Base64-encode bytes using standard alphabet (for AMT endpoint JSON payloads).
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};
    STANDARD.encode(bytes)
}

/// JS setTimeout-based sleep for Cloudflare Workers WASM
pub(crate) async fn sleep_ms(ms: u64) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .expect("setTimeout not found on global");
        let set_timeout_fn: js_sys::Function = set_timeout.into();
        let _ = set_timeout_fn.call2(&JsValue::NULL, &resolve, &JsValue::from(ms as f64));
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

pub(crate) const ALARM_DURATION_MS: i64 = 30 * 60 * 1000; // 30 minutes
pub(crate) const HF_RETRY_DELAYS_MS: &[u64] = &[10_000, 20_000, 40_000]; // retry 503s with backoff (70s total)
pub(crate) const HF_RETRY_DELAYS_ENDING_MS: &[u64] = &[3_000, 5_000]; // shorter retries when session is ending

pub(crate) struct SessionState {
    pub(crate) session_id: String,
    pub(crate) student_id: String,
    pub(crate) baselines: Option<StudentBaselines>,
    pub(crate) baselines_loaded: bool,
    pub(crate) scored_chunks: Vec<ScoredChunk>,
    pub(crate) accumulator: SessionAccumulator,
    pub(crate) inference_failures: usize,
    pub(crate) chunks_in_flight: usize,
    pub(crate) session_ending: bool,
    pub(crate) synthesis_completed: bool,
    pub(crate) dim_stats: DimStats,
    pub(crate) piece_query: Option<String>,
    pub(crate) score_context: Option<ScoreContext>,
    pub(crate) score_context_loaded: bool,
    pub(crate) follower_state: FollowerState,
    pub(crate) mode_detector: ModeDetector,
    pub(crate) conversation_id: Option<String>,
    /// Whether this session is an eval session (export accumulator state in synthesis).
    pub(crate) is_eval: bool,
    /// Encoded WebM bytes from previous chunk (NOT persisted to durable storage).
    /// Used to provide 30s context window for Aria-AMT.
    pub(crate) previous_chunk_audio: Option<Vec<u8>>,
    /// Accumulated AMT notes across chunks for piece fingerprinting.
    pub(crate) accumulated_notes: Vec<PerfNote>,
    /// Result of piece identification (fingerprint or text query).
    pub(crate) piece_identification: Option<PieceIdentification>,
    /// Whether piece identity is locked (confirmed via DTW or text query).
    pub(crate) piece_locked: bool,
    /// Running count of notes fed into identification attempts.
    pub(crate) identification_note_count: u32,
    /// Cached N-gram index from R2 (lazy-loaded on first identification attempt).
    pub(crate) ngram_index: Option<NgramIndex>,
    /// Cached rerank features from R2 (lazy-loaded on first identification attempt).
    pub(crate) rerank_features: Option<RerankFeatures>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            student_id: String::new(),
            baselines: None,
            baselines_loaded: false,
            scored_chunks: Vec::new(),
            accumulator: SessionAccumulator::default(),
            inference_failures: 0,
            chunks_in_flight: 0,
            session_ending: false,
            synthesis_completed: false,
            dim_stats: DimStats::default(),
            piece_query: None,
            score_context: None,
            score_context_loaded: false,
            follower_state: FollowerState::default(),
            mode_detector: ModeDetector::new(),
            conversation_id: None,
            is_eval: false,
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
    pub(crate) state: State,
    pub(crate) env: Env,
    pub(crate) inner: RefCell<SessionState>,
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
        let student_id = url
            .query_pairs()
            .find(|(k, _)| k == "student_id")
            .map(|(_, v)| v.to_string())
            .unwrap_or_default();

        let conversation_id = url
            .query_pairs()
            .find(|(k, _)| k == "conversation_id")
            .map(|(_, v)| v.to_string())
            .filter(|s| !s.is_empty());

        let is_eval = url
            .query_pairs()
            .find(|(k, _)| k == "eval")
            .map(|(_, v)| v == "true")
            .unwrap_or(false);

        console_log!(
            "DO fetch: session_id={}, student_id={}, conversation_id={:?}",
            session_id,
            student_id,
            conversation_id
        );

        // Store session info (on first connect only; reconnections keep existing state)
        {
            let mut s = self.inner.borrow_mut();
            if s.session_id.is_empty() {
                s.session_id = session_id.clone();
                s.student_id = student_id;
                s.conversation_id = conversation_id.clone();
                s.is_eval = is_eval;
            }
        }

        // Persist identity to durable storage (survives DO eviction during long async ops)
        let storage = self.state.storage();
        let _ = storage.put("session_id", &session_id).await;
        let _ = storage
            .put("student_id", &self.inner.borrow().student_id)
            .await;
        if let Some(ref conv_id) = conversation_id {
            let _ = storage.put("conversation_id", conv_id).await;
        }
        if is_eval {
            let _ = storage.put("is_eval", &true).await;
        }

        // Close any existing WebSocket connections (reconnection case)
        let existing_sockets = self.state.get_websockets();
        for old_ws in existing_sockets {
            let _ = old_ws.close(
                Some(1000),
                Some(String::from("New connection replacing old one")),
            );
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
                    console_log!("Last in-flight chunk completed, scheduling synthesis alarm");
                    let _ = self.state.storage().set_alarm(1).await;
                }

                result?;
            }
            "end_session" => {
                {
                    let mut s = self.inner.borrow_mut();
                    s.session_ending = true;
                }
                let _ = self.state.storage().put("session_ending", &true).await;

                let in_flight = self.inner.borrow().chunks_in_flight;
                if in_flight == 0 {
                    // Schedule immediate alarm for synthesis + finalization
                    let _ = self.state.storage().set_alarm(1).await;
                } else {
                    console_log!(
                        "end_session: waiting for {} in-flight chunks",
                        in_flight
                    );
                    // Last completing chunk will set the alarm
                }
            }
            "eval_chunk" => {
                // Only available in eval sessions (matching eval_context export gate)
                let is_eval = self.inner.borrow().is_eval;
                if !is_eval {
                    let _ = ws.send_with_str(
                        r#"{"type":"error","message":"eval_chunk only available in dev"}"#,
                    );
                    return Ok(());
                }

                let chunk_index = parsed
                    .get("chunk_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                // Build HF-response-shaped JSON from the eval payload
                let hf_response = serde_json::json!({
                    "predictions": parsed.get("predictions").cloned().unwrap_or_default(),
                    "midi_notes": parsed.get("midi_notes").cloned().unwrap_or_default(),
                    "pedal_events": parsed.get("pedal_events").cloned().unwrap_or_default(),
                });

                self.process_inference_result(&ws, chunk_index, hf_response)
                    .await?;
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
        // Client disconnected -- mark session as ending and schedule synthesis
        {
            let mut s = self.inner.borrow_mut();
            s.session_ending = true;
        }
        let _ = self.state.storage().put("session_ending", &true).await;
        // Always schedule alarm -- let alarm handler decide if there's content to synthesize.
        // Avoids checking scored_chunks which may be empty after hibernation.
        let _ = self.state.storage().set_alarm(1).await;
        Ok(())
    }

    async fn alarm(&self) -> Result<Response> {
        // Single synthesis + finalization point for all exit paths:
        // - end_session (0ms alarm), websocket_close (0ms alarm),
        // - last in-flight chunk (0ms alarm), inactivity timeout (30min alarm)
        self.ensure_session_state().await;

        let sockets = self.state.get_websockets();
        let ws = sockets.first();

        // Run synthesis (idempotent -- synthesis_completed guard prevents double-run)
        self.run_synthesis_and_persist(ws).await;
        // Finalize (idempotent -- finalized guard prevents double-run)
        self.finalize_session(ws).await;

        Response::ok("alarm handled")
    }
}

// --- Pipeline methods ---

impl PracticeSession {
    /// Reload session identity and accumulator from durable storage if in-memory
    /// state was lost (happens when the DO hibernates between WebSocket events).
    pub(crate) async fn ensure_session_state(&self) {
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

        if let Ok(Some(true)) = storage.get::<bool>("is_eval").await {
            let mut s = self.inner.borrow_mut();
            s.is_eval = true;
        }

        // Reload accumulator from durable storage
        if let Ok(Some(acc_json)) = storage.get::<String>("accumulator").await {
            if let Ok(acc) = serde_json::from_str::<SessionAccumulator>(&acc_json) {
                let mut s = self.inner.borrow_mut();
                s.accumulator = acc;
                console_log!(
                    "DO accumulator reloaded: {} moments, {} transitions",
                    s.accumulator.teaching_moments.len(),
                    s.accumulator.mode_transitions.len()
                );
            }
        }

        // Reload baselines
        if let Ok(Some(bl_json)) = storage.get::<String>("baselines").await {
            if let Ok(baselines) = serde_json::from_str::<StudentBaselines>(&bl_json) {
                let mut s = self.inner.borrow_mut();
                s.baselines = Some(baselines);
                s.baselines_loaded = true;
            }
        }

        // Reload scored_chunks
        if let Ok(Some(sc_json)) = storage.get::<String>("scored_chunks").await {
            if let Ok(sc) = serde_json::from_str::<Vec<ScoredChunk>>(&sc_json) {
                self.inner.borrow_mut().scored_chunks = sc;
            }
        }

        // Reload session_ending flag
        if let Ok(Some(true)) = storage.get::<bool>("session_ending").await {
            self.inner.borrow_mut().session_ending = true;
        }

        // Reload synthesis_completed flag (survives DO eviction)
        if let Ok(Some(true)) = storage.get::<bool>("synthesis_completed").await {
            self.inner.borrow_mut().synthesis_completed = true;
        }

        let state_debug = {
            let s = self.inner.borrow();
            format!(
                "session_id={}, student_id={}, conv_id={:?}",
                s.session_id, s.student_id, s.conversation_id
            )
        };
        console_log!("DO identity reloaded: {}", state_debug);
    }
}

// Chunk processing, observation generation, and inference result handling
// are in session_processing.rs (same `impl PracticeSession` block, split for readability).
