# DO Practice Session Orchestration -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the PracticeSession Durable Object to orchestrate the full inference-to-observation pipeline: chunk_ready -> R2 fetch -> HF inference -> STOP classifier -> teaching moment selection -> LLM observation -> WebSocket push.

**Architecture:** The DO owns the full pipeline. On `chunk_ready`, it fetches audio from R2, calls the HF inference endpoint, runs the STOP classifier, and (if triggered) calls `handle_ask_inner()` directly as a Rust function for LLM observation generation. D1 persistence happens in `finalize_session()` on `end_session` or alarm timeout. Auth is validated by the Worker before routing to the DO; `student_id` is passed as a query param on the internal DO fetch.

**Tech Stack:** Rust (Cloudflare Workers WASM), Durable Objects, D1 (SQLite), R2 (object storage), HuggingFace Inference Endpoint

**Spec:** `docs/superpowers/specs/2026-03-15-do-practice-session-design.md`

**Scope:** `apps/api/` only. Do NOT touch `apps/web/`, `apps/ios/`, or `model/`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `apps/api/src/services/ask.rs` | Modify | Extract `handle_ask_inner()` from `handle_ask()` -- pure LLM function, no D1 persistence |
| `apps/api/src/server.rs` | Modify | Add JWT validation on WebSocket upgrade path, pass `student_id` to DO |
| `apps/api/src/practice/session.rs` | Major rewrite | Full pipeline orchestration, HF inference, STOP, teaching moments, alarm, finalize |

**Not modified:** `stop.rs`, `teaching_moments.rs`, `teaching_moment_handler.rs`, `dims.rs`, `upload.rs`, `start.rs` -- all already correct.

---

## Chunk 1: Extract handle_ask_inner + WebSocket Auth

### Task 1: Extract handle_ask_inner from ask.rs

**Files:**
- Modify: `apps/api/src/services/ask.rs`

**Context:** The existing `handle_ask()` (lines 64-299) does auth, request parsing, memory context, subagent, teacher LLM, D1 persistence, and HTTP response building. We need to extract the core LLM logic into `handle_ask_inner()` that the DO can call directly without HTTP or auth.

- [ ] **Step 1: Define AskInnerRequest and AskInnerResponse types**

Add after the existing `ElaborateResponse` struct (line 38):

```rust
/// Input for the core LLM pipeline (used by both HTTP handler and DO).
pub struct AskInnerRequest {
    pub teaching_moment: serde_json::Value,
    pub student_id: String,
    pub session_id: String,
    pub piece_context: Option<serde_json::Value>,
}

/// Output from the core LLM pipeline (no D1 side effects).
pub struct AskInnerResponse {
    pub observation_text: String,
    pub dimension: String,
    pub framing: String,
    pub reasoning_trace: String,
    pub is_fallback: bool,
}
```

- [ ] **Step 2: Extract handle_ask_inner function**

Add a new public async function that contains the LLM pipeline logic currently in `handle_ask()` lines 91-218. This function:
- Extracts dimension/scores from the teaching_moment JSON
- Builds memory context (queries D1)
- Runs Stage 1 (Groq subagent)
- Runs Stage 2 (Anthropic teacher)
- Returns `AskInnerResponse` (no D1 writes, no HTTP response)

```rust
/// Core two-stage LLM pipeline. No D1 persistence, no HTTP.
/// Called by both the HTTP handler (handle_ask) and the DO (PracticeSession).
pub async fn handle_ask_inner(
    env: &Env,
    req: &AskInnerRequest,
) -> AskInnerResponse {
    let dimension = req.teaching_moment
        .get("dimension")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Build memory context
    let piece_title = req.piece_context
        .as_ref()
        .and_then(|pc| pc.get("title"))
        .and_then(|v| v.as_str());

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    let today = &now[..10.min(now.len())];
    let memory_ctx = crate::services::memory::build_memory_context(
        env, &req.student_id, piece_title, today, piece_title,
    ).await;
    let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

    let recent_observations: Vec<prompts::ObservationRow> = memory_ctx
        .recent_observations
        .iter()
        .map(|obs| prompts::ObservationRow {
            dimension: obs.dimension.clone(),
            observation_text: obs.observation_text.clone(),
            framing: obs.framing.clone(),
            created_at: obs.created_at.clone(),
        })
        .collect();

    // Build student/session context for subagent prompt
    // The DO doesn't have the full student object, so build a minimal one
    let student_json = serde_json::json!({
        "level": "intermediate",
        "goals": "",
    });
    let session_json = serde_json::json!({
        "id": req.session_id,
    });

    // Stage 1: Subagent (Groq)
    let subagent_user_prompt = prompts::build_subagent_user_prompt(
        &req.teaching_moment,
        &student_json,
        &session_json,
        &req.piece_context,
        &recent_observations,
        &memory_text,
    );

    let subagent_result = llm::call_groq(
        env,
        prompts::SUBAGENT_SYSTEM,
        &subagent_user_prompt,
        0.3,
        800,
    ).await;

    let (subagent_output, framing) = match subagent_result {
        Ok(output) => {
            let framing = extract_framing(&output).unwrap_or_else(|| "correction".to_string());
            (output, framing)
        }
        Err(e) => {
            console_error!("Subagent failed: {}", e);
            return AskInnerResponse {
                observation_text: fallback_observation(&dimension),
                dimension,
                framing: "correction".to_string(),
                reasoning_trace: "{}".to_string(),
                is_fallback: true,
            };
        }
    };

    let (subagent_json, subagent_narrative) = split_subagent_output(&subagent_output);

    // Stage 2: Teacher (Anthropic)
    let teacher_user_prompt = prompts::build_teacher_user_prompt(
        &subagent_json,
        &subagent_narrative,
        "intermediate",
        "",
    );

    let teacher_result = llm::call_anthropic(
        env,
        prompts::TEACHER_SYSTEM,
        &teacher_user_prompt,
        300,
    ).await;

    let (observation_text, is_fallback) = match teacher_result {
        Ok(text) => (post_process_observation(&text), false),
        Err(e) => {
            console_error!("Teacher LLM failed: {}", e);
            (fallback_observation(&dimension), true)
        }
    };

    let reasoning_trace = serde_json::json!({
        "subagent_output": subagent_output,
    }).to_string();

    AskInnerResponse {
        observation_text,
        dimension,
        framing,
        reasoning_trace,
        is_fallback,
    }
}
```

- [ ] **Step 3: Refactor handle_ask to call handle_ask_inner**

Replace lines 91-218 of the existing `handle_ask()` with a call to `handle_ask_inner()`, then handle D1 persistence and HTTP response building in `handle_ask()` as before. The key change:

```rust
// In handle_ask(), after parsing the request and validating auth:
let inner_req = AskInnerRequest {
    teaching_moment: request.teaching_moment.clone(),
    student_id: student_id.clone(),
    session_id: request.session.get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string(),
    piece_context: request.piece_context.clone(),
};

let inner_resp = handle_ask_inner(env, &inner_req).await;

// Then use inner_resp.observation_text, .dimension, .framing, etc.
// for D1 storage and HTTP response building (same as before)
```

- [ ] **Step 4: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | grep -E "error|warning.*ask"`
Expected: No new errors. Existing warnings acceptable.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/ask.rs
git commit -m "extract handle_ask_inner for direct DO calls

Pure LLM function: builds memory context, runs Groq subagent +
Anthropic teacher, returns observation text. No D1 persistence,
no HTTP response building. Called by both handle_ask (HTTP) and
PracticeSession DO."
```

---

### Task 2: Add auth validation to WebSocket path

**Files:**
- Modify: `apps/api/src/server.rs`

**Context:** The WebSocket upgrade path (lines 70-79) routes directly to the DO without validating auth. We need to validate the JWT (from cookie or Bearer header) and pass the `student_id` to the DO as a query parameter.

The challenge: `verify_auth()` returns `Result<String, http::Response<axum::body::Body>>` but the WS path returns `worker::Response`. We need to handle the error type conversion.

- [ ] **Step 1: Add auth validation to WS routing**

Replace lines 70-79 in `server.rs`:

```rust
    if path.starts_with("/api/practice/ws/") && method == http::Method::GET {
        let session_id = path.trim_start_matches("/api/practice/ws/");
        if !session_id.is_empty() && !session_id.contains('/') {
            let namespace = env.durable_object("PRACTICE_SESSION")?;
            let stub = namespace.id_from_name(session_id)?.get_stub()?;
            let url = format!("https://do.internal/ws/{}", session_id);
            let mut worker_req = worker::Request::new(&url, worker::Method::Get)?;
            worker_req.headers_mut()?.set("Upgrade", "websocket")?;
            return stub.fetch_with_request(worker_req).await;
        }
    }
```

With:

```rust
    if path.starts_with("/api/practice/ws/") && method == http::Method::GET {
        let session_id = path.trim_start_matches("/api/practice/ws/");
        if !session_id.is_empty() && !session_id.contains('/') {
            // Validate auth before routing to DO
            let student_id = match crate::auth::verify_auth(req.headers(), &env) {
                Ok(id) => id,
                Err(_) => {
                    return worker::Response::error("Unauthorized", 401);
                }
            };

            let namespace = env.durable_object("PRACTICE_SESSION")?;
            let stub = namespace.id_from_name(session_id)?.get_stub()?;
            let url = format!(
                "https://do.internal/ws/{}?student_id={}",
                session_id, student_id
            );
            let mut worker_req = worker::Request::new(&url, worker::Method::Get)?;
            worker_req.headers_mut()?.set("Upgrade", "websocket")?;
            return stub.fetch_with_request(worker_req).await;
        }
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | grep "error"`
Expected: No new errors.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/server.rs
git commit -m "add JWT auth validation to WebSocket upgrade path

Validates cookie/Bearer JWT before routing to PracticeSession DO.
Passes student_id as query param on internal DO fetch URL."
```

---

## Chunk 2: Rewrite PracticeSession DO

### Task 3: Rewrite session.rs with full pipeline orchestration

**Files:**
- Modify: `apps/api/src/practice/session.rs`

**Context:** This is the main deliverable. The existing session.rs has placeholder logic. We replace it with the full pipeline: R2 fetch, HF inference, STOP classifier, teaching moment selection, LLM observation via `handle_ask_inner()`, alarm-based cleanup, and `finalize_session()` for D1 persistence.

The `#[durable_object]` macro requires `&self` (not `&mut self`) for all trait methods. Interior mutability must use `RefCell` or similar. The `worker` crate's DO API uses `&self` throughout.

- [ ] **Step 1: Update imports and struct definition**

Replace the entire file with the new implementation. Start with imports and struct:

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use worker::*;

use crate::practice::dims::DIMS_6;
use crate::practice::teaching_moment::DimStats;
use crate::services::stop;
use crate::services::teaching_moments::{
    RecentObservation, ScoredChunk, StudentBaselines, TeachingMoment,
};

const ALARM_DURATION_MS: i64 = 30 * 60 * 1000; // 30 minutes
const OBSERVATION_THROTTLE_MS: u64 = 180_000; // 3 minutes

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
    dim_stats: DimStats,
    last_observation_at: Option<u64>,
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
            dim_stats: DimStats::default(),
            last_observation_at: None,
        }
    }
}

#[durable_object]
pub struct PracticeSession {
    state: State,
    env: Env,
    inner: RefCell<SessionState>,
}
```

- [ ] **Step 2: Implement DurableObject::new and fetch**

```rust
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

        // Store session info
        {
            let mut s = self.inner.borrow_mut();
            // On reconnection, keep existing state but update session/student refs
            if s.session_id.is_empty() {
                s.session_id = session_id.clone();
                s.student_id = student_id;
            }
        }

        // Close any existing WebSocket connections (reconnection case)
        let existing_sockets = self.state.get_websockets();
        for old_ws in existing_sockets {
            let _ = old_ws.close(Some(1000), Some("New connection replacing old one".into()));
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
```

- [ ] **Step 3: Implement websocket_message handler**

```rust
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
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let r2_key = parsed.get("r2Key").and_then(|v| v.as_str()).unwrap_or("");
                self.handle_chunk_ready(&ws, index, r2_key).await?;
            }
            "end_session" => {
                self.finalize_session(Some(&ws)).await;
            }
            _ => {}
        }

        Ok(())
    }
```

- [ ] **Step 4: Implement websocket_close and alarm handlers**

```rust
    async fn websocket_close(
        &self,
        _ws: WebSocket,
        _code: usize,
        _reason: String,
        _was_clean: bool,
    ) -> Result<()> {
        // Client disconnected -- finalize if we have data
        let has_chunks = !self.inner.borrow().scored_chunks.is_empty();
        if has_chunks {
            self.finalize_session(None).await;
        }
        Ok(())
    }

    async fn alarm(&self) -> Result<Response> {
        // Alarm fired: session timed out (30 min inactivity)
        // Try to send summary via any open WebSocket
        let sockets = self.state.get_websockets();
        let ws = sockets.first();
        self.finalize_session(ws).await;
        Response::ok("alarm handled")
    }
}
```

- [ ] **Step 5: Implement handle_chunk_ready (the core pipeline)**

Add as `impl PracticeSession` methods (outside the `DurableObject` trait):

```rust
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

        // 2. Call HF inference
        let scores_map = match self.call_hf_inference(&audio_bytes).await {
            Ok(scores) => scores,
            Err(e) => {
                console_error!("HF inference failed for chunk {}: {}", index, e);
                self.send_zeroed_chunk_processed(ws, index)?;
                return Ok(());
            }
        };

        // 3. Convert scores: HashMap -> [f64; 6] for STOP classifier
        let scores_array: [f64; 6] = DIMS_6.map(|dim| {
            scores_map.get(dim).copied().unwrap_or(0.0)
        });

        // 4. Send chunk_processed immediately
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
        ws.send_with_str(&response.to_string())?;

        // 5-6. Update DimStats and store ScoredChunk
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

        // 8. Run STOP classifier on current chunk
        let stop_result = stop::classify(&scores_array);

        // 9. Check if we should generate an observation
        let should_generate = {
            let s = self.inner.borrow();
            stop_result.triggered
                && s.baselines.is_some()
                && self.throttle_allows(&s)
        };

        if should_generate {
            self.generate_observation(ws, &scores_array).await;
        }

        // 10. Reset alarm
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
        ws.send_with_str(&response.to_string())
    }
```

- [ ] **Step 6: Implement generate_observation**

```rust
    async fn generate_observation(&self, ws: &WebSocket, _scores: &[f64; 6]) {
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

        let inner_req = crate::services::ask::AskInnerRequest {
            teaching_moment: tm_json,
            student_id: student_id.clone(),
            session_id: session_id.clone(),
            piece_context: None,
        };

        let inner_resp = crate::services::ask::handle_ask_inner(&self.env, &inner_req).await;

        // Push observation to client
        let obs_event = serde_json::json!({
            "type": "observation",
            "text": inner_resp.observation_text,
            "dimension": inner_resp.dimension,
            "framing": inner_resp.framing,
        });
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
```

- [ ] **Step 7: Implement fetch_audio_from_r2 and call_hf_inference**

```rust
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
    ) -> std::result::Result<HashMap<String, f64>, String> {
        let endpoint = self.env.var("HF_INFERENCE_ENDPOINT")
            .map_err(|e| format!("HF_INFERENCE_ENDPOINT not set: {:?}", e))?
            .to_string();
        let token = self.env.secret("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN not set: {:?}", e))?
            .to_string();

        let mut headers = worker::Headers::new();
        headers.set("Content-Type", "application/octet-stream").map_err(|e| format!("{:?}", e))?;
        headers.set("Authorization", &format!("Bearer {}", token)).map_err(|e| format!("{:?}", e))?;

        let mut init = worker::RequestInit::new();
        init.with_method(worker::Method::Post);
        init.with_headers(headers);
        init.with_body(Some(JsValue::from(js_sys::Uint8Array::from(audio_bytes))));

        let request = worker::Request::new_with_init(&endpoint, &init)
            .map_err(|e| format!("Failed to create request: {:?}", e))?;

        let mut response = worker::Fetch::Request(request)
            .send()
            .await
            .map_err(|e| format!("HF fetch failed: {:?}", e))?;

        if response.status_code() != 200 {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("HF returned {}: {}", response.status_code(), body));
        }

        let body: serde_json::Value = response.json().await
            .map_err(|e| format!("HF response parse failed: {:?}", e))?;

        // Extract predictions from nested response
        let predictions = body.get("predictions")
            .or_else(|| Some(&body)) // Fallback: try top-level if no predictions key
            .ok_or_else(|| "No predictions in HF response".to_string())?;

        let mut scores = HashMap::new();
        for dim in DIMS_6 {
            if let Some(val) = predictions.get(dim).and_then(|v| v.as_f64()) {
                scores.insert(dim.to_string(), val);
            }
        }

        if scores.len() < 6 {
            return Err(format!("HF returned only {} dimensions", scores.len()));
        }

        Ok(scores)
    }
```

- [ ] **Step 8: Implement load_baselines**

```rust
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
```

- [ ] **Step 9: Implement finalize_session**

```rust
    async fn finalize_session(&self, ws: Option<&WebSocket>) {
        let (observations, session_id, student_id) = {
            let s = self.inner.borrow();
            (s.observations.clone(), s.session_id.clone(), s.student_id.clone())
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
            });
            let _ = ws.send_with_str(&summary.to_string());
        }

        // 3. Close all WebSockets
        for ws in self.state.get_websockets() {
            let _ = ws.close(Some(1000), Some("Session ended".into()));
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
                    JsValue::NULL,  // piece_context (deferred)
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
} // end impl PracticeSession
```

- [ ] **Step 10: Make SCALER_MEAN public in stop.rs**

The `load_baselines` method references `crate::services::stop::SCALER_MEAN` which is currently not `pub`. Update `apps/api/src/services/stop.rs`:

Replace:
```rust
const SCALER_MEAN: [f64; 6] = [0.5450, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064];
```

With:
```rust
pub const SCALER_MEAN: [f64; 6] = [0.5450, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064];
```

- [ ] **Step 11: Verify compilation**

Run: `cd apps/api && cargo check 2>&1`
Expected: Compiles with only pre-existing warnings.

- [ ] **Step 12: Run existing tests**

Run: `cd apps/api && cargo test 2>&1`
Expected: All existing STOP classifier tests pass (7 tests).

- [ ] **Step 13: Commit**

```bash
git add apps/api/src/practice/session.rs apps/api/src/services/stop.rs
git commit -m "implement full pipeline orchestration in PracticeSession DO

- R2 fetch -> HF inference -> STOP classifier -> teaching moment
  selection -> handle_ask_inner() -> WebSocket observation push
- D1 baselines query (one-time, cached for session)
- Server-side observation throttle (3-minute window)
- finalize_session: persist observations to D1, send session_summary
- 30-minute alarm with reset on chunk_ready
- WebSocket close handler for clean disconnects
- Close old WebSocket on reconnection"
```

---

## Chunk 3: Build, Deploy, Verify

### Task 4: Final verification and deploy

**Files:** None (verification only)

- [ ] **Step 1: Run cargo check**

Run: `cd apps/api && cargo check 2>&1`
Expected: No errors.

- [ ] **Step 2: Run cargo test**

Run: `cd apps/api && cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 3: Build for production**

Run: `cd apps/api && cargo install -q worker-build && worker-build --release 2>&1`
Expected: WASM build succeeds.

- [ ] **Step 4: Deploy**

Run: `cd apps/api && wrangler deploy 2>&1`
Expected: Deploy succeeds, shows PRACTICE_SESSION DO binding.

- [ ] **Step 5: Push to main**

```bash
git push origin main
```

- [ ] **Step 6: Commit plan completion note**

No commit needed -- plan is complete.
