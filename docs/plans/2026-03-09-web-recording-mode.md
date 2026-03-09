# Web Recording Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a full practice companion to the web app -- a floating recording overlay that captures piano audio, runs cloud inference, and delivers real-time teacher observations.

**Architecture:** Client captures audio via Web Audio API with 15-second MediaRecorder chunks. Chunks upload through the Rust Worker to R2. A Durable Object per session manages state via hibernatable WebSocket: it fetches chunks from R2, calls HF inference, maps 19 dims to 6, runs outlier detection for teaching moments, calls the existing ask pipeline for observations, and pushes toasts back to the client. On session end, the DO ranks observations, generates a summary, writes to D1, and the client posts it to chat.

**Tech Stack:** Rust (worker crate 0.7 with DO + R2), Web Audio API, MediaRecorder, WebSocket, React hooks, Tailwind CSS v4, existing design tokens.

**Design doc:** `docs/plans/2026-03-09-web-recording-mode-design.md`

---

## Key Decisions (MVP)

1. **No pre-signed URLs.** Client uploads chunks through the Worker (`POST /api/practice/chunk`), Worker stores in R2 via binding. Chunks are small (~60-120KB), this avoids SigV4 signing complexity.
2. **DO calls ask pipeline as a function**, not via HTTP. The DO and Worker share the same crate -- import and call `handle_ask_internal()` directly.
3. **WebSocket auth via query parameter.** Pass JWT as `?token=...` on the WS URL. DO validates on connection.
4. **19-to-6 dimension mapping.** Simple averages of related dims (hardcoded mapping table).
5. **No minimized pill for MVP.** Overlay is either open or closed.

## Dependency Graph

```
Task 1 (DO+WS spike) ──> Task 3 (practice DO) ──> Task 5 (inference + teaching moments)
Task 2 (R2 + chunk upload) ──────────────────────> Task 5
                                                    │
Task 4 (audio capture hook) ──────────────────────> Task 6 (full pipeline wiring)
                                                    │
                                              Task 7 (overlay UI)
                                                    │
                                              Task 8 (session end + chat integration)
                                                    │
                                              Task 9 (polish + edge cases)
```

Parallelizable: (1, 2, 4), (3, 4), (7 can start after 4)

---

## Task 1: Durable Object + WebSocket Spike

Validate that the Rust `worker` crate supports Durable Objects with hibernatable WebSockets before building real logic.

**Files:**
- Modify: `apps/api/Cargo.toml`
- Modify: `apps/api/wrangler.toml`
- Modify: `apps/api/src/lib.rs`
- Modify: `apps/api/src/server.rs`
- Create: `apps/api/src/practice/mod.rs`
- Create: `apps/api/src/practice/session.rs`

**Step 1: Add R2 feature flag to worker crate**

In `apps/api/Cargo.toml`, change the worker dependency:

```toml
worker = { version = "0.7", features = ["http", "d1", "r2"] }
```

**Step 2: Add DO and R2 bindings to wrangler.toml**

Append to `apps/api/wrangler.toml`:

```toml
[[r2_buckets]]
binding = "CHUNKS"
bucket_name = "crescendai-chunks"

[durable_objects]
bindings = [
  { name = "PRACTICE_SESSION", class_name = "PracticeSession" }
]

[[migrations]]
tag = "v1"
new_classes = ["PracticeSession"]
```

**Step 3: Create the R2 bucket**

Run: `cd apps/api && npx wrangler r2 bucket create crescendai-chunks`

**Step 4: Create practice module with DO struct**

Create `apps/api/src/practice/mod.rs`:

```rust
pub mod session;
```

Create `apps/api/src/practice/session.rs`:

```rust
use worker::*;

#[durable_object]
pub struct PracticeSession {
    state: State,
    env: Env,
}

#[durable_object]
impl DurableObject for PracticeSession {
    fn new(state: State, env: Env) -> Self {
        Self { state, env }
    }

    async fn fetch(&mut self, req: Request) -> Result<Response> {
        // Accept WebSocket upgrade
        let pair = WebSocketPair::new()?;
        let server = pair.server;
        self.state.accept_web_socket(&server);

        // Send a welcome message
        server.send_with_str(r#"{"type":"connected"}"#)?;

        Response::from_websocket(pair.client)
    }

    async fn websocket_message(&mut self, ws: WebSocket, msg: String) -> Result<()> {
        // Echo for spike
        ws.send_with_str(&format!(r#"{{"type":"echo","data":{}}}"#, msg))?;
        Ok(())
    }

    async fn websocket_close(&mut self, _ws: WebSocket, _code: usize, _reason: String, _was_clean: bool) -> Result<()> {
        Ok(())
    }
}
```

**Step 5: Wire DO into lib.rs**

In `apps/api/src/lib.rs`, add:

```rust
pub mod auth;
pub mod practice;
pub mod server;
pub mod services;
```

**Step 6: Add WS upgrade route to server.rs**

In `apps/api/src/server.rs`, add before the health check (line 204):

```rust
// Practice WebSocket upgrade -- forward to Durable Object
if path.starts_with("/api/practice/ws/") && method == http::Method::GET {
    let session_id = path.trim_start_matches("/api/practice/ws/");
    if !session_id.is_empty() && !session_id.contains('/') {
        let namespace = env.durable_object("PRACTICE_SESSION")?;
        let stub = namespace.id_from_name(session_id)?.get_stub()?;
        // Convert HttpRequest to worker::Request for DO forwarding
        let url = format!("https://do.internal/ws/{}", session_id);
        let worker_req = worker::Request::new(&url, worker::Method::Get)?;
        let response = stub.fetch_with_request(worker_req).await?;
        // Return the WebSocket response directly
        // Note: This requires returning worker::Response, not http::Response.
        // We need to convert or handle this specially.
        // For now, return a placeholder -- the actual conversion is part of the spike.
        return Ok(with_cors(
            http::Response::builder()
                .status(http::StatusCode::SWITCHING_PROTOCOLS)
                .body(axum::body::Body::empty())
                .unwrap(),
            origin.as_deref(),
        ));
    }
}
```

**Important:** The `#[event(fetch)]` handler returns `http::Response<axum::body::Body>`, but WebSocket upgrade requires returning a `worker::Response`. This is the key thing to solve in this spike. Options:
- Change the handler's return type to `worker::Response` and convert all existing routes
- Use `worker::Response::from_websocket()` and find a way to convert it to `http::Response`
- Handle the WS path before entering the `http::Response` flow

This spike will determine which approach works.

**Step 7: Build and test**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown --release`
Expected: Compiles successfully (or reveals what needs fixing for DO+WS in Rust)

If compilation succeeds, deploy to dev and test WS connection:
Run: `cd apps/api && npx wrangler dev`
Then in browser console:
```javascript
const ws = new WebSocket('ws://localhost:8787/api/practice/ws/test-session')
ws.onmessage = (e) => console.log(e.data)
ws.onopen = () => ws.send('"hello"')
```
Expected: receive `{"type":"connected"}` then `{"type":"echo","data":"hello"}`

**Step 8: Commit**

```bash
git add apps/api/src/practice/ apps/api/Cargo.toml apps/api/wrangler.toml apps/api/src/lib.rs apps/api/src/server.rs
git commit -m "feat(api): scaffold Durable Object + WebSocket for practice sessions"
```

---

## Task 2: R2 Chunk Upload Endpoint

Add an authenticated endpoint that accepts audio chunk uploads and stores them in R2.

**Files:**
- Create: `apps/api/src/practice/upload.rs`
- Modify: `apps/api/src/practice/mod.rs`
- Modify: `apps/api/src/server.rs`

**Step 1: Create the upload handler**

Create `apps/api/src/practice/upload.rs`:

```rust
use worker::{console_log, Env};

/// Handle POST /api/practice/chunk?sessionId=X&chunkIndex=N
/// Body: raw audio bytes (WebM/Opus)
pub async fn handle_upload_chunk(
    env: &Env,
    headers: &http::HeaderMap,
    body: Vec<u8>,
    session_id: &str,
    chunk_index: &str,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    if body.is_empty() {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Empty body"}"#))
            .unwrap();
    }

    let r2_key = format!("sessions/{}/chunks/{}.webm", session_id, chunk_index);

    let bucket = match env.bucket("CHUNKS") {
        Ok(b) => b,
        Err(e) => {
            console_log!("Failed to get R2 bucket: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Storage unavailable"}"#))
                .unwrap();
        }
    };

    match bucket.put(&r2_key, body).execute().await {
        Ok(_) => {
            let resp = serde_json::json!({
                "r2Key": r2_key,
                "sessionId": session_id,
                "chunkIndex": chunk_index,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&resp).unwrap()))
                .unwrap()
        }
        Err(e) => {
            console_log!("R2 put failed: {:?}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Failed to store chunk"}"#))
                .unwrap()
        }
    }
}
```

**Step 2: Register the module**

In `apps/api/src/practice/mod.rs`:

```rust
pub mod session;
pub mod upload;
```

**Step 3: Add route to server.rs**

Add before the health check in `apps/api/src/server.rs`:

```rust
// Upload audio chunk to R2 (authenticated)
if path == "/api/practice/chunk" && method == http::Method::POST {
    let headers = req.headers().clone();
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();

    // Parse query params
    let query = req_uri_query.unwrap_or_default();
    let params: std::collections::HashMap<String, String> = query
        .split('&')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            Some((parts.next()?.to_string(), parts.next()?.to_string()))
        })
        .collect();

    let session_id = params.get("sessionId").map(|s| s.as_str()).unwrap_or("");
    let chunk_index = params.get("chunkIndex").map(|s| s.as_str()).unwrap_or("0");

    if session_id.is_empty() {
        return Ok(with_cors(
            http::Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Missing sessionId"}"#))
                .unwrap(),
            origin.as_deref(),
        ));
    }

    return Ok(with_cors(
        crate::practice::upload::handle_upload_chunk(&env, &headers, body, session_id, chunk_index).await,
        origin.as_deref(),
    ));
}
```

Note: You'll need to capture `req.uri().query()` before consuming the request body. Add this near line 38:

```rust
let query_string = req.uri().query().map(|q| q.to_string());
```

Then use `query_string` instead of `req_uri_query` above.

**Step 4: Build and test**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown --release`
Expected: Compiles

Test with wrangler dev:
```bash
curl -X POST http://localhost:8787/api/practice/chunk?sessionId=test-123&chunkIndex=0 \
  -H "Cookie: token=<valid-jwt>" \
  --data-binary @test-audio.webm
```
Expected: `{"r2Key":"sessions/test-123/chunks/0.webm","sessionId":"test-123","chunkIndex":"0"}`

**Step 5: Commit**

```bash
git add apps/api/src/practice/upload.rs apps/api/src/practice/mod.rs apps/api/src/server.rs
git commit -m "feat(api): add R2 chunk upload endpoint for practice sessions"
```

---

## Task 3: Practice Session Durable Object (Full Implementation)

Build the real DO with session state, WebSocket message handling, and the chunk processing pipeline.

**Files:**
- Rewrite: `apps/api/src/practice/session.rs`
- Create: `apps/api/src/practice/dims.rs` (19-to-6 mapping)
- Create: `apps/api/src/practice/teaching_moment.rs` (outlier detection)
- Modify: `apps/api/src/practice/mod.rs`
- Modify: `apps/api/src/services/ask.rs` (extract internal function)

**Step 1: Create dimension mapping**

Create `apps/api/src/practice/dims.rs`:

```rust
use std::collections::HashMap;

/// The 6 teacher-grounded dimensions.
pub const DIMS_6: [&str; 6] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
];

/// Map 19 PerCePiano dimensions to 6 teacher-grounded dimensions.
/// Each output dimension is the mean of its constituent raw dimensions.
pub fn map_19_to_6(raw: &HashMap<String, f64>) -> HashMap<String, f64> {
    let avg = |keys: &[&str]| -> f64 {
        let (sum, count) = keys.iter().fold((0.0, 0u32), |(s, c), k| {
            match raw.get(*k) {
                Some(v) => (s + v, c + 1),
                None => (s, c),
            }
        });
        if count == 0 { 0.0 } else { sum / count as f64 }
    };

    let mut mapped = HashMap::new();
    mapped.insert("dynamics".to_string(), avg(&["dynamics_range", "timbre_loudness"]));
    mapped.insert("timing".to_string(), avg(&["timing", "tempo"]));
    mapped.insert("pedaling".to_string(), avg(&["pedal_amount", "pedal_clarity"]));
    mapped.insert("articulation".to_string(), avg(&["articulation_length", "articulation_touch"]));
    mapped.insert("phrasing".to_string(), avg(&["space", "balance", "drama"]));
    mapped.insert("interpretation".to_string(), avg(&[
        "timbre_variety", "timbre_depth", "timbre_brightness",
        "mood_valence", "mood_energy", "mood_imagination",
        "interpretation_sophistication", "interpretation_overall",
    ]));
    mapped
}
```

**Step 2: Create teaching moment detection**

Create `apps/api/src/practice/teaching_moment.rs`:

```rust
use std::collections::HashMap;

/// Running statistics tracker per dimension.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct DimStats {
    pub counts: HashMap<String, u32>,
    pub means: HashMap<String, f64>,
    pub m2s: HashMap<String, f64>, // Welford's running variance
}

impl DimStats {
    /// Update running stats with new scores (Welford's online algorithm).
    pub fn update(&mut self, scores: &HashMap<String, f64>) {
        for (dim, &value) in scores {
            let count = self.counts.entry(dim.clone()).or_insert(0);
            *count += 1;
            let n = *count;

            let mean = self.means.entry(dim.clone()).or_insert(0.0);
            let m2 = self.m2s.entry(dim.clone()).or_insert(0.0);

            let delta = value - *mean;
            *mean += delta / n as f64;
            let delta2 = value - *mean;
            *m2 += delta * delta2;
        }
    }

    fn stddev(&self, dim: &str) -> f64 {
        let count = self.counts.get(dim).copied().unwrap_or(0);
        if count < 2 { return f64::MAX }
        let m2 = self.m2s.get(dim).copied().unwrap_or(0.0);
        (m2 / (count - 1) as f64).sqrt()
    }

    fn mean(&self, dim: &str) -> f64 {
        self.means.get(dim).copied().unwrap_or(0.0)
    }

    /// Check if any dimension deviates > threshold stddevs from running mean.
    /// Returns the most deviant dimension and its z-score, if any exceed threshold.
    /// Requires at least `min_chunks` data points before flagging.
    pub fn detect_outlier(
        &self,
        scores: &HashMap<String, f64>,
        threshold: f64,
        min_chunks: u32,
    ) -> Option<(String, f64)> {
        let mut worst: Option<(String, f64)> = None;

        for (dim, &value) in scores {
            let count = self.counts.get(dim.as_str()).copied().unwrap_or(0);
            if count < min_chunks { continue; }

            let std = self.stddev(dim);
            if std == 0.0 || std == f64::MAX { continue; }

            let z = (value - self.mean(dim)).abs() / std;
            if z > threshold {
                match &worst {
                    None => worst = Some((dim.clone(), z)),
                    Some((_, wz)) if z > *wz => worst = Some((dim.clone(), z)),
                    _ => {}
                }
            }
        }

        worst
    }
}
```

**Step 3: Extract internal ask function from ask.rs**

In `apps/api/src/services/ask.rs`, add a new function that the DO can call directly without HTTP ceremony. Add after line 61:

```rust
/// Internal ask pipeline callable from Durable Object (no HTTP auth layer).
/// Takes pre-validated student_id and structured data.
pub async fn ask_internal(
    env: &Env,
    student_id: &str,
    teaching_moment: serde_json::Value,
    student: serde_json::Value,
    session: serde_json::Value,
    piece_context: Option<serde_json::Value>,
) -> std::result::Result<AskResponse, String> {
    // Same logic as handle_ask but without auth verification and HTTP response wrapping.
    // Extract the core pipeline from handle_ask into this function,
    // then have handle_ask call ask_internal internally.
    // (The refactoring details depend on the exact handle_ask implementation.)

    let request = AskRequest { teaching_moment, student, session, piece_context };
    // ... (extract body of handle_ask from line 91 onwards into this function)
    // Return AskResponse directly instead of wrapping in http::Response
    todo!("Extract from handle_ask -- see lines 91-295 of ask.rs")
}
```

The actual refactoring: move lines 91-295 of `handle_ask` into `ask_internal`, change `handle_ask` to call `ask_internal` and wrap the result in an HTTP response.

**Step 4: Build the full DO**

Rewrite `apps/api/src/practice/session.rs` with full session state, chunk processing, and observation delivery. Key pieces:

- `PracticeSession` struct with fields: `state`, `env`, `session_id`, `student_id`, `scores: Vec<HashMap<String, f64>>`, `observations: Vec<ObservationRecord>`, `dim_stats: DimStats`
- `fetch()`: parse auth token from query param, validate JWT, accept WebSocket, store student_id
- `websocket_message()`: handle `chunk_ready` (fetch from R2, call inference, map dims, check outlier, maybe generate observation) and `end_session` (rank observations, generate summary, write to D1)
- `websocket_close()`: cleanup

The DO fetches chunks from R2, calls the HF inference endpoint via HTTP, and calls `ask_internal` for observations.

**Step 5: Update practice/mod.rs**

```rust
pub mod dims;
pub mod session;
pub mod teaching_moment;
pub mod upload;
```

**Step 6: Build and test**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown --release`
Expected: Compiles

**Step 7: Commit**

```bash
git add apps/api/src/practice/ apps/api/src/services/ask.rs
git commit -m "feat(api): implement practice session DO with inference, dims mapping, and teaching moment detection"
```

---

## Task 4: Client-Side Audio Capture Hook

Build the React hook that manages the audio capture lifecycle using Web Audio API and MediaRecorder.

**Files:**
- Create: `apps/web/src/hooks/usePracticeSession.ts`
- Create: `apps/web/src/lib/practice-api.ts`
- Modify: `apps/web/src/lib/api.ts` (add API_BASE export)

**Step 1: Add practice API client**

Create `apps/web/src/lib/practice-api.ts`:

```typescript
const API_BASE = import.meta.env.PROD
  ? 'https://api.crescend.ai'
  : 'http://localhost:8787'

const WS_BASE = import.meta.env.PROD
  ? 'wss://api.crescend.ai'
  : 'ws://localhost:8787'

export interface PracticeStartResponse {
  sessionId: string
}

export interface ChunkUploadResponse {
  r2Key: string
  sessionId: string
  chunkIndex: string
}

export interface DimScores {
  dynamics: number
  timing: number
  pedaling: number
  articulation: number
  phrasing: number
  interpretation: number
}

export interface ObservationEvent {
  text: string
  dimension: string
  framing: string
}

export interface SessionSummary {
  observations: ObservationEvent[]
  summary: string
}

export type PracticeWsEvent =
  | { type: 'connected' }
  | { type: 'chunk_processed'; index: number; scores: DimScores }
  | { type: 'observation'; text: string; dimension: string; framing: string }
  | { type: 'session_summary'; observations: ObservationEvent[]; summary: string }
  | { type: 'error'; message: string }

export const practiceApi = {
  async start(): Promise<PracticeStartResponse> {
    const res = await fetch(`${API_BASE}/api/practice/start`, {
      method: 'POST',
      credentials: 'include',
    })
    if (!res.ok) throw new Error(`Failed to start session: ${res.status}`)
    return res.json()
  },

  async uploadChunk(sessionId: string, chunkIndex: number, blob: Blob): Promise<ChunkUploadResponse> {
    const res = await fetch(
      `${API_BASE}/api/practice/chunk?sessionId=${sessionId}&chunkIndex=${chunkIndex}`,
      {
        method: 'POST',
        credentials: 'include',
        body: blob,
      },
    )
    if (!res.ok) throw new Error(`Failed to upload chunk: ${res.status}`)
    return res.json()
  },

  connectWebSocket(sessionId: string): WebSocket {
    return new WebSocket(`${WS_BASE}/api/practice/ws/${sessionId}`)
  },
}
```

**Step 2: Create the practice session hook**

Create `apps/web/src/hooks/usePracticeSession.ts`:

```typescript
import { useState, useRef, useCallback } from 'react'
import { practiceApi } from '../lib/practice-api'
import type { PracticeWsEvent, ObservationEvent, DimScores } from '../lib/practice-api'

export type PracticeState =
  | 'idle'
  | 'requesting-mic'
  | 'connecting'
  | 'recording'
  | 'summarizing'
  | 'error'

export interface UsePracticeSessionReturn {
  state: PracticeState
  elapsedSeconds: number
  observations: ObservationEvent[]
  latestScores: DimScores | null
  summary: string | null
  error: string | null
  analyserNode: AnalyserNode | null
  start: () => Promise<void>
  stop: () => void
}

export function usePracticeSession(): UsePracticeSessionReturn {
  const [state, setState] = useState<PracticeState>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [observations, setObservations] = useState<ObservationEvent[]>([])
  const [latestScores, setLatestScores] = useState<DimScores | null>(null)
  const [summary, setSummary] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null)

  const sessionIdRef = useRef<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chunkIndexRef = useRef(0)

  const cleanup = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop()
    if (audioContextRef.current?.state !== 'closed') audioContextRef.current?.close()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    wsRef.current?.close()

    timerRef.current = null
    mediaRecorderRef.current = null
    audioContextRef.current = null
    streamRef.current = null
    wsRef.current = null
    sessionIdRef.current = null
    chunkIndexRef.current = 0
  }, [])

  const start = useCallback(async () => {
    setState('requesting-mic')
    setElapsedSeconds(0)
    setObservations([])
    setLatestScores(null)
    setSummary(null)
    setError(null)

    // 1. Request mic
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
    } catch (e) {
      setState('error')
      setError('Microphone access denied. Please allow mic access and try again.')
      return
    }

    // 2. Set up AudioContext + AnalyserNode for waveform visualization
    const audioCtx = new AudioContext()
    audioContextRef.current = audioCtx
    const source = audioCtx.createMediaStreamSource(stream)
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    setAnalyserNode(analyser)

    // 3. Start session on server
    setState('connecting')
    let sessionId: string
    try {
      const { sessionId: sid } = await practiceApi.start()
      sessionId = sid
      sessionIdRef.current = sid
    } catch (e) {
      cleanup()
      setState('error')
      setError('Failed to start practice session. Please try again.')
      return
    }

    // 4. Connect WebSocket
    const ws = practiceApi.connectWebSocket(sessionId)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data: PracticeWsEvent = JSON.parse(event.data)
      switch (data.type) {
        case 'chunk_processed':
          setLatestScores(data.scores)
          break
        case 'observation':
          setObservations((prev) => [...prev, {
            text: data.text,
            dimension: data.dimension,
            framing: data.framing,
          }])
          break
        case 'session_summary':
          setSummary(data.summary)
          setObservations(data.observations)
          setState('idle')
          cleanup()
          break
        case 'error':
          setError(data.message)
          break
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection lost.')
      setState('error')
      cleanup()
    }

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('WebSocket failed to connect'))
    })

    // 5. Start MediaRecorder with 15s chunks
    const recorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    })
    mediaRecorderRef.current = recorder
    chunkIndexRef.current = 0

    recorder.ondataavailable = async (event) => {
      if (event.data.size === 0) return
      const idx = chunkIndexRef.current++
      try {
        const { r2Key } = await practiceApi.uploadChunk(sessionId, idx, event.data)
        ws.send(JSON.stringify({ type: 'chunk_ready', index: idx, r2Key }))
      } catch (e) {
        console.error('Chunk upload failed:', e)
      }
    }

    recorder.start(15000) // 15-second timeslice
    setState('recording')

    // 6. Start elapsed timer
    const startTime = Date.now()
    timerRef.current = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
  }, [cleanup])

  const stop = useCallback(() => {
    if (state !== 'recording') return
    setState('summarizing')

    // Stop recording (triggers final ondataavailable)
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }

    // Tell DO to end session
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }))
    }

    // Timer cleanup (WS stays open until summary arrives)
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [state])

  return {
    state,
    elapsedSeconds,
    observations,
    latestScores,
    summary,
    error,
    analyserNode,
    start,
    stop,
  }
}
```

**Step 3: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/lib/practice-api.ts
git commit -m "feat(web): add practice session hook with audio capture, WS, and chunk upload"
```

---

## Task 5: Practice Start Endpoint + Inference Integration in DO

Wire up the `POST /api/practice/start` endpoint and implement the DO's chunk processing pipeline (fetch from R2, call HF inference, map dims).

**Files:**
- Create: `apps/api/src/practice/start.rs`
- Modify: `apps/api/src/practice/session.rs` (add inference calls)
- Modify: `apps/api/src/practice/mod.rs`
- Modify: `apps/api/src/server.rs`

**Step 1: Create start endpoint**

Create `apps/api/src/practice/start.rs`:

```rust
use worker::Env;

pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    let session_id = crate::services::ask::generate_uuid();

    let resp = serde_json::json!({
        "sessionId": session_id,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&resp).unwrap()))
        .unwrap()
}
```

**Step 2: Add route to server.rs**

```rust
if path == "/api/practice/start" && method == http::Method::POST {
    let headers = req.headers().clone();
    return Ok(with_cors(
        crate::practice::start::handle_start(&env, &headers).await,
        origin.as_deref(),
    ));
}
```

**Step 3: Implement inference call in DO**

In `apps/api/src/practice/session.rs`, add the HF inference call inside the `chunk_ready` handler:

```rust
/// Call HF inference endpoint with audio chunk bytes.
async fn call_inference(env: &Env, audio_bytes: &[u8]) -> std::result::Result<HashMap<String, f64>, String> {
    let hf_url = env.var("HF_INFERENCE_ENDPOINT")
        .map_err(|e| format!("Missing HF_INFERENCE_ENDPOINT: {:?}", e))?
        .to_string();

    let encoded = base64::engine::general_purpose::STANDARD.encode(audio_bytes);
    let payload = serde_json::json!({
        "inputs": encoded,
    });

    let mut headers = worker::Headers::new();
    headers.set("Content-Type", "application/json").map_err(|e| format!("{:?}", e))?;

    let mut init = worker::RequestInit::new();
    init.with_method(worker::Method::Post)
        .with_headers(headers)
        .with_body(Some(wasm_bindgen::JsValue::from_str(&payload.to_string())));

    let request = worker::Request::new_with_init(&hf_url, &init)
        .map_err(|e| format!("Failed to build request: {:?}", e))?;

    let mut response = worker::Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Inference call failed: {:?}", e))?;

    let text = response.text().await
        .map_err(|e| format!("Failed to read response: {:?}", e))?;

    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse response: {:?}", e))?;

    let predictions = json.get("predictions")
        .ok_or_else(|| "No predictions in response".to_string())?;

    let scores: HashMap<String, f64> = serde_json::from_value(predictions.clone())
        .map_err(|e| format!("Failed to parse predictions: {:?}", e))?;

    Ok(scores)
}
```

**Step 4: Update mod.rs**

```rust
pub mod dims;
pub mod session;
pub mod start;
pub mod teaching_moment;
pub mod upload;
```

**Step 5: Build and test**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown --release`
Expected: Compiles

**Step 6: Commit**

```bash
git add apps/api/src/practice/
git commit -m "feat(api): add practice start endpoint and inference integration in DO"
```

---

## Task 6: Full Pipeline Wiring (Upload -> DO -> Inference -> Observation)

Wire the complete chunk processing pipeline: client uploads chunk, notifies DO, DO processes, calls ask pipeline if teaching moment, pushes result back.

**Files:**
- Modify: `apps/api/src/practice/session.rs` (full websocket_message handler)
- Modify: `apps/api/src/services/ask.rs` (refactor for internal use)

**Step 1: Refactor ask.rs -- extract core pipeline**

In `apps/api/src/services/ask.rs`, refactor `handle_ask` to call a new `ask_internal` function. Read the full file first, then extract lines 91-295 into `ask_internal` which returns `Result<AskResponse, String>` instead of `http::Response`. Make `handle_ask` a thin wrapper that calls `ask_internal` and maps the result.

**Step 2: Complete DO websocket_message handler**

The DO `websocket_message` handler for `chunk_ready`:

```rust
async fn handle_chunk_ready(&mut self, ws: &WebSocket, index: usize, r2_key: &str) -> Result<()> {
    // 1. Fetch chunk from R2
    let bucket = self.env.bucket("CHUNKS")?;
    let object = bucket.get(&r2_key).execute().await?
        .ok_or_else(|| worker::Error::RustError("Chunk not found in R2".into()))?;
    let bytes = object.body()
        .ok_or_else(|| worker::Error::RustError("Empty chunk body".into()))?
        .bytes().await?;

    // 2. Call inference
    let raw_scores = call_inference(&self.env, &bytes).await
        .map_err(|e| worker::Error::RustError(e))?;

    // 3. Map 19 -> 6 dims
    let scores = crate::practice::dims::map_19_to_6(&raw_scores);

    // 4. Update running stats
    self.dim_stats.update(&scores);

    // 5. Send chunk_processed
    let msg = serde_json::json!({
        "type": "chunk_processed",
        "index": index,
        "scores": scores,
    });
    ws.send_with_str(&msg.to_string())?;

    // 6. Check for teaching moment (skip first 2 chunks)
    if let Some((dimension, z_score)) = self.dim_stats.detect_outlier(&scores, 1.5, 3) {
        // Build teaching moment context for ask pipeline
        let teaching_moment = serde_json::json!({
            "chunk_index": index,
            "dimension": dimension,
            "dimension_score": scores.get(&dimension).unwrap_or(&0.0),
            "all_scores": scores,
        });
        let student_ctx = serde_json::json!({
            "level": "intermediate", // TODO: load from D1
        });
        let session_ctx = serde_json::json!({
            "duration_min": 0, // TODO: calculate
            "total_chunks": self.scores.len(),
        });

        match crate::services::ask::ask_internal(
            &self.env,
            &self.student_id,
            teaching_moment,
            student_ctx,
            session_ctx,
            None,
        ).await {
            Ok(response) => {
                let obs = serde_json::json!({
                    "type": "observation",
                    "text": response.observation,
                    "dimension": response.dimension,
                    "framing": response.framing,
                });
                ws.send_with_str(&obs.to_string())?;
                self.observations.push(ObservationRecord {
                    text: response.observation,
                    dimension: response.dimension,
                    framing: response.framing,
                    chunk_index: index,
                    z_score,
                });
            }
            Err(e) => {
                worker::console_log!("Ask pipeline failed for chunk {}: {}", index, e);
            }
        }
    }

    // 7. Store scores
    self.scores.push(scores);

    Ok(())
}
```

**Step 3: Implement end_session handler**

```rust
async fn handle_end_session(&mut self, ws: &WebSocket) -> Result<()> {
    // Rank observations by z-score (most notable first)
    let mut ranked = self.observations.clone();
    ranked.sort_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap_or(std::cmp::Ordering::Equal));
    let top_observations: Vec<_> = ranked.into_iter().take(3).collect();

    // Generate summary via teacher LLM
    let obs_text = top_observations.iter()
        .map(|o| format!("- {} ({}): {}", o.dimension, o.framing, o.text))
        .collect::<Vec<_>>()
        .join("\n");

    let summary = if !obs_text.is_empty() {
        let prompt = format!(
            "Summarize this piano practice session in 2-3 sentences. \
             Focus on the most important observations:\n\n{}",
            obs_text
        );
        match crate::services::llm::call_anthropic(
            &self.env, "You are a warm piano teacher summarizing a practice session.", &prompt, 200,
        ).await {
            Ok(s) => s,
            Err(_) => "Great session! Keep up the practice.".to_string(),
        }
    } else {
        "I listened to your practice but didn't notice any particular areas to highlight. Keep playing!".to_string()
    };

    // Write session to D1
    // ... (INSERT INTO sessions with scores, observations, summary)

    // Send summary
    let msg = serde_json::json!({
        "type": "session_summary",
        "observations": top_observations.iter().map(|o| serde_json::json!({
            "text": o.text,
            "dimension": o.dimension,
            "framing": o.framing,
        })).collect::<Vec<_>>(),
        "summary": summary,
    });
    ws.send_with_str(&msg.to_string())?;

    Ok(())
}
```

**Step 4: Build and test end-to-end**

Run: `cd apps/api && cargo build --target wasm32-unknown-unknown --release`

Manual test with wrangler dev:
1. `POST /api/practice/start` -> get sessionId
2. Connect WS to `/api/practice/ws/{sessionId}`
3. Upload a chunk via `POST /api/practice/chunk`
4. Send `chunk_ready` via WS
5. Wait for `chunk_processed` and possibly `observation`
6. Send `end_session`
7. Wait for `session_summary`

**Step 5: Commit**

```bash
git add apps/api/src/practice/ apps/api/src/services/ask.rs
git commit -m "feat(api): wire full practice pipeline -- upload, inference, teaching moments, observations"
```

---

## Task 7: Recording Overlay UI

Build the floating overlay component with waveform visualizer, timer, observation toasts, and stop button.

**Files:**
- Create: `apps/web/src/components/RecordingOverlay.tsx`
- Create: `apps/web/src/components/WaveformVisualizer.tsx`
- Create: `apps/web/src/components/ObservationToast.tsx`
- Modify: `apps/web/src/styles/app.css` (add overlay animations)

**Step 1: Add CSS animations**

In `apps/web/src/styles/app.css`, add:

```css
@keyframes slide-in-right {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes slide-out-right {
  from { opacity: 1; transform: translateX(0); }
  to { opacity: 0; transform: translateX(20px); }
}

@keyframes overlay-in {
  from { opacity: 0; transform: translateY(20px) scale(0.95); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

.animate-slide-in-right {
  animation: slide-in-right 400ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

.animate-slide-out-right {
  animation: slide-out-right 300ms ease-in both;
}

.animate-overlay-in {
  animation: overlay-in 500ms cubic-bezier(0.16, 1, 0.3, 1) both;
}
```

**Step 2: Create WaveformVisualizer**

Create `apps/web/src/components/WaveformVisualizer.tsx`:

```tsx
import { useRef, useEffect } from 'react'

interface WaveformVisualizerProps {
  analyserNode: AnalyserNode | null
  width?: number
  height?: number
}

export function WaveformVisualizer({ analyserNode, width = 280, height = 80 }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analyserNode) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const bufferLength = analyserNode.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    function draw() {
      animFrameRef.current = requestAnimationFrame(draw)
      analyserNode!.getByteFrequencyData(dataArray)

      ctx!.clearRect(0, 0, width, height)

      const barCount = 32
      const barWidth = (width / barCount) * 0.6
      const gap = (width / barCount) * 0.4
      const step = Math.floor(bufferLength / barCount)

      for (let i = 0; i < barCount; i++) {
        const value = dataArray[i * step] / 255
        const barHeight = Math.max(2, value * height * 0.85)

        const x = i * (barWidth + gap)
        const y = (height - barHeight) / 2

        ctx!.fillStyle = '#7A9A82' // sage/accent
        ctx!.beginPath()
        ctx!.roundRect(x, y, barWidth, barHeight, 2)
        ctx!.fill()
      }
    }

    draw()

    return () => cancelAnimationFrame(animFrameRef.current)
  }, [analyserNode, width, height])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="block"
    />
  )
}
```

**Step 3: Create ObservationToast**

Create `apps/web/src/components/ObservationToast.tsx`:

```tsx
import { useEffect, useState } from 'react'

interface ObservationToastProps {
  text: string
  dimension: string
  onDismiss: () => void
  autoHideMs?: number
}

export function ObservationToast({ text, dimension, onDismiss, autoHideMs = 8000 }: ObservationToastProps) {
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setExiting(true)
      setTimeout(onDismiss, 300) // Wait for exit animation
    }, autoHideMs)
    return () => clearTimeout(timer)
  }, [autoHideMs, onDismiss])

  return (
    <div
      className={`max-w-sm bg-surface-card border border-border rounded-xl px-4 py-3 shadow-card ${
        exiting ? 'animate-slide-out-right' : 'animate-slide-in-right'
      }`}
    >
      <span className="block text-body-xs text-accent font-medium uppercase tracking-wide mb-1">
        {dimension}
      </span>
      <p className="text-body-sm text-cream leading-relaxed">
        {text}
      </p>
    </div>
  )
}
```

**Step 4: Create RecordingOverlay**

Create `apps/web/src/components/RecordingOverlay.tsx`:

```tsx
import { useState, useCallback } from 'react'
import { Stop, CircleNotch } from '@phosphor-icons/react'
import { WaveformVisualizer } from './WaveformVisualizer'
import { ObservationToast } from './ObservationToast'
import type { PracticeState } from '../hooks/usePracticeSession'
import type { ObservationEvent } from '../lib/practice-api'

interface RecordingOverlayProps {
  state: PracticeState
  elapsedSeconds: number
  observations: ObservationEvent[]
  analyserNode: AnalyserNode | null
  error: string | null
  onStop: () => void
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function RecordingOverlay({
  state,
  elapsedSeconds,
  observations,
  analyserNode,
  error,
  onStop,
}: RecordingOverlayProps) {
  const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set())

  const handleDismiss = useCallback((idx: number) => {
    setDismissedIds((prev) => new Set(prev).add(idx))
  }, [])

  const visibleObservations = observations
    .map((obs, idx) => ({ ...obs, idx }))
    .filter(({ idx }) => !dismissedIds.has(idx))
    .slice(-3) // Show max 3 toasts at once

  const isConnecting = state === 'requesting-mic' || state === 'connecting'
  const isSummarizing = state === 'summarizing'

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center pb-32 pointer-events-none">
      {/* Overlay card */}
      <div className="pointer-events-auto bg-espresso/95 backdrop-blur-md border border-border rounded-3xl px-8 py-6 shadow-card animate-overlay-in flex flex-col items-center gap-4 min-w-[340px]">
        {/* Status text */}
        {isConnecting && (
          <div className="flex items-center gap-2 text-text-secondary text-body-sm">
            <CircleNotch size={16} className="animate-spin" />
            <span>Connecting...</span>
          </div>
        )}

        {isSummarizing && (
          <div className="flex items-center gap-2 text-text-secondary text-body-sm">
            <CircleNotch size={16} className="animate-spin" />
            <span>Generating summary...</span>
          </div>
        )}

        {/* Waveform */}
        {state === 'recording' && (
          <>
            <WaveformVisualizer analyserNode={analyserNode} />

            {/* Timer */}
            <span className="font-display text-display-sm text-cream tabular-nums">
              {formatTime(elapsedSeconds)}
            </span>

            {/* Stop button */}
            <button
              type="button"
              onClick={onStop}
              className="w-14 h-14 flex items-center justify-center rounded-full bg-red-600 hover:bg-red-500 text-cream transition"
              aria-label="Stop recording"
            >
              <Stop size={24} weight="fill" />
            </button>
          </>
        )}

        {/* Error */}
        {error && (
          <p className="text-body-sm text-red-400 text-center max-w-xs">{error}</p>
        )}
      </div>

      {/* Toast stack (positioned to the right) */}
      <div className="pointer-events-auto fixed right-6 bottom-32 flex flex-col gap-3">
        {visibleObservations.map(({ idx, text, dimension }) => (
          <ObservationToast
            key={idx}
            text={text}
            dimension={dimension}
            onDismiss={() => handleDismiss(idx)}
          />
        ))}
      </div>
    </div>
  )
}
```

**Step 5: Commit**

```bash
git add apps/web/src/components/RecordingOverlay.tsx apps/web/src/components/WaveformVisualizer.tsx apps/web/src/components/ObservationToast.tsx apps/web/src/styles/app.css
git commit -m "feat(web): add recording overlay with waveform visualizer and observation toasts"
```

---

## Task 8: Wire Overlay into AppChat + Session End Flow

Integrate the recording overlay and practice session hook into the main chat interface. Handle session end by posting summary to chat.

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`
- Modify: `apps/web/src/components/ChatInput.tsx`

**Step 1: Add onRecord callback to ChatInput**

In `apps/web/src/components/ChatInput.tsx`, add `onRecord` prop:

```tsx
interface ChatInputProps {
  onSend: (message: string) => void
  onRecord?: () => void  // NEW
  disabled: boolean
  placeholder?: string
  centered?: boolean
}
```

Wire the green button's onClick (line 67):

```tsx
<button
  type="button"
  onClick={onRecord}  // was undefined
  className="shrink-0 w-16 h-16 flex items-center justify-center rounded-full bg-accent text-cream hover:brightness-110 transition animate-pop-in"
  aria-label="Record audio"
>
  <Waveform size={24} />
</button>
```

**Step 2: Integrate usePracticeSession into AppChat**

In `apps/web/src/components/AppChat.tsx`, add:

```tsx
import { usePracticeSession } from '../hooks/usePracticeSession'
import { RecordingOverlay } from './RecordingOverlay'

// Inside AppChat component, after existing state declarations:
const practice = usePracticeSession()

// Handle recording button click
function handleRecord() {
  practice.start()
}

// When summary arrives, post it to chat
useEffect(() => {
  if (practice.summary) {
    // Create a system-style message with the session summary
    const summaryMsg: MessageRow = {
      id: `practice-${Date.now()}`,
      role: 'assistant',
      content: practice.summary,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, summaryMsg])
  }
}, [practice.summary])
```

Pass `onRecord` to ChatInput:

```tsx
<ChatInput
  onSend={handleSend}
  onRecord={handleRecord}  // NEW
  disabled={isStreaming || practice.state === 'recording'}
  placeholder="Message your teacher..."
  centered={false}
/>
```

Add overlay rendering before the closing `</div>` of the main content area:

```tsx
{practice.state !== 'idle' && (
  <RecordingOverlay
    state={practice.state}
    elapsedSeconds={practice.elapsedSeconds}
    observations={practice.observations}
    analyserNode={practice.analyserNode}
    error={practice.error}
    onStop={practice.stop}
  />
)}
```

**Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx apps/web/src/components/ChatInput.tsx
git commit -m "feat(web): wire recording overlay into chat -- click green button to start practice session"
```

---

## Task 9: Polish + Edge Cases

Handle edge cases, add session D1 persistence, and verify the full end-to-end flow.

**Files:**
- Modify: `apps/api/src/practice/session.rs` (D1 writes, error handling)
- Modify: `apps/web/src/hooks/usePracticeSession.ts` (edge cases)
- Modify: `apps/web/src/components/RecordingOverlay.tsx` (error recovery)

**Step 1: Add D1 session write on end**

In the DO's `handle_end_session`, after generating summary, write to D1:

```rust
let db = self.env.d1("DB")?;
let stmt = db.prepare(
    "INSERT INTO sessions (id, student_id, started_at, ended_at, avg_dynamics, avg_timing, avg_pedaling, avg_articulation, avg_phrasing, avg_interpretation) \
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"
);
// ... bind values from accumulated scores
stmt.bind(&[/* ... */])?.run().await?;
```

**Step 2: Handle edge cases in hook**

In `usePracticeSession.ts`:

- If WS drops mid-session: show error, allow restart
- If user navigates away (beforeunload): stop session gracefully
- If session is < 15s (no chunks): skip inference, show "Play for at least 15 seconds" message
- If inference timeout (>30s per chunk): skip that chunk, continue

**Step 3: Handle mic permission denied UI**

In `RecordingOverlay.tsx`, show a helpful message when `error` contains "Microphone":

```tsx
{error && error.includes('Microphone') && (
  <p className="text-body-sm text-text-secondary text-center max-w-xs">
    Open browser settings to allow microphone access for this site.
  </p>
)}
```

**Step 4: Full E2E test**

1. Open web app, sign in
2. Click green record button
3. See overlay appear with waveform responding to mic
4. Play piano for 30+ seconds
5. See observation toasts appear
6. Click stop
7. See "Generating summary..." state
8. See summary appear in chat
9. Type a follow-up message about the session

**Step 5: Commit**

```bash
git add apps/api/src/practice/ apps/web/src/hooks/ apps/web/src/components/
git commit -m "feat: polish practice session -- D1 persistence, edge cases, error recovery"
```

---

## Verification Checklist

- [ ] DO spike: WS echo works from browser console
- [ ] R2 upload: chunk stored and retrievable
- [ ] Inference round-trip: real audio chunk returns 19-dim scores
- [ ] Dim mapping: 19 dims correctly averaged to 6
- [ ] Teaching moment: outlier detected after 3+ chunks with deviation
- [ ] Observation: ask pipeline called, toast pushed via WS
- [ ] Overlay UI: waveform animates, timer counts, toasts appear and auto-dismiss
- [ ] Session end: observations ranked, summary generated, posted to chat
- [ ] D1: session record persisted
- [ ] Edge cases: mic denied, WS drop, short session, inference timeout
