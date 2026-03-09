use std::collections::HashMap;
use worker::*;

use crate::practice::teaching_moment::DimStats;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ObservationRecord {
    pub text: String,
    pub dimension: String,
    pub framing: String,
    pub chunk_index: usize,
    pub z_score: f64,
}

#[durable_object]
pub struct PracticeSession {
    state: State,
    env: Env,
    session_id: String,
    student_id: String,
    scores: Vec<HashMap<String, f64>>,
    observations: Vec<ObservationRecord>,
    dim_stats: DimStats,
}

impl DurableObject for PracticeSession {
    fn new(state: State, env: Env) -> Self {
        Self {
            state,
            env,
            session_id: String::new(),
            student_id: String::new(),
            scores: Vec::new(),
            observations: Vec::new(),
            dim_stats: DimStats::default(),
        }
    }

    async fn fetch(&self, req: Request) -> Result<Response> {
        // Extract session_id from URL path
        let url = req.url()?;
        let path = url.path();
        let session_id = path
            .strip_prefix("/ws/")
            .unwrap_or("")
            .to_string();

        // TODO: Extract auth token from query param and validate JWT
        // For now, accept all connections

        // Accept WebSocket upgrade
        let pair = WebSocketPair::new()?;
        let server = pair.server;

        // Tag the WebSocket with session info for later retrieval
        self.state.accept_web_socket(&server);

        // Send welcome message
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
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let _r2_key = parsed.get("r2Key").and_then(|v| v.as_str()).unwrap_or("");

                // For now, send a placeholder chunk_processed event.
                // Real inference integration happens in Task 5/6.
                let placeholder_scores = serde_json::json!({
                    "dynamics": 0.0,
                    "timing": 0.0,
                    "pedaling": 0.0,
                    "articulation": 0.0,
                    "phrasing": 0.0,
                    "interpretation": 0.0,
                });

                let response = serde_json::json!({
                    "type": "chunk_processed",
                    "index": index,
                    "scores": placeholder_scores,
                });
                ws.send_with_str(&response.to_string())?;
            }
            "end_session" => {
                // For now, send a placeholder summary.
                // Real summary generation happens in Task 6.
                let summary = serde_json::json!({
                    "type": "session_summary",
                    "observations": [],
                    "summary": "Practice session complete. Detailed analysis will be available soon.",
                });
                ws.send_with_str(&summary.to_string())?;
            }
            _ => {
                // Unknown message type, echo it back for debugging
                let echo = serde_json::json!({
                    "type": "echo",
                    "data": parsed,
                });
                ws.send_with_str(&echo.to_string())?;
            }
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
        Ok(())
    }
}
