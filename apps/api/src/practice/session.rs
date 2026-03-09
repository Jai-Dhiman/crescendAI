use worker::*;

#[durable_object]
pub struct PracticeSession {
    state: State,
    env: Env,
}

impl DurableObject for PracticeSession {
    fn new(state: State, env: Env) -> Self {
        Self { state, env }
    }

    async fn fetch(&self, _req: Request) -> Result<Response> {
        // Accept WebSocket upgrade
        let pair = WebSocketPair::new()?;
        let server = pair.server;
        self.state.accept_web_socket(&server);

        // Send a welcome message
        server.send_with_str(r#"{"type":"connected"}"#)?;

        Response::from_websocket(pair.client)
    }

    async fn websocket_message(&self, ws: WebSocket, msg: WebSocketIncomingMessage) -> Result<()> {
        // Echo for spike
        let text = match msg {
            WebSocketIncomingMessage::String(s) => s,
            WebSocketIncomingMessage::Binary(b) => {
                format!("\"<binary {} bytes>\"", b.len())
            }
        };
        ws.send_with_str(&format!(r#"{{"type":"echo","data":{}}}"#, text))?;
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
