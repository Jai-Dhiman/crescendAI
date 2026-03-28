//! Durable Object error types.

#[derive(Debug, thiserror::Error)]
pub enum PracticeError {
    #[error("storage: {0}")]
    Storage(String),

    #[error("inference: {0}")]
    Inference(String),

    #[error("piece identification: {0}")]
    PieceId(String),

    #[error("synthesis: {0}")]
    Synthesis(String),

    #[error("websocket: {0}")]
    WebSocket(String),
}
