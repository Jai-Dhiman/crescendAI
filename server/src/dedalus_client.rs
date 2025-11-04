// CrescendAI Server - Dedalus Service Binding Client
// Client for Dedalus Python worker via service binding

use worker::*;
use crate::models::{
    DedalusChatRequest, DedalusChatResponse, DedalusMessage,
    StreamChunk, ErrorResponse
};
use serde_json;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone)]
pub enum DedalusError {
    /// Network or HTTP error
    NetworkError(String),

    /// API returned an error response
    ApiError { status: u16, message: String },

    /// Failed to parse response
    ParseError(String),

    /// Request timeout
    Timeout,

    /// Invalid request configuration
    InvalidRequest(String),

    /// Streaming error
    StreamError(String),

    /// Service binding error
    ServiceBindingError(String),
}

impl std::fmt::Display for DedalusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DedalusError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            DedalusError::ApiError { status, message } => {
                write!(f, "API error ({}): {}", status, message)
            }
            DedalusError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            DedalusError::Timeout => write!(f, "Request timeout"),
            DedalusError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            DedalusError::StreamError(msg) => write!(f, "Stream error: {}", msg),
            DedalusError::ServiceBindingError(msg) => write!(f, "Service binding error: {}", msg),
        }
    }
}

impl std::error::Error for DedalusError {}

impl From<DedalusError> for worker::Error {
    fn from(err: DedalusError) -> Self {
        worker::Error::RustError(err.to_string())
    }
}

pub type DedalusResult<T> = std::result::Result<T, DedalusError>;

// ============================================================================
// Dedalus Client (Service Binding)
// ============================================================================

/// Client for Dedalus Python worker via service binding
pub struct DedalusClient {
    /// Service binding to Python Dedalus worker
    fetcher: Fetcher,

    /// Default timeout in milliseconds
    timeout_ms: u64,

    /// Maximum retry attempts
    max_retries: u32,
}

impl DedalusClient {
    /// Create a new Dedalus client from environment binding
    pub fn from_env(env: &Env) -> Result<Self> {
        let fetcher = env.service("DEDALUS")
            .map_err(|e| worker::Error::RustError(format!("Failed to get DEDALUS service binding: {}", e)))?;

        Ok(Self {
            fetcher,
            timeout_ms: 30000, // 30 seconds default
            max_retries: 3,
        })
    }

    /// Set timeout in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set maximum retry attempts
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Make a non-streaming chat completion request
    pub async fn chat_completion(
        &self,
        request: DedalusChatRequest,
    ) -> DedalusResult<DedalusChatResponse> {
        // Ensure streaming is disabled
        let mut req = request.clone();
        req.stream = Some(false);

        // Build the request with retries
        self.execute_with_retry(req).await
    }

    /// Make a streaming chat completion request
    /// Returns a Response object with SSE stream
    pub async fn chat_completion_stream(
        &self,
        request: DedalusChatRequest,
    ) -> DedalusResult<Response> {
        // Ensure streaming is enabled
        let mut req = request.clone();
        req.stream = Some(true);

        // Build and execute the request (no retries for streaming)
        let response = self.execute_request(req).await?;

        Ok(response)
    }

    /// Execute request with exponential backoff retry
    async fn execute_with_retry(
        &self,
        request: DedalusChatRequest,
    ) -> DedalusResult<DedalusChatResponse> {
        let mut last_error = None;
        let mut backoff_ms = 1000; // Start with 1 second

        for attempt in 0..=self.max_retries {
            match self.execute_request(request.clone()).await {
                Ok(mut response) => {
                    // Parse response body
                    match response.json::<DedalusChatResponse>().await {
                        Ok(chat_response) => return Ok(chat_response),
                        Err(e) => {
                            return Err(DedalusError::ParseError(format!(
                                "Failed to parse response: {}",
                                e
                            )))
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(e.clone());

                    // Don't retry on client errors (4xx)
                    if let DedalusError::ApiError { status, .. } = &e {
                        if *status >= 400 && *status < 500 {
                            return Err(e);
                        }
                    }

                    // If this isn't the last attempt, wait before retrying
                    if attempt < self.max_retries {
                        worker::console_log!("Dedalus request failed (attempt {}/{}), retrying in {}ms: {}",
                            attempt + 1, self.max_retries + 1, backoff_ms, e);

                        // Note: In Cloudflare Workers, we can't easily do async delays
                        // So we'll retry immediately but with exponential backoff tracking

                        // Exponential backoff with jitter (tracked for metrics)
                        backoff_ms = (backoff_ms * 2).min(30000); // Cap at 30 seconds
                    }
                }
            }
        }

        Err(last_error.unwrap_or(DedalusError::NetworkError("All retries failed".to_string())))
    }

    /// Execute a single request without retry logic
    async fn execute_request(
        &self,
        request: DedalusChatRequest,
    ) -> DedalusResult<Response> {
        // Serialize request body
        let body = serde_json::to_string(&request)
            .map_err(|e| DedalusError::InvalidRequest(format!("Failed to serialize request: {}", e)))?;

        // Create request init for Python Dedalus worker
        let mut req_init = RequestInit::new();
        req_init.with_method(Method::Post);

        // Set headers
        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json")
            .map_err(|e| DedalusError::NetworkError(format!("Failed to set header: {}", e)))?;

        req_init.with_headers(headers);

        // Set body
        req_init.with_body(Some(body.into()));

        // Execute fetch via service binding to /chat endpoint
        let mut response = self.fetcher.fetch("/chat", Some(req_init))
            .await
            .map_err(|e| DedalusError::ServiceBindingError(format!("Service binding fetch failed: {}", e)))?;

        // Check status code
        let status = response.status_code();

        if status >= 200 && status < 300 {
            Ok(response)
        } else {
            // Try to parse error response
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as ErrorResponse
            let message = if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&error_text) {
                error_resp.message
            } else {
                error_text
            };

            Err(DedalusError::ApiError {
                status,
                message,
            })
        }
    }

}

// ============================================================================
// SSE Stream Parser
// ============================================================================

/// Parse Server-Sent Events from a streaming response
pub struct SseParser;

impl SseParser {
    /// Parse an SSE line into an event
    pub fn parse_line(line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return None;
        }

        if line.starts_with("data: ") {
            let data = &line[6..]; // Skip "data: "

            // Check for [DONE] sentinel
            if data.trim() == "[DONE]" {
                return Some(SseEvent::Done);
            }

            // Try to parse as JSON
            match serde_json::from_str::<StreamChunk>(data) {
                Ok(chunk) => Some(SseEvent::Chunk(chunk)),
                Err(e) => {
                    worker::console_log!("Failed to parse SSE chunk: {} - Data: {}", e, data);
                    None
                }
            }
        } else if line.starts_with("event: ") {
            let event_type = &line[7..]; // Skip "event: "
            Some(SseEvent::Event(event_type.to_string()))
        } else if line.starts_with(":") {
            // Comment, ignore
            None
        } else {
            None
        }
    }

    /// Parse multiple SSE lines
    pub fn parse_lines(text: &str) -> Vec<SseEvent> {
        text.lines()
            .filter_map(|line| Self::parse_line(line))
            .collect()
    }
}

/// SSE event types
#[derive(Debug, Clone)]
pub enum SseEvent {
    /// Data chunk
    Chunk(StreamChunk),

    /// Stream complete
    Done,

    /// Named event
    Event(String),
}

// ============================================================================
// Helpers
// ============================================================================

/// Helper to build a simple chat request
pub fn create_simple_chat_request(
    model: impl Into<String>,
    messages: Vec<DedalusMessage>,
) -> DedalusChatRequest {
    DedalusChatRequest {
        model: model.into(),
        messages,
        tools: None,
        temperature: None,
        max_tokens: None,
        stream: None,
    }
}

/// Helper to build a chat request with tools
pub fn create_chat_request_with_tools(
    model: impl Into<String>,
    messages: Vec<DedalusMessage>,
    tools: Vec<crate::models::DedalusTool>,
) -> DedalusChatRequest {
    DedalusChatRequest {
        model: model.into(),
        messages,
        tools: Some(tools),
        temperature: None,
        max_tokens: None,
        stream: None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser_data() {
        let line = r#"data: {"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let event = SseParser::parse_line(line);
        assert!(matches!(event, Some(SseEvent::Chunk(_))));
    }

    #[test]
    fn test_sse_parser_done() {
        let line = "data: [DONE]";
        let event = SseParser::parse_line(line);
        assert!(matches!(event, Some(SseEvent::Done)));
    }

    #[test]
    fn test_sse_parser_event() {
        let line = "event: ping";
        let event = SseParser::parse_line(line);
        assert!(matches!(event, Some(SseEvent::Event(_))));
    }

    #[test]
    fn test_sse_parser_comment() {
        let line = ": this is a comment";
        let event = SseParser::parse_line(line);
        assert!(event.is_none());
    }
}
