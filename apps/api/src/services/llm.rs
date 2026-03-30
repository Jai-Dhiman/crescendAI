//! HTTP clients for Groq and Anthropic LLM APIs.
//!
//! Uses `worker::Fetch` for WASM-compatible HTTP requests.

use js_sys;
use serde::{Deserialize, Serialize};
use worker::{console_log, Env, Fetch, Headers, Method, Request, RequestInit, Url};

use crate::error::{ApiError, Result};

// --- AI Gateway ---

/// Gateway IDs (created in CF dashboard)
const TEACHER_GATEWAY: &str = "crescendai-teacher";
const BACKGROUND_GATEWAY: &str = "crescendai-background";

/// Default cache TTL in seconds for all gateway requests
const GATEWAY_CACHE_TTL: u32 = 60;

/// Workers AI model for cheap background tasks (titles, goals, memory extraction)
pub const WORKERS_AI_CHEAP_MODEL: &str = "@cf/qwen/qwen3-30b-a3b-fp8";

/// Workers AI model matching Groq's Llama 3.3 70B (used as Groq fallback + shadow benchmark)
const WORKERS_AI_GROQ_FALLBACK_MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

/// Thin client for routing LLM calls through Cloudflare AI Gateway.
/// Constructs gateway URLs, attaches caching headers, logs response metadata.
#[derive(Debug)]
struct AiGateway {
    account_id: String,
    gateway_id: &'static str,
}

impl AiGateway {
    /// Create a gateway client for the teacher path (Anthropic only).
    fn teacher(env: &Env) -> Result<Self> {
        let account_id = Self::read_account_id(env)?;
        Ok(Self {
            account_id,
            gateway_id: TEACHER_GATEWAY,
        })
    }

    /// Create a gateway client for background tasks (Groq + Workers AI).
    fn background(env: &Env) -> Result<Self> {
        let account_id = Self::read_account_id(env)?;
        Ok(Self {
            account_id,
            gateway_id: BACKGROUND_GATEWAY,
        })
    }

    /// Read and validate `CF_ACCOUNT_ID` from environment.
    fn read_account_id(env: &Env) -> Result<String> {
        let account_id = env
            .var("CF_ACCOUNT_ID")
            .map_err(|_| ApiError::ExternalService("CF_ACCOUNT_ID not configured".to_string()))?
            .to_string();
        if account_id.is_empty() {
            return Err(ApiError::ExternalService(
                "CF_ACCOUNT_ID is empty -- set it via wrangler secret or CF dashboard".to_string(),
            ));
        }
        Ok(account_id)
    }

    /// Build a provider-specific gateway URL.
    /// Example: <https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/anthropic/v1/messages>
    fn provider_url(&self, provider: &str, path: &str) -> Result<Url> {
        let url_str = format!(
            "https://gateway.ai.cloudflare.com/v1/{}/{}/{}/{}",
            self.account_id, self.gateway_id, provider, path
        );
        url_str
            .parse()
            .map_err(|e| ApiError::ExternalService(format!("Invalid gateway URL: {e:?}")))
    }

    /// Attach gateway-specific headers (caching) to an existing Headers object.
    #[allow(clippy::unused_self)] // method for consistency; may use self for per-route config later
    fn attach_headers(&self, headers: &Headers) -> Result<()> {
        headers
            .set("cf-aig-cache-ttl", &GATEWAY_CACHE_TTL.to_string())
            .map_err(|e| {
                ApiError::ExternalService(format!("Failed to set cache TTL header: {e:?}"))
            })?;
        Ok(())
    }

    /// Log gateway response metadata (cache status, provider step).
    fn log_response_metadata(&self, response: &worker::Response, label: &str) {
        let cache_status = response
            .headers()
            .get("cf-aig-cache-status")
            .ok()
            .flatten()
            .unwrap_or_else(|| "unknown".to_string());
        let step = response
            .headers()
            .get("cf-aig-step")
            .ok()
            .flatten()
            .unwrap_or_else(|| "0".to_string());
        console_log!(
            "gateway[{}] {}: cache={}, provider_step={}",
            self.gateway_id,
            label,
            cache_status,
            step
        );
    }
}

/// Check if shadow benchmarking should fire for this request.
/// Returns true with probability = `SHADOW_BENCHMARK_PCT` / 100.
fn should_shadow_benchmark(env: &Env) -> bool {
    let enabled = env
        .var("SHADOW_BENCHMARK_ENABLED")
        .ok()
        .is_some_and(|v| v.to_string() == "true");
    if !enabled {
        return false;
    }
    let pct: u32 = env
        .var("SHADOW_BENCHMARK_PCT")
        .ok()
        .and_then(|v| v.to_string().parse().ok())
        .unwrap_or(10);
    let mut buf = [0u8; 1];
    if getrandom::getrandom(&mut buf).is_err() {
        return false;
    }
    let roll = (u32::from(buf[0]) * 100) / 256;
    roll < pct
}

// --- Groq (Stage 1: Subagent) ---

#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<LlmMessage>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Debug, Deserialize)]
struct GroqMessage {
    content: String,
}

/// Call the Groq API with a system prompt and user prompt.
/// Returns the assistant's response text.
pub async fn call_groq(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String> {
    let gateway = AiGateway::background(env)?;
    let run_shadow = should_shadow_benchmark(env);
    let api_key = env
        .secret("GROQ_API_KEY")
        .map_err(|_| ApiError::ExternalService("GROQ_API_KEY not configured".to_string()))?
        .to_string();

    let request_body = GroqRequest {
        model: "llama-3.3-70b-versatile".to_string(),
        messages: vec![
            LlmMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            LlmMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ],
        temperature,
        max_tokens,
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| {
        ApiError::ExternalService(format!("Failed to serialize Groq request: {e:?}"))
    })?;

    let headers = Headers::new();
    headers
        .set("Authorization", &format!("Bearer {api_key}"))
        .map_err(|e| ApiError::ExternalService(format!("Failed to set auth header: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set content-type: {e:?}")))?;
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("groq", "openai/v1/chat/completions")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| ApiError::ExternalService(format!("Failed to create Groq request: {e:?}")))?;

    let groq_start = js_sys::Date::now();
    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Groq request failed: {e:?}")))?;
    let groq_latency = js_sys::Date::now() - groq_start;
    gateway.log_response_metadata(&response, "groq");

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Failed to read Groq response: {e:?}")))?;

    if status != 200 {
        return Err(ApiError::ExternalService(format!(
            "Groq returned status {status}: {response_text}"
        )));
    }

    let groq_response: GroqResponse = serde_json::from_str(&response_text).map_err(|e| {
        ApiError::ExternalService(format!(
            "Failed to parse Groq response: {e:?} - body: {response_text}"
        ))
    })?;

    // Shadow benchmark: measure Workers AI latency for comparison
    if run_shadow {
        let shadow_start = js_sys::Date::now();
        let _ = call_workers_ai(
            env,
            WORKERS_AI_GROQ_FALLBACK_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
        )
        .await;
        let shadow_elapsed = js_sys::Date::now() - shadow_start;
        console_log!(
            "shadow_benchmark: groq_latency_ms={:.0}, workers_ai_latency_ms={:.0}",
            groq_latency,
            shadow_elapsed
        );
    }

    groq_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| ApiError::ExternalService("No choices in Groq response".to_string()))
}

// --- Anthropic (Stage 2: Teacher) ---

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<LlmMessage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

/// Call the Anthropic API with a system prompt and user prompt.
/// Returns the assistant's response text.
pub async fn call_anthropic(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
) -> Result<String> {
    let gateway = AiGateway::teacher(env)?;
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| ApiError::ExternalService("ANTHROPIC_API_KEY not configured".to_string()))?
        .to_string();

    let request_body = AnthropicRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: system_prompt.to_string(),
        messages: vec![LlmMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }],
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| {
        ApiError::ExternalService(format!("Failed to serialize Anthropic request: {e:?}"))
    })?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| ApiError::ExternalService(format!("Failed to set api-key header: {e:?}")))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set version header: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set content-type: {e:?}")))?;
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("anthropic", "v1/messages")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init).map_err(|e| {
        ApiError::ExternalService(format!("Failed to create Anthropic request: {e:?}"))
    })?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Anthropic request failed: {e:?}")))?;
    gateway.log_response_metadata(&response, "anthropic");

    let status = response.status_code();
    let response_text = response.text().await.map_err(|e| {
        ApiError::ExternalService(format!("Failed to read Anthropic response: {e:?}"))
    })?;

    if status != 200 {
        return Err(ApiError::ExternalService(format!(
            "Anthropic returned status {status}: {response_text}"
        )));
    }

    let anthropic_response: AnthropicResponse =
        serde_json::from_str(&response_text).map_err(|e| {
            ApiError::ExternalService(format!(
                "Failed to parse Anthropic response: {e:?} - body: {response_text}"
            ))
        })?;

    anthropic_response
        .content
        .first()
        .map(|c| c.text.clone())
        .ok_or_else(|| ApiError::ExternalService("No content in Anthropic response".to_string()))
}

// --- Anthropic Tool Use ---

#[derive(Debug, Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicToolRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<LlmMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
pub struct AnthropicToolResponse {
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Parsed result from a tool-enabled Anthropic call.
#[derive(Debug)]
pub struct AnthropicToolResult {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

pub async fn call_anthropic_with_tools(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    tools: Option<Vec<AnthropicTool>>,
) -> Result<AnthropicToolResult> {
    let gateway = AiGateway::teacher(env)?;
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| ApiError::ExternalService("ANTHROPIC_API_KEY not configured".to_string()))?
        .to_string();

    let tool_choice = tools.as_ref().map(|_| serde_json::json!({"type": "auto"}));

    let request_body = AnthropicToolRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: system_prompt.to_string(),
        messages: vec![LlmMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }],
        tools,
        tool_choice,
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| {
        ApiError::ExternalService(format!("Failed to serialize Anthropic tool request: {e:?}"))
    })?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| ApiError::ExternalService(format!("Failed to set api-key header: {e:?}")))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set version header: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set content-type: {e:?}")))?;
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("anthropic", "v1/messages")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init).map_err(|e| {
        ApiError::ExternalService(format!("Failed to create Anthropic tool request: {e:?}"))
    })?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Anthropic tool request failed: {e:?}")))?;
    gateway.log_response_metadata(&response, "anthropic-tools");

    let status = response.status_code();
    let response_text = response.text().await.map_err(|e| {
        ApiError::ExternalService(format!("Failed to read Anthropic tool response: {e:?}"))
    })?;

    if status != 200 {
        return Err(ApiError::ExternalService(format!(
            "Anthropic returned status {status}: {response_text}"
        )));
    }

    let parsed: AnthropicToolResponse = serde_json::from_str(&response_text).map_err(|e| {
        ApiError::ExternalService(format!(
            "Failed to parse Anthropic tool response: {e:?} - body: {response_text}"
        ))
    })?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in parsed.content {
        match block {
            AnthropicContentBlock::Text { text } => text_parts.push(text),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall { id, name, input });
            }
        }
    }

    Ok(AnthropicToolResult {
        text: text_parts.join(""),
        tool_calls,
    })
}

// --- Workers AI (cheap background tasks, via HTTP gateway) ---

/// Call Cloudflare Workers AI through the AI Gateway with a system prompt and user prompt.
/// Uses an OpenAI-compatible format routed via the background gateway for unified observability.
/// Best for background/fire-and-forget tasks where speed is not critical.
pub async fn call_workers_ai(
    env: &Env,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String> {
    let gateway = AiGateway::background(env)?;

    // Workers AI through the gateway uses OpenAI-compatible format
    let request_body = GroqRequest {
        model: model.to_string(),
        messages: vec![
            LlmMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            LlmMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ],
        temperature,
        max_tokens,
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| {
        ApiError::ExternalService(format!("Failed to serialize Workers AI request: {e:?}"))
    })?;

    let cf_api_token = env
        .secret("CF_API_TOKEN")
        .map_err(|_| ApiError::ExternalService("CF_API_TOKEN not configured".to_string()))?
        .to_string();

    let headers = Headers::new();
    headers
        .set("Authorization", &format!("Bearer {cf_api_token}"))
        .map_err(|e| ApiError::ExternalService(format!("Failed to set auth header: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set content-type: {e:?}")))?;
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("workers-ai", "v1/chat/completions")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init).map_err(|e| {
        ApiError::ExternalService(format!("Failed to create Workers AI request: {e:?}"))
    })?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| ApiError::ExternalService(format!("Workers AI request failed: {e:?}")))?;

    gateway.log_response_metadata(&response, "workers-ai");

    let status = response.status_code();
    let response_text = response.text().await.map_err(|e| {
        ApiError::ExternalService(format!("Failed to read Workers AI response: {e:?}"))
    })?;

    if status != 200 {
        return Err(ApiError::ExternalService(format!(
            "Workers AI returned status {status}: {response_text}"
        )));
    }

    // Workers AI through the gateway returns OpenAI-compatible format
    let parsed: GroqResponse = serde_json::from_str(&response_text).map_err(|e| {
        ApiError::ExternalService(format!(
            "Failed to parse Workers AI response: {e:?} - body: {response_text}"
        ))
    })?;

    parsed
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| ApiError::ExternalService("No choices in Workers AI response".to_string()))
}

// --- Shared ---

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

// --- Anthropic Streaming ---

#[derive(Debug, Serialize)]
pub struct AnthropicStreamRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: Vec<AnthropicSystemBlock>,
    pub messages: Vec<LlmMessage>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
}

/// Make a streaming request to Anthropic. Returns the raw `worker::Response`
/// so the caller can process the SSE stream.
pub async fn call_anthropic_stream(
    env: &Env,
    system_prompt: &str,
    messages: Vec<LlmMessage>,
    max_tokens: u32,
) -> Result<worker::Response> {
    let gateway = AiGateway::teacher(env)?;
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| ApiError::ExternalService("ANTHROPIC_API_KEY not configured".to_string()))?
        .to_string();

    let request_body = AnthropicStreamRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: vec![AnthropicSystemBlock {
            block_type: "text".to_string(),
            text: system_prompt.to_string(),
            cache_control: Some(CacheControl {
                control_type: "ephemeral".to_string(),
            }),
        }],
        messages,
        stream: true,
    };

    let body_json = serde_json::to_string(&request_body).map_err(|e| {
        ApiError::ExternalService(format!(
            "Failed to serialize Anthropic stream request: {e:?}"
        ))
    })?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| ApiError::ExternalService(format!("Failed to set api-key header: {e:?}")))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set version header: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| ApiError::ExternalService(format!("Failed to set content-type: {e:?}")))?;
    // Skip cache headers for streaming -- cached SSE replay may not behave correctly

    let url = gateway.provider_url("anthropic", "v1/messages")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init).map_err(|e| {
        ApiError::ExternalService(format!("Failed to create Anthropic stream request: {e:?}"))
    })?;

    let response = Fetch::Request(request).send().await.map_err(|e| {
        ApiError::ExternalService(format!("Anthropic stream request failed: {e:?}"))
    })?;
    gateway.log_response_metadata(&response, "anthropic-stream");

    let status = response.status_code();
    if status != 200 {
        return Err(ApiError::ExternalService(format!(
            "Anthropic streaming returned status {status}"
        )));
    }

    Ok(response)
}
