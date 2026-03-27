//! HTTP clients for Groq and Anthropic LLM APIs.
//!
//! Uses worker::Fetch for WASM-compatible HTTP requests.

use serde::{Deserialize, Serialize};
use worker::{console_log, Env, Fetch, Headers, Method, Request, RequestInit, Url};

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
struct AiGateway {
    account_id: String,
    gateway_id: &'static str,
}

impl AiGateway {
    /// Create a gateway client for the teacher path (Anthropic only).
    fn teacher(env: &Env) -> Result<Self, String> {
        let account_id = env
            .var("CF_ACCOUNT_ID")
            .map_err(|_| "CF_ACCOUNT_ID not configured".to_string())?
            .to_string();
        Ok(Self {
            account_id,
            gateway_id: TEACHER_GATEWAY,
        })
    }

    /// Create a gateway client for background tasks (Groq + Workers AI).
    fn background(env: &Env) -> Result<Self, String> {
        let account_id = env
            .var("CF_ACCOUNT_ID")
            .map_err(|_| "CF_ACCOUNT_ID not configured".to_string())?
            .to_string();
        Ok(Self {
            account_id,
            gateway_id: BACKGROUND_GATEWAY,
        })
    }

    /// Build a provider-specific gateway URL.
    /// Example: https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/anthropic/v1/messages
    fn provider_url(&self, provider: &str, path: &str) -> Result<Url, String> {
        let url_str = format!(
            "https://gateway.ai.cloudflare.com/v1/{}/{}/{}/{}",
            self.account_id, self.gateway_id, provider, path
        );
        url_str
            .parse()
            .map_err(|e| format!("Invalid gateway URL: {:?}", e))
    }

    /// Attach gateway-specific headers (caching) to an existing Headers object.
    fn attach_headers(&self, headers: &Headers) -> Result<(), String> {
        headers
            .set("cf-aig-cache-ttl", &GATEWAY_CACHE_TTL.to_string())
            .map_err(|e| format!("Failed to set cache TTL header: {:?}", e))?;
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

// --- Groq (Stage 1: Subagent) ---

#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<LlmMessage>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Deserialize)]
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
) -> Result<String, String> {
    let api_key = env
        .secret("GROQ_API_KEY")
        .map_err(|_| "GROQ_API_KEY not configured".to_string())?
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

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Groq request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("Authorization", &format!("Bearer {}", api_key))
        .map_err(|e| format!("Failed to set auth header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.groq.com/openai/v1/chat/completions"
        .parse()
        .map_err(|e| format!("Invalid Groq URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Groq request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Groq request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Groq response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Groq returned status {}: {}",
            status, response_text
        ));
    }

    let groq_response: GroqResponse = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Failed to parse Groq response: {:?} - body: {}",
            e, response_text
        )
    })?;

    groq_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| "No choices in Groq response".to_string())
}

// --- Anthropic (Stage 2: Teacher) ---

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<LlmMessage>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
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
) -> Result<String, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
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

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| format!("Failed to set api-key header: {:?}", e))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| format!("Failed to set version header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.anthropic.com/v1/messages"
        .parse()
        .map_err(|e| format!("Invalid Anthropic URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Anthropic response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Anthropic returned status {}: {}",
            status, response_text
        ));
    }

    let anthropic_response: AnthropicResponse =
        serde_json::from_str(&response_text).map_err(|e| {
            format!(
                "Failed to parse Anthropic response: {:?} - body: {}",
                e, response_text
            )
        })?;

    anthropic_response
        .content
        .first()
        .map(|c| c.text.clone())
        .ok_or_else(|| "No content in Anthropic response".to_string())
}

// --- Anthropic Tool Use ---

#[derive(Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Serialize)]
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
pub struct AnthropicToolResult {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

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
) -> Result<AnthropicToolResult, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
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

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic tool request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| format!("Failed to set api-key header: {:?}", e))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| format!("Failed to set version header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.anthropic.com/v1/messages"
        .parse()
        .map_err(|e| format!("Invalid Anthropic URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic tool request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic tool request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Anthropic tool response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Anthropic returned status {}: {}",
            status, response_text
        ));
    }

    let parsed: AnthropicToolResponse = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Failed to parse Anthropic tool response: {:?} - body: {}",
            e, response_text
        )
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

// --- Workers AI (cheap background tasks) ---

#[derive(Serialize)]
struct WorkersAiRequest {
    messages: Vec<WorkersAiMessage>,
    max_tokens: u32,
    temperature: f64,
}

#[derive(Serialize)]
struct WorkersAiMessage {
    role: String,
    content: String,
}

/// Call Cloudflare Workers AI with a system prompt and user prompt.
/// Uses Llama 3.3 70B (FP8) -- cheap, no external API call, runs in the same runtime.
/// Best for background/fire-and-forget tasks where speed is not critical.
pub async fn call_workers_ai(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String, String> {
    let ai = env
        .ai("AI")
        .map_err(|e| format!("AI binding failed: {:?}", e))?;

    let request = WorkersAiRequest {
        messages: vec![
            WorkersAiMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            WorkersAiMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ],
        max_tokens,
        temperature,
    };

    let result: serde_json::Value = ai
        .run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", request)
        .await
        .map_err(|e| format!("Workers AI inference failed: {:?}", e))?;

    // Workers AI returns {"response": "..."} for text generation models.
    // When the model outputs valid JSON, serde_wasm_bindgen may deserialize
    // the response field as an object instead of a string.
    match result.get("response") {
        Some(serde_json::Value::String(text)) => Ok(text.clone()),
        Some(value) => {
            // Response came back as parsed object -- re-serialize to string
            Ok(serde_json::to_string(value)
                .unwrap_or_else(|_| format!("{:?}", value)))
        }
        None => {
            // No response field -- return the whole thing
            Ok(serde_json::to_string(&result)
                .unwrap_or_else(|_| format!("{:?}", result)))
        }
    }
}

// --- Shared ---

#[derive(Serialize, Deserialize, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

// --- Anthropic Streaming ---

#[derive(Serialize)]
pub struct AnthropicStreamRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: Vec<AnthropicSystemBlock>,
    pub messages: Vec<LlmMessage>,
    pub stream: bool,
}

#[derive(Serialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
}

/// Make a streaming request to Anthropic. Returns the raw worker::Response
/// so the caller can process the SSE stream.
pub async fn call_anthropic_stream(
    env: &Env,
    system_prompt: &str,
    messages: Vec<LlmMessage>,
    max_tokens: u32,
) -> Result<worker::Response, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
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

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic stream request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| format!("Failed to set api-key header: {:?}", e))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| format!("Failed to set version header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.anthropic.com/v1/messages"
        .parse()
        .map_err(|e| format!("Invalid Anthropic URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic stream request: {:?}", e))?;

    let response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic stream request failed: {:?}", e))?;

    let status = response.status_code();
    if status != 200 {
        return Err(format!("Anthropic streaming returned status {}", status));
    }

    Ok(response)
}
