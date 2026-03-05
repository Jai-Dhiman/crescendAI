//! HTTP clients for Groq and Anthropic LLM APIs.
//!
//! Uses worker::Fetch for WASM-compatible HTTP requests.

use serde::{Deserialize, Serialize};
use worker::{console_log, Env, Fetch, Headers, Method, Request, RequestInit, Url};

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
        return Err(format!("Groq returned status {}: {}", status, response_text));
    }

    let groq_response: GroqResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse Groq response: {:?} - body: {}", e, response_text))?;

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
        model: "claude-sonnet-4-6-20250514".to_string(),
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

    let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)
        .map_err(|e| {
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

// --- Shared ---

#[derive(Serialize, Deserialize)]
struct LlmMessage {
    role: String,
    content: String,
}
