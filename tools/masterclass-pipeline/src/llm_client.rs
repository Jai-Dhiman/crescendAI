use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Generic LLM client that talks the OpenAI chat completions protocol.
/// Works with Ollama, llama.cpp server, vLLM, LM Studio, or any
/// OpenAI-compatible endpoint.
pub struct LlmClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: Option<serde_json::Value>,
}

const MAX_RETRIES: u32 = 3;
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

impl LlmClient {
    pub fn new(base_url: Option<&str>, model: &str) -> Result<Self> {
        let base_url = base_url
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_string();

        // Use a long timeout -- local models can be slow
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .with_context(|| "Failed to create HTTP client")?;

        tracing::info!("LLM client: {} at {}", model, base_url);

        Ok(Self {
            client,
            base_url,
            model: model.to_string(),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn message(&self, system: &str, user: &str) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            stream: false,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                },
            ],
        };

        let mut last_error = None;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = Duration::from_secs(2u64.pow(attempt));
                tracing::info!(
                    "Retrying LLM request in {:?} (attempt {})",
                    delay,
                    attempt + 1
                );
                tokio::time::sleep(delay).await;
            }

            match self.send_request(&request).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("429")
                        || err_str.contains("500")
                        || err_str.contains("503")
                        || err_str.contains("timed out")
                        || err_str.contains("connection refused")
                    {
                        tracing::warn!(
                            "Retryable error on attempt {}: {}",
                            attempt + 1,
                            err_str
                        );
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("LLM request failed after {} retries", MAX_RETRIES)
        }))
    }

    async fn send_request(&self, request: &ChatRequest) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to LLM at {}. Is the server running?",
                    self.base_url
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let error_msg = serde_json::from_str::<ErrorResponse>(&body)
                .ok()
                .and_then(|e| e.error)
                .map(|e| e.to_string())
                .unwrap_or_else(|| body.clone());
            anyhow::bail!("LLM API error ({}): {}", status, error_msg);
        }

        let resp: ChatResponse = response
            .json()
            .await
            .with_context(|| "Failed to parse LLM response")?;

        let text = resp
            .choices
            .into_iter()
            .filter_map(|c| c.message.content)
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            anyhow::bail!("LLM returned empty response");
        }

        Ok(text)
    }
}
