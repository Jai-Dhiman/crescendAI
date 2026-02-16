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
    api_key: Option<String>,
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

const MAX_RETRIES: u32 = 6;
#[cfg(not(test))]
const INITIAL_BACKOFF_SECS: u64 = 4;
#[cfg(test)]
const INITIAL_BACKOFF_SECS: u64 = 0;
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

impl LlmClient {
    pub fn new(base_url: Option<&str>, model: &str, api_key: Option<String>) -> Result<Self> {
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
            api_key,
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
                let default_delay = INITIAL_BACKOFF_SECS * 2u64.pow(attempt - 1);

                // Check for Retry-After hint from previous error
                let delay_secs = last_error
                    .as_ref()
                    .and_then(|e: &anyhow::Error| parse_retry_after(&e.to_string()))
                    .unwrap_or(default_delay);

                let delay = Duration::from_secs(delay_secs);
                tracing::info!(
                    "Retrying LLM request in {}s (attempt {}/{})",
                    delay.as_secs(),
                    attempt + 1,
                    MAX_RETRIES
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

        let mut req = self
            .client
            .post(&url)
            .header("content-type", "application/json");

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = req
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
            // Capture Retry-After header before consuming the response body
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok());

            let body = response.text().await.unwrap_or_default();
            let error_msg = serde_json::from_str::<ErrorResponse>(&body)
                .ok()
                .and_then(|e| e.error)
                .map(|e| e.to_string())
                .unwrap_or_else(|| body.clone());

            if let Some(secs) = retry_after {
                anyhow::bail!(
                    "LLM API error ({}): {} [retry-after:{}]",
                    status,
                    error_msg,
                    secs
                );
            }
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

/// Parse a `[retry-after:N]` tag from an error message string.
fn parse_retry_after(err_msg: &str) -> Option<u64> {
    let marker = "[retry-after:";
    let start = err_msg.find(marker)? + marker.len();
    let end = err_msg[start..].find(']')? + start;
    err_msg[start..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_present() {
        let msg = "LLM API error (429): rate limited [retry-after:30]";
        assert_eq!(parse_retry_after(msg), Some(30));
    }

    #[test]
    fn parse_retry_after_missing() {
        let msg = "LLM API error (429): rate limited";
        assert_eq!(parse_retry_after(msg), None);
    }

    #[test]
    fn parse_retry_after_non_numeric() {
        let msg = "error [retry-after:abc]";
        assert_eq!(parse_retry_after(msg), None);
    }

    #[tokio::test]
    async fn llm_client_success() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;

        let server = MockServer::start().await;
        let body = serde_json::json!({
            "choices": [{"message": {"content": "Hello from LLM"}}]
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = LlmClient::new(Some(&server.uri()), "test-model", None).unwrap();
        let result = client.message("system", "user").await.unwrap();
        assert_eq!(result, "Hello from LLM");
    }

    #[tokio::test]
    async fn llm_client_retry_on_429() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_string("rate limited")
                    .append_header("retry-after", "1"),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;

        let body = serde_json::json!({
            "choices": [{"message": {"content": "OK"}}]
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = LlmClient::new(Some(&server.uri()), "test-model", None).unwrap();
        let result = client.message("system", "user").await.unwrap();
        assert_eq!(result, "OK");
    }

    #[tokio::test]
    async fn llm_client_400_fails_immediately() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let client = LlmClient::new(Some(&server.uri()), "test-model", None).unwrap();
        let result = client.message("system", "user").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("400"));
    }

    #[tokio::test]
    async fn llm_client_empty_response_error() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;

        let server = MockServer::start().await;
        let body = serde_json::json!({
            "choices": [{"message": {"content": null}}]
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = LlmClient::new(Some(&server.uri()), "test-model", None).unwrap();
        let result = client.message("system", "user").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }
}
