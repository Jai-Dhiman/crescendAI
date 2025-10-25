use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct WorkersAIClient {
    http_client: Client,
    account_id: String,
    api_token: String,
}

impl WorkersAIClient {
    pub fn new(account_id: String, api_token: String) -> Self {
        Self {
            http_client: Client::new(),
            account_id,
            api_token,
        }
    }

    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/@cf/baai/bge-base-en-v1.5",
            self.account_id
        );

        let request_body = EmbeddingRequest {
            text: vec![text.to_string()],
        };

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&request_body)
            .send()
            .await
            .context("Failed to call Workers AI embedding API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Workers AI embedding API failed with status {}: {}",
                status,
                error_text
            );
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse embedding response")?;

        if embedding_response.success && !embedding_response.result.data.is_empty() {
            Ok(embedding_response.result.data[0].clone())
        } else {
            anyhow::bail!("Failed to generate embedding: {:?}", embedding_response.errors)
        }
    }

    pub async fn batch_embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/@cf/baai/bge-base-en-v1.5",
            self.account_id
        );

        let text_strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let request_body = EmbeddingRequest { text: text_strings };

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&request_body)
            .send()
            .await
            .context("Failed to call Workers AI batch embedding API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Workers AI batch embedding API failed with status {}: {}",
                status,
                error_text
            );
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse batch embedding response")?;

        if embedding_response.success {
            Ok(embedding_response.result.data)
        } else {
            anyhow::bail!(
                "Failed to generate batch embeddings: {:?}",
                embedding_response.errors
            )
        }
    }

    pub async fn rerank(
        &self,
        query: &str,
        candidates: Vec<&str>,
    ) -> Result<Vec<RerankResult>> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/@cf/baai/bge-reranker-base",
            self.account_id
        );

        let request_body = RerankRequest {
            query: query.to_string(),
            documents: candidates.iter().map(|s| s.to_string()).collect(),
        };

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&request_body)
            .send()
            .await
            .context("Failed to call Workers AI rerank API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Workers AI rerank API failed with status {}: {}",
                status,
                error_text
            );
        }

        let rerank_response: RerankResponse = response
            .json()
            .await
            .context("Failed to parse rerank response")?;

        if rerank_response.success {
            Ok(rerank_response.result)
        } else {
            anyhow::bail!("Failed to rerank: {:?}", rerank_response.errors)
        }
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    text: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    success: bool,
    result: EmbeddingResult,
    errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResult {
    data: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    success: bool,
    result: Vec<RerankResult>,
    errors: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}
