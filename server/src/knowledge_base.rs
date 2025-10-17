use serde::{Deserialize, Serialize};
use worker::*;

/// Knowledge Base chunk metadata stored in Vectorize
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KBChunk {
    pub id: String,     // e.g., "hanon_01::c0"
    pub doc_id: String, // e.g., "hanon_01"
    pub title: String,
    pub tags: Vec<String>,
    pub source: String,
    pub url: Option<String>,
    pub text: String,
    pub chunk_id: u32,
}

/// Embeds a text using Workers AI via HTTP API (no fallbacks)
pub async fn embed_text(env: &Env, text: &str) -> Result<Vec<f32>> {
    let account_id = env.var("CF_ACCOUNT_ID")
        .map_err(|_| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let cf_model = env.var("CF_EMBED_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "@cf/google/embeddinggemma-300m".to_string());

    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
        account_id.to_string(), cf_model
    );
    
    // Send as array to match API expectation
    let payload = serde_json::json!({ "text": [text] });
    
    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", token.to_string()))?;
    headers.set("Content-Type", "application/json")?;
    
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(
        serde_json::to_string(&payload)
            .map_err(|e| worker::Error::RustError(e.to_string()))?
            .into(),
    ));
    
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        let error_text = resp.text().await.unwrap_or_default();
        return Err(worker::Error::RustError(format!(
            "CF AI API error {}: {}", resp.status_code(), error_text
        )));
    }
    
    let response_json: serde_json::Value = resp.json().await?;
    
    // Parse embedding from response
    if let Some(result) = response_json.get("result") {
        if let Some(data) = result.get("data") {
            if let Some(arr) = data.as_array() {
                if !arr.is_empty() {
                    if let Some(first) = arr[0].as_array() {
                        // Response format: {"result": {"data": [[embedding]]}}
                        let embedding: Vec<f32> = first
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect();
                        if !embedding.is_empty() {
                            return Ok(embedding);
                        }
                    }
                }
            }
        }
    }
    
    Err(worker::Error::RustError(
        "Invalid embedding response format from CF AI".to_string()
    ))
}

/// Queries Vectorize using HTTP API (no fallbacks)
pub async fn query_top_k(env: &Env, query: &str, k: usize) -> Result<Vec<KBChunk>> {
    let vector = embed_text(env, query).await?;
    
    let account_id = env.var("CF_ACCOUNT_ID")
        .map_err(|_| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let index_name = env.var("VECTORIZE_INDEX_NAME")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "crescendai-piano-pedagogy".to_string());

    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/vectorize/indexes/{}/query",
        account_id.to_string(), index_name
    );
    
    let payload = serde_json::json!({
        "vector": vector,
        "topK": k,
        "includeVectors": false,
        "includeMetadata": true
    });
    
    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", token.to_string()))?;
    headers.set("Content-Type", "application/json")?;
    
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(
        serde_json::to_string(&payload)
            .map_err(|e| worker::Error::RustError(e.to_string()))?
            .into(),
    ));
    
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        let error_text = resp.text().await.unwrap_or_default();
        return Err(worker::Error::RustError(format!(
            "Vectorize query error {}: {}", resp.status_code(), error_text
        )));
    }
    
    let response_json: serde_json::Value = resp.json().await?;
    
    let mut chunks: Vec<KBChunk> = Vec::new();
    
    if let Some(result) = response_json.get("result") {
        if let Some(matches) = result.get("matches").and_then(|m| m.as_array()) {
            for match_item in matches {
                if let Some(metadata) = match_item.get("metadata") {
                    match serde_json::from_value::<KBChunk>(metadata.clone()) {
                        Ok(chunk) => chunks.push(chunk),
                        Err(e) => {
                            console_log!("Warning: Failed to parse chunk metadata: {:?}", e);
                            continue;
                        }
                    }
                }
            }
        }
    }
    
    Ok(chunks)
}
