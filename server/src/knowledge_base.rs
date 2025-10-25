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

/// Embeds a text using Workers AI binding (native Wrangler binding)
pub async fn embed_text(env: &Env, text: &str) -> Result<Vec<f32>> {
    let ai = env.ai("AI")?;
    let cf_model = env.var("CF_EMBED_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "@cf/google/embeddinggemma-300m".to_string());

    // Use native AI binding
    let payload = serde_json::json!({ "text": [text] });
    
    let response: serde_json::Value = ai
        .run(&cf_model, payload)
        .await
        .map_err(|e| worker::Error::RustError(format!("AI embedding failed: {}", e)))?;
    
    // Parse embedding from response
    if let Some(data) = response.get("data") {
        if let Some(arr) = data.as_array() {
            if !arr.is_empty() {
                if let Some(first) = arr[0].as_array() {
                    // Response format: {"data": [[embedding]]}
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
    
    Err(worker::Error::RustError(
        "Invalid embedding response format from CF AI".to_string()
    ))
}

/// Queries Vectorize - disabled in this build (worker crate lacks Vectorize binding)
pub async fn query_top_k(_env: &Env, _query: &str, _k: usize) -> Result<Vec<KBChunk>> {
    Err(worker::Error::RustError(
        "Vectorize binding is unavailable in the current workers-rs version".to_string(),
    ))
}
