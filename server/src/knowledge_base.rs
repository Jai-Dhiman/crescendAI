use serde::{Deserialize, Serialize};
use worker::*;

/// Knowledge Base chunk metadata stored in Vectorize
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KBChunk {
    pub id: String,       // e.g., "hanon_01::c0"
    pub doc_id: String,   // e.g., "hanon_01"
    pub title: String,
    pub tags: Vec<String>,
    pub source: String,
    pub url: Option<String>,
    pub text: String,
    pub chunk_id: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct CfEmbeddingResponse {
    result: Option<CfEmbeddingResult>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CfEmbeddingResult {
    data: Option<Vec<f32>>, // Some models return flat vector
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAiEmbeddingItem {
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct VectorizeQueryRequest {
    vector: Vec<f32>,
    #[serde(rename = "topK")]
    top_k: usize,
    #[serde(rename = "includeVectors")]
    include_vectors: bool,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct VectorizeQueryResponse {
    result: Option<VectorizeQueryResult>,
}

#[derive(Serialize, Deserialize, Debug)]
struct VectorizeQueryResult {
    #[serde(default)]
    matches: Vec<VectorizeMatch>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VectorizeMatch {
    id: String,
    #[serde(default)]
    metadata: Option<KBChunk>,
}

fn get_env_var(env: &Env, key: &str) -> Option<String> {
    env.var(key).ok().map(|v| v.to_string())
}

/// Embeds a text using Workers AI (primary via CF REST) or OpenAI as fallback
pub async fn embed_text(env: &Env, text: &str) -> Result<Vec<f32>> {
    // Try Cloudflare AI embeddings first (requires CF_ACCOUNT_ID and CF_API_TOKEN)
    let cf_account = get_env_var(env, "CF_ACCOUNT_ID");
    let cf_token = env.secret("CF_API_TOKEN").ok().map(|s| s.to_string());
    let cf_model = get_env_var(env, "CF_EMBED_MODEL").unwrap_or_else(|| "@cf/baai/bge-small-en-v1.5".to_string());

    if let (Some(account_id), Some(token)) = (cf_account, cf_token) {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            account_id, cf_model
        );
        let payload = serde_json::json!({ "text": text });
        let mut headers = Headers::new();
        headers.set("Authorization", &format!("Bearer {}", token)).ok();
        headers.set("Content-Type", "application/json").ok();
        let mut init = RequestInit::new();
        init.with_method(Method::Post);
        init.with_headers(headers);
        init.with_body(Some(serde_json::to_string(&payload).map_err(|e| worker::Error::RustError(e.to_string()))?.into()));
        let req = Request::new_with_init(&url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        if resp.status_code() / 100 == 2 {
            let cf: CfEmbeddingResponse = resp.json().await?;
            if let Some(res) = cf.result {
                if let Some(vec) = res.data {
                    return Ok(vec);
                }
            }
        }
        // If CF path fails, continue to fallback
    }

    // Fallback: OpenAI embeddings
    let openai_key = env.secret("OPENAI_API_KEY")
        .map_err(|_| worker::Error::RustError("OPENAI_API_KEY not configured".to_string()))?
        .to_string();
    let embed_model = get_env_var(env, "OPENAI_EMBED_MODEL").unwrap_or_else(|| "text-embedding-3-small".to_string());
    let url = "https://api.openai.com/v1/embeddings";
    let payload = serde_json::json!({
        "model": embed_model,
        "input": text,
    });
    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", openai_key)).ok();
    headers.set("Content-Type", "application/json").ok();
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(serde_json::to_string(&payload).map_err(|e| worker::Error::RustError(e.to_string()))?.into()));
    let req = Request::new_with_init(url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    if resp.status_code() / 100 != 2 {
        return Err(worker::Error::RustError(format!("OpenAI embeddings HTTP {}", resp.status_code())));
    }
    let body: OpenAiEmbeddingResponse = resp.json().await?;
    let first = body.data.into_iter().next().ok_or_else(|| worker::Error::RustError("OpenAI embeddings empty".to_string()))?;
    Ok(first.embedding)
}

/// Queries Vectorize using an embedded query vector
pub async fn query_top_k(env: &Env, query: &str, k: usize) -> Result<Vec<KBChunk>> {
    let vector = embed_text(env, query).await?;

    // Cloudflare Vectorize REST
    let account_id = get_env_var(env, "CF_ACCOUNT_ID")
        .ok_or_else(|| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let index_name = get_env_var(env, "VECTORIZE_INDEX_NAME").unwrap_or_else(|| "crescendai-piano-pedagogy".to_string());

    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/vectorize/indexes/{}/query",
        account_id, index_name
    );
    let payload = VectorizeQueryRequest {
        vector,
        top_k: k,
        include_vectors: false,
        include_metadata: true,
    };

    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", token)).ok();
    headers.set("Content-Type", "application/json").ok();
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(serde_json::to_string(&payload).map_err(|e| worker::Error::RustError(e.to_string()))?.into()));
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    if resp.status_code() / 100 != 2 {
        return Err(worker::Error::RustError(format!("Vectorize query HTTP {}", resp.status_code())));
    }

    let parsed: VectorizeQueryResponse = resp.json().await?;
    let mut chunks: Vec<KBChunk> = vec![];
    if let Some(res) = parsed.result {
        for m in res.matches {
            if let Some(meta) = m.metadata.clone() {
                chunks.push(meta);
            }
        }
    }
    Ok(chunks)
}
