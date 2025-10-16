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
pub enum EmbeddingProvider {
    Local,
    CloudflareAI,
    OpenAI,
}

/// Local embedding service request/response
#[derive(Serialize, Deserialize, Debug)]
struct LocalEmbedRequest {
    text: String,
    model: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct LocalEmbedResponse {
    embedding: Vec<f32>,
    model: String,
    dimensions: usize,
    processing_time_ms: f32,
}

/// Batch embedding for efficiency
#[derive(Serialize, Deserialize, Debug)]
struct LocalBatchEmbedRequest {
    texts: Vec<String>,
    model: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct LocalBatchEmbedResponse {
    embeddings: Vec<Vec<f32>>,
    model: String,
    dimensions: usize,
    processing_time_ms: f32,
    batch_size: usize,
}

// Existing Cloudflare AI types (unchanged)
#[derive(Serialize, Deserialize, Debug)]
struct CfEmbeddingResponse {
    result: Option<CfEmbeddingResult>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CfEmbeddingResult {
    data: Option<Vec<f32>>,
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

fn get_local_embedding_url(env: &Env) -> String {
    get_env_var(env, "LOCAL_EMBEDDING_URL")
        .unwrap_or_else(|| "http://localhost:8001".to_string())
}

fn get_local_embedding_timeout(env: &Env) -> u32 {
    get_env_var(env, "LOCAL_EMBEDDING_TIMEOUT_MS")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000)
}

fn is_local_embeddings_enabled(env: &Env) -> bool {
    get_env_var(env, "USE_LOCAL_EMBEDDINGS")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(true) // Default to enabled
}

/// Try local embedding service first, with fallback to cloud providers
pub async fn embed_text(env: &Env, text: &str) -> Result<Vec<f32>> {
    let providers = get_embedding_provider_order(env);
    let mut last_error = None;
    
    for provider in providers {
        match provider {
            EmbeddingProvider::Local => {
                match embed_text_local(env, text).await {
                    Ok(embedding) => {
                        console_log!("Local embedding successful (dim={})", embedding.len());
                        return Ok(embedding);
                    }
                    Err(e) => {
                        console_log!("Local embedding failed: {:?}", e);
                        last_error = Some(e);
                    }
                }
            }
            EmbeddingProvider::CloudflareAI => {
                match embed_text_cloudflare(env, text).await {
                    Ok(embedding) => {
                        console_log!("Cloudflare AI embedding successful (dim={})", embedding.len());
                        return Ok(embedding);
                    }
                    Err(e) => {
                        console_log!("Cloudflare AI embedding failed: {:?}", e);
                        last_error = Some(e);
                    }
                }
            }
            EmbeddingProvider::OpenAI => {
                match embed_text_openai(env, text).await {
                    Ok(embedding) => {
                        console_log!("OpenAI embedding successful (dim={})", embedding.len());
                        return Ok(embedding);
                    }
                    Err(e) => {
                        console_log!("OpenAI embedding failed: {:?}", e);
                        last_error = Some(e);
                    }
                }
            }
        }
    }
    
    // All providers failed
    Err(last_error.unwrap_or_else(|| worker::Error::RustError("No embedding providers available".to_string())))
}

/// Batch embedding with automatic fallback
pub async fn embed_texts_batch(env: &Env, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let providers = get_embedding_provider_order(env);
    
    // Try local batch first if enabled
    if let Some(EmbeddingProvider::Local) = providers.first() {
        if let Ok(embeddings) = embed_texts_batch_local(env, texts).await {
            console_log!("Local batch embedding successful ({} texts)", texts.len());
            return Ok(embeddings);
        }
    }
    
    // Fall back to individual embeddings
    let mut embeddings = Vec::new();
    for text in texts {
        embeddings.push(embed_text(env, text).await?);
    }
    Ok(embeddings)
}

fn get_embedding_provider_order(env: &Env) -> Vec<EmbeddingProvider> {
    let mut providers = Vec::new();
    
    if is_local_embeddings_enabled(env) {
        providers.push(EmbeddingProvider::Local);
    }
    
    // Always include cloud fallbacks
    if get_env_var(env, "CF_ACCOUNT_ID").is_some() && env.secret("CF_API_TOKEN").is_ok() {
        providers.push(EmbeddingProvider::CloudflareAI);
    }
    
    if env.secret("OPENAI_API_KEY").is_ok() {
        providers.push(EmbeddingProvider::OpenAI);
    }
    
    providers
}

/// Local embedding service implementation
async fn embed_text_local(env: &Env, text: &str) -> Result<Vec<f32>> {
    let base_url = get_local_embedding_url(env);
    let timeout = get_local_embedding_timeout(env);
    let url = format!("{}/embed", base_url);
    
    let payload = LocalEmbedRequest {
        text: text.to_string(),
        model: get_env_var(env, "LOCAL_EMBEDDING_MODEL"), // e.g., "google/gemma-2-2b-it"
    };
    
    let mut headers = Headers::new();
    headers.set("Content-Type", "application/json").ok();
    
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(serde_json::to_string(&payload)
        .map_err(|e| worker::Error::RustError(e.to_string()))?
        .into()));
    
    let req = Request::new_with_init(&url, &init)?;
    
    // Set timeout using AbortSignal (Web API)
    let abort_controller = web_sys::AbortController::new()
        .map_err(|_| worker::Error::RustError("Failed to create AbortController".to_string()))?;
    
    // Create timeout
    let abort_signal = abort_controller.signal();
    web_sys::window()
        .unwrap()
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            &wasm_bindgen::closure::Closure::wrap(Box::new(move || {
                abort_controller.abort();
            }) as Box<dyn FnMut()>)
            .as_ref()
            .unchecked_ref(),
            timeout as i32,
        )
        .map_err(|_| worker::Error::RustError("Failed to set timeout".to_string()))?;
    
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        let error_body = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(worker::Error::RustError(format!(
            "Local embedding service error (HTTP {}): {}", 
            resp.status_code(), 
            error_body
        )));
    }
    
    let response: LocalEmbedResponse = resp.json().await?;
    Ok(response.embedding)
}

/// Batch local embedding
async fn embed_texts_batch_local(env: &Env, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    
    // Split into smaller batches to avoid timeouts
    const BATCH_SIZE: usize = 50;
    let mut all_embeddings = Vec::new();
    
    for batch in texts.chunks(BATCH_SIZE) {
        let batch_embeddings = embed_batch_local(env, batch).await?;
        all_embeddings.extend(batch_embeddings);
    }
    
    Ok(all_embeddings)
}

async fn embed_batch_local(env: &Env, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let base_url = get_local_embedding_url(env);
    let timeout = get_local_embedding_timeout(env) * 2; // Double timeout for batches
    let url = format!("{}/embed/batch", base_url);
    
    let payload = LocalBatchEmbedRequest {
        texts: texts.to_vec(),
        model: get_env_var(env, "LOCAL_EMBEDDING_MODEL"),
    };
    
    let mut headers = Headers::new();
    headers.set("Content-Type", "application/json").ok();
    
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(serde_json::to_string(&payload)
        .map_err(|e| worker::Error::RustError(e.to_string()))?
        .into()));
    
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        let error_body = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(worker::Error::RustError(format!(
            "Local batch embedding error (HTTP {}): {}", 
            resp.status_code(), 
            error_body
        )));
    }
    
    let response: LocalBatchEmbedResponse = resp.json().await?;
    Ok(response.embeddings)
}

/// Cloudflare AI embedding (unchanged from original)
async fn embed_text_cloudflare(env: &Env, text: &str) -> Result<Vec<f32>> {
    let cf_account = get_env_var(env, "CF_ACCOUNT_ID")
        .ok_or_else(|| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let cf_token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let cf_model = get_env_var(env, "CF_EMBED_MODEL")
        .unwrap_or_else(|| "@cf/baai/bge-small-en-v1.5".to_string());
    
    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
        cf_account, cf_model
    );
    
    let payload = serde_json::json!({ "text": text });
    
    let mut headers = Headers::new();
    headers.set("Authorization", &format!("Bearer {}", cf_token)).ok();
    headers.set("Content-Type", "application/json").ok();
    
    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(payload.to_string().into()));
    
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        return Err(worker::Error::RustError(format!(
            "Cloudflare AI error: HTTP {}", resp.status_code()
        )));
    }
    
    let cf: CfEmbeddingResponse = resp.json().await?;
    
    if let Some(result) = cf.result {
        if let Some(embedding) = result.data {
            return Ok(embedding);
        }
    }
    
    Err(worker::Error::RustError("No embedding data in CF response".to_string()))
}

/// OpenAI embedding (unchanged from original)
async fn embed_text_openai(env: &Env, text: &str) -> Result<Vec<f32>> {
    let openai_key = env.secret("OPENAI_API_KEY")
        .map_err(|_| worker::Error::RustError("OPENAI_API_KEY not configured".to_string()))?
        .to_string();
    
    let embed_model = get_env_var(env, "OPENAI_EMBED_MODEL")
        .unwrap_or_else(|| "text-embedding-3-small".to_string());
    
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
    init.with_body(Some(payload.to_string().into()));
    
    let req = Request::new_with_init(url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        return Err(worker::Error::RustError(format!(
            "OpenAI embeddings HTTP {}", resp.status_code()
        )));
    }
    
    let body: OpenAiEmbeddingResponse = resp.json().await?;
    let first = body.data.into_iter().next()
        .ok_or_else(|| worker::Error::RustError("OpenAI embeddings empty".to_string()))?;
    
    Ok(first.embedding)
}

/// Health check for local embedding service
pub async fn check_local_embedding_health(env: &Env) -> Result<bool> {
    if !is_local_embeddings_enabled(env) {
        return Ok(false);
    }
    
    let base_url = get_local_embedding_url(env);
    let url = format!("{}/health", base_url);
    
    let req = Request::new(&url, Method::Get)?;
    
    match Fetch::Request(req).send().await {
        Ok(mut resp) => {
            if resp.status_code() == 200 {
                if let Ok(health) = resp.json::<serde_json::Value>().await {
                    return Ok(health["status"] == "healthy");
                }
            }
            Ok(false)
        }
        Err(_) => Ok(false),
    }
}

/// Main query function (unchanged interface, updated implementation)
pub async fn query_top_k(env: &Env, query: &str, k: usize) -> Result<Vec<KBChunk>> {
    let vector = embed_text(env, query).await?;
    
    // Cloudflare Vectorize REST (unchanged)
    let account_id = get_env_var(env, "CF_ACCOUNT_ID")
        .ok_or_else(|| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let index_name = get_env_var(env, "VECTORIZE_INDEX_NAME")
        .unwrap_or_else(|| "crescendai-piano-pedagogy".to_string());
    
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
    init.with_body(Some(serde_json::to_string(&payload)
        .map_err(|e| worker::Error::RustError(e.to_string()))?
        .into()));
    
    let req = Request::new_with_init(&url, &init)?;
    let mut resp = Fetch::Request(req).send().await?;
    
    if resp.status_code() / 100 != 2 {
        return Err(worker::Error::RustError(format!(
            "Vectorize query HTTP {}", resp.status_code()
        )));
    }
    
    let parsed: VectorizeQueryResponse = resp.json().await?;
    let mut chunks: Vec<KBChunk> = vec![];
    
    if let Some(res) = parsed.result {
        for m in res.matches {
            if let Some(meta) = m.metadata {
                chunks.push(meta);
            }
        }
    }
    
    Ok(chunks)
}