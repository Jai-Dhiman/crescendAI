use worker::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;
use crate::utils;

/// RAG query with 3-layer caching
pub async fn rag_query(mut req: Request, ctx: RouteContext<Env>) -> Result<Response> {
    let body: RagQueryRequest = req.json().await?;

    // Get KV namespaces
    let embedding_cache = ctx.kv("EMBEDDING_CACHE")?;
    let llm_cache = ctx.kv("LLM_CACHE")?;

    // Layer 1: Check if embedding is cached
    let embedding_key = utils::cache_key_embedding(&body.query);
    let _embedding: Vec<f32> = match get_cached_embedding(&embedding_cache, &embedding_key).await? {
        Some(cached) => {
            console_log!("Embedding cache hit");
            cached
        }
        None => {
            console_log!("Embedding cache miss - generating");
            // Generate embedding using Workers AI binding
            let ai = ctx.env.ai("AI")?;
            let embedding = generate_embedding_with_ai(&ai, &body.query).await?;

            // Cache for 24 hours
            put_cached_embedding(&embedding_cache, &embedding_key, &embedding).await?;
            embedding
        }
    };

    // Layer 2: Check if full response is cached
    let search_key = utils::cache_key_search(&body.query, "default");
    if let Some(cached_response) = get_cached_response(&llm_cache, &search_key).await? {
        console_log!("Full response cached - returning immediately");
        return Response::from_json(&cached_response);
    }

    // Layer 3: Cache miss - call GCP API for full RAG pipeline
    console_log!("Full cache miss - calling GCP API");
    let gcp_api_url = ctx.env.var("GCP_API_URL")?.to_string();

    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post)
        .with_headers(headers)
        .with_body(Some(JsValue::from_str(&serde_json::to_string(&body)?)));

    let gcp_request = Request::new_with_init(&format!("{}/api/chat/query", gcp_api_url), &init)?;
    let mut gcp_response = Fetch::Request(gcp_request).send().await?;

    let rag_response: RagResponse = gcp_response.json().await?;

    // Cache the response for 1 hour
    put_cached_response(&llm_cache, &search_key, &rag_response).await?;

    Response::from_json(&rag_response)
}

/// Stream a project PDF directly from R2 (zero-latency edge access)
///
/// This endpoint provides the fastest download experience by streaming directly
/// from R2 using Worker bindings. Recommended for frequently accessed files.
///
/// For infrequent access or temporary shares, use presigned URLs instead.
pub async fn stream_project(_req: Request, ctx: RouteContext<Env>) -> Result<Response> {
    let project_id = ctx.param("id").ok_or_else(|| Error::RustError("Missing project ID".to_string()))?;

    // First, get the project metadata from GCP API to get the R2 key
    let gcp_api_url = ctx.env.var("GCP_API_URL")?.to_string();

    let mut init = RequestInit::new();
    init.with_method(Method::Get);

    let request = Request::new_with_init(
        &format!("{}/api/projects/{}", gcp_api_url, project_id),
        &init
    )?;
    let mut response = Fetch::Request(request).send().await?;

    if response.status_code() != 200 {
        return Ok(response);
    }

    let project: serde_json::Value = response.json().await?;
    let r2_key = project["project"]["r2_key"]
        .as_str()
        .ok_or_else(|| Error::RustError("Missing r2_key in project".to_string()))?;

    // Stream directly from R2 using binding
    let pdfs_bucket = ctx.env.r2("PDFS_BUCKET")?;
    let object = pdfs_bucket.get(r2_key).execute().await?;

    match object {
        Some(obj) => {
            // Stream the object body with proper headers
            let mut headers = worker::Headers::new();
            headers.set("Content-Type", "application/pdf")?;
            headers.set("Content-Disposition", &format!("inline; filename=\"{}.pdf\"", project_id))?;

            // Add cache headers for edge caching
            headers.set("Cache-Control", "public, max-age=3600")?;
            headers.set("ETag", &obj.http_etag())?;

            console_log!("Streaming project {} from R2 (key: {})", project_id, r2_key);

            Response::from_stream(obj.body())?.with_headers(headers)
        }
        None => {
            Response::error("PDF file not found in R2", 404)
        }
    }
}

/// Generate embedding with caching
pub async fn generate_embedding(mut req: Request, ctx: RouteContext<Env>) -> Result<Response> {
    let body: EmbeddingRequest = req.json().await?;

    let embedding_cache = ctx.kv("EMBEDDING_CACHE")?;
    let key = utils::cache_key_embedding(&body.text);

    let embedding: Vec<f32> = match get_cached_embedding(&embedding_cache, &key).await? {
        Some(cached) => {
            console_log!("Embedding cache hit");
            cached
        }
        None => {
            console_log!("Embedding cache miss - generating");
            let ai = ctx.env.ai("AI")?;
            let embedding = generate_embedding_with_ai(&ai, &body.text).await?;
            put_cached_embedding(&embedding_cache, &key, &embedding).await?;
            embedding
        }
    };

    Response::from_json(&EmbeddingResponse { embedding })
}

/// Proxy all other requests to GCP API
pub async fn proxy_to_gcp(req: Request, ctx: RouteContext<Env>) -> Result<Response> {
    let gcp_api_url = ctx.env.var("GCP_API_URL")?.to_string();
    let path = req.path();

    console_log!("Proxying to GCP API: {}", path);

    let url = format!("{}{}", gcp_api_url, path);

    let mut init = RequestInit::new();
    init.with_method(req.method());

    // Forward headers
    let headers = req.headers().clone();
    init.with_headers(headers);

    let gcp_request = Request::new_with_init(&url, &init)?;
    let response = Fetch::Request(gcp_request).send().await?;

    Ok(response)
}

// Helper function to call Workers AI for embeddings
async fn generate_embedding_with_ai(_ai: &Ai, text: &str) -> Result<Vec<f32>> {
    // PRODUCTION IMPLEMENTATION:
    // The Workers AI API in Rust worker crate v0.6 doesn't expose stable types yet
    // for embeddings. Once the API stabilizes, implement like this:
    //
    // let input = serde_json::json!({ "text": [text] });
    // let response = ai.run("@cf/baai/bge-base-en-v1.5", input).await?;
    // let output: AiTextEmbeddingsOutput = response.as_json()?;
    // return Ok(output.data[0].clone());
    //
    // For now, we use a placeholder that returns a deterministic embedding based on text hash
    // This allows the system to work end-to-end while we wait for the Worker AI Rust API to stabilize

    console_log!("Generating deterministic embedding for: {}", &text[..text.len().min(50)]);

    // Create a deterministic 768-dim vector based on text hash
    // This is a placeholder - real embeddings will be generated by Workers AI once API stabilizes
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();

    // Expand hash to 768 dimensions (BGE-base-v1.5 output size)
    let mut embedding = Vec::with_capacity(768);
    for i in 0..768 {
        let byte_idx = i % hash.len();
        let normalized = (hash[byte_idx] as f32) / 255.0; // Normalize to [0, 1]
        embedding.push(normalized);
    }

    Ok(embedding)
}

// Cache helper functions
async fn get_cached_embedding(kv: &kv::KvStore, key: &str) -> Result<Option<Vec<f32>>> {
    match kv.get(key).text().await? {
        Some(cached) => {
            match serde_json::from_str::<Vec<f32>>(&cached) {
                Ok(value) => Ok(Some(value)),
                Err(_) => Ok(None),
            }
        }
        None => Ok(None),
    }
}

async fn put_cached_embedding(kv: &kv::KvStore, key: &str, embedding: &Vec<f32>) -> Result<()> {
    let serialized = serde_json::to_string(embedding)
        .map_err(|e| Error::RustError(format!("Serialization error: {}", e)))?;

    kv.put(key, serialized)?
        .expiration_ttl(86400) // 24 hours
        .execute()
        .await?;

    Ok(())
}

async fn get_cached_response(kv: &kv::KvStore, key: &str) -> Result<Option<RagResponse>> {
    match kv.get(key).text().await? {
        Some(cached) => {
            match serde_json::from_str::<RagResponse>(&cached) {
                Ok(value) => Ok(Some(value)),
                Err(_) => Ok(None),
            }
        }
        None => Ok(None),
    }
}

async fn put_cached_response(kv: &kv::KvStore, key: &str, response: &RagResponse) -> Result<()> {
    let serialized = serde_json::to_string(response)
        .map_err(|e| Error::RustError(format!("Serialization error: {}", e)))?;

    kv.put(key, serialized)?
        .expiration_ttl(3600) // 1 hour
        .execute()
        .await?;

    Ok(())
}

// Request/Response types
#[derive(Deserialize, Serialize)]
struct RagQueryRequest {
    query: String,
    project_id: Option<String>,
    session_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct RagResponse {
    content: String,
    sources: Vec<Source>,
    confidence: f32,
}

#[derive(Serialize, Deserialize)]
struct Source {
    chunk_id: String,
    content: String,
    score: f32,
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    text: String,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

// Workers AI types matching the actual API response format
// Response: { "shape": [batch_size, embedding_dim], "data": [[...floats...]] }
#[derive(Deserialize)]
struct AiTextEmbeddingsOutput {
    shape: Vec<usize>,  // e.g., [1, 768] for single text with BGE-base-en-v1.5
    data: Vec<Vec<f32>>, // e.g., [[0.1, 0.2, ..., 0.768]]
}
