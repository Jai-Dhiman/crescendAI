use worker::*;
use serde::{Deserialize, Serialize};
use crate::{cache, utils};

/// RAG query with 3-layer caching
pub async fn rag_query(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let body: RagQueryRequest = req.json().await?;

    // Get KV namespaces
    let embedding_cache = ctx.kv("EMBEDDING_CACHE")?;
    let search_cache = ctx.kv("SEARCH_CACHE")?;
    let llm_cache = ctx.kv("LLM_CACHE")?;

    // Layer 1: Check if embedding is cached
    let embedding_key = utils::cache_key_embedding(&body.query);
    let embedding: Vec<f32> = match cache::get(&embedding_cache, &embedding_key).await? {
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
            cache::put(&embedding_cache, &embedding_key, &embedding, 86400).await?;
            embedding
        }
    };

    // Layer 2: Check if search results are cached
    let search_key = utils::cache_key_search(&body.query, "default");
    if let Some(cached_response): Option<RagResponse> = cache::get(&llm_cache, &search_key).await? {
        console_log!("Full response cached - returning immediately");
        return Response::from_json(&cached_response);
    }

    // Layer 3: Cache miss - call GCP API for full RAG pipeline
    console_log!("Full cache miss - calling GCP API");
    let gcp_api_url = ctx.var("GCP_API_URL")?.to_string();

    let gcp_response = Fetch::Url(format!("{}/api/chat/query", gcp_api_url).parse()?)
        .post(&body)?
        .send()
        .await?;

    let rag_response: RagResponse = gcp_response.json().await?;

    // Cache the response for 1 hour
    cache::put(&llm_cache, &search_key, &rag_response, 3600).await?;

    Response::from_json(&rag_response)
}

/// Generate presigned upload URL (R2 direct access)
pub async fn generate_upload_url(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let bucket = ctx.bucket("PDFS_BUCKET")?;

    // Generate a unique key
    let key = format!("projects/{}/{}.pdf", uuid::Uuid::new_v4(), uuid::Uuid::new_v4());

    // Note: R2 binding doesn't directly support presigned URLs from Workers
    // Best practice: Generate from GCP API which has AWS SDK
    let gcp_api_url = ctx.var("GCP_API_URL")?.to_string();

    let response = Fetch::Url(format!("{}/api/projects/upload-url", gcp_api_url).parse()?)
        .post()?
        .send()
        .await?;

    Response::from_json(&response.json::<serde_json::Value>().await?)
}

/// Generate presigned download URL
pub async fn generate_download_url(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let project_id = ctx.param("id").unwrap();

    // Forward to GCP API for presigned URL generation
    let gcp_api_url = ctx.var("GCP_API_URL")?.to_string();

    let response = Fetch::Url(
        format!("{}/api/projects/{}/download-url", gcp_api_url, project_id).parse()?
    )
    .get()?
    .send()
    .await?;

    Response::from_json(&response.json::<serde_json::Value>().await?)
}

/// Generate embedding with caching
pub async fn generate_embedding(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let body: EmbeddingRequest = req.json().await?;

    let embedding_cache = ctx.kv("EMBEDDING_CACHE")?;
    let key = utils::cache_key_embedding(&body.text);

    let embedding: Vec<f32> = cache::with_cache(
        &embedding_cache,
        &key,
        86400, // 24 hours
        async {
            let ai = ctx.env.ai("AI")?;
            generate_embedding_with_ai(&ai, &body.text).await
        },
    )
    .await?;

    Response::from_json(&EmbeddingResponse { embedding })
}

/// Proxy all other requests to GCP API
pub async fn proxy_to_gcp(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let gcp_api_url = ctx.var("GCP_API_URL")?.to_string();
    let path = req.path();

    console_log!("Proxying to GCP API: {}", path);

    let url = format!("{}{}", gcp_api_url, path);

    let mut gcp_req = Fetch::Url(url.parse()?);

    // Forward method
    gcp_req = match req.method() {
        Method::Get => gcp_req.get()?,
        Method::Post => gcp_req.post()?,
        Method::Put => gcp_req.put()?,
        Method::Delete => gcp_req.delete()?,
        _ => gcp_req.get()?,
    };

    // Forward headers
    for (name, value) in req.headers() {
        if let Ok(value_str) = value.to_str() {
            gcp_req = gcp_req.header(&name, value_str)?;
        }
    }

    let response = gcp_req.send().await?;

    Ok(response)
}

// Helper function to call Workers AI
async fn generate_embedding_with_ai(ai: &Ai, text: &str) -> Result<Vec<f32>> {
    // Use Workers AI binding to generate embedding
    // Note: The exact API depends on the worker crate version
    // This is a placeholder - adjust based on actual worker::Ai API

    let result = ai.run(
        "@cf/baai/bge-base-en-v1.5",
        serde_json::json!({ "text": [text] })
    ).await?;

    // Parse the result
    let data: serde_json::Value = result.json().await?;
    let embedding: Vec<f32> = serde_json::from_value(
        data["data"][0].clone()
    ).map_err(|e| Error::RustError(format!("Failed to parse embedding: {}", e)))?;

    Ok(embedding)
}

// Request/Response types
#[derive(Deserialize)]
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
