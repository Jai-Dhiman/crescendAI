use serde::{Deserialize, Serialize};
use worker::*;
use crate::knowledge_base::{embed_text, query_top_k, KBChunk};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Document metadata for ingestion
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
    pub tags: Vec<String>,
    pub source: String,
    pub url: Option<String>,
    pub pdf_filename: Option<String>,
}

/// Ingestion request payload
#[derive(Serialize, Deserialize, Debug)]
pub struct IngestionRequest {
    pub documents: Vec<DocumentInput>,
    pub chunking_config: Option<ChunkingConfig>,
}

/// Individual document input
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentInput {
    pub id: String,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub source: String,
    pub url: Option<String>,
    pub pdf_data: Option<String>, // Base64 encoded PDF
    pub pdf_filename: Option<String>,
}

/// Chunking configuration
#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkingConfig {
    pub target_chars: usize,
    pub overlap_ratio: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_chars: 1000,
            overlap_ratio: 0.2,
        }
    }
}

/// Ingestion result
#[derive(Serialize, Deserialize, Debug)]
pub struct IngestionResult {
    pub success: bool,
    pub documents_processed: usize,
    pub chunks_created: usize,
    pub errors: Vec<String>,
    pub manifest_path: Option<String>,
}

/// Chunk text into overlapping segments
pub fn chunk_text(text: &str, config: &ChunkingConfig) -> Vec<String> {
    if text.len() <= config.target_chars {
        return vec![text.trim().to_string()];
    }

    let step_chars = ((config.target_chars as f32) * (1.0 - config.overlap_ratio)) as usize;
    let step_chars = step_chars.max(1);

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + config.target_chars).min(text.len());
        let chunk = text[start..end].trim();
        
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        
        if end >= text.len() {
            break;
        }
        
        start += step_chars;
    }

    chunks
}

/// Process a single document for ingestion
pub async fn process_document(
    env: &Env,
    doc: &DocumentInput,
    config: &ChunkingConfig,
) -> Result<Vec<KBChunk>> {
    let chunks = chunk_text(&doc.content, config);
    let mut kb_chunks = Vec::new();

    for (idx, text) in chunks.into_iter().enumerate() {
        let chunk_id = format!("{}::c{}", doc.id, idx);
        
        let kb_chunk = KBChunk {
            id: chunk_id,
            doc_id: doc.id.clone(),
            title: doc.title.clone(),
            tags: doc.tags.clone(),
            source: doc.source.clone(),
            url: doc.url.clone(),
            text,
            chunk_id: idx as u32,
        };
        
        kb_chunks.push(kb_chunk);
    }

    Ok(kb_chunks)
}

/// Embed and store chunks in Vectorize using HTTP API
pub async fn store_chunks(env: &Env, chunks: &[KBChunk]) -> Result<usize> {
    let account_id = env.var("CF_ACCOUNT_ID")
        .map_err(|_| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let index_name = env.var("VECTORIZE_INDEX_NAME")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "crescendai-piano-pedagogy".to_string());
    let kv = env.kv("CRESCENDAI_METADATA")?;
    
    let mut stored_count = 0;

    for chunk in chunks {
        // Generate embedding
        let vector = embed_text(env, &chunk.text).await?;
        
        // Store in Vectorize with metadata
        let metadata = serde_json::to_value(chunk)
            .map_err(|e| worker::Error::RustError(format!("Metadata serialization failed: {}", e)))?;
        
        // Upsert to Vectorize via HTTP API
        let vectorize_url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/vectorize/indexes/{}/upsert",
            account_id.to_string(), index_name
        );
        
        let upsert_payload = serde_json::json!({
            "vectors": [{
                "id": chunk.id,
                "values": vector,
                "metadata": metadata
            }]
        });
        
        let mut headers = Headers::new();
        headers.set("Authorization", &format!("Bearer {}", token.to_string()))?;
        headers.set("Content-Type", "application/json")?;
        
        let mut init = RequestInit::new();
        init.with_method(Method::Post);
        init.with_headers(headers);
        init.with_body(Some(
            serde_json::to_string(&upsert_payload)
                .map_err(|e| worker::Error::RustError(e.to_string()))?
                .into(),
        ));
        
        let req = Request::new_with_init(&vectorize_url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        
        if resp.status_code() / 100 != 2 {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(worker::Error::RustError(format!(
                "Vectorize upsert error {}: {}", resp.status_code(), error_text
            )));
        }
        
        // Store chunk JSON in KV for retrieval
        let kv_key = format!("doc:{}:chunk:{}", chunk.doc_id, chunk.chunk_id);
        let chunk_json = serde_json::to_string(chunk)
            .map_err(|e| worker::Error::RustError(format!("Chunk serialization failed: {}", e)))?;
        
        kv.put(&kv_key, &chunk_json)?.execute().await?;
        
        stored_count += 1;
    }

    Ok(stored_count)
}

/// Store PDF data in KV (temporary solution until R2 binding works)
pub async fn store_pdf(
    env: &Env,
    doc_id: &str,
    pdf_data: &str,
    filename: &str,
) -> Result<String> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    
    // Validate base64 PDF data
    match base64_simd::STANDARD.decode_to_vec(pdf_data) {
        Ok(_) => {}, // Valid base64
        Err(_) => {
            return Err(worker::Error::RustError("PDF decode failed: invalid base64".to_string()));
        }
    };
    
    // Store PDF data in KV with metadata
    let kv_key = format!("pdf:{}:{}", doc_id, filename);
    let pdf_metadata = serde_json::json!({
        "doc_id": doc_id,
        "filename": filename,
        "data": pdf_data,
        "stored_at": chrono::Utc::now()
    });
    
    let pdf_json = serde_json::to_string(&pdf_metadata)
        .map_err(|e| worker::Error::RustError(format!("PDF metadata serialization failed: {}", e)))?;
    
    kv.put(&kv_key, &pdf_json)?.execute().await?;
    
    let kv_path = format!("kv:{}", kv_key);
    Ok(kv_path)
}

/// Create and store ingestion manifest in KV
pub async fn create_manifest(
    env: &Env,
    documents: &[DocumentInput],
    chunks_created: usize,
) -> Result<String> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let timestamp = Utc::now();
    
    let manifest = serde_json::json!({
        "timestamp": timestamp,
        "documents_count": documents.len(),
        "chunks_created": chunks_created,
        "documents": documents.iter().map(|d| serde_json::json!({
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "tags": d.tags,
            "url": d.url,
            "pdf_filename": d.pdf_filename,
        })).collect::<Vec<_>>()
    });
    
    let manifest_id = Uuid::new_v4().to_string();
    let manifest_key = format!("manifest:{}:{}", timestamp.format("%Y%m%d-%H%M%S"), manifest_id);
    
    let manifest_json = serde_json::to_string(&manifest)
        .map_err(|e| worker::Error::RustError(format!("Manifest serialization failed: {}", e)))?;
    
    kv.put(&manifest_key, &manifest_json)?
        .expiration_ttl(86400 * 30) // Keep for 30 days
        .execute()
        .await?;
    
    Ok(format!("kv:{}", manifest_key))
}

/// Main ingestion function
pub async fn ingest_documents(env: &Env, request: IngestionRequest) -> Result<IngestionResult> {
    let config = request.chunking_config.unwrap_or_default();
    let mut errors = Vec::new();
    let mut total_chunks = 0;
    let mut processed_docs = 0;

    for doc in &request.documents {
        match process_document(env, doc, &config).await {
            Ok(chunks) => {
                match store_chunks(env, &chunks).await {
                    Ok(stored_count) => {
                        total_chunks += stored_count;
                        processed_docs += 1;
                        
                        // Store PDF if provided
                        if let (Some(pdf_data), Some(filename)) = (&doc.pdf_data, &doc.pdf_filename) {
                            if let Err(e) = store_pdf(env, &doc.id, pdf_data, filename).await {
                                errors.push(format!("Failed to store PDF for {}: {}", doc.id, e));
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Failed to store chunks for {}: {}", doc.id, e));
                    }
                }
            }
            Err(e) => {
                errors.push(format!("Failed to process document {}: {}", doc.id, e));
            }
        }
    }

    // Create ingestion manifest
    let manifest_path = match create_manifest(env, &request.documents, total_chunks).await {
        Ok(path) => Some(path),
        Err(e) => {
            errors.push(format!("Failed to create manifest: {}", e));
            None
        }
    };

    Ok(IngestionResult {
        success: errors.is_empty(),
        documents_processed: processed_docs,
        chunks_created: total_chunks,
        errors,
        manifest_path,
    })
}

/// Validate ingestion setup
pub async fn validate_setup(env: &Env) -> Result<serde_json::Value> {
    let mut checks = serde_json::Map::new();
    
    // Check CF credentials
    match (env.var("CF_ACCOUNT_ID"), env.secret("CF_API_TOKEN")) {
        (Ok(_), Ok(_)) => checks.insert("cf_credentials".to_string(), serde_json::json!({"status": "ok"})),
        _ => checks.insert("cf_credentials".to_string(), serde_json::json!({"status": "error", "message": "CF_ACCOUNT_ID or CF_API_TOKEN not configured"})),
    };
    
    // Check KV binding
    match env.kv("CRESCENDAI_METADATA") {
        Ok(_) => checks.insert("kv_binding".to_string(), serde_json::json!({"status": "ok"})),
        Err(e) => checks.insert("kv_binding".to_string(), serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    
    // Test embedding
    match embed_text(env, "test embedding validation").await {
        Ok(vec) => checks.insert("embedding_test".to_string(), serde_json::json!({"status": "ok", "dimensions": vec.len()})),
        Err(e) => checks.insert("embedding_test".to_string(), serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    
    // Test Vectorize connectivity
    let vectorize_test = match query_top_k(env, "test query", 1).await {
        Ok(_) => serde_json::json!({"status": "ok"}),
        Err(e) => serde_json::json!({"status": "error", "message": e.to_string()}),
    };
    checks.insert("vectorize_test".to_string(), vectorize_test);
    
    Ok(serde_json::Value::Object(checks))
}

/// Purge document data using HTTP API
pub async fn purge_document(env: &Env, doc_id: &str) -> Result<serde_json::Value> {
    let account_id = env.var("CF_ACCOUNT_ID")
        .map_err(|_| worker::Error::RustError("CF_ACCOUNT_ID not configured".to_string()))?;
    let token = env.secret("CF_API_TOKEN")
        .map_err(|_| worker::Error::RustError("CF_API_TOKEN not configured".to_string()))?
        .to_string();
    let index_name = env.var("VECTORIZE_INDEX_NAME")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "crescendai-piano-pedagogy".to_string());
    let kv = env.kv("CRESCENDAI_METADATA")?;
    
    let mut results = serde_json::Map::new();
    let mut deleted_vectors = 0;
    let mut deleted_kv_keys = 0;
    
    // Try to delete common chunk patterns (simplified approach)
    for chunk_idx in 0..1000 { // Assume max 1000 chunks per doc
        let chunk_id = format!("{}::c{}", doc_id, chunk_idx);
        
        // Try to delete from Vectorize via HTTP API
        let delete_url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/vectorize/indexes/{}/delete",
            account_id.to_string(), index_name
        );
        
        let delete_payload = serde_json::json!({
            "ids": [chunk_id.clone()]
        });
        
        let mut headers = Headers::new();
        headers.set("Authorization", &format!("Bearer {}", token.to_string())).ok();
        headers.set("Content-Type", "application/json").ok();
        
        let mut init = RequestInit::new();
        init.with_method(Method::Post);
        init.with_headers(headers);
        init.with_body(Some(
            serde_json::to_string(&delete_payload).unwrap_or_default().into(),
        ));
        
        if let Ok(req) = Request::new_with_init(&delete_url, &init) {
            if let Ok(mut resp) = Fetch::Request(req).send().await {
                if resp.status_code() / 100 == 2 {
                    deleted_vectors += 1;
                }
            }
        }
        
        // Try to delete from KV
        let kv_key = format!("doc:{}:chunk:{}", doc_id, chunk_idx);
        if kv.delete(&kv_key).await.is_ok() {
            deleted_kv_keys += 1;
        }
        
        // Also try to delete PDF from KV
        let pdf_key = format!("pdf:{}:*.pdf", doc_id);
        let _ = kv.delete(&pdf_key).await; // Best effort
        
        // If we haven't found any for a while, assume we're done
        if chunk_idx > 10 && deleted_vectors == 0 && deleted_kv_keys == 0 {
            break;
        }
    }
    
    results.insert("deleted_vectors".to_string(), serde_json::json!(deleted_vectors));
    results.insert("deleted_kv_keys".to_string(), serde_json::json!(deleted_kv_keys));
    results.insert("note".to_string(), serde_json::json!("PDF and manifest cleanup attempted via KV"));
    
    Ok(serde_json::Value::Object(results))
}
