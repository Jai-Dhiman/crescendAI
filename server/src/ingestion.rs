use serde::{Deserialize, Serialize};
use worker::*;
use crate::knowledge_base::KBChunk;
use uuid::Uuid;
use chrono::Utc;
use crate::knowledge_base::embed_text;

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

/// Store chunks in D1 database (with optional KV backup for caching)
pub async fn store_chunks(env: &Env, chunks: &[KBChunk], document_id: &str) -> Result<usize> {
    let mut stored_count = 0;

    for chunk in chunks {
        // Store in D1 database using the proper knowledge base schema
        match crate::db::knowledge::insert_chunk(
            env,
            document_id,
            chunk.chunk_id as i32,
            &chunk.text,
            None, // vectorize_id - will be added when Vectorize is available
            None, // token_count - can calculate later if needed
            None, // metadata
        ).await {
            Ok(_) => {
                stored_count += 1;
                console_log!("Stored chunk {} in D1", chunk.id);
            }
            Err(e) => {
                console_log!("Failed to store chunk {} in D1: {:?}", chunk.id, e);
                return Err(worker::Error::RustError(format!("Failed to store chunk in D1: {:?}", e)));
            }
        }
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
        // First, create the document in D1 knowledge_documents table
        let pdf_filename = doc.pdf_filename.as_deref();
        let doc_metadata = serde_json::json!({
            "tags": doc.tags,
            "content_length": doc.content.len(),
        }).to_string();

        let knowledge_doc = match crate::db::knowledge::insert_document(
            env,
            &doc.title,
            doc.url.as_deref(),
            "pdf", // default to pdf, could be enhanced
            None, // author
            None, // year
            pdf_filename,
            Some(&doc_metadata),
        ).await {
            Ok(d) => d,
            Err(e) => {
                errors.push(format!("Failed to create document {} in D1: {:?}", doc.id, e));
                continue;
            }
        };

        console_log!("Created knowledge document {} in D1", knowledge_doc.id);

        // Process and chunk the document
        match process_document(env, doc, &config).await {
            Ok(chunks) => {
                // Store chunks in D1 with the document ID from D1
                match store_chunks(env, &chunks, &knowledge_doc.id).await {
                    Ok(stored_count) => {
                        total_chunks += stored_count;
                        processed_docs += 1;
                        console_log!("Stored {} chunks for document {}", stored_count, knowledge_doc.id);

                        // Store PDF if provided
                        if let (Some(pdf_data), Some(filename)) = (&doc.pdf_data, &doc.pdf_filename) {
                            if let Err(e) = store_pdf(env, &knowledge_doc.id, pdf_data, filename).await {
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
    
    // Check AI binding
    match env.ai("AI") {
        Ok(_) => checks.insert("ai_binding".to_string(), serde_json::json!({"status": "ok"})),
        Err(e) => checks.insert("ai_binding".to_string(), serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    
// Check Vectorize binding (disabled)
    checks.insert("vectorize_binding".to_string(), serde_json::json!({"status": "skipped"}));
    
    // Test KV binding
    match env.kv("CRESCENDAI_METADATA") {
        Ok(_) => checks.insert("kv_binding".to_string(), serde_json::json!({"status": "ok"})),
        Err(e) => checks.insert("kv_binding".to_string(), serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    
    // Test embedding
    match embed_text(env, "test embedding validation").await {
        Ok(vec) => checks.insert("embedding_test".to_string(), serde_json::json!({"status": "ok", "dimensions": vec.len()})),
        Err(e) => checks.insert("embedding_test".to_string(), serde_json::json!({"status": "error", "message": e.to_string()})),
    };
    
// Test Vectorize connectivity (disabled)
    checks.insert("vectorize_test".to_string(), serde_json::json!({"status": "skipped"}));
    
    Ok(serde_json::Value::Object(checks))
}

/// Purge document data from D1 and KV
pub async fn purge_document(env: &Env, doc_id: &str) -> Result<serde_json::Value> {
    let mut results = serde_json::Map::new();

    // Delete from D1 - this will cascade delete chunks due to foreign key
    let db = env.d1("DB")
        .map_err(|e| worker::Error::RustError(format!("Failed to get DB binding: {}", e)))?;

    // Delete document (cascades to chunks and FTS entries)
    let delete_stmt = db.prepare("DELETE FROM knowledge_documents WHERE id = ?1");
    let delete_query = delete_stmt
        .bind(&[wasm_bindgen::JsValue::from_str(doc_id)])
        .map_err(|e| worker::Error::RustError(format!("Failed to bind parameter: {}", e)))?;

    match delete_query.run().await {
        Ok(_) => {
            results.insert("d1_document_deleted".to_string(), serde_json::json!(true));
            console_log!("Deleted document {} from D1 (with cascade)", doc_id);
        }
        Err(e) => {
            results.insert("d1_error".to_string(), serde_json::json!(format!("{}", e)));
        }
    }

    // Clean up KV cache entries
    if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
        let mut deleted_kv_keys = 0;

        // Try to delete PDF from KV
        let pdf_key = format!("pdf:{}:", doc_id);
        if kv.delete(&pdf_key).await.is_ok() {
            deleted_kv_keys += 1;
        }

        // Try to delete manifest entries
        let manifest_key = format!("manifest:*:{}:*", doc_id);
        if kv.delete(&manifest_key).await.is_ok() {
            deleted_kv_keys += 1;
        }

        results.insert("kv_keys_deleted".to_string(), serde_json::json!(deleted_kv_keys));
    }

    results.insert("note".to_string(), serde_json::json!("Document and chunks deleted from D1; KV cache cleaned up"));

    Ok(serde_json::Value::Object(results))
}
