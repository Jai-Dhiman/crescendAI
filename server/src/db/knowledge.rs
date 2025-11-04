// CrescendAI Server - Knowledge Base Database Queries

use worker::*;
use wasm_bindgen::JsValue;
use serde::{Deserialize, Serialize};
use super::{DbError, DbResult, current_timestamp_ms, generate_id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDocument {
    pub id: String,
    pub title: String,
    pub source: Option<String>,
    pub doc_type: String, // "pdf", "article", "book", "other"
    pub author: Option<String>,
    pub year: Option<i32>,
    pub file_path: Option<String>, // R2 key if stored
    pub metadata: Option<String>, // JSON object
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeChunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: i32,
    pub content: String,
    pub vectorize_id: Option<String>, // ID in Vectorize index
    pub token_count: Option<i32>,
    pub metadata: Option<String>, // JSON object (e.g., page number)
    pub created_at: i64,
}

// Insert a new knowledge document
pub async fn insert_document(
    env: &Env,
    title: &str,
    source: Option<&str>,
    doc_type: &str,
    author: Option<&str>,
    year: Option<i32>,
    file_path: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<KnowledgeDocument> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    // Validate doc_type
    if !["pdf", "article", "book", "other"].contains(&doc_type) {
        return Err(DbError::InvalidInput(format!("Invalid doc_type: {}", doc_type)));
    }

    let doc_id = generate_id();
    let now = current_timestamp_ms();

    let stmt = db.prepare("
        INSERT INTO knowledge_documents (id, title, source, doc_type, author, year, file_path, metadata, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(&doc_id),
            JsValue::from_str(title),
            source.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            JsValue::from_str(doc_type),
            author.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            year.map(|y| JsValue::from_f64(y as f64)).unwrap_or(JsValue::NULL),
            file_path.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            metadata.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            JsValue::from_f64(now as f64),
            JsValue::from_f64(now as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert document: {}", e)))?;

    Ok(KnowledgeDocument {
        id: doc_id,
        title: title.to_string(),
        source: source.map(|s| s.to_string()),
        doc_type: doc_type.to_string(),
        author: author.map(|s| s.to_string()),
        year,
        file_path: file_path.map(|s| s.to_string()),
        metadata: metadata.map(|s| s.to_string()),
        created_at: now,
        updated_at: now,
    })
}

// Insert a new knowledge chunk
pub async fn insert_chunk(
    env: &Env,
    document_id: &str,
    chunk_index: i32,
    content: &str,
    vectorize_id: Option<&str>,
    token_count: Option<i32>,
    metadata: Option<&str>,
) -> DbResult<KnowledgeChunk> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let chunk_id = generate_id();
    let now = current_timestamp_ms();

    let stmt = db.prepare("
        INSERT INTO knowledge_chunks (id, document_id, chunk_index, content, vectorize_id, token_count, metadata, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    ");

    let query = stmt
        .bind(&[
            JsValue::from_str(&chunk_id),
            JsValue::from_str(document_id),
            JsValue::from_f64(chunk_index as f64),
            JsValue::from_str(content),
            vectorize_id.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            token_count.map(|t| JsValue::from_f64(t as f64)).unwrap_or(JsValue::NULL),
            metadata.map(|s| JsValue::from_str(s)).unwrap_or(JsValue::NULL),
            JsValue::from_f64(now as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert chunk: {}", e)))?;

    // Also insert into FTS index
    let fts_stmt = db.prepare("
        INSERT INTO knowledge_chunks_fts (chunk_id, content)
        VALUES (?1, ?2)
    ");

    let fts_query = fts_stmt
        .bind(&[
            JsValue::from_str(&chunk_id),
            JsValue::from_str(content),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind FTS parameters: {}", e)))?;

    fts_query.run().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to insert into FTS index: {}", e)))?;

    Ok(KnowledgeChunk {
        id: chunk_id,
        document_id: document_id.to_string(),
        chunk_index,
        content: content.to_string(),
        vectorize_id: vectorize_id.map(|s| s.to_string()),
        token_count,
        metadata: metadata.map(|s| s.to_string()),
        created_at: now,
    })
}

// Get chunks by their IDs (useful after Vectorize query)
pub async fn get_chunks_by_ids(
    env: &Env,
    chunk_ids: &[String],
) -> DbResult<Vec<KnowledgeChunk>> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }

    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    // Build placeholders for IN clause
    let placeholders: Vec<String> = (1..=chunk_ids.len())
        .map(|i| format!("?{}", i))
        .collect();
    let placeholders_str = placeholders.join(", ");

    let query_str = format!(
        "SELECT id, document_id, chunk_index, content, vectorize_id, token_count, metadata, created_at
         FROM knowledge_chunks
         WHERE id IN ({})",
        placeholders_str
    );

    let stmt = db.prepare(&query_str);

    let bindings: Vec<JsValue> = chunk_ids.iter()
        .map(|id| JsValue::from_str(id))
        .collect();

    let query = stmt
        .bind(&bindings)
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query chunks: {}", e)))?;

    result.results::<KnowledgeChunk>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize chunks: {}", e)))
}

// Full-text search on knowledge chunks
pub async fn search_chunks_fulltext(
    env: &Env,
    query: &str,
    limit: u32,
) -> DbResult<Vec<KnowledgeChunk>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT kc.id, kc.document_id, kc.chunk_index, kc.content, kc.vectorize_id, kc.token_count, kc.metadata, kc.created_at
        FROM knowledge_chunks kc
        INNER JOIN knowledge_chunks_fts fts ON kc.id = fts.chunk_id
        WHERE fts.content MATCH ?1
        ORDER BY rank
        LIMIT ?2
    ");

    let query_result = stmt
        .bind(&[
            JsValue::from_str(query),
            JsValue::from_f64(limit as f64),
        ])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query_result.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to search chunks: {}", e)))?;

    result.results::<KnowledgeChunk>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize chunks: {}", e)))
}

// Get chunks for a specific document
pub async fn get_chunks_by_document(
    env: &Env,
    document_id: &str,
) -> DbResult<Vec<KnowledgeChunk>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, document_id, chunk_index, content, vectorize_id, token_count, metadata, created_at
        FROM knowledge_chunks
        WHERE document_id = ?1
        ORDER BY chunk_index ASC
    ");

    let query = stmt
        .bind(&[JsValue::from_str(document_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query chunks: {}", e)))?;

    result.results::<KnowledgeChunk>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize chunks: {}", e)))
}

// Get a document by ID
pub async fn get_document(env: &Env, document_id: &str) -> DbResult<KnowledgeDocument> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let stmt = db.prepare("
        SELECT id, title, source, doc_type, author, year, file_path, metadata, created_at, updated_at
        FROM knowledge_documents
        WHERE id = ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_str(document_id)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.first::<KnowledgeDocument>(None).await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query document: {}", e)))?;

    result.ok_or_else(|| DbError::NotFound(format!("Document not found: {}", document_id)))
}

// List all documents
pub async fn list_documents(
    env: &Env,
    limit: Option<u32>,
) -> DbResult<Vec<KnowledgeDocument>> {
    let db = env.d1("DB")
        .map_err(|e| DbError::DatabaseError(format!("Failed to get DB binding: {}", e)))?;

    let limit_val = limit.unwrap_or(50);

    let stmt = db.prepare("
        SELECT id, title, source, doc_type, author, year, file_path, metadata, created_at, updated_at
        FROM knowledge_documents
        ORDER BY created_at DESC
        LIMIT ?1
    ");

    let query = stmt
        .bind(&[JsValue::from_f64(limit_val as f64)])
        .map_err(|e| DbError::DatabaseError(format!("Failed to bind parameters: {}", e)))?;

    let result = query.all().await
        .map_err(|e| DbError::DatabaseError(format!("Failed to query documents: {}", e)))?;

    result.results::<KnowledgeDocument>()
        .map_err(|e| DbError::SerializationError(format!("Failed to deserialize documents: {}", e)))
}
