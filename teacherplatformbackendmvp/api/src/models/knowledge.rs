use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

/// Knowledge base document
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct KnowledgeDoc {
    pub id: Uuid,
    pub title: String,
    pub source_type: String, // 'pdf', 'video', 'text', 'web'
    pub source_url: Option<String>,
    pub owner_id: Uuid,
    pub is_public: bool,
    pub status: ProcessingStatus,
    pub total_chunks: i32,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
}

/// Processing status enum
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "processing_status", rename_all = "lowercase")]
pub enum ProcessingStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// Document chunk with embedding
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentChunk {
    pub id: Uuid,
    pub doc_id: Uuid,
    pub chunk_index: i32,
    pub content: String,
    #[sqlx(skip)] // Handled separately due to pgvector type
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Request to create a knowledge base document
#[derive(Debug, Deserialize)]
pub struct CreateKnowledgeRequest {
    pub title: String,
    pub source_type: String, // 'pdf', 'video', 'text', 'web'
    pub source_url: Option<String>,
    pub is_public: bool,
}

/// Response after creating a knowledge base document
#[derive(Debug, Serialize)]
pub struct CreateKnowledgeResponse {
    pub doc: KnowledgeDoc,
    pub upload_url: Option<String>, // Presigned R2 URL for PDF uploads
}

/// Processing status response
#[derive(Debug, Serialize)]
pub struct ProcessingStatusResponse {
    pub status: ProcessingStatus,
    pub progress: i32,      // Number of chunks processed
    pub total_chunks: i32,
    pub error_message: Option<String>,
}

/// Search result from hybrid search
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub chunk_id: Uuid,
    pub doc_id: Uuid,
    pub content: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Chunk metadata stored in JSONB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub page: Option<i32>,
    pub start_char: i32,
    pub end_char: i32,
    pub teacher_id: Option<Uuid>,
    pub is_public: bool,
}

impl ChunkMetadata {
    pub fn new(page: Option<i32>, start_char: i32, end_char: i32) -> Self {
        Self {
            page,
            start_char,
            end_char,
            teacher_id: None,
            is_public: false,
        }
    }
}
