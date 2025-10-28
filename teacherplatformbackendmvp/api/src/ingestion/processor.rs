use crate::ai::workers_ai::WorkersAIClient;
use crate::errors::Result;
use crate::ingestion::{
    chunk_pages, extract_pdf_text, generate_embeddings, store_chunks, ChunkConfig,
};
use sqlx::PgPool;
use uuid::Uuid;

/// Process a PDF document: extract → chunk → embed → store
pub async fn process_pdf_document(
    pool: &PgPool,
    workers_ai: &WorkersAIClient,
    doc_id: Uuid,
    pdf_bytes: &[u8],
) -> Result<usize> {
    // Update status to processing
    sqlx::query(
        r#"
        UPDATE knowledge_base_docs
        SET status = 'processing'
        WHERE id = $1
        "#,
    )
    .bind(doc_id)
    .execute(pool)
    .await?;

    // Extract text from PDF
    let pages = match extract_pdf_text(pdf_bytes) {
        Ok(pages) => pages,
        Err(e) => {
            // Mark as failed with error message
            sqlx::query(
                r#"
                UPDATE knowledge_base_docs
                SET status = 'failed', error_message = $2
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
            .bind(format!("Failed to extract PDF text: {}", e))
            .execute(pool)
            .await?;
            return Err(e);
        }
    };

    // Chunk the pages
    let config = ChunkConfig::default(); // 512 tokens, 128 overlap
    let chunks = match chunk_pages(pages, &config) {
        Ok(chunks) => chunks,
        Err(e) => {
            sqlx::query(
                r#"
                UPDATE knowledge_base_docs
                SET status = 'failed', error_message = $2
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
            .bind(format!("Failed to chunk text: {}", e))
            .execute(pool)
            .await?;
            return Err(e);
        }
    };

    // Generate embeddings using Workers AI
    let chunks_with_embeddings = match generate_embeddings(workers_ai, chunks).await {
        Ok(embeddings) => embeddings,
        Err(e) => {
            // Store error message in database
            sqlx::query(
                r#"
                UPDATE knowledge_base_docs
                SET status = 'failed', error_message = $2
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
            .bind(format!("Failed to generate embeddings: {}", e))
            .execute(pool)
            .await?;
            return Err(e);
        }
    };

    // Store chunks to database
    let stored_count = store_chunks(pool, doc_id, chunks_with_embeddings).await?;

    Ok(stored_count)
}
