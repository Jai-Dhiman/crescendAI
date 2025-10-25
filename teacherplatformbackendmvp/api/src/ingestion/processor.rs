use crate::errors::{AppError, Result};
use crate::ingestion::{
    chunk_pages, extract_pdf_text, generate_embeddings, store_chunks, ChunkConfig,
};
use crate::models::ProcessingStatus;
use sqlx::PgPool;
use uuid::Uuid;

/// Process a PDF document: extract → chunk → embed → store
pub async fn process_pdf_document(
    pool: &PgPool,
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
            // Mark as failed
            sqlx::query(
                r#"
                UPDATE knowledge_base_docs
                SET status = 'failed'
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
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
                SET status = 'failed'
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
            .execute(pool)
            .await?;
            return Err(e);
        }
    };

    // Generate embeddings
    let chunks_with_embeddings = match generate_embeddings(chunks).await {
        Ok(embeddings) => embeddings,
        Err(e) => {
            sqlx::query(
                r#"
                UPDATE knowledge_base_docs
                SET status = 'failed'
                WHERE id = $1
                "#,
            )
            .bind(doc_id)
            .execute(pool)
            .await?;
            return Err(e);
        }
    };

    // Store chunks to database
    let stored_count = store_chunks(pool, doc_id, chunks_with_embeddings).await?;

    Ok(stored_count)
}
