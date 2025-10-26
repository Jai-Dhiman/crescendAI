use crate::ai::workers_ai::WorkersAIClient;
use crate::errors::{AppError, Result};
use crate::ingestion::Chunk;
use std::time::Duration;

/// Generate embeddings for a batch of chunks using Cloudflare Workers AI
pub async fn generate_embeddings(
    workers_ai: &WorkersAIClient,
    chunks: Vec<Chunk>,
) -> Result<Vec<(Chunk, Vec<f32>)>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    tracing::info!("Generating embeddings for {} chunks", chunks.len());

    // Process chunks in batches to respect rate limits and avoid overwhelming the API
    const BATCH_SIZE: usize = 50;
    let mut results = Vec::with_capacity(chunks.len());

    for (batch_idx, chunk_batch) in chunks.chunks(BATCH_SIZE).enumerate() {
        tracing::debug!(
            "Processing embedding batch {}/{} ({} chunks)",
            batch_idx + 1,
            (chunks.len() + BATCH_SIZE - 1) / BATCH_SIZE,
            chunk_batch.len()
        );

        // Extract text from chunks for embedding
        let texts: Vec<&str> = chunk_batch.iter().map(|c| c.content.as_str()).collect();

        // Generate embeddings with retry logic
        let embeddings = generate_embeddings_with_retry(workers_ai, texts).await?;

        // Pair chunks with their embeddings
        for (chunk, embedding) in chunk_batch.iter().zip(embeddings.iter()) {
            results.push((chunk.clone(), embedding.clone()));
        }

        // Small delay between batches to avoid rate limiting
        if batch_idx < (chunks.len() + BATCH_SIZE - 1) / BATCH_SIZE - 1 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    tracing::info!("Successfully generated {} embeddings", results.len());
    Ok(results)
}

/// Generate embeddings with exponential backoff retry logic
async fn generate_embeddings_with_retry(
    workers_ai: &WorkersAIClient,
    texts: Vec<&str>,
) -> Result<Vec<Vec<f32>>> {
    const MAX_RETRIES: u32 = 3;
    let mut retry_count = 0;

    loop {
        match workers_ai.batch_embed(texts.clone()).await {
            Ok(embeddings) => {
                // Validate embedding dimensions (should be 768 for BGE-base-en-v1.5)
                for (idx, embedding) in embeddings.iter().enumerate() {
                    if embedding.len() != 768 {
                        return Err(AppError::Internal(format!(
                            "Invalid embedding dimension for text {}: expected 768, got {}",
                            idx,
                            embedding.len()
                        )));
                    }
                }
                return Ok(embeddings);
            }
            Err(e) => {
                retry_count += 1;
                if retry_count >= MAX_RETRIES {
                    tracing::error!(
                        "Failed to generate embeddings after {} retries: {}",
                        MAX_RETRIES,
                        e
                    );
                    return Err(AppError::Internal(format!(
                        "Failed to generate embeddings: {}",
                        e
                    )));
                }

                // Exponential backoff: 1s, 2s, 4s
                let delay = Duration::from_secs(2u64.pow(retry_count - 1));
                tracing::warn!(
                    "Embedding generation failed (attempt {}/{}), retrying in {:?}: {}",
                    retry_count,
                    MAX_RETRIES,
                    delay,
                    e
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Store chunks with embeddings to database
/// Returns the number of chunks stored
pub async fn store_chunks(
    pool: &sqlx::PgPool,
    doc_id: uuid::Uuid,
    chunks_with_embeddings: Vec<(Chunk, Vec<f32>)>,
) -> Result<usize> {
    if chunks_with_embeddings.is_empty() {
        return Ok(0);
    }

    let mut stored_count = 0;

    // Store chunks in batches to avoid overwhelming the database
    const BATCH_SIZE: usize = 100;

    for (chunk_idx, batch) in chunks_with_embeddings.chunks(BATCH_SIZE).enumerate() {
        let mut tx = pool.begin().await?;

        for (i, (chunk, embedding)) in batch.iter().enumerate() {
            let global_idx = chunk_idx * BATCH_SIZE + i;

            // Create metadata JSON
            let metadata = serde_json::json!({
                "page": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            });

            // Convert embedding to pgvector format string
            // pgvector expects format: [1.0,2.0,3.0]
            let embedding_str = format!(
                "[{}]",
                embedding
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            // Insert chunk
            sqlx::query(
                r#"
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5)
                "#,
            )
            .bind(doc_id)
            .bind(global_idx as i32)
            .bind(&chunk.content)
            .bind(&embedding_str)
            .bind(&metadata)
            .execute(&mut *tx)
            .await?;

            stored_count += 1;
        }

        tx.commit().await?;
    }

    // Update the knowledge base doc with total chunk count
    sqlx::query(
        r#"
        UPDATE knowledge_base_docs
        SET total_chunks = $1, processed_at = NOW(), status = 'completed'
        WHERE id = $2
        "#,
    )
    .bind(stored_count as i32)
    .bind(doc_id)
    .execute(pool)
    .await?;

    Ok(stored_count)
}
