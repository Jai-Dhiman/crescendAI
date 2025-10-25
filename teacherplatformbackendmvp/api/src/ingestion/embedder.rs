use crate::errors::{AppError, Result};
use crate::ingestion::Chunk;

/// Generate embeddings for a batch of chunks
/// For MVP without Cloudflare credentials, this creates mock embeddings
pub async fn generate_embeddings(chunks: Vec<Chunk>) -> Result<Vec<(Chunk, Vec<f32>)>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    // For MVP, generate mock embeddings (768 dimensions for BGE-base-v1.5)
    // In production, this would call Workers AI API
    let mut results = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        // Create a deterministic but varied embedding based on chunk content
        let embedding = create_mock_embedding(&chunk.content);
        results.push((chunk, embedding));
    }

    Ok(results)
}

/// Create a mock embedding for testing (768 dimensions)
/// In production, this would be replaced with actual Workers AI call
fn create_mock_embedding(text: &str) -> Vec<f32> {
    const EMBEDDING_DIM: usize = 768;

    // Create a simple hash-based embedding
    // This is deterministic and varies based on content
    let mut embedding = vec![0.0; EMBEDDING_DIM];

    // Use text bytes to create variation
    let bytes = text.as_bytes();
    for (i, val) in embedding.iter_mut().enumerate() {
        // Create pseudo-random but deterministic values
        let idx = i % bytes.len().max(1);
        let byte_val = bytes.get(idx).copied().unwrap_or(0) as f32;
        *val = (byte_val / 255.0) * 2.0 - 1.0; // Normalize to [-1, 1]
    }

    // Add some variation based on position
    for (i, val) in embedding.iter_mut().enumerate() {
        *val += (i as f32 / EMBEDDING_DIM as f32) * 0.1;
    }

    // Normalize the embedding vector (L2 normalization)
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mock_embedding() {
        let embedding1 = create_mock_embedding("test text");
        let embedding2 = create_mock_embedding("test text");
        let embedding3 = create_mock_embedding("different text");

        // Same text should produce same embedding
        assert_eq!(embedding1, embedding2);

        // Different text should produce different embedding
        assert_ne!(embedding1, embedding3);

        // Check dimensions
        assert_eq!(embedding1.len(), 768);

        // Check normalization (L2 norm should be approximately 1.0)
        let magnitude: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }
}
