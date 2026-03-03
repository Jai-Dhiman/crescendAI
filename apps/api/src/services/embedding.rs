//! Workers AI embedding service using BGE-base-en-v1.5
//!
//! Generates 768-dimensional embeddings for text using Cloudflare Workers AI.
//! Model: @cf/baai/bge-base-en-v1.5

use serde::{Deserialize, Serialize};
use worker::Ai;

/// The embedding model to use
pub const EMBEDDING_MODEL: &str = "@cf/baai/bge-base-en-v1.5";

/// Embedding dimension for BGE-base-en-v1.5
pub const EMBEDDING_DIM: usize = 768;

/// Request format for Workers AI text embedding
#[derive(Serialize)]
struct EmbeddingRequest {
    text: Vec<String>,
}

/// Response format from Workers AI
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<Vec<f32>>,
}

/// Generate embeddings for a single text
pub async fn generate_embedding(ai: &Ai, text: &str) -> Result<Vec<f32>, String> {
    let embeddings = generate_embeddings(ai, &[text.to_string()]).await?;
    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| "No embedding returned".to_string())
}

/// Generate embeddings for multiple texts (batch)
pub async fn generate_embeddings(ai: &Ai, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // Workers AI expects a specific format for text embeddings
    let request = serde_json::json!({
        "text": texts
    });

    let result = ai
        .run(EMBEDDING_MODEL, request)
        .await
        .map_err(|e| format!("Workers AI embedding failed: {:?}", e))?;

    // Parse the response
    let response: EmbeddingResponse = serde_json::from_value(result)
        .map_err(|e| format!("Failed to parse embedding response: {:?}", e))?;

    // Validate dimensions
    for (i, embedding) in response.data.iter().enumerate() {
        if embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "Embedding {} has wrong dimension: expected {}, got {}",
                i,
                EMBEDDING_DIM,
                embedding.len()
            ));
        }
    }

    Ok(response.data)
}

/// Normalize an embedding vector to unit length (for cosine similarity)
pub fn normalize_embedding(embedding: &mut [f32]) {
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in embedding.iter_mut() {
            *x /= magnitude;
        }
    }
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot_product / (mag_a * mag_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_embedding() {
        let mut embedding = vec![3.0, 4.0];
        normalize_embedding(&mut embedding);

        // Should be unit vector: [0.6, 0.8]
        assert!((embedding[0] - 0.6).abs() < 0.0001);
        assert!((embedding[1] - 0.8).abs() < 0.0001);

        // Magnitude should be 1
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vector should have similarity 1.0
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 0.0001);

        // Orthogonal vectors should have similarity 0.0
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.0001);

        // Opposite vectors should have similarity -1.0
        let c = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &c) + 1.0).abs() < 0.0001);
    }
}
