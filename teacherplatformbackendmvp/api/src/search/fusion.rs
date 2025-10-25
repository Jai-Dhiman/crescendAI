use crate::errors::Result;
use crate::models::SearchResult;
use crate::search::{bm25_search, vector_search, UserContext};
use sqlx::PgPool;
use std::collections::HashMap;
use uuid::Uuid;

/// RRF constant (k parameter)
const RRF_K: f32 = 60.0;

/// Perform hybrid search using Reciprocal Rank Fusion
/// Combines vector similarity search and BM25 keyword search
pub async fn hybrid_search(
    pool: &PgPool,
    query_embedding: Vec<f32>,
    query_text: &str,
    user_context: &UserContext,
    final_limit: usize,
) -> Result<Vec<SearchResult>> {
    // Fetch more results from each method to ensure good coverage after fusion
    let intermediate_limit = final_limit * 2;

    // Perform both searches in parallel
    let (vector_results, bm25_results) = tokio::try_join!(
        vector_search(pool, query_embedding, user_context, intermediate_limit),
        bm25_search(pool, query_text, user_context, intermediate_limit),
    )?;

    // Merge results using RRF
    let merged = reciprocal_rank_fusion(vector_results, bm25_results);

    // Return top results
    Ok(merged.into_iter().take(final_limit).collect())
}

/// Merge two ranked lists using Reciprocal Rank Fusion
/// RRF score = sum(1 / (k + rank)) for each result list
fn reciprocal_rank_fusion(
    vector_results: Vec<SearchResult>,
    bm25_results: Vec<SearchResult>,
) -> Vec<SearchResult> {
    let mut rrf_scores: HashMap<Uuid, f32> = HashMap::new();
    let mut chunk_map: HashMap<Uuid, SearchResult> = HashMap::new();

    // Process vector search results
    for (rank, result) in vector_results.into_iter().enumerate() {
        let chunk_id = result.chunk_id;
        *rrf_scores.entry(chunk_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32);
        chunk_map.insert(chunk_id, result);
    }

    // Process BM25 search results
    for (rank, result) in bm25_results.into_iter().enumerate() {
        let chunk_id = result.chunk_id;
        *rrf_scores.entry(chunk_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f32);
        // Update chunk_map if not already present
        chunk_map.entry(chunk_id).or_insert(result);
    }

    // Sort by RRF score descending
    let mut scored_chunks: Vec<(Uuid, f32)> = rrf_scores.into_iter().collect();
    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build final results with RRF scores
    scored_chunks
        .into_iter()
        .filter_map(|(chunk_id, rrf_score)| {
            chunk_map.get(&chunk_id).map(|result| SearchResult {
                chunk_id: result.chunk_id,
                doc_id: result.doc_id,
                content: result.content.clone(),
                score: rrf_score, // Replace original score with RRF score
                metadata: result.metadata.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_result(chunk_id: Uuid, doc_id: Uuid, content: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id,
            doc_id,
            content: content.to_string(),
            score,
            metadata: json!({}),
        }
    }

    #[test]
    fn test_rrf_merge() {
        let chunk1 = Uuid::new_v4();
        let chunk2 = Uuid::new_v4();
        let chunk3 = Uuid::new_v4();
        let doc = Uuid::new_v4();

        let vector_results = vec![
            create_test_result(chunk1, doc, "chunk 1", 0.9),
            create_test_result(chunk2, doc, "chunk 2", 0.8),
        ];

        let bm25_results = vec![
            create_test_result(chunk2, doc, "chunk 2", 0.7),
            create_test_result(chunk3, doc, "chunk 3", 0.6),
        ];

        let merged = reciprocal_rank_fusion(vector_results, bm25_results);

        // chunk2 should be first (appears in both lists)
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].chunk_id, chunk2);
    }

    #[test]
    fn test_rrf_empty() {
        let merged = reciprocal_rank_fusion(vec![], vec![]);
        assert!(merged.is_empty());
    }
}
