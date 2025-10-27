use crate::ai::workers_ai::WorkersAIClient;
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

/// Perform hybrid search with re-ranking using cross-encoder model
pub async fn hybrid_search_with_rerank(
    pool: &PgPool,
    query_embedding: Vec<f32>,
    query_text: &str,
    user_context: &UserContext,
    workers_ai: &WorkersAIClient,
    final_limit: usize,
) -> Result<Vec<SearchResult>> {
    // Fetch more candidates for re-ranking (3-4x final limit for best results)
    let candidate_limit = final_limit * 3;

    // Perform both searches in parallel
    let (vector_results, bm25_results) = tokio::try_join!(
        vector_search(pool, query_embedding, user_context, candidate_limit),
        bm25_search(pool, query_text, user_context, candidate_limit),
    )?;

    // Merge results using RRF to get candidates
    let rrf_candidates = reciprocal_rank_fusion(vector_results, bm25_results);

    // Take top candidates for re-ranking (2x final limit)
    let rerank_candidates: Vec<_> = rrf_candidates
        .into_iter()
        .take(final_limit * 2)
        .collect();

    if rerank_candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Re-rank using cross-encoder model
    tracing::debug!("Re-ranking {} candidates", rerank_candidates.len());
    let candidate_texts: Vec<&str> = rerank_candidates
        .iter()
        .map(|r| r.content.as_str())
        .collect();

    let rerank_results = workers_ai
        .rerank(query_text, candidate_texts)
        .await
        .map_err(|e| {
            tracing::warn!("Re-ranking failed, falling back to RRF scores: {}", e);
            e
        });

    // If re-ranking succeeds, use those scores; otherwise fall back to RRF
    let final_results = match rerank_results {
        Ok(rerank_scores) => {
            // Map re-rank scores back to search results
            let mut scored_results: Vec<(SearchResult, f32)> = rerank_scores
                .into_iter()
                .filter_map(|rr| {
                    rerank_candidates.get(rr.id).map(|sr| (sr.clone(), rr.score))
                })
                .collect();

            // Sort by re-rank score descending
            scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Update scores and take top results
            scored_results
                .into_iter()
                .take(final_limit)
                .map(|(mut result, rerank_score)| {
                    result.score = rerank_score;
                    result
                })
                .collect()
        }
        Err(_) => {
            // Fall back to RRF scores
            rerank_candidates
                .into_iter()
                .take(final_limit)
                .collect()
        }
    };

    Ok(final_results)
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
                doc_title: result.doc_title.clone(),
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
            doc_title: "Test Document".to_string(),
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
