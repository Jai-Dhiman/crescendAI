//! Cross-encoder reranking service using Workers AI BGE-reranker-base
//!
//! Reranks retrieval results using a cross-encoder model that jointly processes
//! query-document pairs for fine-grained relevance scoring. This provides
//! 15-25% precision improvement over embedding similarity alone.

use serde::{Deserialize, Serialize};
use worker::Env;

/// The reranker model to use
pub const RERANKER_MODEL: &str = "@cf/baai/bge-reranker-base";

/// Maximum number of contexts to rerank in a single call
/// (Workers AI may have limits; 50-100 is typical)
pub const MAX_RERANK_CONTEXTS: usize = 50;

/// Context item for reranker input
#[derive(Serialize)]
struct RerankerContext {
    text: String,
}

/// Request format for Workers AI reranker
#[derive(Serialize)]
struct RerankerRequest {
    query: String,
    contexts: Vec<RerankerContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

/// Individual reranked result
#[derive(Deserialize, Debug)]
pub struct RerankedItem {
    /// Original index in the contexts array
    pub id: usize,
    /// Relevance score (higher = more relevant)
    pub score: f32,
}

/// Response format from Workers AI reranker
#[derive(Deserialize)]
struct RerankerResponse {
    response: Vec<RerankedItem>,
}

/// Rerank a list of text passages against a query
///
/// Returns indices and scores sorted by relevance (highest first).
/// If reranking fails, returns None so caller can fall back to original order.
pub async fn rerank_passages(
    env: &Env,
    query: &str,
    passages: &[String],
    top_k: Option<usize>,
) -> Result<Vec<RerankedItem>, String> {
    if passages.is_empty() {
        return Ok(Vec::new());
    }

    // Limit contexts to avoid timeouts
    let limited_passages: Vec<_> = passages
        .iter()
        .take(MAX_RERANK_CONTEXTS)
        .cloned()
        .collect();

    let ai = env
        .ai("AI")
        .map_err(|e| format!("Failed to get AI binding: {:?}", e))?;

    let request = RerankerRequest {
        query: query.to_string(),
        contexts: limited_passages
            .iter()
            .map(|text| RerankerContext { text: text.clone() })
            .collect(),
        top_k,
    };

    let request_json = serde_json::to_value(&request)
        .map_err(|e| format!("Failed to serialize reranker request: {:?}", e))?;

    let result = ai
        .run(RERANKER_MODEL, request_json)
        .await
        .map_err(|e| format!("Workers AI reranker call failed: {:?}", e))?;

    let response: RerankerResponse = serde_json::from_value(result)
        .map_err(|e| format!("Failed to parse reranker response: {:?}", e))?;

    Ok(response.response)
}

/// Rerank retrieval results and return reordered by relevance
///
/// This function:
/// 1. Extracts text from retrieval results
/// 2. Calls Workers AI reranker to score each against the query
/// 3. Returns results sorted by reranker score (descending)
///
/// Type parameter T must be cloneable and provide a text extraction method.
pub async fn rerank_results<T, F>(
    env: &Env,
    query: &str,
    results: Vec<T>,
    extract_text: F,
    top_k: usize,
) -> Result<Vec<(T, f32)>, String>
where
    T: Clone,
    F: Fn(&T) -> String,
{
    if results.is_empty() {
        return Ok(Vec::new());
    }

    // Extract text from each result
    let passages: Vec<String> = results.iter().map(&extract_text).collect();

    // Get reranked scores
    let reranked = rerank_passages(env, query, &passages, Some(top_k)).await?;

    // Map scores back to original results
    let mut scored_results: Vec<(T, f32)> = reranked
        .into_iter()
        .filter_map(|item| {
            results.get(item.id).map(|r| (r.clone(), item.score))
        })
        .collect();

    // Sort by score descending (reranker returns sorted but let's be safe)
    scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to top_k
    scored_results.truncate(top_k);

    Ok(scored_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_request_serialization() {
        let request = RerankerRequest {
            query: "How to practice scales?".to_string(),
            contexts: vec![
                RerankerContext {
                    text: "Scales build technique.".to_string(),
                },
                RerankerContext {
                    text: "Mozart wrote many sonatas.".to_string(),
                },
            ],
            top_k: Some(5),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("How to practice scales?"));
        assert!(json.contains("Scales build technique"));
        assert!(json.contains("top_k"));
    }

    #[test]
    fn test_reranked_item_deserialization() {
        let json = r#"{"id": 2, "score": 0.87}"#;
        let item: RerankedItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.id, 2);
        assert!((item.score - 0.87).abs() < 0.001);
    }
}
