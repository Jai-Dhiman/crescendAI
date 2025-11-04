use serde::{Deserialize, Serialize};
use worker::*;
use crate::db::knowledge::{KnowledgeChunk, search_chunks_fulltext, get_chunks_by_ids};
use sha2::{Sha256, Digest};

/// Knowledge Base chunk metadata stored in Vectorize
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KBChunk {
    pub id: String,     // e.g., "hanon_01::c0"
    pub doc_id: String, // e.g., "hanon_01"
    pub title: String,
    pub tags: Vec<String>,
    pub source: String,
    pub url: Option<String>,
    pub text: String,
    pub chunk_id: u32,
}

/// Search result with relevance score and citation info
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    pub chunk: KnowledgeChunk,
    pub relevance_score: f32,
    pub document_title: String,
    pub source: Option<String>,
    pub rank: usize,
}

/// Formatted search results with citations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FormattedSearchResults {
    pub results: Vec<SearchResult>,
    pub context: String, // Concatenated chunk contents for LLM
    pub citations: Vec<Citation>,
    pub total_chunks: usize,
}

/// Citation information for a knowledge source
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Citation {
    pub id: String,
    pub title: String,
    pub source: Option<String>,
    pub chunk_id: String,
}

/// Convert KnowledgeChunk (from D1) to KBChunk (legacy format) for backward compatibility
pub async fn knowledge_chunk_to_kb_chunk(env: &Env, chunk: KnowledgeChunk) -> Result<KBChunk> {
    // Get document metadata from D1
    let doc = crate::db::knowledge::get_document(env, &chunk.document_id).await
        .map_err(|e| worker::Error::RustError(format!("Failed to get document: {:?}", e)))?;

    // Parse metadata if available
    let tags: Vec<String> = if let Some(metadata_str) = &chunk.metadata {
        serde_json::from_str::<serde_json::Value>(metadata_str)
            .ok()
            .and_then(|v| v.get("tags").and_then(|t| t.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            }))
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok(KBChunk {
        id: chunk.id.clone(),
        doc_id: chunk.document_id.clone(),
        title: doc.title,
        tags,
        source: doc.source.unwrap_or_default(),
        url: doc.file_path, // Use file_path as URL
        text: chunk.content,
        chunk_id: chunk.chunk_index as u32,
    })
}

/// Generate cache key for embeddings
fn generate_embedding_cache_key(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    format!("embed:v1:{:x}", hash)
}

/// Generate cache key for search results
fn generate_search_cache_key(query: &str, limit: u32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.as_bytes());
    hasher.update(&limit.to_le_bytes());
    let hash = hasher.finalize();
    format!("search:v1:{:x}", hash)
}

/// Embeds a text using Workers AI binding with KV caching
pub async fn embed_text(env: &Env, text: &str) -> Result<Vec<f32>> {
    // Try to get from cache first
    if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
        let cache_key = generate_embedding_cache_key(text);
        if let Ok(Some(cached)) = kv.get(&cache_key).text().await {
            if let Ok(embedding) = serde_json::from_str::<Vec<f32>>(&cached) {
                console_log!("Embedding cache hit for text (len={})", text.len());
                return Ok(embedding);
            }
        }
    }

    // Generate embedding using AI binding
    let ai = env.ai("AI")?;
    let cf_model = env.var("CF_EMBED_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "@cf/google/embeddinggemma-300m".to_string());

    let payload = serde_json::json!({ "text": [text] });

    let response: serde_json::Value = ai
        .run(&cf_model, payload)
        .await
        .map_err(|e| worker::Error::RustError(format!("AI embedding failed: {}", e)))?;

    // Parse embedding from response
    if let Some(data) = response.get("data") {
        if let Some(arr) = data.as_array() {
            if !arr.is_empty() {
                if let Some(first) = arr[0].as_array() {
                    let embedding: Vec<f32> = first
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();

                    if !embedding.is_empty() {
                        // Cache the embedding
                        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
                            let cache_key = generate_embedding_cache_key(text);
                            if let Ok(json) = serde_json::to_string(&embedding) {
                                // Cache for 24 hours
                                let _ = kv.put(&cache_key, &json)?
                                    .expiration_ttl(86400)
                                    .execute()
                                    .await;
                            }
                        }
                        return Ok(embedding);
                    }
                }
            }
        }
    }

    Err(worker::Error::RustError(
        "Invalid embedding response format from CF AI".to_string()
    ))
}

/// Queries Vectorize - disabled in this build (worker crate lacks Vectorize binding)
/// This is a placeholder for when Vectorize support is added
pub async fn query_vectorize(_env: &Env, _query_embedding: &[f32], _k: usize) -> Result<Vec<String>> {
    // Returns chunk IDs from Vectorize similarity search
    // TODO: Implement when worker-rs adds Vectorize binding support
    Err(worker::Error::RustError(
        "Vectorize binding is unavailable in the current workers-rs version".to_string(),
    ))
}

/// Hybrid search: combines D1 FTS search with optional Vectorize semantic search
pub async fn hybrid_search(
    env: &Env,
    query: &str,
    limit: u32,
    use_cache: bool,
) -> Result<Vec<KnowledgeChunk>> {
    // Check cache first if enabled
    if use_cache {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            let cache_key = generate_search_cache_key(query, limit);
            if let Ok(Some(cached)) = kv.get(&cache_key).text().await {
                if let Ok(results) = serde_json::from_str::<Vec<KnowledgeChunk>>(&cached) {
                    console_log!("Search cache hit for query: {}", query);
                    return Ok(results);
                }
            }
        }
    }

    // For now, use D1 FTS only since Vectorize binding is unavailable
    // In the future, this will combine D1 FTS + Vectorize semantic search
    console_log!("Performing D1 FTS search for: {}", query);

    let fts_results = search_chunks_fulltext(env, query, limit * 2).await
        .map_err(|e| worker::Error::RustError(format!("FTS search failed: {}", e)))?;

    // TODO: When Vectorize is available, add semantic search:
    // 1. Generate query embedding
    // 2. Query Vectorize for top-k similar chunks
    // 3. Combine FTS + Vectorize results using reciprocal rank fusion
    // 4. Rerank the combined results

    // Limit to requested number
    let mut results: Vec<KnowledgeChunk> = fts_results.into_iter().take(limit as usize).collect();

    // Rerank results for better relevance
    results = rerank_chunks(query, results);

    // Cache the results if enabled
    if use_cache {
        if let Ok(kv) = env.kv("CRESCENDAI_METADATA") {
            let cache_key = generate_search_cache_key(query, limit);
            if let Ok(json) = serde_json::to_string(&results) {
                // Cache for 1 hour
                let _ = kv.put(&cache_key, &json)?
                    .expiration_ttl(3600)
                    .execute()
                    .await;
            }
        }
    }

    Ok(results)
}

/// Rerank chunks based on query relevance
/// Uses simple scoring for now, can be enhanced with a reranking model later
fn rerank_chunks(query: &str, mut chunks: Vec<KnowledgeChunk>) -> Vec<KnowledgeChunk> {
    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    // Score each chunk
    let mut scored_chunks: Vec<(f32, KnowledgeChunk)> = chunks
        .drain(..)
        .map(|chunk| {
            let content_lower = chunk.content.to_lowercase();

            // Simple relevance scoring based on term frequency and position
            let mut score = 0.0f32;

            for (i, term) in query_terms.iter().enumerate() {
                // Count occurrences
                let count = content_lower.matches(term).count();
                score += count as f32;

                // Bonus for exact phrase match
                if content_lower.contains(&query_lower) {
                    score += 5.0;
                }

                // Bonus for term appearing in the first 100 characters
                if let Some(pos) = content_lower.find(term) {
                    if pos < 100 {
                        score += 2.0 / (i as f32 + 1.0); // Earlier terms weighted more
                    }
                }
            }

            // Length normalization (prefer shorter, more focused chunks)
            let length_penalty = (chunk.content.len() as f32 / 1000.0).min(1.0);
            score = score / (1.0 + length_penalty * 0.5);

            (score, chunk)
        })
        .collect();

    // Sort by score descending
    scored_chunks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Return reranked chunks
    scored_chunks.into_iter().map(|(_, chunk)| chunk).collect()
}

/// Format search results with citations for LLM consumption
pub async fn format_search_results(
    env: &Env,
    chunks: Vec<KnowledgeChunk>,
) -> Result<FormattedSearchResults> {
    let total_chunks = chunks.len();
    let mut results = Vec::new();
    let mut citations = Vec::new();
    let mut context_parts = Vec::new();

    for (rank, chunk) in chunks.into_iter().enumerate() {
        // Get document metadata for citation
        let doc_title = if let Ok(doc) = crate::db::knowledge::get_document(env, &chunk.document_id).await {
            doc.title.clone()
        } else {
            "Unknown Source".to_string()
        };

        let source = if let Ok(doc) = crate::db::knowledge::get_document(env, &chunk.document_id).await {
            doc.source.clone()
        } else {
            None
        };

        // Add to citations
        citations.push(Citation {
            id: chunk.id.clone(),
            title: doc_title.clone(),
            source: source.clone(),
            chunk_id: chunk.id.clone(),
        });

        // Add to context with citation marker
        context_parts.push(format!(
            "[{}] {}\n\nSource: {} ({})\n",
            rank + 1,
            chunk.content,
            doc_title,
            source.as_deref().unwrap_or("No source provided")
        ));

        // Add to results
        results.push(SearchResult {
            chunk,
            relevance_score: 1.0 / (rank as f32 + 1.0), // Reciprocal rank
            document_title: doc_title,
            source,
            rank: rank + 1,
        });
    }

    Ok(FormattedSearchResults {
        results,
        context: context_parts.join("\n---\n\n"),
        citations,
        total_chunks,
    })
}
