use crate::models::{PedagogyChunk, Performance, PerformanceDimensions, RetrievalResult, SourceType};
use std::collections::HashMap;
use worker::d1::D1Database;
use worker::Env;

#[cfg(feature = "ssr")]
use super::embedding::generate_embedding;
#[cfg(feature = "ssr")]
use super::vectorize_binding::{get_vectorize_index, query_vectors, VectorMetadata};

/// RRF constant that dampens high-rank dominance
const RRF_K: f64 = 60.0;

/// Dimension name to technique keyword mapping
fn dimension_to_techniques(dimension: &str) -> Vec<&'static str> {
    match dimension {
        "timing" => vec!["timing", "rhythm", "pulse", "rubato"],
        "articulation_length" | "articulation_touch" => {
            vec!["articulation", "legato", "staccato", "touch"]
        }
        "pedal_amount" | "pedal_clarity" => vec!["pedal", "pedaling", "sustain", "damper"],
        "timbre_variety" | "timbre_depth" | "timbre_brightness" | "timbre_loudness" => {
            vec!["tone", "timbre", "color", "voicing", "singing tone"]
        }
        "dynamics_range" => vec!["dynamics", "forte", "piano", "crescendo", "diminuendo"],
        "tempo" => vec!["tempo", "speed", "pacing"],
        "space" => vec!["space", "silence", "pause", "breath"],
        "balance" => vec!["balance", "voicing", "texture"],
        "drama" => vec!["drama", "expression", "emotion", "intensity"],
        "mood_valence" | "mood_energy" | "mood_imagination" => {
            vec!["mood", "character", "expression", "imagination"]
        }
        "interpretation_sophistication" | "interpretation_overall" => {
            vec!["interpretation", "style", "phrasing", "musicality"]
        }
        _ => vec![],
    }
}

/// Build a retrieval query from performance context and weak dimensions
pub fn build_retrieval_query(
    performance: &Performance,
    dimensions: &PerformanceDimensions,
    weakness_threshold: f64,
) -> String {
    let mut query_parts = Vec::new();

    // Add composer and piece context
    query_parts.push(performance.composer.clone());
    if !performance.piece_title.is_empty() {
        query_parts.push(performance.piece_title.clone());
    }

    // Find weak dimensions (below threshold) and add technique keywords
    let dimension_scores = [
        ("timing", dimensions.timing),
        ("articulation_length", dimensions.articulation_length),
        ("articulation_touch", dimensions.articulation_touch),
        ("pedal_amount", dimensions.pedal_amount),
        ("pedal_clarity", dimensions.pedal_clarity),
        ("timbre_variety", dimensions.timbre_variety),
        ("timbre_depth", dimensions.timbre_depth),
        ("dynamics_range", dimensions.dynamics_range),
        ("drama", dimensions.drama),
        ("mood_imagination", dimensions.mood_imagination),
        ("interpretation_overall", dimensions.interpretation_overall),
    ];

    for (name, score) in dimension_scores {
        if score < weakness_threshold {
            for technique in dimension_to_techniques(name) {
                if !query_parts.contains(&technique.to_string()) {
                    query_parts.push(technique.to_string());
                }
            }
        }
    }

    // Limit query length
    query_parts.truncate(10);
    query_parts.join(" ")
}

/// BM25 full-text search using D1 FTS5 with column weighting
///
/// Column weights (higher = more important):
/// - text: 1.0 (base content)
/// - source_title: 2.0 (title matches are highly relevant)
/// - source_author: 1.5 (author matches important for authority)
/// - composers: 1.5 (composer context crucial for piano pedagogy)
/// - pieces: 1.5 (specific piece references valuable)
/// - techniques: 1.2 (technique keyword matches useful)
pub async fn bm25_search(
    db: &D1Database,
    query: &str,
    limit: usize,
) -> Result<Vec<(PedagogyChunk, usize)>, worker::Error> {
    // FTS5 match query with weighted BM25 ranking
    // Column order: text, source_title, source_author, composers, pieces, techniques
    // Weights:      1.0,  2.0,          1.5,           1.5,       1.5,    1.2
    let sql = r#"
        SELECT pc.*, bm25(pedagogy_chunks_fts, 1.0, 2.0, 1.5, 1.5, 1.5, 1.2) as rank
        FROM pedagogy_chunks_fts
        JOIN pedagogy_chunks pc ON pedagogy_chunks_fts.rowid = pc.rowid
        WHERE pedagogy_chunks_fts MATCH ?1
        ORDER BY rank
        LIMIT ?2
    "#;

    let statement = db.prepare(sql);
    let query_result = statement
        .bind(&[query.into(), (limit as i32).into()])?
        .all()
        .await?;

    let results = query_result.results::<serde_json::Value>()?;
    let mut chunks = Vec::new();

    for (rank, row_value) in results.iter().enumerate() {
        if let Ok(chunk) = parse_chunk_from_json(row_value) {
            chunks.push((chunk, rank + 1)); // 1-indexed rank
        }
    }

    Ok(chunks)
}

/// Parse PedagogyChunk from JSON value
fn parse_chunk_from_json(value: &serde_json::Value) -> Result<PedagogyChunk, &'static str> {
    let chunk_id = value
        .get("chunk_id")
        .and_then(|v| v.as_str())
        .ok_or("missing chunk_id")?
        .to_string();
    let text = value
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or("missing text")?
        .to_string();
    let text_with_context = value
        .get("text_with_context")
        .and_then(|v| v.as_str())
        .ok_or("missing text_with_context")?
        .to_string();
    let source_type_str = value
        .get("source_type")
        .and_then(|v| v.as_str())
        .ok_or("missing source_type")?;
    let source_title = value
        .get("source_title")
        .and_then(|v| v.as_str())
        .ok_or("missing source_title")?
        .to_string();
    let source_author = value
        .get("source_author")
        .and_then(|v| v.as_str())
        .ok_or("missing source_author")?
        .to_string();
    let source_url = value.get("source_url").and_then(|v| v.as_str()).map(String::from);
    let page_number = value.get("page_number").and_then(|v| v.as_i64()).map(|v| v as i32);
    let section_title = value.get("section_title").and_then(|v| v.as_str()).map(String::from);
    let paragraph_index = value.get("paragraph_index").and_then(|v| v.as_i64()).map(|v| v as i32);
    let char_start = value.get("char_start").and_then(|v| v.as_i64()).map(|v| v as i32);
    let char_end = value.get("char_end").and_then(|v| v.as_i64()).map(|v| v as i32);
    let timestamp_start = value.get("timestamp_start").and_then(|v| v.as_f64());
    let timestamp_end = value.get("timestamp_end").and_then(|v| v.as_f64());
    let speaker = value.get("speaker").and_then(|v| v.as_str()).map(String::from);
    let ingested_at = value
        .get("ingested_at")
        .and_then(|v| v.as_str())
        .ok_or("missing ingested_at")?
        .to_string();
    let source_hash = value
        .get("source_hash")
        .and_then(|v| v.as_str())
        .ok_or("missing source_hash")?
        .to_string();

    // Parse JSON arrays
    let composers: Vec<String> = value
        .get("composers")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let pieces: Vec<String> = value
        .get("pieces")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let techniques: Vec<String> = value
        .get("techniques")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    let source_type = SourceType::from_str(source_type_str).unwrap_or(SourceType::Book);

    Ok(PedagogyChunk {
        chunk_id,
        text,
        text_with_context,
        source_type,
        source_title,
        source_author,
        source_url,
        page_number,
        section_title,
        paragraph_index,
        char_start,
        char_end,
        timestamp_start,
        timestamp_end,
        speaker,
        composers,
        pieces,
        techniques,
        ingested_at,
        source_hash,
    })
}

/// Vector search using Cloudflare Vectorize via JS interop
///
/// Queries the Vectorize index with the provided embedding and returns
/// matching chunk IDs with their ranks.
#[cfg(feature = "ssr")]
pub async fn vector_search(
    env: &Env,
    query_embedding: &[f32],
    limit: usize,
    filter: Option<VectorMetadata>,
) -> Result<Vec<(String, usize, f32)>, worker::Error> {
    // Get Vectorize index binding
    let index = get_vectorize_index(env, "VECTORIZE")
        .map_err(|e| worker::Error::from(format!("Failed to get Vectorize binding: {:?}", e)))?;

    // Query Vectorize
    let matches = query_vectors(&index, query_embedding, limit, filter)
        .await
        .map_err(|e| worker::Error::from(format!("Vectorize query failed: {}", e)))?;

    // Convert to (chunk_id, rank, score) tuples
    let results: Vec<(String, usize, f32)> = matches
        .into_iter()
        .enumerate()
        .map(|(rank, m)| (m.id, rank + 1, m.score)) // 1-indexed rank
        .collect();

    Ok(results)
}

/// Fallback vector search when SSR feature is not enabled
#[cfg(not(feature = "ssr"))]
pub async fn vector_search(
    _env: &Env,
    _query_embedding: &[f32],
    _limit: usize,
    _filter: Option<()>,
) -> Result<Vec<(String, usize, f32)>, worker::Error> {
    Ok(Vec::new())
}

/// Compute RRF score for a document appearing in multiple rankings
fn compute_rrf_score(ranks: &[Option<usize>]) -> f64 {
    ranks
        .iter()
        .filter_map(|r| r.map(|rank| 1.0 / (RRF_K + rank as f64)))
        .sum()
}

/// Fetch a chunk from D1 by its ID
async fn fetch_chunk_by_id(
    db: &D1Database,
    chunk_id: &str,
) -> Result<Option<PedagogyChunk>, worker::Error> {
    let sql = "SELECT * FROM pedagogy_chunks WHERE chunk_id = ?1";
    let result = db
        .prepare(sql)
        .bind(&[chunk_id.into()])?
        .first::<serde_json::Value>(None)
        .await?;

    match result {
        Some(value) => Ok(parse_chunk_from_json(&value).ok()),
        None => Ok(None),
    }
}

/// Hybrid retrieval combining BM25 and vector search with RRF
///
/// When `env` is provided, performs true hybrid search using Vectorize.
/// Otherwise, falls back to BM25-only search.
///
/// Both BM25 and vector searches run concurrently for improved latency.
#[cfg(feature = "ssr")]
pub async fn hybrid_retrieve(
    env: &Env,
    db: &D1Database,
    query: &str,
    query_embedding: Option<&[f32]>,
    top_k: usize,
) -> Result<Vec<RetrievalResult>, worker::Error> {
    // Run BM25 and vector search concurrently for better latency
    let bm25_future = bm25_search(db, query, 20);

    let (bm25_results, vector_results): (
        Result<Vec<(PedagogyChunk, usize)>, worker::Error>,
        Vec<(String, usize, f32)>,
    ) = if let Some(embedding) = query_embedding {
        // Both searches run in parallel
        let vector_future = vector_search(env, embedding, 20, None);
        let (bm25_res, vector_res) = futures::join!(bm25_future, vector_future);
        (bm25_res, vector_res.unwrap_or_default())
    } else {
        // BM25 only - no vector search needed
        (bm25_future.await, Vec::new())
    };

    let bm25_results = bm25_results?;

    // Build chunk_id -> (bm25_rank, vector_rank, chunk) map
    let mut chunk_ranks: HashMap<String, (Option<usize>, Option<usize>, Option<PedagogyChunk>)> =
        HashMap::new();

    // Add BM25 results
    for (chunk, bm25_rank) in bm25_results {
        chunk_ranks.insert(
            chunk.chunk_id.clone(),
            (Some(bm25_rank), None, Some(chunk)),
        );
    }

    // Add vector results
    for (chunk_id, vector_rank, _score) in vector_results {
        if let Some(entry) = chunk_ranks.get_mut(&chunk_id) {
            // Chunk already in BM25 results, just add vector rank
            entry.1 = Some(vector_rank);
        } else {
            // Chunk only in vector results, need to fetch from D1
            chunk_ranks.insert(chunk_id, (None, Some(vector_rank), None));
        }
    }

    // Fetch missing chunks from D1 (those only in vector results)
    let mut results: Vec<RetrievalResult> = Vec::new();
    for (chunk_id, (bm25_rank, vector_rank, chunk_opt)) in chunk_ranks {
        let chunk = match chunk_opt {
            Some(c) => c,
            None => {
                // Fetch from D1
                match fetch_chunk_by_id(db, &chunk_id).await? {
                    Some(c) => c,
                    None => continue, // Skip if chunk not found
                }
            }
        };

        let rrf_score = compute_rrf_score(&[bm25_rank, vector_rank]);
        results.push(RetrievalResult {
            chunk,
            bm25_rank,
            vector_rank,
            rrf_score,
        });
    }

    // Sort by RRF score and truncate
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap());
    results.truncate(top_k);

    Ok(results)
}

/// BM25-only hybrid retrieve for non-SSR contexts
#[cfg(not(feature = "ssr"))]
pub async fn hybrid_retrieve(
    _env: &Env,
    db: &D1Database,
    query: &str,
    _query_embedding: Option<&[f32]>,
    top_k: usize,
) -> Result<Vec<RetrievalResult>, worker::Error> {
    let bm25_results = bm25_search(db, query, 20).await?;

    let mut results: Vec<RetrievalResult> = bm25_results
        .into_iter()
        .map(|(chunk, bm25_rank)| {
            let rrf_score = compute_rrf_score(&[Some(bm25_rank), None]);
            RetrievalResult {
                chunk,
                bm25_rank: Some(bm25_rank),
                vector_rank: None,
                rrf_score,
            }
        })
        .collect();

    results.truncate(top_k);
    Ok(results)
}

/// Retrieve pedagogy chunks for a performance analysis
///
/// Pipeline:
/// 1. Build retrieval query from performance context and weak dimensions
/// 2. Generate query embedding using Workers AI
/// 3. Hybrid retrieve: BM25 + vector search with RRF fusion
/// 4. Cross-encoder reranking using BGE-reranker-base
/// 5. Return top-k reranked results
#[cfg(feature = "ssr")]
pub async fn retrieve_for_analysis(
    env: &Env,
    db: &D1Database,
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> Result<Vec<RetrievalResult>, worker::Error> {
    use super::reranker::rerank_passages;

    let query = build_retrieval_query(performance, dimensions, 0.5);

    // Generate embedding for the query using Workers AI
    let ai = env.ai("AI")?;
    let embedding = match generate_embedding(&ai, &query).await {
        Ok(emb) => Some(emb),
        Err(e) => {
            // Log error but continue with BM25-only search
            worker::console_log!("Failed to generate embedding: {}. Falling back to BM25-only.", e);
            None
        }
    };

    // Perform hybrid retrieval - get more candidates for reranking
    let candidates = hybrid_retrieve(
        env,
        db,
        &query,
        embedding.as_deref(),
        20, // Get 20 candidates for reranking
    )
    .await?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Cross-encoder reranking for precision improvement
    let passages: Vec<String> = candidates
        .iter()
        .map(|r| r.chunk.text.clone())
        .collect();

    let reranked = match rerank_passages(env, &query, &passages, Some(5)).await {
        Ok(ranked) => {
            worker::console_log!("Reranked {} candidates, returning top {}", candidates.len(), ranked.len());
            ranked
        }
        Err(e) => {
            // Log error but return original order
            worker::console_log!("Reranking failed: {}. Using RRF order.", e);
            return Ok(candidates.into_iter().take(5).collect());
        }
    };

    // Map reranked indices back to results
    let mut results: Vec<RetrievalResult> = reranked
        .into_iter()
        .filter_map(|item| {
            candidates.get(item.id).cloned().map(|mut r| {
                // Store reranker score for debugging/analysis
                r.rrf_score = item.score as f64;
                r
            })
        })
        .collect();

    results.truncate(5);
    Ok(results)
}

/// BM25-only retrieve_for_analysis for non-SSR contexts
#[cfg(not(feature = "ssr"))]
pub async fn retrieve_for_analysis(
    env: &Env,
    db: &D1Database,
    performance: &Performance,
    dimensions: &PerformanceDimensions,
) -> Result<Vec<RetrievalResult>, worker::Error> {
    let query = build_retrieval_query(performance, dimensions, 0.5);
    hybrid_retrieve(env, db, &query, None, 5).await
}

// ============================================================================
// Ingestion Functions
// ============================================================================

#[cfg(feature = "ssr")]
use super::vectorize_binding::{upsert_vectors, VectorRecord};
#[cfg(feature = "ssr")]
use super::embedding::generate_embeddings;

/// Ingest a pedagogy chunk into both D1 (for BM25) and Vectorize (for semantic search)
///
/// This function:
/// 1. Inserts the chunk into D1 (pedagogy_chunks table)
/// 2. Generates an embedding for the text_with_context
/// 3. Upserts the vector into Vectorize with metadata
#[cfg(feature = "ssr")]
pub async fn ingest_chunk(
    env: &Env,
    db: &D1Database,
    chunk: &PedagogyChunk,
    metadata: VectorMetadata,
) -> Result<(), worker::Error> {
    // 1. Insert into D1
    let composers_json = serde_json::to_string(&chunk.composers).unwrap_or_else(|_| "[]".to_string());
    let pieces_json = serde_json::to_string(&chunk.pieces).unwrap_or_else(|_| "[]".to_string());
    let techniques_json = serde_json::to_string(&chunk.techniques).unwrap_or_else(|_| "[]".to_string());

    let sql = r#"
        INSERT OR REPLACE INTO pedagogy_chunks (
            chunk_id, text, text_with_context, source_type, source_title, source_author,
            source_url, page_number, section_title, paragraph_index, char_start, char_end,
            timestamp_start, timestamp_end, speaker, composers, pieces, techniques, source_hash
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
    "#;

    db.prepare(sql)
        .bind(&[
            chunk.chunk_id.clone().into(),
            chunk.text.clone().into(),
            chunk.text_with_context.clone().into(),
            chunk.source_type.as_str().into(),
            chunk.source_title.clone().into(),
            chunk.source_author.clone().into(),
            chunk.source_url.clone().unwrap_or_default().into(),
            chunk.page_number.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.section_title.clone().unwrap_or_default().into(),
            chunk.paragraph_index.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.char_start.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.char_end.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.timestamp_start.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.timestamp_end.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
            chunk.speaker.clone().unwrap_or_default().into(),
            composers_json.into(),
            pieces_json.into(),
            techniques_json.into(),
            chunk.source_hash.clone().into(),
        ])?
        .run()
        .await?;

    // 2. Generate embedding for the text_with_context (includes context header)
    let ai = env.ai("AI")?;
    let embedding = generate_embedding(&ai, &chunk.text_with_context)
        .await
        .map_err(|e| worker::Error::from(format!("Failed to generate embedding: {}", e)))?;

    // 3. Upsert vector into Vectorize
    let index = get_vectorize_index(env, "VECTORIZE")
        .map_err(|e| worker::Error::from(format!("Failed to get Vectorize binding: {:?}", e)))?;

    let vector_record = VectorRecord {
        id: chunk.chunk_id.clone(),
        values: embedding,
        metadata: Some(metadata),
    };

    upsert_vectors(&index, vec![vector_record])
        .await
        .map_err(|e| worker::Error::from(format!("Failed to upsert vector: {}", e)))?;

    Ok(())
}

/// Batch ingest multiple chunks
///
/// More efficient than calling ingest_chunk multiple times as it batches
/// the embedding generation and vector upserts.
#[cfg(feature = "ssr")]
pub async fn ingest_chunks_batch(
    env: &Env,
    db: &D1Database,
    chunks: &[PedagogyChunk],
    metadata: Vec<VectorMetadata>,
) -> Result<usize, worker::Error> {
    if chunks.is_empty() {
        return Ok(0);
    }

    // 1. Insert all chunks into D1
    for chunk in chunks {
        let composers_json = serde_json::to_string(&chunk.composers).unwrap_or_else(|_| "[]".to_string());
        let pieces_json = serde_json::to_string(&chunk.pieces).unwrap_or_else(|_| "[]".to_string());
        let techniques_json = serde_json::to_string(&chunk.techniques).unwrap_or_else(|_| "[]".to_string());

        let sql = r#"
            INSERT OR REPLACE INTO pedagogy_chunks (
                chunk_id, text, text_with_context, source_type, source_title, source_author,
                source_url, page_number, section_title, paragraph_index, char_start, char_end,
                timestamp_start, timestamp_end, speaker, composers, pieces, techniques, source_hash
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
        "#;

        db.prepare(sql)
            .bind(&[
                chunk.chunk_id.clone().into(),
                chunk.text.clone().into(),
                chunk.text_with_context.clone().into(),
                chunk.source_type.as_str().into(),
                chunk.source_title.clone().into(),
                chunk.source_author.clone().into(),
                chunk.source_url.clone().unwrap_or_default().into(),
                chunk.page_number.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.section_title.clone().unwrap_or_default().into(),
                chunk.paragraph_index.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.char_start.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.char_end.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.timestamp_start.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.timestamp_end.map(|n| wasm_bindgen::JsValue::from(n)).unwrap_or(wasm_bindgen::JsValue::NULL),
                chunk.speaker.clone().unwrap_or_default().into(),
                composers_json.into(),
                pieces_json.into(),
                techniques_json.into(),
                chunk.source_hash.clone().into(),
            ])?
            .run()
            .await?;
    }

    // 2. Generate embeddings in batch
    let ai = env.ai("AI")?;
    let texts: Vec<String> = chunks.iter().map(|c| c.text_with_context.clone()).collect();
    let embeddings = generate_embeddings(&ai, &texts)
        .await
        .map_err(|e| worker::Error::from(format!("Failed to generate embeddings: {}", e)))?;

    // 3. Upsert vectors in batch
    let index = get_vectorize_index(env, "VECTORIZE")
        .map_err(|e| worker::Error::from(format!("Failed to get Vectorize binding: {:?}", e)))?;

    let vector_records: Vec<VectorRecord> = chunks
        .iter()
        .zip(embeddings.into_iter())
        .zip(metadata.into_iter())
        .map(|((chunk, embedding), meta)| VectorRecord {
            id: chunk.chunk_id.clone(),
            values: embedding,
            metadata: Some(meta),
        })
        .collect();

    let count = upsert_vectors(&index, vector_records)
        .await
        .map_err(|e| worker::Error::from(format!("Failed to upsert vectors: {}", e)))?;

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_retrieval_query() {
        let performance = Performance {
            id: "test".to_string(),
            composer: "Chopin".to_string(),
            piece_title: "Nocturne Op. 9 No. 2".to_string(),
            performer: "Test".to_string(),
            audio_url: "".to_string(),
            duration_seconds: 0.0,
            year_recorded: None,
        };

        let dimensions = PerformanceDimensions {
            timing: 0.8,
            articulation_length: 0.3, // weak
            articulation_touch: 0.4,  // weak
            pedal_amount: 0.7,
            pedal_clarity: 0.6,
            timbre_variety: 0.7,
            timbre_depth: 0.6,
            timbre_brightness: 0.7,
            timbre_loudness: 0.6,
            dynamics_range: 0.7,
            tempo: 0.8,
            space: 0.7,
            balance: 0.6,
            drama: 0.4, // weak
            mood_valence: 0.6,
            mood_energy: 0.7,
            mood_imagination: 0.5,
            interpretation_sophistication: 0.6,
            interpretation_overall: 0.6,
        };

        let query = build_retrieval_query(&performance, &dimensions, 0.5);

        assert!(query.contains("Chopin"));
        assert!(query.contains("Nocturne"));
        assert!(query.contains("legato") || query.contains("articulation"));
    }

    #[test]
    fn test_rrf_score() {
        // Document appearing at rank 1 in both searches
        let score1 = compute_rrf_score(&[Some(1), Some(1)]);
        // Document appearing at rank 10 in one search only
        let score2 = compute_rrf_score(&[Some(10), None]);

        assert!(score1 > score2);
    }
}
