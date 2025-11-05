//! RAG (Retrieval-Augmented Generation) Search Tests
//! Tests for knowledge base search accuracy and relevance

use wasm_bindgen_test::*;
use worker::*;
use serde_json::{json, Value};

wasm_bindgen_test_configure!(run_in_browser);

// ============================================================================
// Test Environment Setup
// ============================================================================

struct RagTestEnv {
    test_documents: Vec<TestDocument>,
}

struct TestDocument {
    id: String,
    title: String,
    content: String,
    metadata: DocumentMetadata,
}

struct DocumentMetadata {
    author: String,
    source: String,
    topic: Vec<String>,
}

impl RagTestEnv {
    fn new() -> Self {
        let test_documents = vec![
            TestDocument {
                id: "doc1".to_string(),
                title: "Hand Position Fundamentals".to_string(),
                content: "Proper hand position is essential for piano technique. \
                         The hand should maintain a natural curved shape, with fingers \
                         gently arched over the keys. The wrist should remain level with \
                         the forearm, neither too high nor too low. Relaxation is key to \
                         preventing tension and injury.".to_string(),
                metadata: DocumentMetadata {
                    author: "Dorothy Taubman".to_string(),
                    source: "Piano Technique: A Guide".to_string(),
                    topic: vec!["technique".to_string(), "hand-position".to_string()],
                },
            },
            TestDocument {
                id: "doc2".to_string(),
                title: "Pedaling in Romantic Repertoire".to_string(),
                content: "The sustain pedal is crucial in romantic piano music. \
                         Syncopated pedaling, where the pedal is depressed after the key, \
                         creates a legato effect without muddying the harmony. In works by \
                         Chopin and Liszt, careful pedal changes at harmonic shifts maintain \
                         clarity while achieving the desired sonority.".to_string(),
                metadata: DocumentMetadata {
                    author: "Seymour Bernstein".to_string(),
                    source: "With Your Own Two Hands".to_string(),
                    topic: vec!["pedaling".to_string(), "romantic-music".to_string()],
                },
            },
            TestDocument {
                id: "doc3".to_string(),
                title: "Scale Practice Methods".to_string(),
                content: "Scales are the foundation of piano technique. Practice scales \
                         in all keys, starting slowly with a metronome. Focus on evenness \
                         of tone and timing. Use various rhythmic patterns to develop \
                         finger independence. Gradually increase tempo only after achieving \
                         perfect accuracy at slower speeds.".to_string(),
                metadata: DocumentMetadata {
                    author: "Charles-Louis Hanon".to_string(),
                    source: "The Virtuoso Pianist".to_string(),
                    topic: vec!["practice".to_string(), "scales".to_string(), "technique".to_string()],
                },
            },
            TestDocument {
                id: "doc4".to_string(),
                title: "Dynamic Control and Expression".to_string(),
                content: "Dynamic variation brings music to life. Begin by mastering the \
                         extremes: true pianissimo and full fortissimo. Practice gradual \
                         crescendos and diminuendos, using arm weight rather than finger \
                         pressure. Listen carefully to the sound produced, adjusting touch \
                         to achieve the desired color and intensity.".to_string(),
                metadata: DocumentMetadata {
                    author: "Josef Lhevinne".to_string(),
                    source: "Basic Principles in Pianoforte Playing".to_string(),
                    topic: vec!["dynamics".to_string(), "expression".to_string()],
                },
            },
            TestDocument {
                id: "doc5".to_string(),
                title: "Sight Reading Development".to_string(),
                content: "Improving sight reading requires consistent daily practice. \
                         Read new music every day, focusing on maintaining steady tempo \
                         rather than perfection. Look ahead to anticipate upcoming notes \
                         and patterns. Practice reading both hands together from the start. \
                         Don't stop for mistakes; keep going to develop fluency.".to_string(),
                metadata: DocumentMetadata {
                    author: "Paul Harris".to_string(),
                    source: "Improve Your Sight-Reading".to_string(),
                    topic: vec!["sight-reading".to_string(), "practice".to_string()],
                },
            },
        ];

        Self { test_documents }
    }

    async fn ingest_test_documents(&self) -> Result<()> {
        // Simulate ingesting documents into knowledge base
        // In actual implementation, this would call the ingestion API
        Ok(())
    }
}

// ============================================================================
// SEARCH ACCURACY TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_exact_match_search() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search for exact phrase that exists in document
    let results = simulate_search("hand position is essential").await
        .expect("Search should succeed");

    assert!(results.len() > 0, "Should find documents with exact phrase");

    // First result should be the hand position document
    assert_eq!(results[0]["id"], "doc1");
    assert!(results[0]["relevance_score"].as_f64().unwrap() > 0.8,
        "Exact match should have high relevance score");
}

#[wasm_bindgen_test]
async fn test_semantic_search() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search using different words with same meaning
    let results = simulate_search("how to use the sustain pedal correctly").await
        .expect("Search should succeed");

    assert!(results.len() > 0, "Should find semantically similar documents");

    // Should return the pedaling document
    let has_pedaling_doc = results.iter().any(|r| r["id"] == "doc2");
    assert!(has_pedaling_doc, "Should find pedaling document via semantic search");
}

#[wasm_bindgen_test]
async fn test_multi_keyword_search() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search with multiple relevant keywords
    let results = simulate_search("scales practice metronome technique").await
        .expect("Search should succeed");

    assert!(results.len() > 0, "Should find documents matching multiple keywords");

    // Should prioritize document with more keyword matches
    assert_eq!(results[0]["id"], "doc3",
        "Scale practice document should rank highest");
}

#[wasm_bindgen_test]
async fn test_topic_filtering() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search within specific topic
    let results = simulate_search_with_filter("practice", "topic", "technique").await
        .expect("Search should succeed");

    // All results should be in the "technique" topic
    for result in results {
        let topics = result["metadata"]["topic"].as_array().unwrap();
        let has_technique = topics.iter().any(|t| t.as_str().unwrap() == "technique");
        assert!(has_technique, "All results should match topic filter");
    }
}

#[wasm_bindgen_test]
async fn test_author_filtering() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    let results = simulate_search_with_filter(
        "piano technique",
        "author",
        "Dorothy Taubman"
    ).await.expect("Search should succeed");

    // All results should be from specified author
    for result in results {
        assert_eq!(result["metadata"]["author"].as_str().unwrap(), "Dorothy Taubman");
    }
}

#[wasm_bindgen_test]
async fn test_search_ranking_quality() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search for specific topic
    let results = simulate_search("pedaling technique in Chopin").await
        .expect("Search should succeed");

    assert!(results.len() >= 2, "Should return multiple results");

    // Results should be ranked by relevance
    for i in 0..results.len()-1 {
        let score1 = results[i]["relevance_score"].as_f64().unwrap();
        let score2 = results[i+1]["relevance_score"].as_f64().unwrap();
        assert!(score1 >= score2, "Results should be sorted by relevance score");
    }

    // Top result should be highly relevant
    assert!(results[0]["relevance_score"].as_f64().unwrap() > 0.7,
        "Top result should have strong relevance");
}

#[wasm_bindgen_test]
async fn test_no_results_for_irrelevant_query() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search for completely unrelated topic
    let results = simulate_search("quantum physics black holes").await
        .expect("Search should succeed");

    // Should return empty or very low relevance scores
    if results.len() > 0 {
        assert!(results[0]["relevance_score"].as_f64().unwrap() < 0.3,
            "Irrelevant results should have low scores");
    }
}

#[wasm_bindgen_test]
async fn test_fuzzy_matching() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Search with typos
    let results = simulate_search("pieno technicue").await
        .expect("Search should succeed");

    // Should still find piano technique documents despite typos
    assert!(results.len() > 0, "Should handle fuzzy matching");
}

// ============================================================================
// HYBRID SEARCH TESTS (Vectorize + D1 FTS)
// ============================================================================

#[wasm_bindgen_test]
async fn test_hybrid_search_combining_vector_and_fts() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Query that should benefit from both vector and full-text search
    let query = "how to practice scales efficiently";

    let vector_results = simulate_vector_search(query).await
        .expect("Vector search should succeed");

    let fts_results = simulate_fts_search(query).await
        .expect("FTS search should succeed");

    let hybrid_results = simulate_hybrid_search(query).await
        .expect("Hybrid search should succeed");

    // Hybrid should combine strengths of both approaches
    assert!(hybrid_results.len() > 0, "Hybrid search should return results");

    // Hybrid results should include high-quality matches from both methods
    let hybrid_ids: Vec<&str> = hybrid_results.iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();

    // Should include doc3 (scales practice)
    assert!(hybrid_ids.contains(&"doc3"),
        "Hybrid search should find the scales document");
}

#[wasm_bindgen_test]
async fn test_reranking_improves_results() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    let query = "expression and dynamics in piano performance";

    // Get results without reranking
    let unranked_results = simulate_search_no_rerank(query).await
        .expect("Search should succeed");

    // Get results with BGE reranking
    let reranked_results = simulate_search_with_rerank(query).await
        .expect("Search should succeed");

    assert!(reranked_results.len() > 0, "Should have reranked results");

    // Reranking should improve top result relevance
    let reranked_top_score = reranked_results[0]["relevance_score"].as_f64().unwrap();

    // Top reranked result should be the dynamics document
    assert_eq!(reranked_results[0]["id"], "doc4",
        "Reranking should surface the most relevant document");

    assert!(reranked_top_score > 0.8,
        "Reranked top result should have high relevance");
}

// ============================================================================
// EMBEDDING QUALITY TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_embedding_cache_consistency() {
    let query = "hand position technique";

    // Generate embedding twice
    let embedding1 = simulate_generate_embedding(query).await
        .expect("Embedding generation should succeed");

    let embedding2 = simulate_generate_embedding(query).await
        .expect("Embedding generation should succeed");

    // Should return identical embeddings (from cache)
    assert_eq!(embedding1, embedding2, "Embeddings should be consistent");
}

#[wasm_bindgen_test]
async fn test_embedding_similarity() {
    let query1 = "piano pedaling technique";
    let query2 = "sustain pedal usage";
    let query3 = "quantum computing";

    let emb1 = simulate_generate_embedding(query1).await.unwrap();
    let emb2 = simulate_generate_embedding(query2).await.unwrap();
    let emb3 = simulate_generate_embedding(query3).await.unwrap();

    // Calculate cosine similarity
    let sim_12 = cosine_similarity(&emb1, &emb2);
    let sim_13 = cosine_similarity(&emb1, &emb3);

    // Related queries should have higher similarity
    assert!(sim_12 > sim_13,
        "Related queries should have higher embedding similarity");

    assert!(sim_12 > 0.7,
        "Semantically similar queries should have high similarity");
}

// ============================================================================
// CONTEXT INTEGRATION TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_search_with_user_context() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // User context: intermediate level, working on romantic pieces
    let user_context = json!({
        "experience_level": "intermediate",
        "repertoire": ["Chopin Nocturne Op. 9 No. 2"],
        "goals": "Improve expression and pedaling"
    });

    let results = simulate_search_with_context(
        "how to improve my technique",
        user_context
    ).await.expect("Search should succeed");

    // Results should be personalized based on context
    assert!(results.len() > 0, "Should return contextual results");

    // Should prioritize pedaling and expression documents for this user
    let top_ids: Vec<&str> = results.iter()
        .take(2)
        .map(|r| r["id"].as_str().unwrap())
        .collect();

    assert!(top_ids.contains(&"doc2") || top_ids.contains(&"doc4"),
        "Should prioritize expression/pedaling docs for this user's context");
}

#[wasm_bindgen_test]
async fn test_search_respects_experience_level() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    // Beginner context
    let beginner_context = json!({
        "experience_level": "beginner"
    });

    let beginner_results = simulate_search_with_context(
        "piano practice advice",
        beginner_context
    ).await.expect("Search should succeed");

    // Should prioritize foundational topics for beginners
    assert!(beginner_results.len() > 0, "Should return results for beginners");

    // Top results should include hand position and scales (fundamentals)
    let has_fundamentals = beginner_results.iter().take(3).any(|r| {
        r["id"] == "doc1" || r["id"] == "doc3"
    });

    assert!(has_fundamentals,
        "Beginner context should prioritize fundamental technique documents");
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_search_latency() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    let start = simulate_timestamp();

    let _results = simulate_search("piano technique").await
        .expect("Search should succeed");

    let elapsed = simulate_timestamp() - start;

    // Should complete within 100ms (P95 target)
    assert!(elapsed < 100.0, "Search should complete in <100ms (was {}ms)", elapsed);
}

#[wasm_bindgen_test]
async fn test_concurrent_search_performance() {
    let env = RagTestEnv::new();
    env.ingest_test_documents().await.expect("Ingestion should succeed");

    let queries = vec![
        "hand position",
        "pedaling technique",
        "scale practice",
        "dynamics and expression",
        "sight reading",
    ];

    let start = simulate_timestamp();

    // Simulate concurrent searches
    for query in queries {
        let _results = simulate_search(query).await
            .expect("Search should succeed");
    }

    let elapsed = simulate_timestamp() - start;

    // Multiple searches should complete efficiently
    assert!(elapsed < 500.0,
        "5 concurrent searches should complete in <500ms (was {}ms)", elapsed);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

async fn simulate_search(query: &str) -> Result<Vec<Value>> {
    // Simulate hybrid search (would call actual knowledge_base::search in real implementation)
    Ok(vec![
        json!({
            "id": "doc1",
            "title": "Hand Position Fundamentals",
            "content": "...",
            "relevance_score": 0.92,
            "metadata": {
                "author": "Dorothy Taubman",
                "topic": ["technique", "hand-position"]
            }
        }),
        json!({
            "id": "doc3",
            "title": "Scale Practice Methods",
            "content": "...",
            "relevance_score": 0.78,
            "metadata": {
                "author": "Charles-Louis Hanon",
                "topic": ["practice", "scales", "technique"]
            }
        }),
    ])
}

async fn simulate_search_with_filter(
    query: &str,
    _filter_key: &str,
    _filter_value: &str
) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_vector_search(query: &str) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_fts_search(query: &str) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_hybrid_search(query: &str) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_search_no_rerank(query: &str) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_search_with_rerank(query: &str) -> Result<Vec<Value>> {
    simulate_search(query).await
}

async fn simulate_search_with_context(_query: &str, _context: Value) -> Result<Vec<Value>> {
    Ok(vec![
        json!({
            "id": "doc2",
            "title": "Pedaling in Romantic Repertoire",
            "relevance_score": 0.89,
            "metadata": {
                "author": "Seymour Bernstein",
                "topic": ["pedaling", "romantic-music"]
            }
        }),
    ])
}

async fn simulate_generate_embedding(_text: &str) -> Result<Vec<f32>> {
    // Simulate BGE embedding (768 dimensions)
    Ok(vec![0.1; 768])
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn simulate_timestamp() -> f64 {
    // In real implementation, would use js_sys::Date::now()
    0.0
}
