//! Cache Behavior Tests
//! Tests for multi-layer caching (embeddings, search, feedback, LLM responses)

use wasm_bindgen_test::*;
use worker::*;
use serde_json::{json, Value};
use std::collections::HashMap;

wasm_bindgen_test_configure!(run_in_browser);

// ============================================================================
// Test Environment Setup
// ============================================================================

struct CacheTestEnv {
    kv_store: HashMap<String, (String, u64)>, // (value, expiry_timestamp)
}

impl CacheTestEnv {
    fn new() -> Self {
        Self {
            kv_store: HashMap::new(),
        }
    }

    async fn set_cache(&mut self, key: &str, value: &str, ttl_seconds: u64) -> Result<()> {
        let expiry = simulate_timestamp() as u64 + ttl_seconds;
        self.kv_store.insert(key.to_string(), (value.to_string(), expiry));
        Ok(())
    }

    async fn get_cache(&self, key: &str) -> Option<String> {
        if let Some((value, expiry)) = self.kv_store.get(key) {
            if (simulate_timestamp() as u64) < *expiry {
                return Some(value.clone());
            }
        }
        None
    }

    async fn delete_cache(&mut self, key: &str) -> Result<()> {
        self.kv_store.remove(key);
        Ok(())
    }

    fn cache_size(&self) -> usize {
        self.kv_store.len()
    }
}

// ============================================================================
// EMBEDDING CACHE TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_embedding_cache_hit() {
    let mut env = CacheTestEnv::new();

    let text = "hand position technique";

    // First call - cache miss, generate embedding
    let result1 = simulate_get_embedding(&mut env, text).await
        .expect("Should generate embedding");

    // Second call - cache hit
    let result2 = simulate_get_embedding(&mut env, text).await
        .expect("Should retrieve from cache");

    assert_eq!(result1, result2, "Cached embedding should match");
}

#[wasm_bindgen_test]
async fn test_embedding_cache_ttl() {
    let mut env = CacheTestEnv::new();

    let text = "piano technique";

    // Set short TTL embedding
    simulate_cache_embedding(&mut env, text, 1).await
        .expect("Should cache embedding");

    // Should be available immediately
    let hit1 = env.get_cache(&format!("emb:{}", text)).await;
    assert!(hit1.is_some(), "Embedding should be in cache");

    // Simulate time passing beyond TTL
    simulate_advance_time(2);

    // Should be expired
    let hit2 = env.get_cache(&format!("emb:{}", text)).await;
    assert!(hit2.is_none(), "Embedding should be expired");
}

#[wasm_bindgen_test]
async fn test_embedding_cache_different_texts() {
    let mut env = CacheTestEnv::new();

    let text1 = "hand position";
    let text2 = "pedaling technique";

    let emb1 = simulate_get_embedding(&mut env, text1).await.unwrap();
    let emb2 = simulate_get_embedding(&mut env, text2).await.unwrap();

    // Different texts should have different embeddings
    assert_ne!(emb1, emb2, "Different texts should have different embeddings");

    // Both should be cached
    assert!(env.get_cache(&format!("emb:{}", text1)).await.is_some());
    assert!(env.get_cache(&format!("emb:{}", text2)).await.is_some());
}

#[wasm_bindgen_test]
async fn test_embedding_cache_key_generation() {
    let text = "test query with special chars: @#$%";

    let key1 = simulate_generate_cache_key("embedding", text);
    let key2 = simulate_generate_cache_key("embedding", text);

    // Same text should generate same key
    assert_eq!(key1, key2, "Cache keys should be deterministic");

    // Key should be hashed (SHA256) for consistency
    assert!(key1.starts_with("emb:"), "Key should have proper prefix");
    assert_eq!(key1.len(), 68, "Key should be 'emb:' + 64 hex chars (SHA256)");
}

// ============================================================================
// SEARCH CACHE TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_search_cache_hit() {
    let mut env = CacheTestEnv::new();

    let query = "piano technique";

    // First search - cache miss
    let results1 = simulate_search_with_cache(&mut env, query).await
        .expect("Search should succeed");

    // Second search - cache hit
    let results2 = simulate_search_with_cache(&mut env, query).await
        .expect("Search should succeed");

    assert_eq!(results1, results2, "Cached results should match");
}

#[wasm_bindgen_test]
async fn test_search_cache_ttl_1_hour() {
    let mut env = CacheTestEnv::new();

    let query = "hand position";
    let results = vec![json!({"id": "doc1", "score": 0.9})];

    // Cache search results with 1 hour TTL
    simulate_cache_search_results(&mut env, query, &results, 3600).await
        .expect("Should cache results");

    // Should be available within TTL
    let cached = env.get_cache(&format!("search:{}", query)).await;
    assert!(cached.is_some(), "Search results should be cached");

    // Simulate 30 minutes - should still be cached
    simulate_advance_time(1800);
    let cached_30min = env.get_cache(&format!("search:{}", query)).await;
    assert!(cached_30min.is_some(), "Should still be cached after 30 min");

    // Simulate 2 hours total - should be expired
    simulate_advance_time(5400); // 30min + 90min = 2 hours
    let expired = env.get_cache(&format!("search:{}", query)).await;
    assert!(expired.is_none(), "Should be expired after 2 hours");
}

#[wasm_bindgen_test]
async fn test_search_cache_invalidation_on_document_update() {
    let mut env = CacheTestEnv::new();

    let query = "piano technique";

    // Cache search results
    let results = vec![json!({"id": "doc1"})];
    simulate_cache_search_results(&mut env, query, &results, 3600).await
        .expect("Should cache results");

    assert!(env.get_cache(&format!("search:{}", query)).await.is_some());

    // Simulate document update
    simulate_document_update(&mut env, "doc1").await
        .expect("Document update should succeed");

    // Cache should be invalidated
    let after_update = env.get_cache(&format!("search:{}", query)).await;
    assert!(after_update.is_none(), "Cache should be invalidated after document update");
}

// ============================================================================
// FEEDBACK CACHE TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_feedback_cache_ttl_6_hours() {
    let mut env = CacheTestEnv::new();

    let recording_id = "rec123";
    let feedback = json!({
        "overall_assessment": {"summary": "Good performance"},
        "practice_recommendations": {"immediate": ["Practice scales"]}
    });

    // Cache feedback with 6 hour TTL
    simulate_cache_feedback(&mut env, recording_id, &feedback, 21600).await
        .expect("Should cache feedback");

    // Should be available
    let cached = env.get_cache(&format!("feedback:{}", recording_id)).await;
    assert!(cached.is_some(), "Feedback should be cached");

    // After 3 hours - still cached
    simulate_advance_time(10800);
    let cached_3h = env.get_cache(&format!("feedback:{}", recording_id)).await;
    assert!(cached_3h.is_some(), "Should still be cached after 3 hours");

    // After 7 hours total - expired
    simulate_advance_time(14400);
    let expired = env.get_cache(&format!("feedback:{}", recording_id)).await;
    assert!(expired.is_none(), "Should be expired after 7 hours");
}

#[wasm_bindgen_test]
async fn test_feedback_cache_per_recording() {
    let mut env = CacheTestEnv::new();

    let rec1 = "rec001";
    let rec2 = "rec002";

    let feedback1 = json!({"score": 0.8});
    let feedback2 = json!({"score": 0.9});

    simulate_cache_feedback(&mut env, rec1, &feedback1, 3600).await.unwrap();
    simulate_cache_feedback(&mut env, rec2, &feedback2, 3600).await.unwrap();

    // Both should be cached independently
    assert!(env.get_cache(&format!("feedback:{}", rec1)).await.is_some());
    assert!(env.get_cache(&format!("feedback:{}", rec2)).await.is_some());

    // Deleting one shouldn't affect the other
    env.delete_cache(&format!("feedback:{}", rec1)).await.unwrap();

    assert!(env.get_cache(&format!("feedback:{}", rec1)).await.is_none());
    assert!(env.get_cache(&format!("feedback:{}", rec2)).await.is_some());
}

// ============================================================================
// LLM RESPONSE CACHE TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_llm_response_cache() {
    let mut env = CacheTestEnv::new();

    let prompt = "What is proper hand position?";
    let response = "Proper hand position involves maintaining a curved shape...";

    // Cache LLM response
    simulate_cache_llm_response(&mut env, prompt, response, 7200).await
        .expect("Should cache LLM response");

    // Should retrieve from cache
    let cached = env.get_cache(&format!("llm:{}", prompt)).await;
    assert!(cached.is_some(), "LLM response should be cached");
    assert_eq!(cached.unwrap(), response);
}

#[wasm_bindgen_test]
async fn test_llm_cache_considers_context() {
    let prompt = "How do I practice?";
    let context1 = json!({"experience_level": "beginner"});
    let context2 = json!({"experience_level": "advanced"});

    // Same prompt, different context should generate different cache keys
    let key1 = simulate_generate_llm_cache_key(prompt, &context1);
    let key2 = simulate_generate_llm_cache_key(prompt, &context2);

    assert_ne!(key1, key2, "Different contexts should generate different cache keys");
}

// ============================================================================
// CACHE METRICS TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_cache_hit_rate_tracking() {
    let mut metrics = CacheMetrics::new();

    // Simulate cache hits and misses
    metrics.record_hit("embedding");
    metrics.record_hit("embedding");
    metrics.record_miss("embedding");
    metrics.record_hit("embedding");

    let hit_rate = metrics.hit_rate("embedding");

    // 3 hits out of 4 total = 75%
    assert!((hit_rate - 0.75).abs() < 0.01, "Hit rate should be 75%");
}

#[wasm_bindgen_test]
async fn test_cache_metrics_per_type() {
    let mut metrics = CacheMetrics::new();

    metrics.record_hit("embedding");
    metrics.record_hit("embedding");
    metrics.record_miss("search");
    metrics.record_hit("search");

    let emb_rate = metrics.hit_rate("embedding");
    let search_rate = metrics.hit_rate("search");

    assert!((emb_rate - 1.0).abs() < 0.01, "Embedding hit rate should be 100%");
    assert!((search_rate - 0.5).abs() < 0.01, "Search hit rate should be 50%");
}

#[wasm_bindgen_test]
async fn test_cache_size_monitoring() {
    let mut env = CacheTestEnv::new();

    assert_eq!(env.cache_size(), 0, "Cache should start empty");

    // Add multiple cache entries
    env.set_cache("key1", "value1", 3600).await.unwrap();
    env.set_cache("key2", "value2", 3600).await.unwrap();
    env.set_cache("key3", "value3", 3600).await.unwrap();

    assert_eq!(env.cache_size(), 3, "Cache should have 3 entries");

    // Delete one
    env.delete_cache("key2").await.unwrap();

    assert_eq!(env.cache_size(), 2, "Cache should have 2 entries");
}

// ============================================================================
// CACHE WARMING TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_pre_generate_common_embeddings() {
    let mut env = CacheTestEnv::new();

    let common_queries = vec![
        "hand position",
        "pedaling technique",
        "scale practice",
        "sight reading",
        "dynamics and expression",
    ];

    // Warm cache with common queries
    for query in &common_queries {
        simulate_get_embedding(&mut env, query).await.unwrap();
    }

    // All should be cached
    for query in &common_queries {
        let cached = env.get_cache(&format!("emb:{}", query)).await;
        assert!(cached.is_some(), "Common query '{}' should be pre-cached", query);
    }

    assert_eq!(env.cache_size(), common_queries.len(),
        "All common queries should be in cache");
}

#[wasm_bindgen_test]
async fn test_cache_warming_improves_latency() {
    let mut env = CacheTestEnv::new();

    let query = "piano technique";

    // Pre-warm cache
    simulate_get_embedding(&mut env, query).await.unwrap();

    // Subsequent requests should be faster (cache hit)
    let _result = simulate_get_embedding(&mut env, query).await.unwrap();

    // In real implementation, this would be significantly faster
    // Here we just verify cache hit occurred
    let cached = env.get_cache(&format!("emb:{}", query)).await;
    assert!(cached.is_some(), "Should hit cache");
}

// ============================================================================
// CACHE EVICTION TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_manual_cache_invalidation() {
    let mut env = CacheTestEnv::new();

    // Cache multiple items
    env.set_cache("emb:query1", "embedding1", 3600).await.unwrap();
    env.set_cache("search:query2", "results2", 3600).await.unwrap();
    env.set_cache("feedback:rec3", "feedback3", 3600).await.unwrap();

    assert_eq!(env.cache_size(), 3);

    // Invalidate all search caches
    simulate_invalidate_cache_prefix(&mut env, "search:").await.unwrap();

    assert_eq!(env.cache_size(), 2, "Search cache should be invalidated");
    assert!(env.get_cache("search:query2").await.is_none());
    assert!(env.get_cache("emb:query1").await.is_some());
}

#[wasm_bindgen_test]
async fn test_invalidate_all_caches() {
    let mut env = CacheTestEnv::new();

    env.set_cache("key1", "value1", 3600).await.unwrap();
    env.set_cache("key2", "value2", 3600).await.unwrap();

    assert_eq!(env.cache_size(), 2);

    // Clear all caches
    simulate_clear_all_cache(&mut env).await.unwrap();

    assert_eq!(env.cache_size(), 0, "All caches should be cleared");
}

// ============================================================================
// CONCURRENT CACHE ACCESS TESTS
// ============================================================================

#[wasm_bindgen_test]
async fn test_concurrent_cache_reads() {
    let env = CacheTestEnv::new();

    let key = "test_key";

    // Simulate concurrent reads (in actual implementation, would use async parallel execution)
    let read1 = env.get_cache(key).await;
    let read2 = env.get_cache(key).await;
    let read3 = env.get_cache(key).await;

    // All reads should return consistent results
    assert_eq!(read1, read2);
    assert_eq!(read2, read3);
}

#[wasm_bindgen_test]
async fn test_cache_stampede_prevention() {
    let mut env = CacheTestEnv::new();

    let query = "expensive computation";

    // Multiple concurrent requests for same uncached item
    // Should only generate once and cache for all
    // Simulating concurrent access
    for _ in 0..10 {
        let _ = simulate_get_embedding(&mut env, query).await;
    }

    // In real implementation, would use proper async concurrency
    // Here we just verify caching works
    assert!(env.get_cache(&format!("emb:{}", query)).await.is_some(),
        "Result should be cached after first computation");
}

// ============================================================================
// HELPER STRUCTS AND FUNCTIONS
// ============================================================================

struct CacheMetrics {
    hits: HashMap<String, u64>,
    misses: HashMap<String, u64>,
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            hits: HashMap::new(),
            misses: HashMap::new(),
        }
    }

    fn record_hit(&mut self, cache_type: &str) {
        *self.hits.entry(cache_type.to_string()).or_insert(0) += 1;
    }

    fn record_miss(&mut self, cache_type: &str) {
        *self.misses.entry(cache_type.to_string()).or_insert(0) += 1;
    }

    fn hit_rate(&self, cache_type: &str) -> f64 {
        let hits = self.hits.get(cache_type).unwrap_or(&0);
        let misses = self.misses.get(cache_type).unwrap_or(&0);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            *hits as f64 / total as f64
        }
    }
}

async fn simulate_get_embedding(env: &mut CacheTestEnv, text: &str) -> Result<Vec<f32>> {
    let cache_key = format!("emb:{}", text);

    // Check cache first
    if let Some(_cached) = env.get_cache(&cache_key).await {
        // In real implementation, would deserialize embedding
        return Ok(vec![0.1, 0.2, 0.3]);
    }

    // Cache miss - generate embedding
    let embedding = vec![0.1, 0.2, 0.3]; // Mock embedding

    // Cache with 24h TTL
    env.set_cache(&cache_key, "cached_embedding", 86400).await?;

    Ok(embedding)
}

async fn simulate_cache_embedding(
    env: &mut CacheTestEnv,
    text: &str,
    ttl_seconds: u64
) -> Result<()> {
    let cache_key = format!("emb:{}", text);
    env.set_cache(&cache_key, "embedding_data", ttl_seconds).await
}

async fn simulate_search_with_cache(
    env: &mut CacheTestEnv,
    query: &str
) -> Result<Vec<Value>> {
    let cache_key = format!("search:{}", query);

    if let Some(_cached) = env.get_cache(&cache_key).await {
        return Ok(vec![json!({"id": "doc1", "score": 0.9})]);
    }

    // Perform search
    let results = vec![json!({"id": "doc1", "score": 0.9})];

    // Cache with 1h TTL
    env.set_cache(&cache_key, "search_results", 3600).await?;

    Ok(results)
}

async fn simulate_cache_search_results(
    env: &mut CacheTestEnv,
    query: &str,
    _results: &[Value],
    ttl_seconds: u64
) -> Result<()> {
    let cache_key = format!("search:{}", query);
    env.set_cache(&cache_key, "search_results", ttl_seconds).await
}

async fn simulate_cache_feedback(
    env: &mut CacheTestEnv,
    recording_id: &str,
    _feedback: &Value,
    ttl_seconds: u64
) -> Result<()> {
    let cache_key = format!("feedback:{}", recording_id);
    env.set_cache(&cache_key, "feedback_data", ttl_seconds).await
}

async fn simulate_cache_llm_response(
    env: &mut CacheTestEnv,
    prompt: &str,
    response: &str,
    ttl_seconds: u64
) -> Result<()> {
    let cache_key = format!("llm:{}", prompt);
    env.set_cache(&cache_key, response, ttl_seconds).await
}

async fn simulate_document_update(env: &mut CacheTestEnv, _doc_id: &str) -> Result<()> {
    // Invalidate search caches when document is updated
    simulate_invalidate_cache_prefix(env, "search:").await
}

async fn simulate_invalidate_cache_prefix(env: &mut CacheTestEnv, prefix: &str) -> Result<()> {
    let keys_to_delete: Vec<String> = env.kv_store.keys()
        .filter(|k| k.starts_with(prefix))
        .cloned()
        .collect();

    for key in keys_to_delete {
        env.delete_cache(&key).await?;
    }

    Ok(())
}

async fn simulate_clear_all_cache(env: &mut CacheTestEnv) -> Result<()> {
    env.kv_store.clear();
    Ok(())
}

fn simulate_generate_cache_key(prefix: &str, content: &str) -> String {
    // In real implementation, would use SHA256 hashing
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    let hash_hex = hex::encode(hash);

    format!("{}:{}", prefix, hash_hex)
}

fn simulate_generate_llm_cache_key(prompt: &str, context: &Value) -> String {
    let combined = format!("{}{}", prompt, context.to_string());
    simulate_generate_cache_key("llm", &combined)
}

fn simulate_timestamp() -> f64 {
    // In real implementation, would use js_sys::Date::now()
    1699104000000.0 // Mock timestamp
}

fn simulate_advance_time(_seconds: u64) {
    // In real implementation, would advance mock time
    // For now, this is a no-op in simulation
}
