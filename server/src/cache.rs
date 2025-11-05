//! Unified Caching System
//!
//! Provides a centralized cache manager for embeddings, search results, and LLM responses.
//! Uses Cloudflare KV for persistent caching with configurable TTLs.

use worker::*;
use worker::kv::KvStore;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

// ============================================================================
// Cache Configuration
// ============================================================================

/// Cache TTL configurations (in seconds)
pub struct CacheTTL {
    /// Embeddings cache TTL (24 hours)
    pub embeddings: u64,
    /// Search results cache TTL (1 hour)
    pub search: u64,
    /// Feedback response cache TTL (6 hours)
    pub feedback: u64,
    /// LLM response cache TTL (1 hour)
    pub llm: u64,
}

impl Default for CacheTTL {
    fn default() -> Self {
        Self {
            embeddings: 86400,  // 24 hours
            search: 3600,        // 1 hour
            feedback: 21600,     // 6 hours
            llm: 3600,           // 1 hour
        }
    }
}

// ============================================================================
// Cache Types
// ============================================================================

/// Cache type for namespacing and metrics
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    Embedding,
    Search,
    Feedback,
    LLM,
}

impl CacheType {
    /// Get cache key prefix
    pub fn prefix(&self) -> &'static str {
        match self {
            CacheType::Embedding => "embed:v1:",
            CacheType::Search => "search:v1:",
            CacheType::Feedback => "feedback:v1:",
            CacheType::LLM => "llm:v1:",
        }
    }

    /// Get default TTL for this cache type
    pub fn default_ttl(&self) -> u64 {
        match self {
            CacheType::Embedding => 86400,  // 24 hours
            CacheType::Search => 3600,       // 1 hour
            CacheType::Feedback => 21600,    // 6 hours
            CacheType::LLM => 3600,          // 1 hour
        }
    }
}

// ============================================================================
// Cache Manager
// ============================================================================

/// Unified cache manager for all caching operations
pub struct CacheManager<'a> {
    kv: KvStore,
    ttl_config: CacheTTL,
    _env: &'a Env,
}

impl<'a> CacheManager<'a> {
    /// Create a new cache manager
    pub fn new(env: &'a Env) -> Result<Self> {
        let kv = env.kv("CRESCENDAI_METADATA")
            .map_err(|e| worker::Error::RustError(format!("Failed to get KV binding: {}", e)))?;

        Ok(Self {
            kv,
            ttl_config: CacheTTL::default(),
            _env: env,
        })
    }

    /// Create a cache manager with custom TTL configuration
    pub fn with_ttl(env: &'a Env, ttl_config: CacheTTL) -> Result<Self> {
        let kv = env.kv("CRESCENDAI_METADATA")
            .map_err(|e| worker::Error::RustError(format!("Failed to get KV binding: {}", e)))?;

        Ok(Self {
            kv,
            ttl_config,
            _env: env,
        })
    }

    /// Generate a cache key with type prefix and hash
    pub fn generate_key(&self, cache_type: CacheType, data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let hash = hasher.finalize();
        format!("{}{:x}", cache_type.prefix(), hash)
    }

    /// Generate a cache key with additional parameters
    pub fn generate_key_with_params(&self, cache_type: CacheType, data: &str, params: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.update(params);
        let hash = hasher.finalize();
        format!("{}{:x}", cache_type.prefix(), hash)
    }

    /// Get a value from cache (generic deserialization)
    pub async fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        match self.kv.get(key).text().await {
            Ok(Some(cached)) => {
                match serde_json::from_str::<T>(&cached) {
                    Ok(value) => {
                        console_log!("Cache hit: {}", key);
                        Ok(Some(value))
                    }
                    Err(e) => {
                        console_error!("Cache deserialization error for key {}: {}", key, e);
                        Ok(None)
                    }
                }
            }
            Ok(None) => {
                console_log!("Cache miss: {}", key);
                Ok(None)
            }
            Err(e) => {
                console_error!("Cache read error for key {}: {}", key, e);
                Ok(None)
            }
        }
    }

    /// Set a value in cache (generic serialization) with default TTL
    pub async fn set<T: Serialize>(&self, cache_type: CacheType, key: &str, value: &T) -> Result<()> {
        let ttl = cache_type.default_ttl();
        self.set_with_ttl(key, value, ttl).await
    }

    /// Set a value in cache with custom TTL
    pub async fn set_with_ttl<T: Serialize>(&self, key: &str, value: &T, ttl: u64) -> Result<()> {
        match serde_json::to_string(value) {
            Ok(json) => {
                self.kv.put(key, &json)?
                    .expiration_ttl(ttl)
                    .execute()
                    .await?;
                console_log!("Cache set: {} (TTL: {}s)", key, ttl);
                Ok(())
            }
            Err(e) => {
                console_error!("Cache serialization error: {}", e);
                Err(worker::Error::RustError(format!("Failed to serialize cache value: {}", e)))
            }
        }
    }

    /// Delete a value from cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.kv.delete(key).await?;
        console_log!("Cache delete: {}", key);
        Ok(())
    }

    /// Check if a key exists in cache
    pub async fn exists(&self, key: &str) -> bool {
        matches!(self.kv.get(key).text().await, Ok(Some(_)))
    }

    // ========================================================================
    // Type-specific helpers
    // ========================================================================

    /// Get embedding from cache
    pub async fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let key = self.generate_key(CacheType::Embedding, text);
        self.get(&key).await
    }

    /// Set embedding in cache
    pub async fn set_embedding(&self, text: &str, embedding: &Vec<f32>) -> Result<()> {
        let key = self.generate_key(CacheType::Embedding, text);
        self.set(CacheType::Embedding, &key, embedding).await
    }

    /// Get search results from cache
    pub async fn get_search_results(&self, query: &str, limit: u32) -> Result<Option<Vec<String>>> {
        let params = limit.to_le_bytes();
        let key = self.generate_key_with_params(CacheType::Search, query, &params);
        self.get(&key).await
    }

    /// Set search results in cache
    pub async fn set_search_results(&self, query: &str, limit: u32, results: &Vec<String>) -> Result<()> {
        let params = limit.to_le_bytes();
        let key = self.generate_key_with_params(CacheType::Search, query, &params);
        self.set(CacheType::Search, &key, results).await
    }

    /// Get feedback from cache
    pub async fn get_feedback(&self, recording_id: &str) -> Result<Option<serde_json::Value>> {
        let key = self.generate_key(CacheType::Feedback, recording_id);
        self.get(&key).await
    }

    /// Set feedback in cache
    pub async fn set_feedback(&self, recording_id: &str, feedback: &serde_json::Value) -> Result<()> {
        let key = self.generate_key(CacheType::Feedback, recording_id);
        self.set(CacheType::Feedback, &key, feedback).await
    }

    /// Get LLM response from cache
    pub async fn get_llm_response(&self, prompt_hash: &str) -> Result<Option<String>> {
        let key = self.generate_key(CacheType::LLM, prompt_hash);
        self.get(&key).await
    }

    /// Set LLM response in cache
    pub async fn set_llm_response(&self, prompt_hash: &str, response: &str) -> Result<()> {
        let key = self.generate_key(CacheType::LLM, prompt_hash);
        self.set(CacheType::LLM, &key, &response.to_string()).await
    }
}

// ============================================================================
// Cache Metrics
// ============================================================================

/// Cache metrics for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl CacheMetrics {
    pub fn new(hits: u64, misses: u64) -> Self {
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        Self {
            hits,
            misses,
            hit_rate,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a hash key for arbitrary data
pub fn hash_key(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}
