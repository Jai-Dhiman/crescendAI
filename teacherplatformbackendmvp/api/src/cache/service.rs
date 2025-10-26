use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cache::{CacheKey, KVClient};
use crate::config::CacheConfig;
use crate::models::SearchResult;

/// Caching service with 3-layer strategy
#[derive(Clone)]
pub struct CacheService {
    kv_client: Option<KVClient>,
    config: CacheConfig,
}

impl CacheService {
    pub fn new(kv_client: Option<KVClient>, config: CacheConfig) -> Self {
        Self { kv_client, config }
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.kv_client.is_some()
    }

    /// Get cached embedding for a query
    pub async fn get_embedding(&self, query: &str) -> Result<Option<Vec<f32>>> {
        let Some(kv) = &self.kv_client else {
            return Ok(None);
        };

        let key = CacheKey::embedding(query);
        let namespace = kv.embedding_namespace();

        match kv.get(namespace, &key).await {
            Ok(Some(bytes)) => {
                let embedding: Vec<f32> = bincode::deserialize(&bytes)?;
                tracing::debug!("Cache HIT for embedding: {}", query);
                Ok(Some(embedding))
            }
            Ok(None) => {
                tracing::debug!("Cache MISS for embedding: {}", query);
                Ok(None)
            }
            Err(e) => {
                tracing::warn!("Failed to get embedding from cache: {}", e);
                Ok(None)
            }
        }
    }

    /// Cache an embedding with TTL
    pub async fn put_embedding(&self, query: &str, embedding: Vec<f32>) -> Result<()> {
        let Some(kv) = &self.kv_client else {
            return Ok(());
        };

        let key = CacheKey::embedding(query);
        let namespace = kv.embedding_namespace();
        let ttl_seconds = self.config.embedding_cache_ttl_hours * 3600;

        let bytes = bincode::serialize(&embedding)?;
        kv.put(namespace, &key, bytes, Some(ttl_seconds)).await?;

        tracing::debug!("Cached embedding for query: {}", query);
        Ok(())
    }

    /// Get cached search results
    pub async fn get_search_results(
        &self,
        query: &str,
        filters: &str,
    ) -> Result<Option<Vec<SearchResult>>> {
        let Some(kv) = &self.kv_client else {
            return Ok(None);
        };

        let key = CacheKey::search(query, filters);
        let namespace = kv.search_namespace();

        match kv.get(namespace, &key).await {
            Ok(Some(bytes)) => {
                let results: Vec<SearchResult> = bincode::deserialize(&bytes)?;
                tracing::debug!("Cache HIT for search results: {}", query);
                Ok(Some(results))
            }
            Ok(None) => {
                tracing::debug!("Cache MISS for search results: {}", query);
                Ok(None)
            }
            Err(e) => {
                tracing::warn!("Failed to get search results from cache: {}", e);
                Ok(None)
            }
        }
    }

    /// Cache search results with TTL
    pub async fn put_search_results(
        &self,
        query: &str,
        filters: &str,
        results: &[SearchResult],
    ) -> Result<()> {
        let Some(kv) = &self.kv_client else {
            return Ok(());
        };

        let key = CacheKey::search(query, filters);
        let namespace = kv.search_namespace();
        let ttl_seconds = self.config.search_cache_ttl_hours * 3600;

        let bytes = bincode::serialize(&results)?;
        kv.put(namespace, &key, bytes, Some(ttl_seconds)).await?;

        tracing::debug!("Cached search results for query: {}", query);
        Ok(())
    }

    /// Get cached LLM response
    pub async fn get_llm_response(&self, query: &str, context: &str) -> Result<Option<String>> {
        let Some(kv) = &self.kv_client else {
            return Ok(None);
        };

        let key = CacheKey::llm(query, context);
        let namespace = kv.llm_namespace();

        match kv.get(namespace, &key).await {
            Ok(Some(bytes)) => {
                let response = String::from_utf8(bytes)?;
                tracing::debug!("Cache HIT for LLM response: {}", query);
                Ok(Some(response))
            }
            Ok(None) => {
                tracing::debug!("Cache MISS for LLM response: {}", query);
                Ok(None)
            }
            Err(e) => {
                tracing::warn!("Failed to get LLM response from cache: {}", e);
                Ok(None)
            }
        }
    }

    /// Cache LLM response with TTL
    pub async fn put_llm_response(
        &self,
        query: &str,
        context: &str,
        response: &str,
    ) -> Result<()> {
        let Some(kv) = &self.kv_client else {
            return Ok(());
        };

        let key = CacheKey::llm(query, context);
        let namespace = kv.llm_namespace();
        let ttl_seconds = self.config.llm_cache_ttl_hours * 3600;

        let bytes = response.as_bytes().to_vec();
        kv.put(namespace, &key, bytes, Some(ttl_seconds)).await?;

        tracing::debug!("Cached LLM response for query: {}", query);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_service_disabled() {
        let config = CacheConfig {
            embedding_cache_ttl_hours: 24,
            search_cache_ttl_hours: 1,
            llm_cache_ttl_hours: 24,
        };
        let service = CacheService::new(None, config);
        assert!(!service.is_enabled());
    }
}
