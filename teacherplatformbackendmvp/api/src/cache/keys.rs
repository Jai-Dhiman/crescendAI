use sha2::{Digest, Sha256};

pub struct CacheKey;

impl CacheKey {
    pub fn embedding(query: &str) -> String {
        let hash = Self::hash(query);
        format!("embed:v1:{}", hash)
    }

    pub fn search(query: &str, filters: &str) -> String {
        let combined = format!("{}{}", query, filters);
        let hash = Self::hash(&combined);
        format!("search:v1:{}", hash)
    }

    pub fn llm(query: &str, context: &str) -> String {
        let combined = format!("{}{}", query, context);
        let hash = Self::hash(&combined);
        format!("llm:v1:{}", hash)
    }

    fn hash(input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cache_key() {
        let key = CacheKey::embedding("How do I improve finger independence?");
        assert!(key.starts_with("embed:v1:"));
        assert_eq!(key.len(), "embed:v1:".len() + 64); // SHA256 = 64 hex chars
    }

    #[test]
    fn test_cache_key_consistency() {
        let query = "test query";
        let key1 = CacheKey::embedding(query);
        let key2 = CacheKey::embedding(query);
        assert_eq!(key1, key2);
    }
}
