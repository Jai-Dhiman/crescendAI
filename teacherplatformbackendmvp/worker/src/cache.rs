use worker::*;
use crate::utils;

/// Check cache and return if exists, otherwise call the provided function and cache the result
pub async fn with_cache<F, T>(
    kv: &kv::KvStore,
    key: &str,
    ttl_seconds: u64,
    f: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    // Try to get from cache
    if let Some(cached) = kv.get(key).text().await? {
        if let Ok(value) = serde_json::from_str(&cached) {
            console_log!("Cache hit: {}", key);
            return Ok(value);
        }
    }

    console_log!("Cache miss: {}", key);

    // Cache miss - call the function
    let result = f.await?;

    // Store in cache
    let serialized = serde_json::to_string(&result)
        .map_err(|e| Error::RustError(format!("Serialization error: {}", e)))?;

    kv.put(key, serialized)?
        .expiration_ttl(ttl_seconds)
        .execute()
        .await?;

    Ok(result)
}

/// Get from cache without fallback
pub async fn get<T>(kv: &kv::KvStore, key: &str) -> Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    if let Some(cached) = kv.get(key).text().await? {
        if let Ok(value) = serde_json::from_str(&cached) {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Put into cache
pub async fn put<T>(
    kv: &kv::KvStore,
    key: &str,
    value: &T,
    ttl_seconds: u64,
) -> Result<()>
where
    T: serde::Serialize,
{
    let serialized = serde_json::to_string(value)
        .map_err(|e| Error::RustError(format!("Serialization error: {}", e)))?;

    kv.put(key, serialized)?
        .expiration_ttl(ttl_seconds)
        .execute()
        .await?;

    Ok(())
}
