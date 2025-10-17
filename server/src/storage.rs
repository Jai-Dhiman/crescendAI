use crate::{AnalysisResult, ComparisonResult, JobStatus, UserPreference};
use serde_json::json;
use worker::*;

/// Cache configuration for different data types
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub ttl_seconds: u64,
    pub use_cache_bust: bool,
    pub cache_warming: bool,
}

impl CacheConfig {
    pub fn for_job_status(status: &str) -> Self {
        match status {
            "completed" | "failed" => Self {
                ttl_seconds: 3600, // 1 hour for final states
                use_cache_bust: false,
                cache_warming: true,
            },
            _ => Self {
                ttl_seconds: 60, // 1 minute for in-progress jobs
                use_cache_bust: true,
                cache_warming: false,
            },
        }
    }

    pub fn for_analysis_results() -> Self {
        Self {
            ttl_seconds: 7200, // 2 hours for analysis results
            use_cache_bust: false,
            cache_warming: true,
        }
    }

    pub fn for_comparison_results() -> Self {
        Self {
            ttl_seconds: 3600, // 1 hour for comparison results
            use_cache_bust: false,
            cache_warming: true,
        }
    }

    pub fn for_file_metadata() -> Self {
        Self {
            ttl_seconds: 1800, // 30 minutes for file metadata
            use_cache_bust: false,
            cache_warming: false,
        }
    }
}

/// Enhanced cache metrics for monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheMetrics {
    pub hit_count: u64,
    pub miss_count: u64,
    pub last_access: f64,
    pub data_size: usize,
}

impl CacheMetrics {
    pub fn new(data_size: usize) -> Self {
        Self {
            hit_count: 0,
            miss_count: 1, // First access is a miss
            last_access: js_sys::Date::now(),
            data_size,
        }
    }

    pub fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_access = js_sys::Date::now();
    }

    pub fn record_miss(&mut self) {
        self.miss_count += 1;
        self.last_access = js_sys::Date::now();
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}

/// Update cache metrics for monitoring and optimization
async fn update_cache_metrics(env: &Env, metrics_key: &str, is_hit: bool) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;

    // Get existing metrics or create new ones
    let mut metrics = match kv.get(metrics_key).text().await? {
        Some(data) => {
            serde_json::from_str::<CacheMetrics>(&data).unwrap_or_else(|_| CacheMetrics::new(0))
        }
        None => CacheMetrics::new(0),
    };

    // Update metrics
    if is_hit {
        metrics.record_hit();
    } else {
        metrics.record_miss();
    }

    // Store updated metrics (with short TTL to avoid bloat)
    let metrics_json = serde_json::to_string(&metrics)?;
    kv.put(metrics_key, &metrics_json)?
        .expiration_ttl(3600) // 1 hour TTL for metrics
        .execute()
        .await?;

    Ok(())
}

/// Warm cache for frequently accessed items
pub async fn warm_cache_for_completed_job(env: &Env, job_id: &str) -> Result<()> {
    console_log!("Warming cache for completed job: {}", job_id);

    // Pre-populate cache with job status and related data
    match get_job_status(env, job_id).await {
        Ok(status) if status.status == "completed" => {
            // Also try to warm analysis result cache if it exists
            if let Ok(_) = get_analysis_result(env, job_id).await {
                console_log!("Cache warmed for job {} and its analysis result", job_id);
            }
        }
        _ => {}
    }

    Ok(())
}

/// Intelligent cache invalidation for related data
pub async fn invalidate_related_cache(
    env: &Env,
    job_id: &str,
    invalidation_type: &str,
) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;

    match invalidation_type {
        "job_completed" => {
            // When a job completes, just log it - the status was already written
            // No need to delete and rewrite, which creates a race condition
            console_log!("Job completion handled for: {}", job_id);
        }
        "analysis_updated" => {
            // When analysis is updated, invalidate result cache
            console_log!("Invalidating analysis result cache: {}", job_id);
            kv.delete(&format!("result:{}", job_id)).await?;
        }
        _ => {}
    }

    Ok(())
}

pub async fn upload_to_r2(env: &Env, file_id: &str, data: Vec<u8>) -> Result<()> {
    let bucket = env.bucket("AUDIO_BUCKET")?;
    let key = format!("audio/{}.wav", file_id);

    let data_size = data.len();
    bucket.put(&key, data).execute().await?;

    // Store metadata in KV
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let metadata = json!({
        "id": file_id,
        "uploaded_at": js_sys::Date::now(),
        "size": data_size,
        "status": "uploaded"
    });

    kv.put(&format!("file:{}", file_id), &metadata.to_string())?
        .execute()
        .await?;

    Ok(())
}

pub async fn get_job_status(env: &Env, job_id: &str) -> Result<JobStatus> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let key = format!("job:{}", job_id);

    // Use a shorter TTL for faster updates, but still cache to reduce KV reads
    match kv.get(&key).cache_ttl(60).text().await? {
        Some(data) => {
            let status: JobStatus =
                serde_json::from_str(&data).map_err(|e| worker::Error::RustError(e.to_string()))?;

            // Update cache metrics
            update_cache_metrics(env, &format!("metrics:job:{}", job_id), true)
                .await
                .ok();

            console_log!(
                "Retrieved job status for {}: {} (progress: {}) [CACHE HIT]",
                job_id,
                status.status,
                status.progress
            );
            Ok(status)
        }
        None => {
            // Record cache miss
            update_cache_metrics(env, &format!("metrics:job:{}", job_id), false)
                .await
                .ok();

            console_log!("Job status not found for {} [CACHE MISS]", job_id);
            Err(worker::Error::RustError("Job not found".to_string()))
        }
    }
}

pub async fn update_job_status(env: &Env, job_id: &str, status: &JobStatus) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let status_json =
        serde_json::to_string(status).map_err(|e| worker::Error::RustError(e.to_string()))?;

    // Get intelligent cache configuration based on status
    let cache_config = CacheConfig::for_job_status(&status.status);

    // Store with optimized TTL
    let mut put_builder = kv.put(&format!("job:{}", job_id), &status_json)?;
    put_builder = put_builder.expiration_ttl(cache_config.ttl_seconds);
    put_builder.execute().await?;

    // Handle cache invalidation and warming for final states
    if status.status == "completed" {
        // Invalidate related caches and warm for completed jobs
        invalidate_related_cache(env, job_id, "job_completed")
            .await
            .ok();

        // Warm cache for this completed job
        if cache_config.cache_warming {
            warm_cache_for_completed_job(env, job_id).await.ok();
        }
    } else if status.status == "failed" {
        // Invalidate caches for failed jobs
        invalidate_related_cache(env, job_id, "job_failed")
            .await
            .ok();
    }

    console_log!(
        "Updated job status for {}: {} (progress: {}) [TTL: {}s]",
        job_id,
        status.status,
        status.progress,
        cache_config.ttl_seconds
    );
    Ok(())
}

pub async fn store_analysis_result(
    env: &Env,
    result_id: &str,
    result: &AnalysisResult,
) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let result_json =
        serde_json::to_string(result).map_err(|e| worker::Error::RustError(e.to_string()))?;

    // Use intelligent caching for analysis results
    let cache_config = CacheConfig::for_analysis_results();

    kv.put(&format!("result:{}", result_id), &result_json)?
        .expiration_ttl(cache_config.ttl_seconds)
        .execute()
        .await?;

    console_log!(
        "Stored analysis result for {}: [TTL: {}s]",
        result_id,
        cache_config.ttl_seconds
    );
    Ok(())
}

pub async fn get_analysis_result(env: &Env, result_id: &str) -> Result<AnalysisResult> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let cache_config = CacheConfig::for_analysis_results();

    match kv
        .get(&format!("result:{}", result_id))
        .cache_ttl(cache_config.ttl_seconds)
        .text()
        .await?
    {
        Some(data) => {
            let result: AnalysisResult =
                serde_json::from_str(&data).map_err(|e| worker::Error::RustError(e.to_string()))?;

            // Update cache metrics
            update_cache_metrics(env, &format!("metrics:result:{}", result_id), true)
                .await
                .ok();

            console_log!("Retrieved analysis result for {}: [CACHE HIT]", result_id);
            Ok(result)
        }
        None => {
            // Record cache miss
            update_cache_metrics(env, &format!("metrics:result:{}", result_id), false)
                .await
                .ok();

            console_log!("Analysis result not found for {} [CACHE MISS]", result_id);
            Err(worker::Error::RustError("Result not found".to_string()))
        }
    }
}

/// Store temporal analysis result in KV
///
/// # Arguments
/// * `env` - Worker environment
/// * `job_id` - Job identifier
/// * `result` - Temporal analysis result
///
/// # Errors
/// Returns error if KV write fails
pub async fn store_temporal_analysis_result(
    env: &Env,
    job_id: &str,
    result: &crate::TemporalAnalysisResult,
) -> Result<()> {
    console_log!("Storing temporal analysis result for job: {}", job_id);

    let kv = env.kv("CRESCENDAI_METADATA").map_err(|e| {
        console_log!("Failed to access KV namespace: {}", e);
        e
    })?;

    let key = format!("result:{}", job_id);
    let value = serde_json::to_string(result).map_err(|e| {
        console_log!("Failed to serialize result: {}", e);
        worker::Error::RustError(format!("Serialization failed: {}", e))
    })?;

    kv.put(&key, value)
        .map_err(|e| {
            console_log!("Failed to put to KV: {}", e);
            e
        })?
        .expiration_ttl(7200) // 2 hours
        .execute()
        .await
        .map_err(|e| {
            console_log!("Failed to execute KV put: {}", e);
            e
        })?;

    console_log!("Stored temporal result for: {}", job_id);
    Ok(())
}

/// Retrieve temporal analysis result from KV
///
/// # Arguments
/// * `env` - Worker environment
/// * `job_id` - Job identifier
///
/// # Returns
/// Temporal analysis result if found
///
/// # Errors
/// Returns error if not found or deserialization fails
pub async fn get_temporal_analysis_result(
    env: &Env,
    job_id: &str,
) -> Result<crate::TemporalAnalysisResult> {
    console_log!("Retrieving temporal analysis result for: {}", job_id);

    let kv = env.kv("CRESCENDAI_METADATA")?;
    let key = format!("result:{}", job_id);

    let value = kv.get(&key).text().await.map_err(|e| {
        console_log!("KV get failed for {}: {}", key, e);
        e
    })?;

    match value {
        Some(json_str) => {
            let result =
                serde_json::from_str::<crate::TemporalAnalysisResult>(&json_str).map_err(|e| {
                    console_log!("Failed to deserialize result: {}", e);
                    worker::Error::RustError(format!("Deserialization failed: {}", e))
                })?;

            console_log!("Retrieved temporal result for: {}", job_id);
            Ok(result)
        }
        None => {
            console_log!("No result found for: {}", job_id);
            Err(worker::Error::RustError(format!(
                "Result not found: {}",
                job_id
            )))
        }
    }
}

pub async fn get_audio_from_r2(env: &Env, file_id: &str) -> Result<Vec<u8>> {
    let bucket = env.bucket("AUDIO_BUCKET")?;
    let key = format!("audio/{}.wav", file_id);

    let object = bucket
        .get(&key)
        .execute()
        .await?
        .ok_or_else(|| worker::Error::RustError("File not found".to_string()))?;

    let data = object
        .body()
        .ok_or_else(|| worker::Error::RustError("No body in object".to_string()))?
        .bytes()
        .await?;
    Ok(data)
}

pub async fn store_comparison_result(
    env: &Env,
    comparison_id: &str,
    result: &ComparisonResult,
) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let result_json =
        serde_json::to_string(result).map_err(|e| worker::Error::RustError(e.to_string()))?;

    kv.put(&format!("comparison:{}", comparison_id), &result_json)?
        .execute()
        .await?;

    console_log!(
        "Stored comparison result for comparison_id: {}",
        comparison_id
    );
    Ok(())
}

pub async fn get_comparison_result(env: &Env, comparison_id: &str) -> Result<ComparisonResult> {
    let kv = env.kv("CRESCENDAI_METADATA")?;

    match kv
        .get(&format!("comparison:{}", comparison_id))
        .text()
        .await?
    {
        Some(data) => {
            let result: ComparisonResult =
                serde_json::from_str(&data).map_err(|e| worker::Error::RustError(e.to_string()))?;
            console_log!(
                "Retrieved comparison result for comparison_id: {}",
                comparison_id
            );
            Ok(result)
        }
        None => Err(worker::Error::RustError(
            "Comparison result not found".to_string(),
        )),
    }
}

pub async fn save_user_preference(env: &Env, preference: &UserPreference) -> Result<()> {
    let kv = env.kv("CRESCENDAI_METADATA")?;
    let preference_json =
        serde_json::to_string(preference).map_err(|e| worker::Error::RustError(e.to_string()))?;

    // Store preference with unique key based on comparison ID and timestamp
    let preference_key = format!(
        "preference:{}:{}",
        preference.comparison_id,
        js_sys::Date::now() as u64
    );

    kv.put(&preference_key, &preference_json)?.execute().await?;

    // Also store aggregated preferences for analytics
    let analytics_key = format!("analytics:{}", preference.comparison_id);
    match kv.get(&analytics_key).text().await? {
        Some(existing_data) => {
            // Update existing analytics
            let mut analytics: serde_json::Value = serde_json::from_str(&existing_data)
                .unwrap_or_else(
                    |_| json!({"model_a_votes": 0, "model_b_votes": 0, "total_votes": 0}),
                );

            if preference.preferred_model == "model_a" {
                analytics["model_a_votes"] =
                    json!(analytics["model_a_votes"].as_u64().unwrap_or(0) + 1);
            } else {
                analytics["model_b_votes"] =
                    json!(analytics["model_b_votes"].as_u64().unwrap_or(0) + 1);
            }
            analytics["total_votes"] = json!(analytics["total_votes"].as_u64().unwrap_or(0) + 1);

            kv.put(&analytics_key, &analytics.to_string())?
                .execute()
                .await?;
        }
        None => {
            // Create new analytics entry
            let analytics = if preference.preferred_model == "model_a" {
                json!({"model_a_votes": 1, "model_b_votes": 0, "total_votes": 1})
            } else {
                json!({"model_a_votes": 0, "model_b_votes": 1, "total_votes": 1})
            };

            kv.put(&analytics_key, &analytics.to_string())?
                .execute()
                .await?;
        }
    }

    console_log!(
        "Saved user preference for comparison_id: {} (prefers: {})",
        preference.comparison_id,
        preference.preferred_model
    );
    Ok(())
}
