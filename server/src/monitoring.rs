// Phase 3.4: Production Monitoring and Observability
// Implements structured logging, metrics collection, and health checks

use worker::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

#[cfg(not(test))]
use js_sys::Date;

// Test-only Date implementation  
#[cfg(test)]
struct Date;

#[cfg(test)]
impl Date {
    fn now() -> f64 {
        1640995200000.0 // Fixed timestamp for tests
    }
}

// Conditional logging macro for test vs WASM environments
#[cfg(test)]
macro_rules! console_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(test))]
macro_rules! console_log {
    ($($arg:tt)*) => {
        worker::console_log!($($arg)*);
    };
}

/// Structured log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
}

/// Request context for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    pub request_id: String,
    pub method: String,
    pub path: String,
    pub user_agent: Option<String>,
    pub client_ip: String,
    pub timestamp: f64,
    pub api_key_hash: Option<String>, // Hash of API key for correlation without exposure
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub request_duration_ms: f64,
    pub processing_time_ms: Option<f64>,
    pub memory_usage_mb: Option<usize>,
    pub cache_hit: Option<bool>,
    pub file_size_bytes: Option<usize>,
    pub error_count: u32,
}

/// Security metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub auth_failures: u32,
    pub rate_limit_hits: u32,
    pub invalid_file_attempts: u32,
    pub malicious_requests: u32,
    pub cors_violations: u32,
}

/// Cache metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub cache_errors: u32,
    pub avg_lookup_time_ms: f64,
}

/// Request logger with structured output
pub struct RequestLogger {
    context: RequestContext,
    start_time: f64,
    metrics: PerformanceMetrics,
}

impl RequestLogger {
    /// Create a new request logger
    pub fn new(req: &Request) -> Self {
        let request_id = uuid::Uuid::new_v4().to_string();
        let timestamp = Date::now();
        
        let context = RequestContext {
            request_id: request_id.clone(),
            method: req.method().to_string(),
            path: req.path(),
            user_agent: req.headers().get("User-Agent").ok().flatten(),
            client_ip: get_client_ip_from_request(req),
            timestamp,
            api_key_hash: get_api_key_hash(req),
        };

        let metrics = PerformanceMetrics {
            request_duration_ms: 0.0,
            processing_time_ms: None,
            memory_usage_mb: None,
            cache_hit: None,
            file_size_bytes: None,
            error_count: 0,
        };

        Self {
            context,
            start_time: timestamp,
            metrics,
        }
    }

    /// Log structured message
    pub fn log(&self, level: LogLevel, message: &str, additional_data: Option<Value>) {
        let log_entry = json!({
            "timestamp": Date::now(),
            "level": level,
            "message": message,
            "request_id": self.context.request_id,
            "method": self.context.method,
            "path": self.context.path,
            "client_ip": self.context.client_ip,
            "additional_data": additional_data.unwrap_or(json!({}))
        });

        // In production, this would send to logging service
        web_sys::console::log_1(&format!("{}", log_entry).into());
    }

    /// Log request start
    pub fn log_request_start(&self) {
        self.log(LogLevel::INFO, "Request started", Some(json!({
            "user_agent": self.context.user_agent,
            "api_key_present": self.context.api_key_hash.is_some()
        })));
    }

    /// Log request completion
    pub fn log_request_complete(&mut self, status_code: u16, response_size: Option<usize>) {
        self.metrics.request_duration_ms = Date::now() - self.start_time;
        
        self.log(LogLevel::INFO, "Request completed", Some(json!({
            "status_code": status_code,
            "duration_ms": self.metrics.request_duration_ms,
            "response_size_bytes": response_size,
            "processing_time_ms": self.metrics.processing_time_ms,
            "cache_hit": self.metrics.cache_hit,
            "file_size_bytes": self.metrics.file_size_bytes
        })));
    }

    /// Log error
    pub fn log_error(&mut self, error: &str, error_type: &str, stack_trace: Option<&str>) {
        self.metrics.error_count += 1;
        
        self.log(LogLevel::ERROR, "Request error", Some(json!({
            "error": error,
            "error_type": error_type,
            "stack_trace": stack_trace,
            "error_count": self.metrics.error_count
        })));
    }

    /// Update processing metrics
    pub fn update_processing_metrics(&mut self, processing_time_ms: f64, file_size: Option<usize>) {
        self.metrics.processing_time_ms = Some(processing_time_ms);
        self.metrics.file_size_bytes = file_size;
    }

    /// Update cache metrics
    pub fn update_cache_metrics(&mut self, cache_hit: bool) {
        self.metrics.cache_hit = Some(cache_hit);
    }

    /// Get request ID for correlation
    pub fn get_request_id(&self) -> &str {
        &self.context.request_id
    }
}

/// Performance metrics collector
pub struct PerformanceMetricsCollector {
    kv_store: kv::KvStore,
}

impl PerformanceMetricsCollector {
    pub fn new(kv_store: kv::KvStore) -> Self {
        Self { kv_store }
    }

    /// Record API latency metric
    pub async fn record_api_latency(&self, endpoint: &str, duration_ms: f64) -> Result<()> {
        let key = format!("metrics:api_latency:{}", endpoint);
        let timestamp = Date::now() as u64;
        
        let metric_data = json!({
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "timestamp": timestamp
        });

        // Store metric with TTL (24 hours)
        self.kv_store
            .put(&key, metric_data.to_string())?
            .expiration_ttl(86400)
            .execute()
            .await?;

        Ok(())
    }

    /// Record processing time metric
    pub async fn record_processing_time(&self, operation: &str, duration_ms: f64) -> Result<()> {
        let key = format!("metrics:processing:{}", operation);
        let timestamp = Date::now() as u64;
        
        let metric_data = json!({
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": timestamp
        });

        self.kv_store
            .put(&key, metric_data.to_string())?
            .expiration_ttl(86400)
            .execute()
            .await?;

        Ok(())
    }

    /// Record error metric
    pub async fn record_error(&self, error_type: &str, endpoint: &str) -> Result<()> {
        let key = format!("metrics:errors:{}:{}", error_type, endpoint);
        let timestamp = Date::now() as u64;
        
        // Get current error count
        let current_count = self.get_error_count(&key).await.unwrap_or(0);
        
        let metric_data = json!({
            "error_type": error_type,
            "endpoint": endpoint,
            "count": current_count + 1,
            "last_occurrence": timestamp
        });

        self.kv_store
            .put(&key, metric_data.to_string())?
            .expiration_ttl(86400)
            .execute()
            .await?;

        Ok(())
    }

    /// Get error count
    async fn get_error_count(&self, key: &str) -> Result<u32> {
        if let Some(data) = self.kv_store.get(key).text().await? {
            if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
                return Ok(parsed["count"].as_u64().unwrap_or(0) as u32);
            }
        }
        Ok(0)
    }

    /// Get API latency statistics
    pub async fn get_api_latency_stats(&self, endpoint: &str) -> Result<Option<Value>> {
        let key = format!("metrics:api_latency:{}", endpoint);
        
        if let Some(data) = self.kv_store.get(&key).text().await? {
            if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
                return Ok(Some(parsed));
            }
        }
        
        Ok(None)
    }
}

/// Error rate tracker
pub struct ErrorTracker {
    kv_store: kv::KvStore,
    window_size: u64, // Time window in seconds
}

impl ErrorTracker {
    pub fn new(kv_store: kv::KvStore, window_size: u64) -> Self {
        Self { kv_store, window_size }
    }

    /// Track error occurrence
    pub async fn track_error(&self, error_type: &str, endpoint: &str) -> Result<()> {
        let now = Date::now() as u64 / 1000;
        let key = format!("errors:{}:{}", error_type, endpoint);
        
        // Get current error timestamps
        let current_data = self.kv_store.get(&key).text().await?;
        let mut error_timestamps: Vec<u64> = current_data
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default();

        // Remove old timestamps outside the window
        let window_start = now - self.window_size;
        error_timestamps.retain(|&timestamp| timestamp > window_start);

        // Add new error timestamp
        error_timestamps.push(now);

        // Store updated timestamps
        let data = serde_json::to_string(&error_timestamps)
            .map_err(|_| worker::Error::RustError("Failed to serialize error timestamps".to_string()))?;
        
        self.kv_store
            .put(&key, data)?
            .expiration_ttl(self.window_size + 60) // TTL with buffer
            .execute()
            .await?;

        Ok(())
    }

    /// Get error rate for a specific type/endpoint
    pub async fn get_error_rate(&self, error_type: &str, endpoint: &str) -> Result<f64> {
        let now = Date::now() as u64 / 1000;
        let key = format!("errors:{}:{}", error_type, endpoint);
        
        let current_data = self.kv_store.get(&key).text().await?;
        let error_timestamps: Vec<u64> = current_data
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default();

        // Count errors in current window
        let window_start = now - self.window_size;
        let errors_in_window = error_timestamps
            .iter()
            .filter(|&&timestamp| timestamp > window_start)
            .count();

        // Calculate rate (errors per second)
        Ok(errors_in_window as f64 / self.window_size as f64)
    }

    /// Check if error rate exceeds threshold
    pub async fn is_error_rate_exceeded(&self, error_type: &str, endpoint: &str, threshold: f64) -> Result<bool> {
        let current_rate = self.get_error_rate(error_type, endpoint).await?;
        Ok(current_rate > threshold)
    }
}

/// Health check system
pub struct HealthChecker {
    kv_store: Option<kv::KvStore>,
}

impl HealthChecker {
    pub fn new(kv_store: Option<kv::KvStore>) -> Self {
        Self { kv_store }
    }

    /// Perform basic health check
    pub async fn basic_health_check(&self) -> HealthCheckResult {
        let mut result = HealthCheckResult {
            status: "healthy".to_string(),
            timestamp: Date::now(),
            checks: HashMap::new(),
            details: HashMap::new(),
        };

        // Check system time
        result.checks.insert("system_time".to_string(), true);
        result.details.insert("system_time".to_string(), json!({
            "current_time": Date::now(),
            "status": "ok"
        }));

        // Check memory (simulated)
        let memory_ok = self.check_memory_usage().await;
        result.checks.insert("memory".to_string(), memory_ok);
        result.details.insert("memory".to_string(), json!({
            "status": if memory_ok { "ok" } else { "warning" },
            "usage_mb": 64 // Simulated value
        }));

        // If any check fails, mark as unhealthy
        if !result.checks.values().all(|&check| check) {
            result.status = "unhealthy".to_string();
        }

        result
    }

    /// Perform detailed health check including dependencies
    pub async fn detailed_health_check(&self) -> HealthCheckResult {
        let mut result = self.basic_health_check().await;

        // Check KV store if available
        if let Some(kv) = &self.kv_store {
            let kv_ok = self.check_kv_store(kv).await;
            result.checks.insert("kv_store".to_string(), kv_ok);
            result.details.insert("kv_store".to_string(), json!({
                "status": if kv_ok { "ok" } else { "error" },
                "latency_ms": 10 // Simulated latency
            }));
        }

        // Check external dependencies
        let modal_ok = self.check_modal_api().await;
        result.checks.insert("modal_api".to_string(), modal_ok);
        result.details.insert("modal_api".to_string(), json!({
            "status": if modal_ok { "ok" } else { "degraded" },
            "response_time_ms": 250 // Simulated response time
        }));

        // Update overall status
        if !result.checks.values().all(|&check| check) {
            result.status = "degraded".to_string();
        }

        result
    }

    /// Check memory usage
    async fn check_memory_usage(&self) -> bool {
        // In a real implementation, this would check actual memory usage
        // For now, simulate a memory check
        true
    }

    /// Check KV store connectivity
    async fn check_kv_store(&self, kv: &kv::KvStore) -> bool {
        // Try a simple KV operation
        let test_key = "health_check_test";
        let test_value = "ok";
        
        match kv.put(test_key, test_value) {
            Ok(put_request) => {
                match put_request.execute().await {
                    Ok(_) => {
                        // Clean up test key
                        let _ = kv.delete(test_key).await;
                        true
                    }
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    /// Check Modal API connectivity
    async fn check_modal_api(&self) -> bool {
        // In a real implementation, this would ping the Modal API
        // For now, simulate the check
        true
    }
}

/// Health check result structure
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub status: String, // "healthy", "degraded", "unhealthy"
    pub timestamp: f64,
    pub checks: HashMap<String, bool>,
    pub details: HashMap<String, Value>,
}

/// System information provider
pub struct SystemInfo;

impl SystemInfo {
    /// Get system information
    pub fn get_system_info() -> Value {
        json!({
            "service": "crescendai-backend",
            "version": env!("CARGO_PKG_VERSION"),
            "build_time": std::env::var("BUILD_TIMESTAMP").unwrap_or_else(|_| "unknown".to_string()),
            "runtime": "cloudflare-workers",
            "wasm_target": "wasm32-unknown-unknown",
            "timestamp": Date::now()
        })
    }

    /// Get runtime metrics
    pub fn get_runtime_metrics() -> Value {
        json!({
            "uptime_ms": Date::now(), // Simplified - actual uptime would be different
            "memory_usage_estimate_mb": 32, // Estimated
            "request_count": 0, // Would be tracked separately
            "cache_size_estimate": 1024, // Estimated
            "timestamp": Date::now()
        })
    }
}

/// Helper functions
fn get_client_ip_from_request(req: &Request) -> String {
    req.headers()
        .get("CF-Connecting-IP")
        .ok()
        .flatten()
        .or_else(|| req.headers().get("X-Forwarded-For").ok().flatten())
        .or_else(|| req.headers().get("X-Real-IP").ok().flatten())
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_api_key_hash(req: &Request) -> Option<String> {
    // Get API key and create a hash for correlation without exposure
    let api_key = req.headers()
        .get("Authorization")
        .ok()
        .flatten()
        .and_then(|auth| auth.strip_prefix("Bearer ").map(|key| key.to_string()))
        .or_else(|| req.headers().get("X-API-Key").ok().flatten());

    api_key.map(|key| {
        // Create a simple hash of the API key for correlation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    })
}

/// Alerting system for critical events
pub struct AlertingSystem {
    webhook_url: Option<String>,
}

impl AlertingSystem {
    pub fn new(webhook_url: Option<String>) -> Self {
        Self { webhook_url }
    }

    /// Send critical alert
    pub async fn send_critical_alert(&self, title: &str, message: &str, details: Option<Value>) -> Result<()> {
        let alert = json!({
            "level": "critical",
            "title": title,
            "message": message,
            "details": details.unwrap_or(json!({})),
            "timestamp": Date::now(),
            "service": "crescendai-backend"
        });

        // Log the alert
        web_sys::console::error_1(&format!("CRITICAL ALERT: {}", alert).into());

        // Send to webhook if configured
        if let Some(webhook_url) = &self.webhook_url {
            self.send_webhook_alert(webhook_url, &alert).await?;
        }

        Ok(())
    }

    /// Send warning alert
    pub async fn send_warning_alert(&self, title: &str, message: &str) -> Result<()> {
        let alert = json!({
            "level": "warning",
            "title": title,
            "message": message,
            "timestamp": Date::now(),
            "service": "crescendai-backend"
        });

        web_sys::console::warn_1(&format!("WARNING ALERT: {}", alert).into());
        Ok(())
    }

    /// Send webhook alert
    async fn send_webhook_alert(&self, webhook_url: &str, alert: &Value) -> Result<()> {
        // In a real implementation, this would send HTTP POST to webhook URL
        // For now, just log that it would be sent
        web_sys::console::log_1(&format!("Would send webhook to {}: {}", webhook_url, alert).into());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_serialization() {
        let levels = vec![LogLevel::DEBUG, LogLevel::INFO, LogLevel::WARN, LogLevel::ERROR];
        
        for level in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            let deserialized: LogLevel = serde_json::from_str(&serialized).unwrap();
            
            // Use Debug trait for comparison since LogLevel doesn't implement PartialEq
            assert_eq!(format!("{:?}", level), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::get_system_info();
        
        assert!(info["service"].as_str() == Some("crescendai-backend"));
        assert!(info["version"].is_string());
        assert!(info["runtime"].as_str() == Some("cloudflare-workers"));
        assert!(info["timestamp"].is_number());
    }

    #[test]
    fn test_runtime_metrics() {
        let metrics = SystemInfo::get_runtime_metrics();
        
        assert!(metrics["uptime_ms"].is_number());
        assert!(metrics["memory_usage_estimate_mb"].is_number());
        assert!(metrics["timestamp"].is_number());
    }

    #[test]
    fn test_health_check_result_serialization() {
        let mut checks = HashMap::new();
        checks.insert("test".to_string(), true);
        
        let mut details = HashMap::new();
        details.insert("test".to_string(), json!({"status": "ok"}));
        
        let result = HealthCheckResult {
            status: "healthy".to_string(),
            timestamp: 1609459200.0,
            checks,
            details,
        };
        
        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: HealthCheckResult = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(result.status, deserialized.status);
        assert_eq!(result.timestamp, deserialized.timestamp);
        assert_eq!(result.checks.len(), deserialized.checks.len());
    }
}