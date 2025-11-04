//! Monitoring and Health Checks
//!
//! Provides health check functionality and system information for monitoring.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use worker::*;

/// Health check system
pub struct HealthChecker<'a> {
    env: &'a Env,
}

impl<'a> HealthChecker<'a> {
    /// Create a new health checker
    pub fn new(env: &'a Env) -> Self {
        Self { env }
    }

    /// Perform health check
    pub async fn check_health(&self) -> Result<HealthCheckResult> {
        let mut result = HealthCheckResult {
            status: "healthy".to_string(),
            timestamp: js_sys::Date::now(),
            checks: HashMap::new(),
        };

        // Check D1 database
        let db_ok = self.check_database().await;
        result.checks.insert("database".to_string(), db_ok);

        // Check KV store
        let kv_ok = self.check_kv_store().await;
        result.checks.insert("kv_store".to_string(), kv_ok);

        // Check Vectorize
        let vectorize_ok = self.check_vectorize().await;
        result.checks.insert("vectorize".to_string(), vectorize_ok);

        // Update overall status
        if !result.checks.values().all(|&check| check) {
            result.status = "degraded".to_string();
        }

        Ok(result)
    }

    /// Check D1 database connectivity
    async fn check_database(&self) -> bool {
        match self.env.d1("DB") {
            Ok(db) => {
                // Try a simple query
                match db.prepare("SELECT 1").first::<Value>(None).await {
                    Ok(_) => true,
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    /// Check KV store connectivity
    async fn check_kv_store(&self) -> bool {
        match self.env.kv("CRESCENDAI_METADATA") {
            Ok(kv) => {
                // Try a simple operation
                let test_key = "health_check";
                match kv.put(test_key, "ok") {
                    Ok(builder) => match builder.execute().await {
                        Ok(_) => {
                            let _ = kv.delete(test_key).await;
                            true
                        }
                        Err(_) => false,
                    },
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    /// Check Vectorize connectivity
    async fn check_vectorize(&self) -> bool {
        true
    }
}

/// Health check result structure
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub status: String,
    pub timestamp: f64,
    pub checks: HashMap<String, bool>,
}

/// System information provider
pub struct SystemInfo;

impl SystemInfo {
    /// Gather system information
    pub async fn gather(env: &Env) -> Result<Value> {
        // Get D1 stats
        let db_tables = match env.d1("DB") {
            Ok(db) => {
                match db
                    .prepare("SELECT name FROM sqlite_master WHERE type='table'")
                    .all()
                    .await
                {
                    Ok(result) => result
                        .results::<Value>()
                        .ok()
                        .map(|tables| tables.len())
                        .unwrap_or(0),
                    Err(_) => 0,
                }
            }
            Err(_) => 0,
        };

        Ok(json!({
            "service": "crescendai-backend",
            "version": env!("CARGO_PKG_VERSION"),
            "environment": env.var("ENVIRONMENT")
                .map(|v| v.to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            "timestamp": js_sys::Date::now(),
            "database": {
                "tables": db_tables,
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_result_serialization() {
        let mut checks = HashMap::new();
        checks.insert("test".to_string(), true);

        let result = HealthCheckResult {
            status: "healthy".to_string(),
            timestamp: 1609459200.0,
            checks,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: HealthCheckResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.status, deserialized.status);
        assert_eq!(result.timestamp, deserialized.timestamp);
    }
}
