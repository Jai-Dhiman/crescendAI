//! Security Module
//!
//! Handles authentication, rate limiting, input validation, and secure error responses.

use worker::*;
use std::collections::HashMap;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum SecurityError {
    InvalidApiKey,
    RateLimitExceeded,
    InvalidInput(String),
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityError::InvalidApiKey => write!(f, "Invalid or missing API key"),
            SecurityError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            SecurityError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for SecurityError {}

impl From<SecurityError> for worker::Error {
    fn from(error: SecurityError) -> Self {
        worker::Error::RustError(format!("SecurityError: {}", error))
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Rate limiter using Cloudflare KV for distributed rate limiting
pub struct RateLimiter {
    kv: kv::KvStore,
    window_size: u64,
    max_requests: u32,
}

impl RateLimiter {
    /// Create a new rate limiter from environment
    ///
    /// Defaults: 60 second window, 100 requests max
    pub fn new(env: &Env) -> Self {
        let kv = env.kv("CRESCENDAI_METADATA").expect("KV binding required");
        Self {
            kv,
            window_size: 60,      // 60 second window
            max_requests: 100,    // 100 requests per minute
        }
    }

    /// Check if a request from the given IP is within rate limits
    ///
    /// Returns true if allowed, false if rate limit exceeded
    pub async fn check_rate_limit(&self, client_ip: &str) -> Result<bool> {
        let now = js_sys::Date::now() as u64 / 1000;
        let window_start = now - self.window_size;

        let key = format!("rate_limit:{}", client_ip);

        // Get current request count
        let current_data = self.kv.get(&key).text().await?;
        let mut requests: HashMap<u64, u32> = current_data
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default();

        // Remove old entries outside window
        requests.retain(|&timestamp, _| timestamp > window_start);

        // Count current requests in window
        let current_count: u32 = requests.values().sum();

        if current_count >= self.max_requests {
            return Ok(false);
        }

        // Increment counter for current second
        *requests.entry(now).or_insert(0) += 1;

        // Store updated data with expiration
        let data = serde_json::to_string(&requests)
            .map_err(|_| worker::Error::RustError("Failed to serialize rate limit data".to_string()))?;

        self.kv
            .put(&key, data)?
            .expiration_ttl(self.window_size + 60) // TTL with buffer
            .execute()
            .await?;

        Ok(true)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract client IP from request headers
///
/// Checks CF-Connecting-IP (Cloudflare), X-Forwarded-For, and X-Real-IP
pub fn get_client_ip(req: &Request) -> String {
    req.headers()
        .get("CF-Connecting-IP")
        .ok()
        .flatten()
        .or_else(|| req.headers().get("X-Forwarded-For").ok().flatten())
        .or_else(|| req.headers().get("X-Real-IP").ok().flatten())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Create a secure error response that doesn't leak sensitive information
pub fn secure_error_response(message: &str, status: u16) -> Result<Response> {
    let json = serde_json::json!({
        "error": message,
        "status": status
    });

    Response::from_json(&json)
        .map(|r| r.with_status(status))
}

/// Validate request body size
pub fn validate_body_size(size: usize, max_size: usize) -> Result<()> {
    if size > max_size {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            format!("Request body too large: {} bytes (max: {} bytes)", size, max_size)
        )));
    }
    Ok(())
}

/// Validate filename for security (no path traversal)
pub fn validate_filename(filename: &str) -> Result<()> {
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Invalid filename: path traversal detected".to_string()
        )));
    }
    Ok(())
}

/// Validate UUID format
pub fn validate_uuid(uuid: &str) -> Result<()> {
    // Simple UUID validation (8-4-4-4-12 format)
    let parts: Vec<&str> = uuid.split('-').collect();
    if parts.len() != 5
        || parts[0].len() != 8
        || parts[1].len() != 4
        || parts[2].len() != 4
        || parts[3].len() != 4
        || parts[4].len() != 12 {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Invalid UUID format".to_string()
        )));
    }

    // Check all characters are hex
    for part in parts {
        if !part.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(worker::Error::from(SecurityError::InvalidInput(
                "Invalid UUID format: non-hex characters".to_string()
            )));
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_filename() {
        assert!(validate_filename("test.wav").is_ok());
        assert!(validate_filename("song-123.mp3").is_ok());

        assert!(validate_filename("../etc/passwd").is_err());
        assert!(validate_filename("test/../secret").is_err());
        assert!(validate_filename("path/to/file").is_err());
        assert!(validate_filename("C:\\Windows\\file").is_err());
    }

    #[test]
    fn test_validate_body_size() {
        assert!(validate_body_size(100, 1000).is_ok());
        assert!(validate_body_size(1000, 1000).is_ok());
        assert!(validate_body_size(1001, 1000).is_err());
    }
}
