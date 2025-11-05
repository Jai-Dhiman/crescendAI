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
// API Key Validation
// ============================================================================

/// Validate API key for development/testing
///
/// Checks if the provided API key matches one of the allowed keys in the environment.
/// This is primarily for development and testing. Production auth should use proper OAuth/JWT.
pub fn validate_api_key(env: &Env, api_key: &str) -> Result<bool> {
    // Get allowed API keys from environment
    let allowed_keys = env
        .var("ALLOWED_API_KEYS")
        .ok()
        .map(|v| v.to_string())
        .unwrap_or_default();

    if allowed_keys.is_empty() {
        // If no keys configured, allow all (development mode)
        console_log!("Warning: No API keys configured, allowing all requests");
        return Ok(true);
    }

    // Split comma-separated keys
    let keys: Vec<&str> = allowed_keys.split(',').map(|s| s.trim()).collect();

    Ok(keys.contains(&api_key))
}

/// Extract API key from request headers
///
/// Checks Authorization header (Bearer token) and X-API-Key header
pub fn get_api_key(req: &Request) -> Option<String> {
    // Check Authorization header (Bearer token)
    if let Ok(Some(auth)) = req.headers().get("Authorization") {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            return Some(token.to_string());
        }
    }

    // Check X-API-Key header
    req.headers().get("X-API-Key").ok().flatten()
}

// ============================================================================
// File Validation (Magic Bytes)
// ============================================================================

/// Validate file type by checking magic bytes (file signature)
///
/// This provides more security than just checking file extensions, as it verifies
/// the actual file format by reading the first few bytes.
pub fn validate_file_magic_bytes(file_bytes: &[u8], expected_mime: &str) -> Result<bool> {
    if file_bytes.len() < 12 {
        return Ok(false); // File too small to have valid header
    }

    match expected_mime {
        // WAV files start with "RIFF" and have "WAVE" at bytes 8-11
        "audio/wav" | "audio/wave" => {
            let is_valid = file_bytes.starts_with(b"RIFF") && &file_bytes[8..12] == b"WAVE";
            Ok(is_valid)
        }

        // MP3 files start with ID3 tag or MPEG frame sync
        "audio/mpeg" | "audio/mp3" => {
            let has_id3 = file_bytes.starts_with(b"ID3");
            let has_mpeg_sync = file_bytes[0] == 0xFF && (file_bytes[1] & 0xE0) == 0xE0;
            Ok(has_id3 || has_mpeg_sync)
        }

        // M4A files (MPEG-4 Audio) start with ftyp box
        "audio/mp4" | "audio/x-m4a" => {
            let has_ftyp = file_bytes.len() >= 12
                && &file_bytes[4..8] == b"ftyp"
                && (&file_bytes[8..12] == b"M4A " || &file_bytes[8..12] == b"mp42");
            Ok(has_ftyp)
        }

        // Unknown MIME type - allow by default (extension check already done)
        _ => {
            console_log!("Warning: Unknown MIME type for magic byte validation: {}", expected_mime);
            Ok(true)
        }
    }
}

/// Sanitize user input to prevent injection attacks
///
/// Removes potentially dangerous characters and limits length
pub fn sanitize_user_input(input: &str, max_length: usize) -> String {
    input
        .chars()
        .take(max_length)
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-' || *c == '_' || *c == '.')
        .collect()
}

// ============================================================================
// Security Headers
// ============================================================================

/// Add security headers to a response
pub fn add_security_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();

    // Prevent MIME type sniffing
    let _ = headers.set("X-Content-Type-Options", "nosniff");

    // Prevent clickjacking
    let _ = headers.set("X-Frame-Options", "DENY");

    // Basic CSP for API (no scripts needed)
    let _ = headers.set("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'");

    // Remove sensitive headers
    let _ = headers.delete("Server");
    let _ = headers.delete("X-Powered-By");

    response
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
