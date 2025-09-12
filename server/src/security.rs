use worker::*;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;
use regex::Regex;
use once_cell::sync::Lazy;

/// HMAC type alias for webhook signature verification
type HmacSha256 = Hmac<Sha256>;

/// Error types for security operations
#[derive(Debug)]
pub enum SecurityError {
    InvalidApiKey,
    InvalidSignature,
    ReplayAttack,
    RateLimitExceeded,
    InvalidInput(String),
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityError::InvalidApiKey => write!(f, "Invalid or missing API key"),
            SecurityError::InvalidSignature => write!(f, "Invalid webhook signature"),
            SecurityError::ReplayAttack => write!(f, "Request timestamp too old"),
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

/// Regex patterns for input validation
static FILENAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z0-9_.-]+$").unwrap()
});

static UUID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$").unwrap()
});

/// Validates API key from request headers
pub fn validate_api_key(req: &Request, env: &Env) -> Result<()> {
    // Check for API key in Authorization header or X-API-Key header
    let valid_api_key = env.secret("API_KEY")?.to_string();
    
    // Try Authorization header first
    if let Ok(Some(auth)) = req.headers().get("Authorization") {
        if let Some(key) = auth.strip_prefix("Bearer ") {
            if constant_time_compare(key, &valid_api_key) {
                return Ok(());
            }
        }
    }
    
    // Try X-API-Key header
    if let Ok(Some(key)) = req.headers().get("X-API-Key") {
        if constant_time_compare(&key, &valid_api_key) {
            return Ok(());
        }
    }
    
    Err(worker::Error::from(SecurityError::InvalidApiKey))
}

/// Constant-time string comparison to prevent timing attacks
fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    a.bytes()
        .zip(b.bytes())
        .fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Validates webhook signature using HMAC-SHA256
pub fn validate_webhook_signature(
    payload: &[u8],
    signature_header: &str,
    secret: &str,
    timestamp_header: &str,
) -> Result<()> {
    // Parse timestamp and validate it's not too old (5 minutes)
    let timestamp: i64 = timestamp_header.parse()
        .map_err(|_| worker::Error::from(SecurityError::InvalidInput("Invalid timestamp".to_string())))?;
    
    let current_timestamp = js_sys::Date::now() as i64 / 1000;
    if current_timestamp - timestamp > 300 {
        return Err(worker::Error::from(SecurityError::ReplayAttack));
    }

    // Create payload with timestamp for signature verification
    let signed_payload = format!("{}.{}", timestamp, std::str::from_utf8(payload)
        .map_err(|_| worker::Error::from(SecurityError::InvalidInput("Invalid payload encoding".to_string())))?);

    // Calculate expected signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|_| worker::Error::from(SecurityError::InvalidSignature))?;
    mac.update(signed_payload.as_bytes());
    let expected_signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    // Compare signatures in constant time
    if constant_time_compare(signature_header, &expected_signature) {
        Ok(())
    } else {
        Err(worker::Error::from(SecurityError::InvalidSignature))
    }
}

/// Validates file type based on content and extension
pub fn validate_file_type(filename: &str, content: &[u8]) -> Result<()> {
    // Check file extension
    let extension = filename.split('.').last().unwrap_or("").to_lowercase();
    if !matches!(extension.as_str(), "wav" | "mp3" | "ogg" | "m4a" | "aac" | "flac" | "mp4") {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Only audio files are allowed (WAV, MP3, OGG, M4A, AAC, FLAC, MP4)".to_string()
        )));
    }

    // Validate content based on magic numbers (basic validation for common formats)
    match extension.as_str() {
        "wav" => {
            if content.len() < 12 || &content[0..4] != b"RIFF" || &content[8..12] != b"WAVE" {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    "Invalid WAV file format".to_string()
                )));
            }
        }
        "mp3" => {
            if content.len() < 3 || 
               (!content.starts_with(b"ID3") && 
                !content.starts_with(&[0xFF, 0xFB]) && 
                !content.starts_with(&[0xFF, 0xFA])) {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    "Invalid MP3 file format".to_string()
                )));
            }
        }
        "ogg" => {
            if content.len() < 4 || !content.starts_with(b"OggS") {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    "Invalid OGG file format".to_string()
                )));
            }
        }
        "flac" => {
            if content.len() < 4 || !content.starts_with(b"fLaC") {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    "Invalid FLAC file format".to_string()
                )));
            }
        }
        "m4a" | "aac" | "mp4" => {
            // Basic validation for MP4 container (used by M4A/AAC/MP4)
            if content.len() < 8 {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    format!("Invalid {} file format", extension.to_uppercase())
                )));
            }
            // Look for 'ftyp' box in MP4 container
            let has_ftyp = content.windows(4).any(|window| window == b"ftyp");
            if !has_ftyp {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    format!("Invalid {} file format", extension.to_uppercase())
                )));
            }
        }
        _ => {
            // For other formats, just check minimum file size
            if content.len() < 100 {
                return Err(worker::Error::from(SecurityError::InvalidInput(
                    "Audio file appears to be too small or corrupted".to_string()
                )));
            }
        },
    }

    Ok(())
}

/// Validates and sanitizes filename to prevent path traversal
pub fn sanitize_filename(filename: &str) -> Result<String> {
    if filename.is_empty() || filename.len() > 255 {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Invalid filename length".to_string()
        )));
    }

    // Check for path traversal attempts
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Filename contains invalid characters".to_string()
        )));
    }

    // Validate against allowed pattern
    if !FILENAME_REGEX.is_match(filename) {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "Filename contains invalid characters".to_string()
        )));
    }

    Ok(filename.to_string())
}

/// Validates UUID format
pub fn validate_uuid(uuid: &str) -> Result<()> {
    if UUID_REGEX.is_match(uuid) {
        Ok(())
    } else {
        Err(worker::Error::from(SecurityError::InvalidInput(
            "Invalid UUID format".to_string()
        )))
    }
}

/// Rate limiting implementation using Cloudflare KV
pub struct RateLimiter {
    kv: kv::KvStore,
    window_size: u64,
    max_requests: u32,
}

impl RateLimiter {
    pub fn new(kv: kv::KvStore, window_size: u64, max_requests: u32) -> Self {
        Self {
            kv,
            window_size,
            max_requests,
        }
    }

    /// Check if request is within rate limit
    pub async fn check_rate_limit(&self, client_ip: &str) -> Result<()> {
        let now = js_sys::Date::now() as u64 / 1000;
        let window_start = now - self.window_size;
        
        let key = format!("rate_limit:{}", client_ip);
        
        // Get current request count
        let current_data = self.kv.get(&key).text().await?;
        let mut requests: HashMap<u64, u32> = current_data
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default();

        // Remove old entries
        requests.retain(|&timestamp, _| timestamp > window_start);

        // Count current requests in window
        let current_count: u32 = requests.values().sum();

        if current_count >= self.max_requests {
            return Err(worker::Error::from(SecurityError::RateLimitExceeded));
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

        Ok(())
    }
}

/// Validates request body size
pub fn validate_body_size(size: usize, max_size: usize) -> Result<()> {
    if size > max_size {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            format!("Request body too large: {} bytes (max: {} bytes)", size, max_size)
        )));
    }
    Ok(())
}

/// Gets client IP address from request
pub fn get_client_ip(req: &Request) -> String {
    req.headers()
        .get("CF-Connecting-IP")
        .ok()
        .flatten()
        .or_else(|| req.headers().get("X-Forwarded-For").ok().flatten())
        .or_else(|| req.headers().get("X-Real-IP").ok().flatten())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Secure error response that doesn't leak sensitive information
pub fn secure_error_response(error: &worker::Error, include_details: bool) -> Response {
    let (status, message) = match error {
        worker::Error::RustError(ref e) if e.contains("SecurityError") => {
            if include_details {
                (401, e.clone())
            } else {
                (401, "Authentication required".to_string())
            }
        }
        _ => {
            if include_details {
                (500, "Internal server error".to_string())
            } else {
                (500, "Request could not be processed".to_string())
            }
        }
    };

    let error_response = serde_json::json!({
        "error": "Request failed",
        "message": message,
        "code": status
    });

    Response::from_json(&error_response)
        .map(|response| response.with_status(status))
        .unwrap_or_else(|_| Response::error("Request could not be processed", 500).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare("same", "same"));
        assert!(!constant_time_compare("different", "strings"));
        assert!(!constant_time_compare("short", "longer"));
        assert!(!constant_time_compare("", "nonempty"));
    }

    #[test]
    fn test_filename_sanitization() {
        // Valid filenames
        assert!(sanitize_filename("audio.wav").is_ok());
        assert!(sanitize_filename("test_file-1.mp3").is_ok());
        assert!(sanitize_filename("file123.wav").is_ok());
        assert!(sanitize_filename("audio.flac").is_ok());
        assert!(sanitize_filename("music.m4a").is_ok());

        // Invalid filenames
        assert!(sanitize_filename("../etc/passwd").is_err());
        assert!(sanitize_filename("file/path.wav").is_err());
        assert!(sanitize_filename("file\\path.wav").is_err());
        assert!(sanitize_filename("").is_err());
        assert!(sanitize_filename("file with spaces.wav").is_err());
        assert!(sanitize_filename(&"a".repeat(256)).is_err());
    }

    #[test]
    fn test_uuid_validation() {
        // Valid UUIDs
        assert!(validate_uuid("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(validate_uuid("6ba7b810-9dad-11d1-80b4-00c04fd430c8").is_ok());

        // Invalid UUIDs
        assert!(validate_uuid("invalid-uuid").is_err());
        assert!(validate_uuid("550e8400-e29b-41d4-a716-44665544000").is_err()); // Too short
        assert!(validate_uuid("550e8400-e29b-41d4-a716-4466554400000").is_err()); // Too long
        assert!(validate_uuid("").is_err());
    }

    #[test]
    fn test_file_type_validation() {
        // Valid WAV file
        let mut wav_data = Vec::new();
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&[0u8; 4]);
        wav_data.extend_from_slice(b"WAVE");
        wav_data.extend_from_slice(&[0u8; 20]);
        
        assert!(validate_file_type("audio.wav", &wav_data).is_ok());

        // Valid MP3 file (ID3 tag)
        let mut mp3_data = Vec::new();
        mp3_data.extend_from_slice(b"ID3");
        mp3_data.extend_from_slice(&[0u8; 20]);
        
        assert!(validate_file_type("audio.mp3", &mp3_data).is_ok());

        // Invalid file type
        assert!(validate_file_type("audio.txt", b"text content").is_err());
        
        // Invalid WAV content
        assert!(validate_file_type("audio.wav", b"not wav").is_err());
    }

    #[test]
    fn test_body_size_validation() {
        assert!(validate_body_size(1000, 2000).is_ok());
        assert!(validate_body_size(2000, 2000).is_ok());
        assert!(validate_body_size(2001, 2000).is_err());
        assert!(validate_body_size(0, 1000).is_ok());
    }

    #[test]
    fn test_security_error_display() {
        let errors = vec![
            SecurityError::InvalidApiKey,
            SecurityError::InvalidSignature,
            SecurityError::ReplayAttack,
            SecurityError::RateLimitExceeded,
            SecurityError::InvalidInput("test".to_string()),
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
            assert!(!display.contains("Debug") && !display.contains("{"));
        }
    }

    #[test]
    fn test_filename_regex() {
        // Valid patterns
        assert!(FILENAME_REGEX.is_match("audio.wav"));
        assert!(FILENAME_REGEX.is_match("test_file-123.mp3"));
        assert!(FILENAME_REGEX.is_match("file.ext"));

        // Invalid patterns
        assert!(!FILENAME_REGEX.is_match("file with space.wav"));
        assert!(!FILENAME_REGEX.is_match("file@symbol.wav"));
        assert!(!FILENAME_REGEX.is_match("../path.wav"));
        assert!(!FILENAME_REGEX.is_match(""));
    }

    #[test]
    fn test_uuid_regex() {
        // Valid UUIDs
        assert!(UUID_REGEX.is_match("550e8400-e29b-41d4-a716-446655440000"));
        assert!(UUID_REGEX.is_match("6ba7b810-9dad-11d1-80b4-00c04fd430c8"));

        // Invalid UUIDs
        assert!(!UUID_REGEX.is_match("invalid-uuid"));
        assert!(!UUID_REGEX.is_match("550e8400e29b41d4a716446655440000")); // No dashes
        assert!(!UUID_REGEX.is_match("550e8400-e29b-41d4-a716")); // Too short
        assert!(!UUID_REGEX.is_match(""));
    }
}