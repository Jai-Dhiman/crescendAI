//! R2 storage service for audio file uploads.
//!
//! Handles uploading user audio files to Cloudflare R2 for inference.

use worker::{Bucket, console_log, HttpMetadata};

/// Maximum allowed file size (50MB)
pub const MAX_FILE_SIZE: usize = 50 * 1024 * 1024;

/// Allowed MIME types for audio uploads
pub const ALLOWED_MIME_TYPES: &[&str] = &[
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/webm",
];

/// Upload result containing the R2 key and public URL
#[derive(Debug, Clone)]
pub struct UploadResult {
    pub key: String,
    pub url: String,
}

/// Validate that the content type is an allowed audio format
pub fn validate_content_type(content_type: &str) -> bool {
    ALLOWED_MIME_TYPES.iter().any(|&allowed| {
        content_type.eq_ignore_ascii_case(allowed)
            || content_type.starts_with(allowed)
    })
}

/// Get file extension from content type
pub fn extension_from_content_type(content_type: &str) -> &'static str {
    if content_type.contains("mpeg") || content_type.contains("mp3") {
        "mp3"
    } else if content_type.contains("wav") {
        "wav"
    } else if content_type.contains("m4a") || content_type.contains("mp4") {
        "m4a"
    } else if content_type.contains("webm") {
        "webm"
    } else {
        "mp3" // default
    }
}

/// Generate a unique key for the upload
pub fn generate_upload_key(extension: &str) -> String {
    let timestamp = js_sys::Date::now() as u64;
    let random: u32 = (js_sys::Math::random() * 1_000_000.0) as u32;
    format!("user-uploads/{}-{}.{}", timestamp, random, extension)
}

/// Upload audio file to R2 bucket
///
/// Returns the R2 key and a URL that can be used to access the file.
pub async fn upload_audio(
    bucket: &Bucket,
    data: Vec<u8>,
    content_type: &str,
) -> Result<UploadResult, String> {
    // Validate file size
    if data.len() > MAX_FILE_SIZE {
        return Err(format!(
            "File too large: {} bytes (max {} bytes)",
            data.len(),
            MAX_FILE_SIZE
        ));
    }

    // Validate content type
    if !validate_content_type(content_type) {
        return Err(format!(
            "Invalid content type: {}. Allowed: {:?}",
            content_type, ALLOWED_MIME_TYPES
        ));
    }

    let extension = extension_from_content_type(content_type);
    let key = generate_upload_key(extension);

    console_log!("Uploading {} bytes to R2 key: {}", data.len(), key);

    // Upload to R2 with content type metadata
    let metadata = HttpMetadata {
        content_type: Some(content_type.to_string()),
        content_language: None,
        content_disposition: None,
        content_encoding: None,
        cache_control: None,
        cache_expiry: None,
    };

    bucket
        .put(&key, data)
        .http_metadata(metadata)
        .execute()
        .await
        .map_err(|e| format!("R2 upload failed: {:?}", e))?;

    console_log!("Upload complete: {}", key);

    // Generate URL (R2 public URL pattern)
    // Note: This assumes the bucket has a public domain configured
    // For Cloudflare R2, the URL pattern depends on your configuration:
    // - Custom domain: https://your-domain.com/{key}
    // - Workers binding: Accessed via bucket.get() not URL
    // For inference, we pass the R2 key and let the inference handler fetch it
    let url = format!("/r2/{}", key);

    Ok(UploadResult { key, url })
}

/// Delete an uploaded file from R2
#[allow(dead_code)]
pub async fn delete_upload(bucket: &Bucket, key: &str) -> Result<(), String> {
    bucket
        .delete(key)
        .await
        .map_err(|e| format!("R2 delete failed: {:?}", e))
}

/// Get audio file from R2 bucket
pub async fn get_audio(bucket: &Bucket, key: &str) -> Result<Vec<u8>, String> {
    let object = bucket
        .get(key)
        .execute()
        .await
        .map_err(|e| format!("R2 get failed: {:?}", e))?;

    match object {
        Some(obj) => {
            let body = obj
                .body()
                .ok_or_else(|| "Object has no body".to_string())?;
            body.bytes()
                .await
                .map_err(|e| format!("Failed to read body: {:?}", e))
        }
        None => Err("Object not found".to_string()),
    }
}
