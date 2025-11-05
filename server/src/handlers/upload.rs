//! Upload Handler
//!
//! Handles audio file uploads to R2 storage and metadata persistence in D1.

use worker::*;
use serde::{Deserialize, Serialize};
use crate::db::recordings;

// ============================================================================
// Constants
// ============================================================================

const MAX_FILE_SIZE: usize = 50 * 1024 * 1024; // 50MB
const ALLOWED_MIME_TYPES: &[&str] = &["audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a"];
const ALLOWED_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a"];

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct UploadMetadata {
    #[serde(rename = "originalName")]
    original_name: String,
    size: usize,
    #[serde(rename = "type")]
    mime_type: String,
    hash: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub id: String,
    pub status: String,
    pub message: String,
    pub original_filename: String,
}

// ============================================================================
// Main Upload Handler
// ============================================================================

/// POST /api/v1/upload - Upload audio file
pub async fn upload_handler(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    console_log!("Upload request received");

    // Get user_id from header or default for now (auth will be added later)
    let user_id = req.headers()
        .get("X-User-Id")?
        .unwrap_or_else(|| "default_user".to_string());

    // Parse multipart form data
    let form = match req.form_data().await {
        Ok(form) => form,
        Err(e) => {
            console_error!("Failed to parse form data: {:?}", e);
            return Response::error("Invalid form data", 400);
        }
    };

    // Extract audio file
    let audio_file = match form.get("audio") {
        Some(FormEntry::File(file)) => file,
        _ => {
            console_error!("No audio file in form data");
            return Response::error("Missing audio file", 400);
        }
    };

    // Extract metadata
    let metadata_json = match form.get("metadata") {
        Some(FormEntry::Field(json)) => json,
        _ => {
            console_error!("No metadata in form data");
            return Response::error("Missing metadata", 400);
        }
    };

    let metadata: UploadMetadata = match serde_json::from_str(&metadata_json) {
        Ok(meta) => meta,
        Err(e) => {
            console_error!("Failed to parse metadata: {:?}", e);
            return Response::error("Invalid metadata format", 400);
        }
    };

    // Validate file
    if let Err(msg) = validate_file(&audio_file, &metadata).await {
        console_error!("Validation failed: {}", msg);
        return Response::error(&msg, 400);
    }

    // Get file bytes
    let file_bytes = match audio_file.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            console_error!("Failed to read file bytes: {:?}", e);
            return Response::error("Failed to read file", 500);
        }
    };

    // Generate file extension from metadata
    let extension = metadata.mime_type.split('/').last().unwrap_or("wav");
    let extension = match extension {
        "mpeg" => "mp3",
        "x-m4a" => "m4a",
        ext => ext,
    };

    // Upload to R2
    let recording_id = crate::db::generate_id();
    let r2_key = format!("recordings/{}/{}.{}", user_id, recording_id, extension);

    if let Err(e) = upload_to_r2(&ctx.env, &r2_key, &file_bytes).await {
        console_error!("Failed to upload to R2: {:?}", e);
        return Response::error("Failed to store file", 500);
    }

    // Store metadata in D1
    let recording = match recordings::insert_recording(
        &ctx.env,
        &user_id,
        &metadata.original_name,
        metadata.size as i64,
        None, // Duration will be set after analysis
        &metadata.mime_type,
        &r2_key,
    ).await {
        Ok(rec) => rec,
        Err(e) => {
            console_error!("Failed to store recording metadata: {:?}", e);
            // Try to clean up R2 file
            let _ = delete_from_r2(&ctx.env, &r2_key).await;
            return Response::error("Failed to save recording metadata", 500);
        }
    };

    console_log!("Upload successful: recording_id={}", recording.id);

    // Return success response
    let response = UploadResponse {
        id: recording.id,
        status: "uploaded".to_string(),
        message: "Recording uploaded successfully".to_string(),
        original_filename: metadata.original_name,
    };

    Response::from_json(&response)
}

// ============================================================================
// Validation
// ============================================================================

async fn validate_file(_file: &File, metadata: &UploadMetadata) -> std::result::Result<(), String> {
    // Validate file size
    if metadata.size > MAX_FILE_SIZE {
        return Err(format!(
            "File too large. Maximum size is {} MB",
            MAX_FILE_SIZE / (1024 * 1024)
        ));
    }

    if metadata.size == 0 {
        return Err("File is empty".to_string());
    }

    // Validate MIME type
    if !ALLOWED_MIME_TYPES.contains(&metadata.mime_type.as_str()) {
        return Err(format!(
            "Invalid file type. Allowed types: {}",
            ALLOWED_MIME_TYPES.join(", ")
        ));
    }

    // Validate file extension from original name
    if let Some(ext) = metadata.original_name.split('.').last() {
        let ext_lower = ext.to_lowercase();
        if !ALLOWED_EXTENSIONS.contains(&ext_lower.as_str()) {
            return Err(format!(
                "Invalid file extension. Allowed extensions: {}",
                ALLOWED_EXTENSIONS.join(", ")
            ));
        }
    } else {
        return Err("File must have an extension".to_string());
    }

    Ok(())
}

// ============================================================================
// R2 Operations
// ============================================================================

async fn upload_to_r2(env: &Env, key: &str, data: &[u8]) -> Result<()> {
    let bucket = env.bucket("crescendai_bucket")
        .map_err(|e| {
            console_error!("Failed to get R2 bucket binding: {:?}", e);
            worker::Error::RustError(format!("R2 bucket not configured: {}", e))
        })?;

    bucket.put(key, data.to_vec()).execute().await?;

    console_log!("Uploaded to R2: key={}, size={}", key, data.len());

    Ok(())
}

async fn delete_from_r2(env: &Env, key: &str) -> Result<()> {
    let bucket = env.bucket("crescendai_bucket")?;
    bucket.delete(key).await?;

    console_log!("Deleted from R2: key={}", key);

    Ok(())
}

// ============================================================================
// Recording Management Endpoints
// ============================================================================

/// GET /api/v1/recordings/:id - Get recording metadata
pub async fn get_recording_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get user_id from header
    let user_id = req.headers()
        .get("X-User-Id")?
        .unwrap_or_else(|| "default_user".to_string());

    // Get recording ID from path
    let recording_id = match ctx.param("id") {
        Some(id) => id,
        None => return Response::error("Missing recording ID", 400),
    };

    // Fetch recording from D1
    let recording = match recordings::get_recording(&ctx.env, &recording_id).await {
        Ok(rec) => rec,
        Err(crate::db::DbError::NotFound(_)) => {
            return Response::error("Recording not found", 404);
        }
        Err(e) => {
            console_error!("Failed to get recording: {:?}", e);
            return Response::error("Failed to retrieve recording", 500);
        }
    };

    // Verify ownership
    if recording.user_id != user_id {
        return Response::error("Unauthorized", 403);
    }

    Response::from_json(&recording)
}

#[derive(Debug, Serialize)]
struct ListRecordingsResponse {
    recordings: Vec<recordings::Recording>,
    total: u32,
    page: u32,
    limit: u32,
}

/// GET /api/v1/recordings - List user recordings
pub async fn list_recordings_handler(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    // Get user_id from header
    let user_id = req.headers()
        .get("X-User-Id")?
        .unwrap_or_else(|| "default_user".to_string());

    // Parse query parameters
    let url = req.url()?;
    let query_pairs = url.query_pairs();

    let mut page = 1u32;
    let mut limit = 20u32;
    let mut status_filter: Option<String> = None;
    let mut date_from: Option<String> = None;
    let mut date_to: Option<String> = None;
    let mut sort_by = "created_at".to_string();
    let mut order = "desc".to_string();

    for (key, value) in query_pairs {
        match key.as_ref() {
            "page" => page = value.parse().unwrap_or(1),
            "limit" => limit = value.parse().unwrap_or(20),
            "status" => status_filter = Some(value.to_string()),
            "date_from" => date_from = Some(value.to_string()),
            "date_to" => date_to = Some(value.to_string()),
            "sort_by" => sort_by = value.to_string(),
            "order" => order = value.to_string(),
            _ => {}
        }
    }

    // Cap limit at 100
    limit = limit.min(100);

    // Validate status filter
    if let Some(ref status) = status_filter {
        if !["uploaded", "processing", "analyzed", "failed"].contains(&status.as_str()) {
            return Response::error("Invalid status filter. Allowed: uploaded, processing, analyzed, failed", 400);
        }
    }

    // Validate sort_by
    if !["created_at", "updated_at", "filename", "file_size", "duration"].contains(&sort_by.as_str()) {
        return Response::error("Invalid sort_by. Allowed: created_at, updated_at, filename, file_size, duration", 400);
    }

    // Validate order
    if !["asc", "desc"].contains(&order.to_lowercase().as_str()) {
        return Response::error("Invalid order. Allowed: asc, desc", 400);
    }

    let offset = (page - 1) * limit;

    // Fetch recordings with filters
    let recordings_list = match recordings::list_recordings_filtered(
        &ctx.env,
        &user_id,
        status_filter.as_deref(),
        date_from.as_deref(),
        date_to.as_deref(),
        &sort_by,
        &order,
        Some(limit),
        Some(offset),
    ).await {
        Ok(list) => list,
        Err(e) => {
            console_error!("Failed to list recordings: {:?}", e);
            return Response::error("Failed to retrieve recordings", 500);
        }
    };

    // Get total count with filters
    let total = match recordings::count_recordings_filtered(
        &ctx.env,
        &user_id,
        status_filter.as_deref(),
        date_from.as_deref(),
        date_to.as_deref(),
    ).await {
        Ok(count) => count,
        Err(e) => {
            console_error!("Failed to count recordings: {:?}", e);
            return Response::error("Failed to count recordings", 500);
        }
    };

    let response = ListRecordingsResponse {
        recordings: recordings_list,
        total,
        page,
        limit,
    };

    Response::from_json(&response)
}
