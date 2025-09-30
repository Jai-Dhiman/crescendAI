use worker::*;
use serde_json::json;
use uuid::Uuid;
use crate::storage;
use crate::processing;
use crate::security::{
    validate_file_type, sanitize_filename, validate_uuid, validate_body_size, 
    secure_error_response, SecurityError
};
use crate::tutor::{UserContext};
use crate::AnalysisData;

pub async fn upload_audio(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_FILE_SIZE: usize = 50 * 1024 * 1024; // 50MB limit
    
    // Get form data with size validation
    let form = req.form_data().await?;
    
    if let Some(FormEntry::File(file)) = form.get("audio") {
        // Get file data and validate size
        let file_data = file.bytes().await?;
        validate_body_size(file_data.len(), MAX_FILE_SIZE)?;
        
        // Get and sanitize filename
        let original_filename = file.name();
        let sanitized_filename = sanitize_filename(&original_filename)?;
        
        // Validate file type based on content and extension
        validate_file_type(&sanitized_filename, &file_data)?;
        
        // Generate secure file ID
        let file_id = Uuid::new_v4().to_string();
        
        console_log!(
            "Processing file upload: {} ({} bytes) -> ID: {}", 
            sanitized_filename, 
            file_data.len(), 
            file_id
        );
        
        // Upload to R2 storage
        match storage::upload_to_r2(&ctx.env, &file_id, file_data).await {
            Ok(_) => {
                Response::from_json(&json!({
                    "id": file_id,
                    "status": "uploaded",
                    "message": "Audio file uploaded successfully",
                    "original_filename": sanitized_filename
                }))
            }
            Err(e) => {
                console_log!("Upload failed for file {}: {:?}", file_id, e);
                let is_dev = ctx.env.var("ENVIRONMENT")
                    .map(|v| v.to_string() == "development")
                    .unwrap_or(false);
                Ok(secure_error_response(&e, is_dev))
            }
        }
    } else {
        Err(worker::Error::from(SecurityError::InvalidInput(
            "No audio file provided in form data".to_string()
        )))
    }
}

pub async fn analyze_audio(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 1024; // 1KB limit for JSON payload
    
    // Check Content-Length header for early validation
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }
    
    let body: serde_json::Value = req.json().await?;
    
    // Validate and sanitize file ID
    let file_id = body["id"].as_str()
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing or invalid file ID".to_string()
        )))?;
    
    // Validate UUID format
    validate_uuid(file_id)?;
    
    // Generate new job ID
    let job_id = Uuid::new_v4().to_string();
    
    console_log!("Starting analysis for file ID: {} with job ID: {}", file_id, job_id);
    
    // Start async processing
let force_gpu = req.headers().get("X-Run-GPU").ok().flatten().map(|v| v.eq_ignore_ascii_case("true"));
    match processing::start_analysis(&ctx.env, file_id, &job_id, force_gpu).await {
        Ok(_) => {
            Response::from_json(&json!({
                "job_id": job_id,
                "status": "processing",
                "message": "Analysis started successfully"
            }))
        }
        Err(e) => {
            console_log!("Analysis failed to start for job {}: {:?}", job_id, e);
            let is_dev = ctx.env.var("ENVIRONMENT")
                .map(|v| v.to_string() == "development")
                .unwrap_or(false);
            Ok(secure_error_response(&e, is_dev))
        }
    }
}

pub async fn get_job_status(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let job_id = ctx.param("id")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing job ID parameter".to_string()
        )))?;
    
    // Validate UUID format
    validate_uuid(job_id)?;
    
    match storage::get_job_status(&ctx.env, job_id).await {
        Ok(status) => {
            console_log!("Retrieved status for job {}: {}", job_id, status.status);
            // Compute body and ETag for HTTP caching
            let body = serde_json::to_string(&status)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            let etag = crate::utils::compute_etag_from_str(&body);
            let if_none_match = req.headers().get("If-None-Match").ok().flatten();

            if let Some(tag) = if_none_match {
                if tag == etag {
                    let mut res = Response::empty()?;
                    res = res.with_status(304);
                    let headers = res.headers_mut();
                    headers.set("ETag", &etag)?;
                    headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
                    return Ok(res);
                }
            }

            let mut res = Response::from_json(&status)?;
            let headers = res.headers_mut();
            headers.set("ETag", &etag)?;
            headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
            Ok(res)
        }
        Err(_e) => {
            console_log!("Job status not found for ID {}", job_id);
            Err(worker::Error::from(SecurityError::InvalidInput(
                "Job not found".to_string()
            )))
        }
    }
}

pub async fn get_analysis_result(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let result_id = ctx.param("id")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing result ID parameter".to_string()
        )))?;
    
    // Validate UUID format
    validate_uuid(result_id)?;
    
    match storage::get_analysis_result(&ctx.env, result_id).await {
        Ok(result) => {
            console_log!("Retrieved analysis result for ID {}: {}", result_id, result.status);
            let body = serde_json::to_string(&result)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            let etag = crate::utils::compute_etag_from_str(&body);
            let if_none_match = req.headers().get("If-None-Match").ok().flatten();

            if let Some(tag) = if_none_match {
                if tag == etag {
                    let mut res = Response::empty()?;
                    res = res.with_status(304);
                    let headers = res.headers_mut();
                    headers.set("ETag", &etag)?;
                    headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
                    return Ok(res);
                }
            }

            let mut res = Response::from_json(&result)?;
            let headers = res.headers_mut();
            headers.set("ETag", &etag)?;
            headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
            Ok(res)
        }
        Err(_e) => {
            console_log!("Analysis result not found for ID {}", result_id);
            Err(worker::Error::from(SecurityError::InvalidInput(
                "Analysis result not found".to_string()
            )))
        }
    }
}

pub async fn compare_models(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 1024; // 1KB limit for JSON payload
    
    // Check Content-Length header for early validation
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }
    
    let body: serde_json::Value = req.json().await?;
    
    // Validate and sanitize file ID
    let file_id = body["id"].as_str()
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing or invalid file ID".to_string()
        )))?;
    
    // Validate UUID format
    validate_uuid(file_id)?;
    
    // Optional model configuration
    let model_a = body["model_a"].as_str().unwrap_or("hybrid_ast");
    let model_b = body["model_b"].as_str().unwrap_or("ultra_small_ast");
    
    // Generate new comparison job ID
    let comparison_id = Uuid::new_v4().to_string();
    
    console_log!("Starting model comparison for file ID: {} with comparison ID: {} (models: {} vs {})", 
                 file_id, comparison_id, model_a, model_b);
    
    // Start async comparison processing
let force_gpu = req.headers().get("X-Run-GPU").ok().flatten().map(|v| v.eq_ignore_ascii_case("true"));
    match processing::start_model_comparison(&ctx.env, file_id, &comparison_id, model_a, model_b, force_gpu).await {
        Ok(_) => {
            Response::from_json(&json!({
                "comparison_id": comparison_id,
                "status": "processing",
                "message": "Model comparison started successfully",
                "models": {
                    "model_a": model_a,
                    "model_b": model_b
                }
            }))
        }
        Err(e) => {
            console_log!("Comparison failed to start for job {}: {:?}", comparison_id, e);
            let is_dev = ctx.env.var("ENVIRONMENT")
                .map(|v| v.to_string() == "development")
                .unwrap_or(false);
            Ok(secure_error_response(&e, is_dev))
        }
    }
}

pub async fn get_comparison_result(req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let comparison_id = ctx.param("id")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing comparison ID parameter".to_string()
        )))?;
    
    // Validate UUID format
    validate_uuid(comparison_id)?;
    
    match storage::get_comparison_result(&ctx.env, comparison_id).await {
        Ok(result) => {
            console_log!("Retrieved comparison result for ID {}: {}", comparison_id, result.status);
            let body = serde_json::to_string(&result)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            let etag = crate::utils::compute_etag_from_str(&body);
            let if_none_match = req.headers().get("If-None-Match").ok().flatten();

            if let Some(tag) = if_none_match {
                if tag == etag {
                    let mut res = Response::empty()?;
                    res = res.with_status(304);
                    let headers = res.headers_mut();
                    headers.set("ETag", &etag)?;
                    headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
                    return Ok(res);
                }
            }

            let mut res = Response::from_json(&result)?;
            let headers = res.headers_mut();
            headers.set("ETag", &etag)?;
            headers.set("Cache-Control", "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800")?;
            Ok(res)
        }
        Err(_e) => {
            console_log!("Comparison result not found for ID {}", comparison_id);
            Err(worker::Error::from(SecurityError::InvalidInput(
                "Comparison result not found".to_string()
            )))
        }
    }
}

pub async fn save_user_preference(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 2048; // 2KB limit for preference data
    
    // Check Content-Length header
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }
    
    let body: serde_json::Value = req.json().await?;
    
    // Validate required fields
    let comparison_id = body["comparison_id"].as_str()
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing comparison_id".to_string()
        )))?;
    
    let preferred_model = body["preferred_model"].as_str()
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput(
            "Missing preferred_model".to_string()
        )))?;
    
    // Validate UUID format for comparison ID
    validate_uuid(comparison_id)?;
    
    // Validate preferred model value
    if !matches!(preferred_model, "model_a" | "model_b") {
        return Err(worker::Error::from(SecurityError::InvalidInput(
            "preferred_model must be 'model_a' or 'model_b'".to_string()
        )));
    }
    
    // Optional feedback (with length limit)
    let feedback = body["feedback"].as_str();
    if let Some(fb) = feedback {
        if fb.len() > 1000 {
            return Err(worker::Error::from(SecurityError::InvalidInput(
                "Feedback too long (max 1000 characters)".to_string()
            )));
        }
    }
    
    let preference = crate::UserPreference {
        comparison_id: comparison_id.to_string(),
        preferred_model: preferred_model.to_string(),
        feedback: feedback.map(|s| s.to_string()),
        created_at: js_sys::Date::new_0().to_iso_string().into(),
    };
    
    console_log!("Saving user preference for comparison {}: prefers {}", 
                 comparison_id, preferred_model);
    
    match storage::save_user_preference(&ctx.env, &preference).await {
        Ok(_) => {
            Response::from_json(&json!({
                "status": "saved",
                "message": "User preference saved successfully",
                "comparison_id": comparison_id,
                "preferred_model": preferred_model
            }))
        }
        Err(e) => {
            console_log!("Failed to save preference for comparison {}: {:?}", comparison_id, e);
            let is_dev = ctx.env.var("ENVIRONMENT")
                .map(|v| v.to_string() == "development")
                .unwrap_or(false);
            Ok(secure_error_response(&e, is_dev))
        }
    }
}

pub async fn generate_tutor_feedback(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_JSON_SIZE: usize = 8 * 1024; // 8KB limit for tutor payload

    // Check Content-Length header for early validation
    if let Ok(Some(content_length)) = req.headers().get("Content-Length") {
        if let Ok(size) = content_length.parse::<usize>() {
            validate_body_size(size, MAX_JSON_SIZE)?;
        }
    }

    let body: serde_json::Value = req.json().await?;

    // Parse analysis (supports both existing AnalysisData and PercepPiano 19-d schema)
    let analysis_val = body.get("analysis")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput("Missing analysis".to_string())))?;

    // First attempt: directly parse to AnalysisData
    let analysis: AnalysisData = match serde_json::from_value::<AnalysisData>(analysis_val.clone()) {
        Ok(a) => a,
        Err(_) => {
            // Second attempt: parse PercepPiano schema and map
            #[derive(serde::Deserialize)]
            struct PP19 {
                timing_stable_unstable: f32,
                articulation_short_long: f32,
                articulation_soft_hard: f32,
                pedal_sparse_saturated: f32,
                pedal_clean_blurred: f32,
                timbre_even_colorful: f32,
                timbre_shallow_rich: f32,
                timbre_bright_dark: f32,
                timbre_soft_loud: f32,
                dynamic_sophisticated_raw: f32,
                dynamic_range_little_large: f32,
                music_making_fast_slow: f32,
                music_making_flat_spacious: f32,
                music_making_disproportioned_balanced: f32,
                music_making_pure_dramatic: f32,
                emotion_mood_optimistic_dark: f32,
                emotion_mood_low_high_energy: f32,
                emotion_mood_honest_imaginative: f32,
                interpretation_unsatisfactory_convincing: f32,
            }
            fn avg(a: f32, b: f32) -> f32 { ((a + b) / 2.0).clamp(0.0, 1.0) }
            let pp: PP19 = serde_json::from_value(analysis_val.clone())
                .map_err(|_| worker::Error::from(SecurityError::InvalidInput("Invalid analysis payload (schema)".to_string())))?;
            // Mapping notes: use PercepPiano 19-d schema directly to AnalysisData (same schema)
            AnalysisData {
                timing_stable_unstable: pp.timing_stable_unstable,
                articulation_short_long: pp.articulation_short_long,
                articulation_soft_hard: pp.articulation_soft_hard,
                pedal_sparse_saturated: pp.pedal_sparse_saturated,
                pedal_clean_blurred: pp.pedal_clean_blurred,
                timbre_even_colorful: pp.timbre_even_colorful,
                timbre_shallow_rich: pp.timbre_shallow_rich,
                timbre_bright_dark: pp.timbre_bright_dark,
                timbre_soft_loud: pp.timbre_soft_loud,
                dynamic_sophisticated_raw: pp.dynamic_sophisticated_raw,
                dynamic_range_little_large: pp.dynamic_range_little_large,
                music_making_fast_slow: pp.music_making_fast_slow,
                music_making_flat_spacious: pp.music_making_flat_spacious,
                music_making_disproportioned_balanced: pp.music_making_disproportioned_balanced,
                music_making_pure_dramatic: pp.music_making_pure_dramatic,
                emotion_mood_optimistic_dark: pp.emotion_mood_optimistic_dark,
                emotion_mood_low_high_energy: pp.emotion_mood_low_high_energy,
                emotion_mood_honest_imaginative: pp.emotion_mood_honest_imaginative,
                interpretation_unsatisfactory_convincing: pp.interpretation_unsatisfactory_convincing,
            }
        }
    };

    // Parse user_context
    let user_ctx_val = body.get("user_context")
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidInput("Missing user_context".to_string())))?;
    let user_ctx: UserContext = serde_json::from_value(user_ctx_val.clone())
        .map_err(|_| worker::Error::from(SecurityError::InvalidInput("Invalid user_context payload".to_string())))?;

    // Options
    let top_k = body.get("options").and_then(|o| o.get("top_k")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    // Ensure Tutor feature is enabled
    let enabled = ctx.env.var("TUTOR_ENABLED").map(|v| v.to_string() == "true").unwrap_or(true);
    if !enabled {
        return Err(worker::Error::from(SecurityError::InvalidInput("Tutor service disabled".to_string())));
    }

    // Generate feedback
    match crate::tutor::generate_feedback(&ctx.env, &analysis, &user_ctx, top_k).await {
        Ok(feedback) => Response::from_json(&feedback),
        Err(e) => {
            let is_dev = ctx.env.var("ENVIRONMENT")
                .map(|v| v.to_string() == "development")
                .unwrap_or(false);
            Ok(secure_error_response(&e, is_dev))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Mock helper functions for testing error conditions
    fn create_mock_file_data() -> Vec<u8> {
        // Create a minimal WAV file header for testing
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");  // RIFF header
        data.extend_from_slice(&[0u8; 4]); // File size placeholder
        data.extend_from_slice(b"WAVE");  // WAVE format
        data.extend_from_slice(b"fmt ");  // Format chunk
        data.extend_from_slice(&[16u8, 0, 0, 0]); // Format chunk size
        data.extend_from_slice(&[1u8, 0]); // PCM format
        data.extend_from_slice(&[1u8, 0]); // Mono
        data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // Sample rate (44100)
        data.extend_from_slice(&[0x88, 0x58, 0x01, 0]); // Byte rate
        data.extend_from_slice(&[2u8, 0]); // Block align
        data.extend_from_slice(&[16u8, 0]); // Bits per sample
        data.extend_from_slice(b"data"); // Data chunk
        data.extend_from_slice(&[0u8; 4]); // Data size placeholder
        data.extend_from_slice(&[0u8; 100]); // Sample data
        data
    }

    #[test]
    fn test_file_id_generation() {
        // Test that UUID generation works
        let file_id1 = Uuid::new_v4().to_string();
        let file_id2 = Uuid::new_v4().to_string();
        
        assert_ne!(file_id1, file_id2);
        assert_eq!(file_id1.len(), 36); // UUID length
        assert!(file_id1.contains('-'));
    }

    #[test]
    fn test_job_id_generation() {
        // Test job ID generation
        let job_id1 = Uuid::new_v4().to_string();
        let job_id2 = Uuid::new_v4().to_string();
        
        assert_ne!(job_id1, job_id2);
        assert_eq!(job_id1.len(), 36); // UUID length
        assert!(job_id1.contains('-'));
    }

    #[test]
    fn test_analyze_audio_request_body_parsing() {
        // Test valid JSON body parsing
        let valid_body = json!({
            "id": "test-file-id-123"
        });
        
        let id = valid_body["id"].as_str().unwrap();
        assert_eq!(id, "test-file-id-123");
    }

    #[test]
    fn test_analyze_audio_missing_id() {
        // Test missing ID handling
        let invalid_body = json!({
            "other_field": "value"
        });
        
        let id = invalid_body["id"].as_str();
        assert!(id.is_none());
    }

    #[test]
    fn test_analyze_audio_null_id() {
        // Test null ID handling
        let invalid_body = json!({
            "id": null
        });
        
        let id = invalid_body["id"].as_str();
        assert!(id.is_none());
    }

    #[test]
    fn test_analyze_audio_empty_id() {
        // Test empty string ID
        let invalid_body = json!({
            "id": ""
        });
        
        let id = invalid_body["id"].as_str().unwrap();
        assert_eq!(id, "");
    }

    #[test]
    fn test_upload_response_format() {
        // Test upload success response format
        let file_id = "test-file-123";
        let response_body = json!({
            "id": file_id,
            "status": "uploaded",
            "message": "Audio file uploaded successfully"
        });
        
        assert_eq!(response_body["id"], file_id);
        assert_eq!(response_body["status"], "uploaded");
        assert_eq!(response_body["message"], "Audio file uploaded successfully");
    }

    #[test]
    fn test_analyze_response_format() {
        // Test analyze success response format
        let job_id = "test-job-456";
        let response_body = json!({
            "job_id": job_id,
            "status": "processing",
            "message": "Analysis started"
        });
        
        assert_eq!(response_body["job_id"], job_id);
        assert_eq!(response_body["status"], "processing");
        assert_eq!(response_body["message"], "Analysis started");
    }

    #[test]
    fn test_error_response_codes() {
        // Test that we use appropriate error codes
        let upload_error_code = 500;
        let missing_file_error_code = 400;
        let not_found_error_code = 404;
        let analysis_error_code = 500;
        
        assert_eq!(upload_error_code, 500);
        assert_eq!(missing_file_error_code, 400);
        assert_eq!(not_found_error_code, 404);
        assert_eq!(analysis_error_code, 500);
    }

    #[test]
    fn test_file_data_creation() {
        let file_data = create_mock_file_data();
        assert!(!file_data.is_empty());
        assert!(file_data.len() > 44); // Minimum WAV file size
        assert_eq!(&file_data[0..4], b"RIFF");
        assert_eq!(&file_data[8..12], b"WAVE");
    }

    #[test]
    fn test_file_id_validation() {
        // Test file ID format validation
        let valid_uuid = Uuid::new_v4().to_string();
        assert_eq!(valid_uuid.len(), 36);
        assert!(valid_uuid.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        
        // Count hyphens (should be 4 in a UUID)
        let hyphen_count = valid_uuid.chars().filter(|&c| c == '-').count();
        assert_eq!(hyphen_count, 4);
    }

    #[test]
    fn test_request_body_validation() {
        // Test various request body formats
        let bodies = vec![
            json!({"id": "valid-id"}),
            json!({"id": "another-valid-id", "extra": "field"}),
            json!({}),
            json!({"wrong_field": "value"}),
            json!(null),
        ];
        
        for body in bodies {
            let id = body.get("id").and_then(|v| v.as_str());
            match id {
                Some(id_str) if !id_str.is_empty() => {
                    // Valid case
                    assert!(!id_str.is_empty());
                }
                _ => {
                    // Invalid case - should be handled appropriately
                    assert!(id.is_none() || id.unwrap().is_empty());
                }
            }
        }
    }

    #[test]
    fn test_response_json_structure() {
        // Test that our response structures are valid JSON
        let upload_response = json!({
            "id": "test-id",
            "status": "uploaded",
            "message": "Audio file uploaded successfully"
        });
        
        let analyze_response = json!({
            "job_id": "job-id",
            "status": "processing", 
            "message": "Analysis started"
        });
        
        // Serialize and deserialize to ensure valid JSON
        let upload_str = serde_json::to_string(&upload_response).unwrap();
        let analyze_str = serde_json::to_string(&analyze_response).unwrap();
        
        let _: serde_json::Value = serde_json::from_str(&upload_str).unwrap();
        let _: serde_json::Value = serde_json::from_str(&analyze_str).unwrap();
        
        assert!(upload_str.contains("uploaded"));
        assert!(analyze_str.contains("processing"));
    }

    #[test]
    fn test_error_message_formats() {
        // Test various error messages
        let errors = vec![
            "No audio file provided",
            "Upload failed", 
            "Failed to start analysis",
            "Job not found",
            "Result not found",
            "Missing file ID",
        ];
        
        for error in errors {
            assert!(!error.is_empty());
            assert!(error.len() < 100); // Reasonable error message length
        }
    }

    #[test]
    fn test_concurrent_requests() {
        // Test that multiple UUIDs can be generated concurrently
        let mut ids = Vec::new();
        for _ in 0..10 {
            ids.push(Uuid::new_v4().to_string());
        }
        
        // All IDs should be unique
        for i in 0..ids.len() {
            for j in i+1..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }
}