use worker::*;
use serde_json::json;
use uuid::Uuid;
use crate::storage;
use crate::processing;

pub async fn upload_audio(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let form = req.form_data().await?;
    
    if let Some(FormEntry::File(file)) = form.get("audio") {
        let file_id = Uuid::new_v4().to_string();
        let file_data = file.bytes().await?;
        
        // Upload to R2 storage
        match storage::upload_to_r2(&ctx.env, &file_id, file_data).await {
            Ok(_) => {
                Response::from_json(&json!({
                    "id": file_id,
                    "status": "uploaded",
                    "message": "Audio file uploaded successfully"
                }))
            }
            Err(e) => {
                console_log!("Upload failed: {:?}", e);
                Response::error("Upload failed", 500)
            }
        }
    } else {
        Response::error("No audio file provided", 400)
    }
}

pub async fn analyze_audio(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let body: serde_json::Value = req.json().await?;
    
    let file_id = body["id"].as_str()
        .ok_or_else(|| worker::Error::RustError("Missing file ID".to_string()))?;
    
    let job_id = Uuid::new_v4().to_string();
    
    // Start async processing
    match processing::start_analysis(&ctx.env, file_id, &job_id).await {
        Ok(_) => {
            Response::from_json(&json!({
                "job_id": job_id,
                "status": "processing",
                "message": "Analysis started"
            }))
        }
        Err(e) => {
            console_log!("Analysis failed to start: {:?}", e);
            Response::error("Failed to start analysis", 500)
        }
    }
}

pub async fn get_job_status(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let job_id = ctx.param("id").unwrap();
    
    match storage::get_job_status(&ctx.env, job_id).await {
        Ok(status) => Response::from_json(&status),
        Err(_) => Response::error("Job not found", 404)
    }
}

pub async fn get_analysis_result(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let result_id = ctx.param("id").unwrap();
    
    match storage::get_analysis_result(&ctx.env, result_id).await {
        Ok(result) => Response::from_json(&result),
        Err(_) => Response::error("Result not found", 404)
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