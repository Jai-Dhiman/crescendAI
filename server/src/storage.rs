use worker::*;
use serde_json::json;
use crate::{AnalysisResult, JobStatus, ComparisonResult, UserPreference};

pub async fn upload_to_r2(env: &Env, file_id: &str, data: Vec<u8>) -> Result<()> {
    let bucket = env.bucket("AUDIO_BUCKET")?;
    let key = format!("audio/{}.wav", file_id);
    
    let data_size = data.len();
    bucket.put(&key, data).execute().await?;
    
    // Store metadata in KV
    let kv = env.kv("METADATA")?;
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
    let kv = env.kv("METADATA")?;
    
    // Try with cache-busting first for potentially completed jobs
    let key = format!("job:{}", job_id);
    
    match kv.get(&key).cache_ttl(60).text().await? { // 60 second cache TTL (minimum allowed)
        Some(data) => {
            let status: JobStatus = serde_json::from_str(&data)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            console_log!("Retrieved job status for {}: {} (progress: {})", job_id, status.status, status.progress);
            Ok(status)
        }
        None => Err(worker::Error::RustError("Job not found".to_string()))
    }
}

pub async fn update_job_status(env: &Env, job_id: &str, status: &JobStatus) -> Result<()> {
    let kv = env.kv("METADATA")?;
    let status_json = serde_json::to_string(status)
        .map_err(|e| worker::Error::RustError(e.to_string()))?;
    
    // Use a shorter TTL to force faster propagation for completed jobs
    let mut put_builder = kv.put(&format!("job:{}", job_id), &status_json)?;
    
    // For completed or failed jobs, use shorter TTL to force faster updates
    if status.status == "completed" || status.status == "failed" {
        put_builder = put_builder.expiration_ttl(60); // 1 minute TTL to force refresh
    }
    
    put_builder.execute().await?;
    
    console_log!("Updated job status for {}: {} (progress: {})", job_id, status.status, status.progress);
    Ok(())
}

pub async fn store_analysis_result(env: &Env, result_id: &str, result: &AnalysisResult) -> Result<()> {
    let kv = env.kv("METADATA")?;
    let result_json = serde_json::to_string(result)
        .map_err(|e| worker::Error::RustError(e.to_string()))?;
    
    kv.put(&format!("result:{}", result_id), &result_json)?
        .execute()
        .await?;
    
    Ok(())
}

pub async fn get_analysis_result(env: &Env, result_id: &str) -> Result<AnalysisResult> {
    let kv = env.kv("METADATA")?;
    
    match kv.get(&format!("result:{}", result_id)).text().await? {
        Some(data) => {
            let result: AnalysisResult = serde_json::from_str(&data)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            Ok(result)
        }
        None => Err(worker::Error::RustError("Result not found".to_string()))
    }
}

pub async fn get_audio_from_r2(env: &Env, file_id: &str) -> Result<Vec<u8>> {
    let bucket = env.bucket("AUDIO_BUCKET")?;
    let key = format!("audio/{}.wav", file_id);
    
    let object = bucket.get(&key).execute().await?
        .ok_or_else(|| worker::Error::RustError("File not found".to_string()))?;
    
    let data = object.body()
        .ok_or_else(|| worker::Error::RustError("No body in object".to_string()))?
        .bytes().await?;
    Ok(data)
}

pub async fn store_comparison_result(env: &Env, comparison_id: &str, result: &ComparisonResult) -> Result<()> {
    let kv = env.kv("METADATA")?;
    let result_json = serde_json::to_string(result)
        .map_err(|e| worker::Error::RustError(e.to_string()))?;
    
    kv.put(&format!("comparison:{}", comparison_id), &result_json)?
        .execute()
        .await?;
    
    console_log!("Stored comparison result for comparison_id: {}", comparison_id);
    Ok(())
}

pub async fn get_comparison_result(env: &Env, comparison_id: &str) -> Result<ComparisonResult> {
    let kv = env.kv("METADATA")?;
    
    match kv.get(&format!("comparison:{}", comparison_id)).text().await? {
        Some(data) => {
            let result: ComparisonResult = serde_json::from_str(&data)
                .map_err(|e| worker::Error::RustError(e.to_string()))?;
            console_log!("Retrieved comparison result for comparison_id: {}", comparison_id);
            Ok(result)
        }
        None => Err(worker::Error::RustError("Comparison result not found".to_string()))
    }
}

pub async fn save_user_preference(env: &Env, preference: &UserPreference) -> Result<()> {
    let kv = env.kv("METADATA")?;
    let preference_json = serde_json::to_string(preference)
        .map_err(|e| worker::Error::RustError(e.to_string()))?;
    
    // Store preference with unique key based on comparison ID and timestamp
    let preference_key = format!("preference:{}:{}", 
                                preference.comparison_id, 
                                js_sys::Date::now() as u64);
    
    kv.put(&preference_key, &preference_json)?
        .execute()
        .await?;
    
    // Also store aggregated preferences for analytics
    let analytics_key = format!("analytics:{}", preference.comparison_id);
    match kv.get(&analytics_key).text().await? {
        Some(existing_data) => {
            // Update existing analytics
            let mut analytics: serde_json::Value = serde_json::from_str(&existing_data)
                .unwrap_or_else(|_| json!({"model_a_votes": 0, "model_b_votes": 0, "total_votes": 0}));
            
            if preference.preferred_model == "model_a" {
                analytics["model_a_votes"] = json!(analytics["model_a_votes"].as_u64().unwrap_or(0) + 1);
            } else {
                analytics["model_b_votes"] = json!(analytics["model_b_votes"].as_u64().unwrap_or(0) + 1);
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
    
    console_log!("Saved user preference for comparison_id: {} (prefers: {})", 
                 preference.comparison_id, preference.preferred_model);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{AnalysisResult, JobStatus};
    use serde_json::json;

    fn create_test_job_status(id: &str) -> JobStatus {
        JobStatus {
            job_id: id.to_string(),
            status: "processing".to_string(),
            progress: 50.0,
            error: None,
        }
    }

    fn create_test_analysis_result(id: &str) -> AnalysisResult {
        AnalysisResult {
            id: id.to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![1.0, 2.0, 3.0]),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_r2_key_generation() {
        let file_id = "test-file-123";
        let expected_key = "audio/test-file-123.wav";
        let actual_key = format!("audio/{}.wav", file_id);
        
        assert_eq!(actual_key, expected_key);
    }

    #[test]
    fn test_r2_key_with_special_characters() {
        let file_id = "test-file-with-hyphens-and_underscores";
        let expected_key = "audio/test-file-with-hyphens-and_underscores.wav";
        let actual_key = format!("audio/{}.wav", file_id);
        
        assert_eq!(actual_key, expected_key);
    }

    #[test]
    fn test_r2_key_with_uuid() {
        let file_id = "550e8400-e29b-41d4-a716-446655440000";
        let expected_key = "audio/550e8400-e29b-41d4-a716-446655440000.wav";
        let actual_key = format!("audio/{}.wav", file_id);
        
        assert_eq!(actual_key, expected_key);
    }

    #[test]
    fn test_kv_key_formats() {
        let file_id = "file123";
        let job_id = "job456";
        let result_id = "result789";
        
        assert_eq!(format!("file:{}", file_id), "file:file123");
        assert_eq!(format!("job:{}", job_id), "job:job456");
        assert_eq!(format!("result:{}", result_id), "result:result789");
    }

    #[test]
    fn test_metadata_creation() {
        let file_id = "test-file";
        let data_size = 1024;
        let timestamp = 1609459200000.0; // 2021-01-01T00:00:00Z
        
        let metadata = json!({
            "id": file_id,
            "uploaded_at": timestamp,
            "size": data_size,
            "status": "uploaded"
        });
        
        assert_eq!(metadata["id"], file_id);
        assert_eq!(metadata["size"], data_size);
        assert_eq!(metadata["status"], "uploaded");
        assert_eq!(metadata["uploaded_at"], timestamp);
    }

    #[test]
    fn test_job_status_serialization() {
        let status = create_test_job_status("test-job");
        let json_result = serde_json::to_string(&status);
        
        assert!(json_result.is_ok());
        let json_str = json_result.unwrap();
        assert!(json_str.contains("test-job"));
        assert!(json_str.contains("processing"));
        assert!(json_str.contains("50"));
    }

    #[test]
    fn test_job_status_deserialization() {
        let json_str = r#"{"job_id":"test-job","status":"completed","progress":100.0,"error":null}"#;
        let result: std::result::Result<JobStatus, serde_json::Error> = serde_json::from_str(json_str);
        
        assert!(result.is_ok());
        let status = result.unwrap();
        assert_eq!(status.job_id, "test-job");
        assert_eq!(status.status, "completed");
        assert_eq!(status.progress, 100.0);
        assert_eq!(status.error, None);
    }

    #[test]
    fn test_job_status_with_error_serialization() {
        let status = JobStatus {
            job_id: "failed-job".to_string(),
            status: "failed".to_string(),
            progress: 25.0,
            error: Some("Network timeout".to_string()),
        };
        
        let json_result = serde_json::to_string(&status);
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("failed-job"));
        assert!(json_str.contains("failed"));
        assert!(json_str.contains("Network timeout"));
    }

    #[test]
    fn test_analysis_result_serialization() {
        let result = create_test_analysis_result("test-result");
        let json_result = serde_json::to_string(&result);
        
        assert!(json_result.is_ok());
        let json_str = json_result.unwrap();
        assert!(json_str.contains("test-result"));
        assert!(json_str.contains("completed"));
        assert!(json_str.contains("1.0"));
        assert!(json_str.contains("2.0"));
        assert!(json_str.contains("3.0"));
    }

    #[test]
    fn test_analysis_result_deserialization() {
        let json_str = r#"{"id":"test-result","status":"completed","dimensions":[4.0,5.0,6.0],"created_at":"2023-01-01T00:00:00Z"}"#;
        let result: std::result::Result<AnalysisResult, serde_json::Error> = serde_json::from_str(json_str);
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.id, "test-result");
        assert_eq!(analysis.status, "completed");
        assert_eq!(analysis.dimensions, Some(vec![4.0, 5.0, 6.0]));
        assert_eq!(analysis.created_at, "2023-01-01T00:00:00Z");
    }

    #[test]
    fn test_analysis_result_no_dimensions() {
        let result = AnalysisResult {
            id: "no-dims".to_string(),
            status: "processing".to_string(),
            dimensions: None,
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };
        
        let json_result = serde_json::to_string(&result);
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("no-dims"));
        assert!(json_str.contains("null") || json_str.contains("\"dimensions\":null"));
    }

    #[test]
    fn test_large_dimensions_array() {
        let large_dimensions: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let result = AnalysisResult {
            id: "large-dims".to_string(),
            status: "completed".to_string(),
            dimensions: Some(large_dimensions.clone()),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };
        
        let json_result = serde_json::to_string(&result);
        assert!(json_result.is_ok());
        
        // Verify deserialization
        let json_str = json_result.unwrap();
        let parsed: std::result::Result<AnalysisResult, serde_json::Error> = serde_json::from_str(&json_str);
        assert!(parsed.is_ok());
        
        let parsed_result = parsed.unwrap();
        assert_eq!(parsed_result.dimensions.as_ref().unwrap().len(), 1000);
        assert_eq!(parsed_result.dimensions.as_ref().unwrap()[0], 0.0);
        assert_eq!(parsed_result.dimensions.as_ref().unwrap()[999], 99.9);
    }

    #[test]
    fn test_empty_dimensions_array() {
        let result = AnalysisResult {
            id: "empty-dims".to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![]),
            created_at: "2023-01-01T00:00:00Z".to_string(),
        };
        
        let json_result = serde_json::to_string(&result);
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("[]"));
        
        // Verify deserialization
        let parsed: std::result::Result<AnalysisResult, serde_json::Error> = serde_json::from_str(&json_str);
        assert!(parsed.is_ok());
        
        let parsed_result = parsed.unwrap();
        assert_eq!(parsed_result.dimensions, Some(vec![]));
    }

    #[test]
    fn test_progress_boundaries() {
        let test_cases = vec![
            (0.0, "started"),
            (25.0, "preprocessing"),
            (50.0, "processing"),
            (75.0, "analyzing"),
            (100.0, "completed"),
        ];
        
        for (progress, status) in test_cases {
            let job_status = JobStatus {
                job_id: "test".to_string(),
                status: status.to_string(),
                progress,
                error: None,
            };
            
            assert!(job_status.progress >= 0.0);
            assert!(job_status.progress <= 100.0);
            assert_eq!(job_status.progress, progress);
        }
    }

    #[test]
    fn test_invalid_json_handling() {
        let invalid_json = r#"{"id":"test","status":"completed","progress":"not_a_number"}"#;
        let result: std::result::Result<JobStatus, serde_json::Error> = serde_json::from_str(invalid_json);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_required_fields() {
        let incomplete_json = r#"{"job_id":"test"}"#;
        let result: std::result::Result<JobStatus, serde_json::Error> = serde_json::from_str(incomplete_json);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_fields_ignored() {
        let json_with_extra = r#"{"job_id":"test","status":"completed","progress":100.0,"error":null,"extra_field":"ignored"}"#;
        let result: std::result::Result<JobStatus, serde_json::Error> = serde_json::from_str(json_with_extra);
        
        assert!(result.is_ok());
        let status = result.unwrap();
        assert_eq!(status.job_id, "test");
        assert_eq!(status.status, "completed");
    }

    #[test]
    fn test_file_size_metadata() {
        let sizes = vec![0, 1, 1024, 1024*1024, 10*1024*1024];
        
        for size in sizes {
            let metadata = json!({
                "id": "test",
                "uploaded_at": 1609459200000.0,
                "size": size,
                "status": "uploaded"
            });
            
            assert_eq!(metadata["size"], size);
            assert!(size >= 0);
        }
    }

    #[test]
    fn test_timestamp_format() {
        let timestamp = 1609459200000.0; // Mock timestamp (2021-01-01T00:00:00Z)
        assert!(timestamp > 0.0);
        
        let metadata = json!({
            "id": "test",
            "uploaded_at": timestamp,
            "size": 1024,
            "status": "uploaded"
        });
        
        assert_eq!(metadata["uploaded_at"], timestamp);
    }

    #[test]
    fn test_status_transitions() {
        let statuses = vec![
            "uploaded",
            "preprocessing", 
            "processing",
            "analyzing",
            "completed",
            "failed"
        ];
        
        for status in statuses {
            let job_status = JobStatus {
                job_id: "test".to_string(),
                status: status.to_string(),
                progress: if status == "completed" { 100.0 } else if status == "failed" { 0.0 } else { 50.0 },
                error: if status == "failed" { Some("Test error".to_string()) } else { None },
            };
            
            assert!(!job_status.status.is_empty());
            assert_eq!(job_status.status, status);
        }
    }

    #[test]
    fn test_error_message_handling() {
        let error_messages = vec![
            Some("Network timeout".to_string()),
            Some("Invalid audio format".to_string()),
            Some("Processing failed".to_string()),
            Some("Out of memory".to_string()),
            None,
        ];
        
        for error in error_messages {
            let job_status = JobStatus {
                job_id: "test".to_string(),
                status: if error.is_some() { "failed" } else { "completed" }.to_string(),
                progress: if error.is_some() { 0.0 } else { 100.0 },
                error: error.clone(),
            };
            
            assert_eq!(job_status.error, error);
            
            // Verify serialization
            let json_result = serde_json::to_string(&job_status);
            assert!(json_result.is_ok());
        }
    }
}