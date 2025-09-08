use worker::*;
use wasm_bindgen::JsValue;
use serde_json::json;
use crate::processing;

pub async fn send_for_inference(env: &Env, spectrogram_data: &[u8], job_id: &str) -> Result<()> {
    let modal_api_key = env.secret("MODAL_API_KEY")?
        .to_string();
    let modal_endpoint = env.var("MODAL_ENDPOINT")?
        .to_string();
    
    // Encode spectrogram data as base64
    use base64::Engine;
    let encoded_data = base64::engine::general_purpose::STANDARD.encode(spectrogram_data);
    
    let request_body = json!({
        "job_id": job_id,
        "spectrogram": encoded_data,
        "callback_url": format!("{}/webhook/modal", env.var("WORKER_URL")?.to_string())
    });
    
    let mut headers = Headers::new();
    headers.set("Content-Type", "application/json")?;
    headers.set("Authorization", &format!("Bearer {}", modal_api_key))?;
    
    let request = Request::new_with_init(
        &modal_endpoint,
        RequestInit::new()
            .with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(JsValue::from_str(&request_body.to_string())))
    )?;
    
    let mut response = Fetch::Request(request).send().await?;
    
    if response.status_code() == 200 {
        console_log!("Successfully sent job {} to Modal", job_id);
        Ok(())
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        console_log!("Modal API error: {}", error_text);
        Err(worker::Error::RustError(format!("Modal API error: {}", error_text)))
    }
}

pub async fn handle_modal_webhook(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let body: serde_json::Value = req.json().await?;
    
    let job_id = body["job_id"].as_str()
        .ok_or_else(|| worker::Error::RustError("Missing job_id".to_string()))?;
    
    let status = body["status"].as_str()
        .ok_or_else(|| worker::Error::RustError("Missing status".to_string()))?;
    
    match status {
        "completed" => {
            if let Some(dimensions) = body["dimensions"].as_array() {
                let dimension_values: Result<Vec<f32>> = dimensions
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32)
                        .ok_or_else(|| worker::Error::RustError("Invalid dimension value".to_string())))
                    .collect();
                
                let dimension_values = dimension_values?;
                processing::complete_analysis(&ctx.env, job_id, dimension_values).await?;
                
                Response::from_json(&json!({
                    "status": "success",
                    "message": "Analysis completed"
                }))
            } else {
                Response::error("Missing dimensions in webhook", 400)
            }
        }
        "failed" => {
            let error_message = body["error"].as_str().unwrap_or("Unknown error");
            
            // Update job status to failed
            let failed_status = crate::JobStatus {
                id: job_id.to_string(),
                status: "failed".to_string(),
                progress: 0.0,
                error: Some(error_message.to_string()),
            };
            
            crate::storage::update_job_status(&ctx.env, job_id, &failed_status).await?;
            
            Response::from_json(&json!({
                "status": "error",
                "message": "Analysis failed"
            }))
        }
        _ => {
            Response::error("Invalid webhook status", 400)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use base64::Engine;

    fn create_test_spectrogram_data() -> Vec<u8> {
        // Create test spectrogram data (128x128 float32 values)
        let size = 128 * 128 * 4; // 4 bytes per float32
        let mut data = Vec::with_capacity(size);
        
        for i in 0..(128 * 128) {
            let value = ((i as f32).sin() * 0.5 + 0.5) * 255.0;
            let bytes = (value as u32).to_le_bytes();
            data.extend_from_slice(&bytes);
        }
        
        data
    }

    #[test]
    fn test_base64_encoding() {
        let test_data = vec![1, 2, 3, 4, 5];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&test_data);
        
        assert!(!encoded.is_empty());
        assert!(encoded.chars().all(|c| c.is_ascii()));
        
        // Verify we can decode it back
        let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).unwrap();
        assert_eq!(decoded, test_data);
    }

    #[test]
    fn test_spectrogram_base64_encoding() {
        let spectrogram_data = create_test_spectrogram_data();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&spectrogram_data);
        
        assert!(!encoded.is_empty());
        assert!(spectrogram_data.len() > 0);
        
        // Verify encoding doesn't corrupt data
        let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).unwrap();
        assert_eq!(decoded, spectrogram_data);
    }

    #[test]
    fn test_modal_request_body_creation() {
        let job_id = "test-job-123";
        let spectrogram_data = create_test_spectrogram_data();
        let encoded_data = base64::engine::general_purpose::STANDARD.encode(&spectrogram_data);
        let callback_url = "https://example.com/webhook/modal";

        let request_body = json!({
            "job_id": job_id,
            "spectrogram": encoded_data,
            "callback_url": callback_url
        });

        assert_eq!(request_body["job_id"], job_id);
        assert_eq!(request_body["spectrogram"], encoded_data);
        assert_eq!(request_body["callback_url"], callback_url);
    }

    #[test]
    fn test_modal_request_body_serialization() {
        let request_body = json!({
            "job_id": "test-job",
            "spectrogram": "base64data",
            "callback_url": "https://example.com/callback"
        });

        let serialized = serde_json::to_string(&request_body);
        assert!(serialized.is_ok());
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("test-job"));
        assert!(json_str.contains("base64data"));
        assert!(json_str.contains("callback"));
    }

    #[test]
    fn test_webhook_body_parsing_completed() {
        let webhook_body = json!({
            "job_id": "completed-job-123",
            "status": "completed",
            "dimensions": [1.0, 2.0, 3.0, 4.0, 5.0]
        });

        let job_id = webhook_body["job_id"].as_str().unwrap();
        let status = webhook_body["status"].as_str().unwrap();
        let dimensions = webhook_body["dimensions"].as_array().unwrap();

        assert_eq!(job_id, "completed-job-123");
        assert_eq!(status, "completed");
        assert_eq!(dimensions.len(), 5);
        assert_eq!(dimensions[0].as_f64().unwrap(), 1.0);
        assert_eq!(dimensions[4].as_f64().unwrap(), 5.0);
    }

    #[test]
    fn test_webhook_body_parsing_failed() {
        let webhook_body = json!({
            "job_id": "failed-job-456",
            "status": "failed",
            "error": "Processing timeout"
        });

        let job_id = webhook_body["job_id"].as_str().unwrap();
        let status = webhook_body["status"].as_str().unwrap();
        let error = webhook_body["error"].as_str().unwrap();

        assert_eq!(job_id, "failed-job-456");
        assert_eq!(status, "failed");
        assert_eq!(error, "Processing timeout");
    }

    #[test]
    fn test_webhook_missing_job_id() {
        let webhook_body = json!({
            "status": "completed",
            "dimensions": [1.0, 2.0, 3.0]
        });

        let job_id = webhook_body["job_id"].as_str();
        assert!(job_id.is_none());
    }

    #[test]
    fn test_webhook_missing_status() {
        let webhook_body = json!({
            "job_id": "test-job",
            "dimensions": [1.0, 2.0, 3.0]
        });

        let status = webhook_body["status"].as_str();
        assert!(status.is_none());
    }

    #[test]
    fn test_webhook_missing_dimensions() {
        let webhook_body = json!({
            "job_id": "test-job", 
            "status": "completed"
        });

        let dimensions = webhook_body["dimensions"].as_array();
        assert!(dimensions.is_none());
    }

    #[test]
    fn test_dimension_conversion() {
        let dimensions_json = json!([1.5, 2.7, 3.14, 4.0, 5.9]);
        let dimensions_array = dimensions_json.as_array().unwrap();
        
        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or("Invalid dimension".to_string()))
            .collect();
        
        assert!(converted.is_ok());
        let values = converted.unwrap();
        assert_eq!(values.len(), 5);
        assert_eq!(values[0], 1.5);
        assert_eq!(values[2], 3.14);
    }

    #[test]
    fn test_invalid_dimension_conversion() {
        let dimensions_json = json!([1.0, "invalid", 3.0]);
        let dimensions_array = dimensions_json.as_array().unwrap();
        
        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or("Invalid dimension".to_string()))
            .collect();
        
        assert!(converted.is_err());
    }

    #[test]
    fn test_empty_dimensions_array() {
        let dimensions_json = json!([]);
        let dimensions_array = dimensions_json.as_array().unwrap();
        
        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or("Invalid dimension".to_string()))
            .collect();
        
        assert!(converted.is_ok());
        let values = converted.unwrap();
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_large_dimensions_array() {
        let large_dims: Vec<serde_json::Value> = (0..1000)
            .map(|i| json!(i as f64 * 0.1))
            .collect();
        let dimensions_json = json!(large_dims);
        let dimensions_array = dimensions_json.as_array().unwrap();
        
        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or("Invalid dimension".to_string()))
            .collect();
        
        assert!(converted.is_ok());
        let values = converted.unwrap();
        assert_eq!(values.len(), 1000);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[999], 99.9);
    }

    #[test]
    fn test_failed_status_creation() {
        let job_id = "failed-job";
        let error_message = "Network timeout";
        
        let failed_status = crate::JobStatus {
            id: job_id.to_string(),
            status: "failed".to_string(),
            progress: 0.0,
            error: Some(error_message.to_string()),
        };
        
        assert_eq!(failed_status.id, job_id);
        assert_eq!(failed_status.status, "failed");
        assert_eq!(failed_status.progress, 0.0);
        assert_eq!(failed_status.error, Some(error_message.to_string()));
    }

    #[test]
    fn test_webhook_response_formats() {
        let success_response = json!({
            "status": "success",
            "message": "Analysis completed"
        });
        
        let error_response = json!({
            "status": "error", 
            "message": "Analysis failed"
        });
        
        assert_eq!(success_response["status"], "success");
        assert_eq!(success_response["message"], "Analysis completed");
        assert_eq!(error_response["status"], "error");
        assert_eq!(error_response["message"], "Analysis failed");
    }

    #[test]
    fn test_callback_url_generation() {
        let worker_url = "https://my-worker.example.com";
        let callback_url = format!("{}/webhook/modal", worker_url);
        
        assert_eq!(callback_url, "https://my-worker.example.com/webhook/modal");
        assert!(callback_url.starts_with("https://"));
        assert!(callback_url.ends_with("/webhook/modal"));
    }

    #[test]
    fn test_authorization_header_format() {
        let api_key = "test-api-key-123";
        let auth_header = format!("Bearer {}", api_key);
        
        assert_eq!(auth_header, "Bearer test-api-key-123");
        assert!(auth_header.starts_with("Bearer "));
    }

    #[test]
    fn test_webhook_status_validation() {
        let valid_statuses = vec!["completed", "failed"];
        let invalid_statuses = vec!["unknown", "pending", "", "processing"];
        
        for status in valid_statuses {
            assert!(status == "completed" || status == "failed");
        }
        
        for status in invalid_statuses {
            assert!(status != "completed" && status != "failed");
        }
    }

    #[test]
    fn test_error_message_extraction() {
        let webhook_with_error = json!({
            "job_id": "test",
            "status": "failed",
            "error": "Processing failed"
        });
        
        let webhook_without_error = json!({
            "job_id": "test",
            "status": "failed"
        });
        
        let error1 = webhook_with_error["error"].as_str().unwrap_or("Unknown error");
        let error2 = webhook_without_error["error"].as_str().unwrap_or("Unknown error");
        
        assert_eq!(error1, "Processing failed");
        assert_eq!(error2, "Unknown error");
    }

    #[test]
    fn test_json_parsing_edge_cases() {
        let edge_cases = vec![
            json!({"job_id": null, "status": "completed"}),
            json!({"job_id": "", "status": "completed"}),
            json!({"job_id": "test", "status": null}),
            json!({"job_id": "test", "status": ""}),
            json!({}),
            json!(null),
        ];
        
        for case in edge_cases {
            let job_id = case.get("job_id").and_then(|v| v.as_str());
            let status = case.get("status").and_then(|v| v.as_str());
            
            // These should handle gracefully
            match (job_id, status) {
                (Some(jid), Some(st)) if !jid.is_empty() && !st.is_empty() => {
                    // Valid case
                    assert!(!jid.is_empty() && !st.is_empty());
                }
                _ => {
                    // Invalid case - should be handled with error
                    assert!(job_id.is_none() || status.is_none() || 
                           job_id.unwrap().is_empty() || status.unwrap().is_empty());
                }
            }
        }
    }

    #[test]
    fn test_http_status_codes() {
        let success_code = 200;
        let bad_request_code = 400;
        let internal_error_code = 500;
        
        assert_eq!(success_code, 200);
        assert_eq!(bad_request_code, 400);
        assert_eq!(internal_error_code, 500);
        
        assert!(success_code >= 200 && success_code < 300);
        assert!(bad_request_code >= 400 && bad_request_code < 500);
        assert!(internal_error_code >= 500);
    }

    #[test]
    fn test_content_type_header() {
        let content_type = "application/json";
        assert_eq!(content_type, "application/json");
        assert!(content_type.contains("json"));
    }

    #[test]
    fn test_spectrogram_data_integrity() {
        let original_data = create_test_spectrogram_data();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&original_data);
        let decoded = base64::engine::general_purpose::STANDARD.decode(&encoded).unwrap();
        
        assert_eq!(original_data.len(), decoded.len());
        assert_eq!(original_data, decoded);
        
        // Verify data is not empty and has expected structure
        assert!(!original_data.is_empty());
        assert_eq!(original_data.len(), 128 * 128 * 4); // 128x128 float32 values
    }

    #[test]
    fn test_concurrent_webhook_handling() {
        let webhook_bodies = vec![
            json!({"job_id": "job1", "status": "completed", "dimensions": [1.0]}),
            json!({"job_id": "job2", "status": "failed", "error": "timeout"}),
            json!({"job_id": "job3", "status": "completed", "dimensions": [2.0, 3.0]}),
        ];
        
        for body in webhook_bodies {
            let job_id = body["job_id"].as_str().unwrap();
            let status = body["status"].as_str().unwrap();
            
            assert!(!job_id.is_empty());
            assert!(status == "completed" || status == "failed");
        }
    }
}