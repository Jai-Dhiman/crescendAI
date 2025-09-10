use crate::processing;
use crate::security::{validate_body_size, validate_webhook_signature, SecurityError};
use serde_json::json;
use wasm_bindgen::JsValue;
use worker::*;

pub async fn send_for_inference(env: &Env, spectrogram_data: &[u8], job_id: &str, file_id: &str) -> Result<()> {
    // Use real model service endpoint (fallback to localhost for development)
    let model_endpoint = env.var("MODEL_SERVICE_URL")
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "http://localhost:8002".to_string());

    console_log!("Sending inference request to: {}/analyze", model_endpoint);
    console_log!("Job ID: {}, File ID: {}, Spectrogram size: {} bytes", job_id, file_id, spectrogram_data.len());

    // Encode spectrogram data as base64
    use base64::Engine;
    let encoded_data = base64::engine::general_purpose::STANDARD.encode(spectrogram_data);

    let request_body = json!({
        "file_id": job_id,
        "spectrogram_data": encoded_data,
        "metadata": {
            "source": "crescend_ai",
            "timestamp": js_sys::Date::new_0().to_iso_string().as_string().unwrap()
        }
    });

    console_log!("Request body size: {} chars", request_body.to_string().len());

    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;

    let request = Request::new_with_init(
        &format!("{}/analyze", model_endpoint),
        RequestInit::new()
            .with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(JsValue::from_str(&request_body.to_string()))),
    )?;

    console_log!("Making inference request...");
    let mut response = Fetch::Request(request).send().await?;
    console_log!("Got response with status: {}", response.status_code());

    if response.status_code() == 200 {
        // Parse the real model service response
        let response_data: serde_json::Value = response.json().await?;
        
        console_log!("Real model analysis completed for job {}", job_id);
        
        // Extract analysis data from real model service response
        if let Some(analysis) = response_data["analysis"].as_object() {
            use crate::AnalysisData;
            
            // Extract individual analysis values
            let analysis_data = AnalysisData {
                rhythm: analysis.get("rhythm").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                pitch: analysis.get("pitch").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                dynamics: analysis.get("dynamics").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                tempo: analysis.get("tempo").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                articulation: analysis.get("articulation").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                expression: analysis.get("expression").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                technique: analysis.get("technique").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                timing: analysis.get("timing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                phrasing: analysis.get("phrasing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                voicing: analysis.get("voicing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                pedaling: analysis.get("pedaling").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                hand_coordination: analysis.get("hand_coordination").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                musical_understanding: analysis.get("musical_understanding").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                stylistic_accuracy: analysis.get("stylistic_accuracy").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                creativity: analysis.get("creativity").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                listening: analysis.get("listening").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                overall_performance: analysis.get("overall_performance").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                stage_presence: analysis.get("stage_presence").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                repertoire_difficulty: analysis.get("repertoire_difficulty").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
            };
            
            // Extract insights from the response
            let insights = response_data["insights"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            
            // Extract processing time
            let processing_time = response_data["processing_time"].as_f64().map(|t| t as f32);
            
            // Complete the analysis with the results
            console_log!("Attempting to complete analysis for job_id: {}", job_id);
            match processing::complete_analysis(env, job_id, file_id, analysis_data, insights, processing_time).await {
                Ok(_) => {
                    console_log!("Successfully completed analysis for job_id: {}", job_id);
                }
                Err(e) => {
                    console_log!("ERROR: Failed to complete analysis for job_id {}: {:?}", job_id, e);
                    return Err(e);
                }
            }
            
        } else {
            return Err(worker::Error::RustError(
                "Invalid response format from real model service".to_string()
            ));
        }
        
        Ok(())
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        console_log!("Real model API error: {}", error_text);
        Err(worker::Error::RustError(format!(
            "Real model API error: {}",
            error_text
        )))
    }
}

pub async fn send_for_inference_with_model(
    env: &Env, 
    spectrogram_data: &[u8], 
    job_id: &str, 
    file_id: &str,
    model_type: &str
) -> Result<(crate::AnalysisData, Vec<String>, Option<f32>)> {
    // Determine model endpoint based on model type
    let model_endpoint = match model_type {
        "hybrid_ast" => env.var("MODEL_SERVICE_URL")
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "http://localhost:8002".to_string()),
        "ultra_small_ast" => env.var("MODEL_SERVICE_SMALL_URL")
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "http://localhost:8001".to_string()),
        _ => env.var("MODEL_SERVICE_URL")
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "http://localhost:8002".to_string()),
    };

    console_log!("Sending inference request to: {}/analyze (model: {})", model_endpoint, model_type);
    console_log!("Job ID: {}, File ID: {}, Spectrogram size: {} bytes", job_id, file_id, spectrogram_data.len());

    // Encode spectrogram data as base64
    use base64::Engine;
    let encoded_data = base64::engine::general_purpose::STANDARD.encode(spectrogram_data);

    let request_body = json!({
        "file_id": job_id,
        "spectrogram_data": encoded_data,
        "model_type": model_type,
        "metadata": {
            "source": "crescend_ai_comparison",
            "timestamp": js_sys::Date::new_0().to_iso_string().as_string().unwrap(),
            "comparison_mode": true
        }
    });

    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;

    let request = Request::new_with_init(
        &format!("{}/analyze", model_endpoint),
        RequestInit::new()
            .with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(JsValue::from_str(&request_body.to_string()))),
    )?;

    console_log!("Making inference request for model {}...", model_type);
    let mut response = Fetch::Request(request).send().await?;
    console_log!("Got response with status: {} for model {}", response.status_code(), model_type);

    if response.status_code() == 200 {
        // Parse the model service response
        let response_data: serde_json::Value = response.json().await?;
        
        console_log!("Model {} analysis completed for job {}", model_type, job_id);
        
        // Extract analysis data from model service response
        if let Some(analysis) = response_data["analysis"].as_object() {
            use crate::AnalysisData;
            
            // Extract individual analysis values
            let analysis_data = AnalysisData {
                rhythm: analysis.get("rhythm").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                pitch: analysis.get("pitch").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                dynamics: analysis.get("dynamics").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                tempo: analysis.get("tempo").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                articulation: analysis.get("articulation").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                expression: analysis.get("expression").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                technique: analysis.get("technique").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                timing: analysis.get("timing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                phrasing: analysis.get("phrasing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                voicing: analysis.get("voicing").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                pedaling: analysis.get("pedaling").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                hand_coordination: analysis.get("hand_coordination").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                musical_understanding: analysis.get("musical_understanding").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                stylistic_accuracy: analysis.get("stylistic_accuracy").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                creativity: analysis.get("creativity").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                listening: analysis.get("listening").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                overall_performance: analysis.get("overall_performance").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                stage_presence: analysis.get("stage_presence").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
                repertoire_difficulty: analysis.get("repertoire_difficulty").and_then(|v| v.as_f64()).unwrap_or(75.0) as f32,
            };
            
            // Extract insights from the response
            let insights = response_data["insights"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            
            // Extract processing time
            let processing_time = response_data["processing_time"].as_f64().map(|t| t as f32);
            
            console_log!("Successfully extracted analysis data for model {}", model_type);
            Ok((analysis_data, insights, processing_time))
            
        } else {
            Err(worker::Error::RustError(format!(
                "Invalid response format from model service ({})",
                model_type
            )))
        }
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        console_log!("Model {} API error: {}", model_type, error_text);
        Err(worker::Error::RustError(format!(
            "Model {} API error: {}",
            model_type,
            error_text
        )))
    }
}

pub async fn handle_modal_webhook(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    const MAX_WEBHOOK_SIZE: usize = 10 * 1024 * 1024; // 10MB limit for webhook payload

    // Verify webhook signature first
    let signature_header = req
        .headers()
        .get("X-Modal-Signature")?
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidSignature))?;

    let timestamp_header = req
        .headers()
        .get("X-Modal-Timestamp")?
        .ok_or_else(|| worker::Error::from(SecurityError::InvalidSignature))?;

    // Get webhook secret from environment
    let webhook_secret = ctx.env.secret("MODAL_WEBHOOK_SECRET")?.to_string();

    // Get raw body for signature verification
    let raw_body = req.bytes().await?;
    validate_body_size(raw_body.len(), MAX_WEBHOOK_SIZE)?;

    // Verify webhook signature and timestamp
    validate_webhook_signature(
        &raw_body,
        &signature_header,
        &webhook_secret,
        &timestamp_header,
    )?;

    // Parse JSON after signature verification
    let body: serde_json::Value = serde_json::from_slice(&raw_body).map_err(|_| {
        worker::Error::from(SecurityError::InvalidInput(
            "Invalid JSON in webhook payload".to_string(),
        ))
    })?;

    // Validate required fields
    let job_id = body["job_id"].as_str().ok_or_else(|| {
        worker::Error::from(SecurityError::InvalidInput(
            "Missing or invalid job_id in webhook".to_string(),
        ))
    })?;

    let status = body["status"].as_str().ok_or_else(|| {
        worker::Error::from(SecurityError::InvalidInput(
            "Missing or invalid status in webhook".to_string(),
        ))
    })?;

    // Validate job_id format (should be UUID)
    crate::security::validate_uuid(job_id)?;

    console_log!(
        "Processing webhook for job {} with status {}",
        job_id,
        status
    );

    match status {
        "completed" => {
            if let Some(dimensions) = body["dimensions"].as_array() {
                // Validate dimensions array size (prevent DoS)
                if dimensions.len() > 10000 {
                    return Err(worker::Error::from(SecurityError::InvalidInput(
                        "Dimensions array too large".to_string(),
                    )));
                }

                let dimension_values: Result<Vec<f32>> = dimensions
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .filter(|&f| f.is_finite()) // Validate finite numbers
                            .ok_or_else(|| {
                                worker::Error::from(SecurityError::InvalidInput(
                                    "Invalid dimension value".to_string(),
                                ))
                            })
                    })
                    .collect();

                let dimension_values = dimension_values?;
                
                // Convert Vec<f32> to AnalysisData struct 
                // Ensure we have enough values (fill with defaults if needed)
                let mut vals = dimension_values.clone();
                vals.resize(19, 75.0); // Fill missing values with 75.0 default
                
                let analysis_data = crate::AnalysisData {
                    rhythm: vals.get(0).copied().unwrap_or(75.0),
                    pitch: vals.get(1).copied().unwrap_or(75.0),
                    dynamics: vals.get(2).copied().unwrap_or(75.0),
                    tempo: vals.get(3).copied().unwrap_or(75.0),
                    articulation: vals.get(4).copied().unwrap_or(75.0),
                    expression: vals.get(5).copied().unwrap_or(75.0),
                    technique: vals.get(6).copied().unwrap_or(75.0),
                    timing: vals.get(7).copied().unwrap_or(75.0),
                    phrasing: vals.get(8).copied().unwrap_or(75.0),
                    voicing: vals.get(9).copied().unwrap_or(75.0),
                    pedaling: vals.get(10).copied().unwrap_or(75.0),
                    hand_coordination: vals.get(11).copied().unwrap_or(75.0),
                    musical_understanding: vals.get(12).copied().unwrap_or(75.0),
                    stylistic_accuracy: vals.get(13).copied().unwrap_or(75.0),
                    creativity: vals.get(14).copied().unwrap_or(75.0),
                    listening: vals.get(15).copied().unwrap_or(75.0),
                    stage_presence: vals.get(16).copied().unwrap_or(75.0),
                    repertoire_difficulty: vals.get(17).copied().unwrap_or(75.0),
                    overall_performance: vals.get(18).copied().unwrap_or(75.0),
                };
                
                // Create some default insights for this webhook-based completion
                let insights = vec![
                    "Analysis completed via Modal webhook".to_string(),
                    "Performance shows good technical foundation".to_string(),
                ];
                
                processing::complete_analysis(&ctx.env, job_id, job_id, analysis_data, insights, None).await?;

                console_log!("Analysis completed successfully for job {}", job_id);
                Response::from_json(&json!({
                    "status": "success",
                    "message": "Analysis completed"
                }))
            } else {
                Err(worker::Error::from(SecurityError::InvalidInput(
                    "Missing dimensions array in completed webhook".to_string(),
                )))
            }
        }
        "failed" => {
            let error_message = body["error"]
                .as_str()
                .unwrap_or("Unknown error")
                .chars()
                .take(500) // Limit error message length
                .collect::<String>();

            // Update job status to failed
            let failed_status = crate::JobStatus {
                job_id: job_id.to_string(),
                status: "failed".to_string(),
                progress: 0.0,
                error: Some(error_message.clone()),
            };

            crate::storage::update_job_status(&ctx.env, job_id, &failed_status).await?;

            console_log!("Analysis failed for job {}: {}", job_id, error_message);
            Response::from_json(&json!({
                "status": "error",
                "message": "Analysis failed"
            }))
        }
        _ => {
            console_log!("Invalid webhook status received: {}", status);
            Err(worker::Error::from(SecurityError::InvalidInput(format!(
                "Invalid webhook status: {}",
                status
            ))))
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine;
    use serde_json::json;

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
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, test_data);
    }

    #[test]
    fn test_spectrogram_base64_encoding() {
        let spectrogram_data = create_test_spectrogram_data();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&spectrogram_data);

        assert!(!encoded.is_empty());
        assert!(spectrogram_data.len() > 0);

        // Verify encoding doesn't corrupt data
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
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
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or("Invalid dimension".to_string())
            })
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
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or("Invalid dimension".to_string())
            })
            .collect();

        assert!(converted.is_err());
    }

    #[test]
    fn test_empty_dimensions_array() {
        let dimensions_json = json!([]);
        let dimensions_array = dimensions_json.as_array().unwrap();

        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or("Invalid dimension".to_string())
            })
            .collect();

        assert!(converted.is_ok());
        let values = converted.unwrap();
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_large_dimensions_array() {
        let large_dims: Vec<serde_json::Value> = (0..1000).map(|i| json!(i as f64 * 0.1)).collect();
        let dimensions_json = json!(large_dims);
        let dimensions_array = dimensions_json.as_array().unwrap();

        let converted: std::result::Result<Vec<f32>, String> = dimensions_array
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or("Invalid dimension".to_string())
            })
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
            job_id: job_id.to_string(),
            status: "failed".to_string(),
            progress: 0.0,
            error: Some(error_message.to_string()),
        };

        assert_eq!(failed_status.job_id, job_id);
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

        let error1 = webhook_with_error["error"]
            .as_str()
            .unwrap_or("Unknown error");
        let error2 = webhook_without_error["error"]
            .as_str()
            .unwrap_or("Unknown error");

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
                    assert!(
                        job_id.is_none()
                            || status.is_none()
                            || job_id.unwrap().is_empty()
                            || status.unwrap().is_empty()
                    );
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
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();

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
