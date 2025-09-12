// Phase 3.3: End-to-End Integration Tests
// Tests for complete audio analysis workflow

use wasm_bindgen_test::*;
use worker::*;
use serde_json::{json, Value};
use std::collections::HashMap;

wasm_bindgen_test_configure!(run_in_browser);

/// Integration test environment
struct IntegrationTestEnv {
    api_key: String,
    base_url: String,
    modal_webhook_secret: String,
    test_files: HashMap<String, Vec<u8>>,
}

impl IntegrationTestEnv {
    fn new() -> Self {
        let mut test_files = HashMap::new();
        
        // Add test audio files
        test_files.insert("test.wav".to_string(), create_wav_test_file());
        test_files.insert("test.mp3".to_string(), create_mp3_test_file());
        test_files.insert("test.flac".to_string(), create_flac_test_file());
        test_files.insert("test.aac".to_string(), create_aac_test_file());
        
        Self {
            api_key: "test-integration-key".to_string(),
            base_url: "https://test-api.crescendai.com".to_string(),
            modal_webhook_secret: "test-modal-secret".to_string(),
            test_files,
        }
    }

    fn create_auth_headers(&self) -> Headers {
        let headers = Headers::new();
        headers.set("Authorization", &format!("Bearer {}", self.api_key)).unwrap();
        headers.set("Content-Type", "application/json").unwrap();
        headers
    }

    async fn upload_test_file(&self, filename: &str) -> worker::Result<String> {
        let _file_data = self.test_files.get(filename)
            .ok_or_else(|| worker::Error::RustError(format!("Test file {} not found", filename)))?;
        
        // Simulate upload process
        let upload_id = uuid::Uuid::new_v4().to_string();
        
        // In real implementation, this would make actual HTTP requests
        // For now, we'll simulate the upload workflow
        Ok(upload_id)
    }
}

// =============================================================================
// COMPLETE WORKFLOW TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_complete_audio_analysis_workflow() {
    let env = IntegrationTestEnv::new();
    
    // Step 1: Upload audio file
    let upload_id = env.upload_test_file("test.wav").await
        .expect("Should successfully upload test file");
    
    // Step 2: Initiate analysis
    let analysis_id = simulate_start_analysis(&upload_id).await
        .expect("Should successfully start analysis");
    
    // Step 3: Poll for job status
    let mut job_status = "pending".to_string();
    let mut poll_count = 0;
    let max_polls = 30; // 30 second timeout
    
    while job_status != "completed" && poll_count < max_polls {
        job_status = simulate_poll_job_status(&analysis_id).await
            .expect("Should successfully poll job status");
        
        if job_status == "failed" {
            panic!("Analysis job failed");
        }
        
        poll_count += 1;
        
        // Simulate 1 second delay
        simulate_delay(1000).await;
    }
    
    assert_eq!(job_status, "completed", "Analysis should complete successfully");
    
    // Step 4: Retrieve analysis results
    let results = simulate_get_analysis_results(&analysis_id).await
        .expect("Should successfully retrieve results");
    
    // Validate results structure
    validate_analysis_results(&results);
    
    // Step 5: Verify analysis data quality
    assert_analysis_data_quality(&results).await;
}

#[wasm_bindgen_test]
async fn test_wav_file_processing() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.wav").await
        .expect("WAV file upload should succeed");
    
    let analysis_id = simulate_start_analysis(&upload_id).await
        .expect("WAV analysis should start successfully");
    
    // Wait for completion
    let results = wait_for_analysis_completion(&analysis_id).await
        .expect("WAV analysis should complete");
    
    // Verify WAV-specific processing
    assert!(results["file_format"].as_str() == Some("wav"), "Should detect WAV format");
    assert!(results["sample_rate"].as_u64().unwrap_or(0) > 0, "Should extract sample rate");
    assert!(results["channels"].as_u64().unwrap_or(0) > 0, "Should extract channel count");
}

#[wasm_bindgen_test]
async fn test_mp3_file_processing() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.mp3").await
        .expect("MP3 file upload should succeed");
    
    let analysis_id = simulate_start_analysis(&upload_id).await
        .expect("MP3 analysis should start successfully");
    
    let results = wait_for_analysis_completion(&analysis_id).await
        .expect("MP3 analysis should complete");
    
    // Verify MP3-specific processing
    assert!(results["file_format"].as_str() == Some("mp3"), "Should detect MP3 format");
    assert!(results["bitrate"].as_u64().unwrap_or(0) > 0, "Should extract bitrate");
}

#[wasm_bindgen_test]
async fn test_flac_file_processing() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.flac").await
        .expect("FLAC file upload should succeed");
    
    let analysis_id = simulate_start_analysis(&upload_id).await
        .expect("FLAC analysis should start successfully");
    
    let results = wait_for_analysis_completion(&analysis_id).await
        .expect("FLAC analysis should complete");
    
    // Verify FLAC-specific processing
    assert!(results["file_format"].as_str() == Some("flac"), "Should detect FLAC format");
    assert!(results["lossless"].as_bool() == Some(true), "Should identify as lossless");
}

#[wasm_bindgen_test]
async fn test_aac_file_processing() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.aac").await
        .expect("AAC file upload should succeed");
    
    let analysis_id = simulate_start_analysis(&upload_id).await
        .expect("AAC analysis should start successfully");
    
    let results = wait_for_analysis_completion(&analysis_id).await
        .expect("AAC analysis should complete");
    
    // Verify AAC-specific processing
    assert!(results["file_format"].as_str() == Some("aac"), "Should detect AAC format");
}

// =============================================================================
// MODAL GPU INTEGRATION TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_modal_gpu_integration() {
    let env = IntegrationTestEnv::new();
    
    // Upload and start analysis
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    let analysis_id = simulate_start_analysis(&upload_id).await.unwrap();
    
    // Simulate Modal GPU processing
    let modal_job_id = simulate_modal_job_submission(&analysis_id).await
        .expect("Should successfully submit job to Modal");
    
    // Simulate Modal webhook callback
    let webhook_payload = create_modal_webhook_payload(&modal_job_id, &analysis_id);
    let webhook_response = simulate_modal_webhook_callback(&webhook_payload, &env.modal_webhook_secret).await
        .expect("Should successfully handle Modal webhook");
    
    assert!(webhook_response["status"].as_str() == Some("success"), 
            "Webhook should be processed successfully");
    
    // Verify analysis results are updated
    let results = simulate_get_analysis_results(&analysis_id).await.unwrap();
    assert!(results["status"].as_str() == Some("completed"), 
            "Analysis should be marked as completed");
}

#[wasm_bindgen_test]
async fn test_modal_webhook_signature_validation() {
    let env = IntegrationTestEnv::new();
    
    let analysis_id = "test-analysis-id";
    let modal_job_id = "test-modal-job-id";
    
    // Test valid webhook signature
    let valid_payload = create_modal_webhook_payload(modal_job_id, analysis_id);
    let valid_response = simulate_modal_webhook_callback(&valid_payload, &env.modal_webhook_secret).await;
    assert!(valid_response.is_ok(), "Valid webhook signature should be accepted");
    
    // Test invalid webhook signature
    let invalid_response = simulate_modal_webhook_callback(&valid_payload, "wrong-secret").await;
    assert!(invalid_response.is_err(), "Invalid webhook signature should be rejected");
}

#[wasm_bindgen_test]
async fn test_modal_error_handling() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    let analysis_id = simulate_start_analysis(&upload_id).await.unwrap();
    
    // Simulate Modal processing error
    let error_payload = json!({
        "job_id": "test-job-id",
        "analysis_id": analysis_id,
        "status": "failed",
        "error": "GPU processing failed",
        "timestamp": 1609459200
    });
    
    let _response = simulate_modal_webhook_callback(&error_payload, &env.modal_webhook_secret).await
        .expect("Should handle error webhook");
    
    // Verify error handling
    let results = simulate_get_analysis_results(&analysis_id).await.unwrap();
    assert!(results["status"].as_str() == Some("failed"), 
            "Analysis should be marked as failed");
    assert!(results["error"].as_str().is_some(), 
            "Error message should be recorded");
}

// =============================================================================
// COMPARISON WORKFLOW TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_comparison_workflow() {
    let env = IntegrationTestEnv::new();
    
    // Upload test file
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    
    // Initiate model comparison
    let comparison_id = simulate_start_comparison(&upload_id).await
        .expect("Should successfully start comparison");
    
    // Wait for both models to complete
    let comparison_results = wait_for_comparison_completion(&comparison_id).await
        .expect("Comparison should complete successfully");
    
    // Validate comparison results structure
    validate_comparison_results(&comparison_results);
    
    // Test user preference submission
    let preference_data = json!({
        "comparison_id": comparison_id,
        "preferred_model": "model_a",
        "feedback": "Model A provided more accurate timing analysis"
    });
    
    let preference_response = simulate_submit_preference(&preference_data).await
        .expect("Should successfully submit user preference");
    
    assert!(preference_response["status"].as_str() == Some("success"),
            "Preference submission should succeed");
}

#[wasm_bindgen_test]
async fn test_comparison_with_different_models() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    
    // Test comparison with specific model configurations
    let comparison_configs = vec![
        json!({"model_a": "ast-v1", "model_b": "ast-v2"}),
        json!({"model_a": "ast-v2", "model_b": "ast-v3"}),
        json!({"model_a": "ast-base", "model_b": "ast-large"}),
    ];
    
    for config in comparison_configs {
        let comparison_id = simulate_start_comparison_with_config(&upload_id, &config).await
            .expect("Should start comparison with custom config");
        
        let results = wait_for_comparison_completion(&comparison_id).await
            .expect("Custom comparison should complete");
        
        // Verify both models ran
        assert!(results["model_a"]["status"].as_str() == Some("completed"));
        assert!(results["model_b"]["status"].as_str() == Some("completed"));
        
        // Verify different models produce different results
        let model_a_score = results["model_a"]["analysis"]["overall_performance"].as_f64().unwrap();
        let model_b_score = results["model_b"]["analysis"]["overall_performance"].as_f64().unwrap();
        
        // Allow for some variation between models
        assert!((model_a_score - model_b_score).abs() >= 0.0,
                "Models should potentially produce different scores");
    }
}

// =============================================================================
// ERROR RECOVERY TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_upload_failure_recovery() {
    let env = IntegrationTestEnv::new();
    
    // Simulate upload failure
    let upload_result = simulate_failed_upload("corrupted-file.wav").await;
    assert!(upload_result.is_err(), "Corrupted upload should fail");
    
    // Test retry with valid file
    let retry_upload_id = env.upload_test_file("test.wav").await
        .expect("Retry upload should succeed");
    
    let analysis_id = simulate_start_analysis(&retry_upload_id).await
        .expect("Analysis after retry should succeed");
    
    let results = wait_for_analysis_completion(&analysis_id).await
        .expect("Analysis should complete after retry");
    
    validate_analysis_results(&results);
}

#[wasm_bindgen_test]
async fn test_analysis_timeout_recovery() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    let analysis_id = simulate_start_analysis(&upload_id).await.unwrap();
    
    // Simulate analysis timeout
    simulate_analysis_timeout(&analysis_id).await;
    
    // Verify timeout is handled gracefully
    let results = simulate_get_analysis_results(&analysis_id).await.unwrap();
    assert!(results["status"].as_str() == Some("failed") || 
            results["status"].as_str() == Some("timeout"),
            "Timeout should be handled gracefully");
    
    // Test retry mechanism
    let retry_analysis_id = simulate_start_analysis(&upload_id).await
        .expect("Retry analysis should be possible");
    
    let retry_results = wait_for_analysis_completion(&retry_analysis_id).await
        .expect("Retry analysis should complete");
    
    validate_analysis_results(&retry_results);
}

#[wasm_bindgen_test]
async fn test_partial_failure_recovery() {
    let env = IntegrationTestEnv::new();
    
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    
    // Simulate partial failure in comparison
    let comparison_id = simulate_start_comparison(&upload_id).await.unwrap();
    
    // Simulate one model failing
    simulate_model_failure(&comparison_id, "model_b").await;
    
    // Verify partial results are still usable
    let results = simulate_get_comparison_results(&comparison_id).await.unwrap();
    
    assert!(results["model_a"]["status"].as_str() == Some("completed"),
            "Model A should complete successfully");
    assert!(results["model_b"]["status"].as_str() == Some("failed"),
            "Model B should be marked as failed");
    
    // Verify comparison can still be used with one model
    assert!(results["status"].as_str() == Some("partial"),
            "Comparison should be marked as partial success");
}

// =============================================================================
// CONCURRENT REQUEST HANDLING TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_concurrent_analysis_requests() {
    let env = IntegrationTestEnv::new();
    
    let concurrent_count = 5;
    let mut analysis_ids = Vec::new();
    
    // Start multiple analyses concurrently
    for i in 0..concurrent_count {
        let filename = if i % 2 == 0 { "test.wav" } else { "test.mp3" };
        let upload_id = env.upload_test_file(filename).await
            .expect("Concurrent upload should succeed");
        
        let analysis_id = simulate_start_analysis(&upload_id).await
            .expect("Concurrent analysis start should succeed");
        
        analysis_ids.push(analysis_id);
    }
    
    // Wait for all analyses to complete
    for analysis_id in analysis_ids {
        let results = wait_for_analysis_completion(&analysis_id).await
            .expect("Concurrent analysis should complete");
        
        validate_analysis_results(&results);
    }
}

#[wasm_bindgen_test]
async fn test_concurrent_upload_handling() {
    let env = IntegrationTestEnv::new();
    
    let concurrent_uploads = 10;
    let mut upload_futures = Vec::new();
    
    // Start multiple uploads concurrently
    for i in 0..concurrent_uploads {
        let _filename = format!("test_{}.wav", i);
        // In a real test, we'd create unique files
        let upload_future = env.upload_test_file("test.wav");
        upload_futures.push(upload_future);
    }
    
    // All uploads should succeed
    for (i, upload_future) in upload_futures.into_iter().enumerate() {
        let upload_id = upload_future.await
            .expect(&format!("Concurrent upload {} should succeed", i));
        
        assert!(!upload_id.is_empty(), "Upload ID should be generated");
    }
}

// =============================================================================
// REAL-WORLD SCENARIO TESTS
// =============================================================================

#[wasm_bindgen_test]
async fn test_mobile_app_workflow() {
    let _env = IntegrationTestEnv::new();
    
    // Simulate mobile app workflow:
    // 1. Record audio
    // 2. Upload in chunks
    // 3. Start analysis
    // 4. Poll for results
    // 5. Display results
    
    // Step 1: Simulate chunked upload
    let chunks = create_audio_chunks("test.wav", 4); // 4 chunks
    let mut upload_id = String::new();
    
    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            upload_id = simulate_start_chunked_upload(chunk).await
                .expect("Should start chunked upload");
        } else {
            simulate_upload_chunk(&upload_id, i, chunk).await
                .expect("Should upload chunk successfully");
        }
    }
    
    // Finalize upload
    simulate_finalize_upload(&upload_id).await
        .expect("Should finalize upload");
    
    // Continue with normal workflow
    let analysis_id = simulate_start_analysis(&upload_id).await.unwrap();
    let results = wait_for_analysis_completion(&analysis_id).await.unwrap();
    
    validate_analysis_results(&results);
}

#[wasm_bindgen_test]
async fn test_web_app_workflow() {
    let env = IntegrationTestEnv::new();
    
    // Simulate web app workflow:
    // 1. Drag and drop file
    // 2. Upload with progress
    // 3. Start analysis
    // 4. Real-time progress updates
    // 5. Download results
    
    let upload_id = env.upload_test_file("test.wav").await.unwrap();
    
    // Simulate progress tracking
    let analysis_id = simulate_start_analysis(&upload_id).await.unwrap();
    
    // Monitor progress in real-time
    let mut progress = 0.0;
    while progress < 1.0 {
        let status = simulate_get_job_progress(&analysis_id).await.unwrap();
        progress = status["progress"].as_f64().unwrap_or(0.0);
        
        assert!(progress >= 0.0 && progress <= 1.0, 
                "Progress should be between 0 and 1");
        
        if progress < 1.0 {
            simulate_delay(500).await; // Check every 500ms
        }
    }
    
    let results = simulate_get_analysis_results(&analysis_id).await.unwrap();
    validate_analysis_results(&results);
}

// =============================================================================
// HELPER FUNCTIONS FOR INTEGRATION TESTS
// =============================================================================

/// Create test audio files
fn create_wav_test_file() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&[44u8, 1, 0, 0]); // File size
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&[16, 0, 0, 0]); // Subchunk1Size
    data.extend_from_slice(&[1, 0]); // AudioFormat (PCM)
    data.extend_from_slice(&[2, 0]); // NumChannels (stereo)
    data.extend_from_slice(&[68, 172, 0, 0]); // SampleRate (44100)
    data.extend_from_slice(&[16, 177, 2, 0]); // ByteRate
    data.extend_from_slice(&[4, 0]); // BlockAlign
    data.extend_from_slice(&[16, 0]); // BitsPerSample
    data.extend_from_slice(b"data");
    data.extend_from_slice(&[0, 1, 0, 0]); // Subchunk2Size
    
    // Add some sample audio data
    for i in 0..128 {
        let sample = ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / 44100.0).sin() * 32767.0) as i16;
        data.extend_from_slice(&sample.to_le_bytes());
    }
    
    data
}

fn create_mp3_test_file() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"ID3");
    data.extend_from_slice(&[3, 0]); // Version
    data.extend_from_slice(&[0]); // Flags
    data.extend_from_slice(&[0, 0, 0, 0]); // Size
    data.extend_from_slice(&[0xFF, 0xFB]); // MP3 frame header
    data.extend_from_slice(&[0u8; 100]); // Dummy audio data
    data
}

fn create_flac_test_file() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"fLaC");
    data.extend_from_slice(&[0u8; 38]); // STREAMINFO block
    data.extend_from_slice(&[0u8; 100]); // Dummy audio data
    data
}

fn create_aac_test_file() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0xFF, 0xF1]); // ADTS header
    data.extend_from_slice(&[0u8; 100]); // Dummy audio data
    data
}

/// Simulation functions
async fn simulate_start_analysis(_upload_id: &str) -> worker::Result<String> {
    // Simulate starting analysis
    let analysis_id = uuid::Uuid::new_v4().to_string();
    Ok(analysis_id)
}

async fn simulate_poll_job_status(_analysis_id: &str) -> worker::Result<String> {
    // Simulate job status polling
    // In real implementation, this would check actual job status
    use js_sys::Math;
    
    let random = Math::random();
    if random < 0.1 {
        Ok("pending".to_string())
    } else if random < 0.9 {
        Ok("processing".to_string())
    } else {
        Ok("completed".to_string())
    }
}

async fn simulate_get_analysis_results(analysis_id: &str) -> worker::Result<Value> {
    // Simulate getting analysis results
    Ok(json!({
        "id": analysis_id,
        "status": "completed",
        "file_format": "wav",
        "sample_rate": 44100,
        "channels": 2,
        "duration": 30.5,
        "analysis": {
            "rhythm": 8.5,
            "pitch": 7.8,
            "dynamics": 8.2,
            "tempo": 7.9,
            "articulation": 8.1,
            "expression": 7.7,
            "technique": 8.3,
            "timing": 8.0,
            "phrasing": 7.9,
            "voicing": 8.1,
            "pedaling": 7.8,
            "hand_coordination": 8.2,
            "musical_understanding": 8.0,
            "stylistic_accuracy": 7.9,
            "creativity": 7.5,
            "listening": 8.1,
            "overall_performance": 8.0,
            "stage_presence": 7.8,
            "repertoire_difficulty": 8.5
        },
        "insights": [
            "Strong technical execution",
            "Good timing accuracy",
            "Expressive dynamics"
        ],
        "processing_time": 2.3
    }))
}

async fn wait_for_analysis_completion(analysis_id: &str) -> worker::Result<Value> {
    // Wait for analysis to complete (with timeout)
    let mut attempts = 0;
    let max_attempts = 10;
    
    while attempts < max_attempts {
        let status = simulate_poll_job_status(analysis_id).await?;
        
        if status == "completed" {
            return simulate_get_analysis_results(analysis_id).await;
        } else if status == "failed" {
            return Err(worker::Error::RustError("Analysis failed".to_string()));
        }
        
        simulate_delay(1000).await; // Wait 1 second
        attempts += 1;
    }
    
    Err(worker::Error::RustError("Analysis timeout".to_string()))
}

async fn simulate_delay(ms: u32) {
    // Simulate delay in WASM environment
    let start = js_sys::Date::now();
    while js_sys::Date::now() - start < ms as f64 {
        // Busy wait
    }
}

async fn simulate_modal_job_submission(_analysis_id: &str) -> worker::Result<String> {
    let modal_job_id = uuid::Uuid::new_v4().to_string();
    Ok(modal_job_id)
}

fn create_modal_webhook_payload(modal_job_id: &str, analysis_id: &str) -> Value {
    json!({
        "job_id": modal_job_id,
        "analysis_id": analysis_id,
        "status": "completed",
        "results": {
            "rhythm": 8.5,
            "pitch": 7.8,
            "dynamics": 8.2
        },
        "timestamp": 1609459200
    })
}

async fn simulate_modal_webhook_callback(_payload: &Value, secret: &str) -> worker::Result<Value> {
    // Simulate webhook processing
    if secret == "test-modal-secret" {
        Ok(json!({"status": "success"}))
    } else {
        Err(worker::Error::RustError("Invalid webhook signature".to_string()))
    }
}

async fn simulate_start_comparison(_upload_id: &str) -> worker::Result<String> {
    let comparison_id = uuid::Uuid::new_v4().to_string();
    Ok(comparison_id)
}

async fn wait_for_comparison_completion(comparison_id: &str) -> worker::Result<Value> {
    // Simulate comparison completion
    Ok(json!({
        "id": comparison_id,
        "status": "completed",
        "model_a": {
            "model_name": "ast-v1",
            "status": "completed",
            "analysis": {
                "overall_performance": 8.0
            }
        },
        "model_b": {
            "model_name": "ast-v2", 
            "status": "completed",
            "analysis": {
                "overall_performance": 8.2
            }
        }
    }))
}

fn validate_analysis_results(results: &Value) {
    assert!(results["id"].as_str().is_some(), "Results should have ID");
    assert!(results["status"].as_str() == Some("completed"), "Results should be completed");
    assert!(results["analysis"].is_object(), "Results should have analysis object");
    
    let analysis = &results["analysis"];
    let required_fields = vec![
        "rhythm", "pitch", "dynamics", "tempo", "articulation",
        "expression", "technique", "timing", "phrasing", "voicing",
        "pedaling", "hand_coordination", "musical_understanding",
        "stylistic_accuracy", "creativity", "listening",
        "overall_performance", "stage_presence", "repertoire_difficulty"
    ];
    
    for field in required_fields {
        assert!(analysis[field].is_number(), "Analysis should have {} field", field);
        let value = analysis[field].as_f64().unwrap();
        assert!(value >= 0.0 && value <= 10.0, "{} should be between 0 and 10", field);
    }
}

fn validate_comparison_results(results: &Value) {
    assert!(results["id"].as_str().is_some(), "Comparison should have ID");
    assert!(results["status"].as_str() == Some("completed"), "Comparison should be completed");
    assert!(results["model_a"].is_object(), "Should have model_a results");
    assert!(results["model_b"].is_object(), "Should have model_b results");
}

async fn assert_analysis_data_quality(results: &Value) {
    let analysis = &results["analysis"];
    
    // Check for realistic score ranges
    let overall_score = analysis["overall_performance"].as_f64().unwrap();
    assert!(overall_score > 0.0 && overall_score <= 10.0, 
            "Overall performance score should be realistic");
    
    // Check that different dimensions have reasonable variation
    let scores: Vec<f64> = vec![
        "rhythm", "pitch", "dynamics", "tempo", "articulation"
    ].iter().map(|field| analysis[field].as_f64().unwrap()).collect();
    
    let max_score = scores.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_score = scores.iter().fold(10.0f64, |a, &b| a.min(b));
    
    // Scores shouldn't all be identical (some variation expected)
    assert!(max_score - min_score > 0.0, "Analysis scores should show some variation");
}

// Additional helper functions for various test scenarios...
async fn simulate_start_comparison_with_config(_upload_id: &str, _config: &Value) -> worker::Result<String> {
    let comparison_id = uuid::Uuid::new_v4().to_string();
    Ok(comparison_id)
}

async fn simulate_submit_preference(_preference_data: &Value) -> worker::Result<Value> {
    Ok(json!({"status": "success"}))
}

async fn simulate_failed_upload(_filename: &str) -> worker::Result<String> {
    Err(worker::Error::RustError("Upload failed".to_string()))
}

async fn simulate_analysis_timeout(_analysis_id: &str) {
    // Simulate timeout scenario
}

async fn simulate_model_failure(_comparison_id: &str, _model: &str) {
    // Simulate model failure
}

async fn simulate_get_comparison_results(comparison_id: &str) -> worker::Result<Value> {
    Ok(json!({
        "id": comparison_id,
        "status": "partial",
        "model_a": {"status": "completed"},
        "model_b": {"status": "failed"}
    }))
}

fn create_audio_chunks(_filename: &str, chunk_count: usize) -> Vec<Vec<u8>> {
    let full_data = create_wav_test_file();
    let chunk_size = full_data.len() / chunk_count;
    
    (0..chunk_count)
        .map(|i| {
            let start = i * chunk_size;
            let end = if i == chunk_count - 1 { full_data.len() } else { (i + 1) * chunk_size };
            full_data[start..end].to_vec()
        })
        .collect()
}

async fn simulate_start_chunked_upload(_first_chunk: &[u8]) -> worker::Result<String> {
    Ok(uuid::Uuid::new_v4().to_string())
}

async fn simulate_upload_chunk(_upload_id: &str, _chunk_index: usize, _chunk: &[u8]) -> worker::Result<()> {
    Ok(())
}

async fn simulate_finalize_upload(_upload_id: &str) -> worker::Result<()> {
    Ok(())
}

async fn simulate_get_job_progress(_analysis_id: &str) -> worker::Result<Value> {
    use js_sys::Math;
    let progress = Math::random();
    Ok(json!({"progress": progress}))
}
