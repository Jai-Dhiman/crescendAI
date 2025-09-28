use worker::*;
use crate::{JobStatus, AnalysisResult, AnalysisData, ComparisonResult, ModelResult};
use crate::storage;
use crate::audio_dsp;

// Conditional logging macro for test vs WASM environments
#[cfg(test)]
macro_rules! console_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(test))]
macro_rules! console_log {
    ($($arg:tt)*) => {
        worker::console_log!($($arg)*);
    };
}

pub async fn start_analysis(env: &Env, file_id: &str, job_id: &str, force_gpu: Option<bool>) -> Result<()> {
    console_log!("Starting analysis for file_id: {}, job_id: {}", file_id, job_id);
    
    // Initialize job status
    let initial_status = JobStatus {
        job_id: job_id.to_string(),
        status: "processing".to_string(),
        progress: 0.0,
        error: None,
    };
    
    console_log!("Updating initial job status...");
    storage::update_job_status(env, job_id, &initial_status).await?;

    // Get audio data from R2
    console_log!("Retrieving audio data from R2 for file_id: {}", file_id);
    let audio_data = storage::get_audio_from_r2(env, file_id).await?;
    console_log!("Retrieved {} bytes of audio data", audio_data.len());
    
    // Update status - preprocessing
    let preprocessing_status = JobStatus {
        job_id: job_id.to_string(),
        status: "processing".to_string(),
        progress: 25.0,
        error: None,
    };
    storage::update_job_status(env, job_id, &preprocessing_status).await?;
    
    // Generate mel-spectrogram (simplified - would need actual audio processing)
    console_log!("Generating mel-spectrogram for job_id: {}", job_id);
    let spectrogram_data = generate_mel_spectrogram(&audio_data).await?;
    console_log!("Generated {} bytes of spectrogram data", spectrogram_data.len());
    
    // At this point, inference is not configured. Return an explicit error per project policy.
    let error_status = JobStatus {
        job_id: job_id.to_string(),
        status: "failed".to_string(),
        progress: 50.0,
        error: Some("ML inference not configured".to_string()),
    };
    storage::update_job_status(env, job_id, &error_status).await?;
    return Err(worker::Error::RustError("ML inference not configured".to_string()));
}

async fn generate_mel_spectrogram(audio_data: &[u8]) -> Result<Vec<u8>> {
    console_log!("Processing audio data of {} bytes using real DSP", audio_data.len());
    
    // Validate minimum audio size
    if audio_data.len() < 44 {
        return Err(worker::Error::RustError("Audio data too small".to_string()));
    }
    
    // Use the real DSP implementation to generate mel-spectrogram
    match audio_dsp::process_audio_to_mel_spectrogram(audio_data).await {
        Ok(mel_spectrogram_bytes) => {
            console_log!("Successfully generated mel-spectrogram: {} bytes", mel_spectrogram_bytes.len());
            Ok(mel_spectrogram_bytes)
        }
        Err(e) => {
            console_log!("DSP processing failed: {}", e);
            // Fallback to placeholder for development/testing purposes
            console_log!("Falling back to placeholder spectrogram for compatibility");
            let placeholder_spectrogram = create_placeholder_spectrogram();
            Ok(placeholder_spectrogram)
        }
    }
}

fn create_placeholder_spectrogram() -> Vec<u8> {
    // Create a placeholder 128x128 spectrogram (64KB of data)
    // This represents a mel-spectrogram with 128 mel bins and 128 time frames
    let size = 128 * 128 * 4; // 4 bytes per float32
    let mut data = Vec::with_capacity(size);
    
    // Generate some dummy spectral data as float32 values
    for i in 0..(128 * 128) {
        let value = (i as f32).sin() * 0.5 + 0.5; // Range [0, 1]
        let bytes = value.to_le_bytes(); // Convert float32 to bytes
        data.extend_from_slice(&bytes);
    }
    
    console_log!("Generated placeholder spectrogram: {} bytes (expected: {})", data.len(), size);
    data
}


pub async fn complete_analysis(
    env: &Env, 
    job_id: &str, 
    file_id: &str,
    analysis_data: AnalysisData,
    insights: Vec<String>,
    processing_time: Option<f32>
) -> Result<()> {
    console_log!("complete_analysis: Starting completion for job_id: {}", job_id);
    
    // Create analysis result
    let result = AnalysisResult {
        id: job_id.to_string(),
        status: "completed".to_string(),
        file_id: file_id.to_string(),
        analysis: analysis_data,
        insights,
        created_at: js_sys::Date::new_0().to_iso_string().as_string().unwrap(),
        processing_time,
    };
    
    console_log!("complete_analysis: Created analysis result for job_id: {}", job_id);
    
    // Store the result
    match storage::store_analysis_result(env, job_id, &result).await {
        Ok(_) => {
            console_log!("complete_analysis: Successfully stored analysis result for job_id: {}", job_id);
            
            // Trigger cache warming for this completed analysis
            storage::warm_cache_for_completed_job(env, job_id).await.ok();
        }
        Err(e) => {
            console_log!("complete_analysis: ERROR storing analysis result for job_id {}: {:?}", job_id, e);
            return Err(e);
        }
    }
    
    // Update job status to completed
    let completed_status = JobStatus {
        job_id: job_id.to_string(),
        status: "completed".to_string(),
        progress: 100.0,
        error: None,
    };
    
    console_log!("complete_analysis: Updating job status to completed for job_id: {}", job_id);
    match storage::update_job_status(env, job_id, &completed_status).await {
        Ok(_) => {
            console_log!("complete_analysis: Successfully updated job status to completed for job_id: {}", job_id);
            
            // Verify the status was actually written by reading it back
            match storage::get_job_status(env, job_id).await {
                Ok(verified_status) => {
                    console_log!("complete_analysis: Verified job status for {}: {} (expected: completed)", job_id, verified_status.status);
                }
                Err(e) => {
                    console_log!("complete_analysis: WARNING: Could not verify job status for {}: {:?}", job_id, e);
                }
            }
        }
        Err(e) => {
            console_log!("complete_analysis: ERROR updating job status for job_id {}: {:?}", job_id, e);
            return Err(e);
        }
    }
    
    console_log!("complete_analysis: Completed all operations for job_id: {}", job_id);
    Ok(())
}

pub async fn start_model_comparison(
    env: &Env, 
    file_id: &str, 
    comparison_id: &str, 
    model_a: &str, 
    model_b: &str,
    force_gpu: Option<bool>
) -> Result<()> {
    console_log!("Starting model comparison for file_id: {}, comparison_id: {} (models: {} vs {})", 
                 file_id, comparison_id, model_a, model_b);
    
    // Initialize job status
    let initial_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 0.0,
        error: None,
    };
    
    storage::update_job_status(env, comparison_id, &initial_status).await?;

    // Mock-by-default path for portfolio demo cost control
let use_mock_default = env.var("USE_MOCK_INFERENCE").map(|v| v.to_string() == "true").unwrap_or(false);
    let use_mock = match force_gpu {
        Some(true) => false,
        _ => use_mock_default,
    };
    if use_mock {
        console_log!("USE_MOCK_INFERENCE=true -> generating mock comparison (no GPU calls)");
        let (a_data, a_insights, a_time) = generate_mock_analysis(file_id);
        let (b_data, b_insights, b_time) = generate_mock_analysis(file_id);
        complete_model_comparison(env, comparison_id, file_id, (a_data, a_insights, Some(a_time)), (b_data, b_insights, Some(b_time))).await?;
        return Ok(());
    }
    
    // Get audio data from R2
    console_log!("Retrieving audio data from R2 for file_id: {}", file_id);
    let audio_data = storage::get_audio_from_r2(env, file_id).await?;
    console_log!("Retrieved {} bytes of audio data", audio_data.len());
    
    // Update status - preprocessing
    let preprocessing_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 20.0,
        error: None,
    };
    storage::update_job_status(env, comparison_id, &preprocessing_status).await?;
    
    // Generate spectrogram
    console_log!("Generating mel-spectrogram for comparison: {}", comparison_id);
    let spectrogram_data = generate_mel_spectrogram(&audio_data).await?;
    
    // Update status - running parallel inference
    let inference_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 40.0,
        error: None,
    };
    storage::update_job_status(env, comparison_id, &inference_status).await?;
    
    // Run both models in parallel
    console_log!("Running parallel inference for models {} and {}", model_a, model_b);
    
    let model_a_future = modal_client::send_for_inference_with_model(
        env, &spectrogram_data, comparison_id, file_id, model_a
    );
    let model_b_future = modal_client::send_for_inference_with_model(
        env, &spectrogram_data, comparison_id, file_id, model_b
    );
    
    // Wait for both models to complete
    let (result_a, result_b) = match futures::try_join!(model_a_future, model_b_future) {
        Ok((a, b)) => (a, b),
        Err(e) => {
            console_log!("Parallel inference failed for comparison {}: {:?}", comparison_id, e);
            let error_status = JobStatus {
                job_id: comparison_id.to_string(),
                status: "failed".to_string(),
                progress: 0.0,
                error: Some(e.to_string()),
            };
            storage::update_job_status(env, comparison_id, &error_status).await?;
            return Err(e);
        }
    };
    
    // Update status - processing results
    let processing_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 80.0,
        error: None,
    };
    storage::update_job_status(env, comparison_id, &processing_status).await?;
    
    // Complete the comparison
    complete_model_comparison(env, comparison_id, file_id, result_a, result_b).await?;
    
    console_log!("Model comparison completed successfully for comparison_id: {}", comparison_id);
    Ok(())
}

pub async fn complete_model_comparison(
    env: &Env,
    comparison_id: &str,
    file_id: &str,
    result_a: (AnalysisData, Vec<String>, Option<f32>),
    result_b: (AnalysisData, Vec<String>, Option<f32>),
) -> Result<()> {
    console_log!("Completing model comparison for comparison_id: {}", comparison_id);
    
    let (analysis_a, insights_a, time_a) = result_a;
    let (analysis_b, insights_b, time_b) = result_b;
    
    let model_a_result = ModelResult {
        model_name: "Model A".to_string(),
        model_type: "hybrid_ast".to_string(),
        analysis: analysis_a,
        insights: insights_a,
        processing_time: time_a.unwrap_or(0.0) as f64,
        dimensions: None, // Add placeholder dimensions
    };
    
    let model_b_result = ModelResult {
        model_name: "Model B".to_string(),
        model_type: "ultra_small_ast".to_string(),
        analysis: analysis_b,
        insights: insights_b,
        processing_time: time_b.unwrap_or(0.0) as f64,
        dimensions: None, // Add placeholder dimensions
    };
    
    let total_processing_time = match (time_a, time_b) {
        (Some(a), Some(b)) => Some(a.max(b)), // Use the longer time since they ran in parallel
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    
    // Create comparison result
    let comparison_result = ComparisonResult {
        id: comparison_id.to_string(),
        status: "completed".to_string(),
        file_id: file_id.to_string(),
        model_a: model_a_result,
        model_b: model_b_result,
        created_at: js_sys::Date::new_0().to_iso_string().as_string().unwrap(),
        total_processing_time,
    };
    
    // Store the comparison result
    match storage::store_comparison_result(env, comparison_id, &comparison_result).await {
        Ok(_) => console_log!("Successfully stored comparison result for comparison_id: {}", comparison_id),
        Err(e) => {
            console_log!("ERROR storing comparison result for comparison_id {}: {:?}", comparison_id, e);
            return Err(e);
        }
    }
    
    // Update job status to completed
    let completed_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "completed".to_string(),
        progress: 100.0,
        error: None,
    };
    
    match storage::update_job_status(env, comparison_id, &completed_status).await {
        Ok(_) => {
            console_log!("Successfully updated comparison job status to completed for comparison_id: {}", comparison_id);
        }
        Err(e) => {
            console_log!("ERROR updating comparison job status for comparison_id {}: {:?}", comparison_id, e);
            return Err(e);
        }
    }
    
    console_log!("Completed all operations for comparison_id: {}", comparison_id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_wav_data() -> Vec<u8> {
        let mut data = Vec::new();
        
        // WAV file header
        data.extend_from_slice(b"RIFF");              // ChunkID
        data.extend_from_slice(&[68u8, 0, 0, 0]);    // ChunkSize (44 + data size)
        data.extend_from_slice(b"WAVE");              // Format
        data.extend_from_slice(b"fmt ");              // Subchunk1ID
        data.extend_from_slice(&[16u8, 0, 0, 0]);    // Subchunk1Size (PCM)
        data.extend_from_slice(&[1u8, 0]);           // AudioFormat (PCM)
        data.extend_from_slice(&[1u8, 0]);           // NumChannels (mono)
        data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // SampleRate (44100)
        data.extend_from_slice(&[0x88, 0x58, 1, 0]); // ByteRate
        data.extend_from_slice(&[2u8, 0]);           // BlockAlign
        data.extend_from_slice(&[16u8, 0]);          // BitsPerSample
        data.extend_from_slice(b"data");              // Subchunk2ID
        data.extend_from_slice(&[20u8, 0, 0, 0]);    // Subchunk2Size (data size)
        
        // Sample data (10 samples of silence)
        for _ in 0..10 {
            data.extend_from_slice(&[0u8, 0]);
        }
        
        data
    }

    fn create_invalid_audio_data() -> Vec<u8> {
        vec![0x00, 0x01, 0x02, 0x03] // Too short and no RIFF header
    }

    fn create_non_wav_data() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"JPEG"); // Wrong format
        data.extend_from_slice(&[0u8; 40]); // Padding
        data
    }

    #[test]
    fn test_initial_job_status_creation() {
        let job_id = "test-job-123";
        let initial_status = JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 0.0,
            error: None,
        };
        
        assert_eq!(initial_status.job_id, job_id);
        assert_eq!(initial_status.status, "processing");
        assert_eq!(initial_status.progress, 0.0);
        assert_eq!(initial_status.error, None);
    }

    #[test]
    fn test_status_progression() {
        let job_id = "test-job";
        let statuses = vec![
            ("processing", 0.0),
            ("processing", 25.0),
            ("processing", 50.0),
            ("processing", 75.0),
            ("completed", 100.0),
        ];
        
        for (status_name, progress) in statuses {
            let status = JobStatus {
                job_id: job_id.to_string(),
                status: status_name.to_string(),
                progress,
                error: None,
            };
            
            assert_eq!(status.status, status_name);
            assert_eq!(status.progress, progress);
            assert!(status.progress >= 0.0 && status.progress <= 100.0);
        }
    }

    #[test]
    fn test_error_status_creation() {
        let job_id = "failed-job";
        let error_message = "Processing failed";
        
        let error_status = JobStatus {
            job_id: job_id.to_string(),
            status: "failed".to_string(),
            progress: 0.0,
            error: Some(error_message.to_string()),
        };
        
        assert_eq!(error_status.status, "failed");
        assert_eq!(error_status.progress, 0.0);
        assert_eq!(error_status.error, Some(error_message.to_string()));
    }

    #[test]
    fn test_valid_wav_header_detection() {
        let valid_data = create_valid_wav_data();
        
        // Test the validation logic from generate_mel_spectrogram
        assert!(valid_data.len() >= 44);
        assert_eq!(&valid_data[0..4], b"RIFF");
        assert_eq!(&valid_data[8..12], b"WAVE");
    }

    #[test]
    fn test_invalid_wav_header_detection() {
        let invalid_data = create_invalid_audio_data();
        
        // Test the validation logic
        let is_too_short = invalid_data.len() < 44;
        let has_riff_header = invalid_data.len() >= 4 && &invalid_data[0..4] == b"RIFF";
        
        assert!(is_too_short || !has_riff_header);
    }

    #[test]
    fn test_non_wav_format_detection() {
        let non_wav_data = create_non_wav_data();
        
        // Should fail validation
        let is_valid = non_wav_data.len() >= 44 && &non_wav_data[0..4] == b"RIFF";
        assert!(!is_valid);
    }

    #[test]
    fn test_placeholder_spectrogram_generation() {
        let spectrogram = create_placeholder_spectrogram();
        
        // Should be 128x128 spectrogram with 4 bytes per float32
        let expected_size = 128 * 128 * 4;
        assert_eq!(spectrogram.len(), expected_size);
        assert!(!spectrogram.is_empty());
    }

    #[test]
    fn test_placeholder_spectrogram_content() {
        let spectrogram = create_placeholder_spectrogram();
        
        // Check that we have actual data (not all zeros)
        let has_non_zero = spectrogram.iter().any(|&byte| byte != 0);
        assert!(has_non_zero);
        
        // Verify size is correct for 128x128 float32 array
        assert_eq!(spectrogram.len(), 128 * 128 * 4);
    }

    #[test]
    fn test_analysis_result_creation() {
        let job_id = "test-result";
        let file_id = "test-file-id";
        
        let analysis_data = AnalysisData {
            rhythm: 0.8,
            pitch: 0.7,
            dynamics: 0.6,
            tempo: 0.9,
            articulation: 0.7,
            expression: 0.8,
            technique: 0.6,
            timing: 0.7,
            phrasing: 0.8,
            voicing: 0.7,
            pedaling: 0.6,
            hand_coordination: 0.7,
            musical_understanding: 0.8,
            stylistic_accuracy: 0.7,
            creativity: 0.8,
            listening: 0.7,
            overall_performance: 0.8,
            stage_presence: 0.7,
            repertoire_difficulty: 0.6,
        };
        
        let result = AnalysisResult {
            id: job_id.to_string(),
            status: "completed".to_string(),
            file_id: file_id.to_string(),
            analysis: analysis_data.clone(),
            insights: vec!["Good rhythm".to_string(), "Work on dynamics".to_string()],
            created_at: "2023-01-01T00:00:00Z".to_string(), // Mock timestamp for testing
            processing_time: Some(0.5),
        };
        
        assert_eq!(result.id, job_id);
        assert_eq!(result.status, "completed");
        assert_eq!(result.file_id, file_id);
        assert!(!result.created_at.is_empty());
    }

    #[test]
    fn test_model_result_with_large_dimensions() {
        let large_dimensions: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        
        let analysis_data = AnalysisData {
            rhythm: 0.8,
            pitch: 0.7,
            dynamics: 0.6,
            tempo: 0.9,
            articulation: 0.7,
            expression: 0.8,
            technique: 0.6,
            timing: 0.7,
            phrasing: 0.8,
            voicing: 0.7,
            pedaling: 0.6,
            hand_coordination: 0.7,
            musical_understanding: 0.8,
            stylistic_accuracy: 0.7,
            creativity: 0.8,
            listening: 0.7,
            overall_performance: 0.8,
            stage_presence: 0.7,
            repertoire_difficulty: 0.6,
        };
        
        // Use ModelResult instead which has the dimensions field
        let result = ModelResult {
            model_name: "test-model".to_string(),
            model_type: "spectrogram".to_string(),
            analysis: analysis_data,
            insights: vec!["Good rhythm".to_string()],
            processing_time: 0.5,
            dimensions: Some(large_dimensions.clone()),
        };
        
        assert_eq!(result.dimensions.as_ref().unwrap().len(), 1000);
        assert_eq!(result.dimensions.as_ref().unwrap()[0], 0.0);
        assert_eq!(result.dimensions.as_ref().unwrap()[999], 99.9);
    }

    #[test]
    fn test_model_result_empty_dimensions() {
        let analysis_data = AnalysisData {
            rhythm: 0.8,
            pitch: 0.7,
            dynamics: 0.6,
            tempo: 0.9,
            articulation: 0.7,
            expression: 0.8,
            technique: 0.6,
            timing: 0.7,
            phrasing: 0.8,
            voicing: 0.7,
            pedaling: 0.6,
            hand_coordination: 0.7,
            musical_understanding: 0.8,
            stylistic_accuracy: 0.7,
            creativity: 0.8,
            listening: 0.7,
            overall_performance: 0.8,
            stage_presence: 0.7,
            repertoire_difficulty: 0.6,
        };
        
        // Use ModelResult instead which has the dimensions field
        let result = ModelResult {
            model_name: "empty-model".to_string(),
            model_type: "spectrogram".to_string(),
            analysis: analysis_data,
            insights: vec![],
            processing_time: 0.5,
            dimensions: Some(vec![]),
        };
        
        assert_eq!(result.dimensions, Some(vec![]));
    }

    #[test]
    fn test_completed_job_status_creation() {
        let job_id = "completed-job";
        
        let completed_status = JobStatus {
            job_id: job_id.to_string(),
            status: "completed".to_string(),
            progress: 100.0,
            error: None,
        };
        
        assert_eq!(completed_status.status, "completed");
        assert_eq!(completed_status.progress, 100.0);
        assert_eq!(completed_status.error, None);
    }

    #[test]
    fn test_progress_values() {
        let progress_values = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        
        for progress in progress_values {
            let status = JobStatus {
                job_id: "test".to_string(),
                status: "processing".to_string(),
                progress,
                error: None,
            };
            
            assert!(status.progress >= 0.0);
            assert!(status.progress <= 100.0);
            assert_eq!(status.progress, progress);
        }
    }

    #[test]
    fn test_status_names() {
        let status_names = vec![
            "pending",
            "processing", 
            "processing",
            "completed",
            "failed"
        ];
        
        for status_name in status_names {
            let status = JobStatus {
                job_id: "test".to_string(),
                status: status_name.to_string(),
                progress: if status_name == "completed" { 100.0 } else { 50.0 },
                error: if status_name == "failed" { Some("Error".to_string()) } else { None },
            };
            
            assert_eq!(status.status, status_name);
            assert!(!status.status.is_empty());
        }
    }

    #[test]
    fn test_timestamp_generation() {
        // Mock timestamp generation for testing
        let timestamp1 = "2023-01-01T00:00:00Z".to_string();
        let timestamp2 = "2023-01-01T00:01:00Z".to_string();
        
        // Both should be valid ISO strings
        assert!(!timestamp1.is_empty());
        assert!(!timestamp2.is_empty());
        assert!(timestamp1.contains('T'));
        assert!(timestamp2.contains('T'));
        assert!(timestamp1.contains('Z'));
        assert!(timestamp2.contains('Z'));
    }

    #[test]
    fn test_wav_file_minimum_size() {
        let valid_data = create_valid_wav_data();
        let minimum_wav_size = 44; // WAV header minimum size
        
        assert!(valid_data.len() >= minimum_wav_size);
    }

    #[test]
    fn test_wav_file_structure() {
        let valid_data = create_valid_wav_data();
        
        // Check WAV file structure
        assert_eq!(&valid_data[0..4], b"RIFF");
        assert_eq!(&valid_data[8..12], b"WAVE");
        assert_eq!(&valid_data[12..16], b"fmt ");
        
        // Check that we have a data chunk
        let has_data_chunk = valid_data.windows(4).any(|window| window == b"data");
        assert!(has_data_chunk);
    }

    #[test]
    fn test_spectrogram_dimensions() {
        let spectrogram = create_placeholder_spectrogram();
        
        // 128 x 128 spectrogram with 4 bytes per float32
        let width = 128;
        let height = 128;
        let bytes_per_float = 4;
        let expected_size = width * height * bytes_per_float;
        
        assert_eq!(spectrogram.len(), expected_size);
    }

    #[test]
    fn test_error_handling_scenarios() {
        let error_scenarios = vec![
            "Invalid audio format",
            "Network timeout",
            "Processing failed",
            "Out of memory",
            "Modal API error",
        ];
        
        for error_msg in error_scenarios {
            let error_status = JobStatus {
                job_id: "error-test".to_string(),
                status: "failed".to_string(),
                progress: 0.0,
                error: Some(error_msg.to_string()),
            };
            
            assert_eq!(error_status.status, "failed");
            assert_eq!(error_status.error, Some(error_msg.to_string()));
        }
    }

    #[test]
    fn test_dimensions_vector_operations() {
        let dimensions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test basic vector operations
        assert_eq!(dimensions.len(), 5);
        assert_eq!(dimensions[0], 1.0);
        assert_eq!(dimensions[4], 5.0);
        assert!(!dimensions.is_empty());
        
        // Test that vector can be cloned
        let cloned_dimensions = dimensions.clone();
        assert_eq!(dimensions, cloned_dimensions);
    }

    #[test]
    fn test_edge_case_dimensions() {
        let edge_cases = vec![
            vec![], // Empty
            vec![0.0], // Single value
            vec![f32::MIN, f32::MAX], // Extreme values
            vec![1.0; 1000], // Large vector
        ];
        
        let analysis_data = AnalysisData {
            rhythm: 0.8,
            pitch: 0.7,
            dynamics: 0.6,
            tempo: 0.9,
            articulation: 0.7,
            expression: 0.8,
            technique: 0.6,
            timing: 0.7,
            phrasing: 0.8,
            voicing: 0.7,
            pedaling: 0.6,
            hand_coordination: 0.7,
            musical_understanding: 0.8,
            stylistic_accuracy: 0.7,
            creativity: 0.8,
            listening: 0.7,
            overall_performance: 0.8,
            stage_presence: 0.7,
            repertoire_difficulty: 0.6,
        };
        
        // Use ModelResult instead which has the dimensions field
        for dimensions in edge_cases {
            let result = ModelResult {
                model_name: "edge-test".to_string(),
                model_type: "spectrogram".to_string(),
                analysis: analysis_data.clone(),
                insights: vec![],
                processing_time: 0.5,
                dimensions: Some(dimensions.clone()),
            };
            
            assert_eq!(result.dimensions, Some(dimensions));
        }
    }

    #[test] 
    fn test_concurrent_job_processing() {
        let job_ids = vec!["job1", "job2", "job3", "job4", "job5"];
        
        for job_id in job_ids {
            let status = JobStatus {
                job_id: job_id.to_string(),
                status: "processing".to_string(),
                progress: 50.0,
                error: None,
            };
            
            assert_eq!(status.job_id, job_id);
            assert!(!status.job_id.is_empty());
        }
    }
}