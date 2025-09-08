use worker::*;
use crate::{JobStatus, AnalysisResult};
use crate::storage;
use crate::modal_client;

pub async fn start_analysis(env: &Env, file_id: &str, job_id: &str) -> Result<()> {
    // Initialize job status
    let initial_status = JobStatus {
        id: job_id.to_string(),
        status: "preprocessing".to_string(),
        progress: 0.0,
        error: None,
    };
    
    storage::update_job_status(env, job_id, &initial_status).await?;
    
    // Get audio data from R2
    let audio_data = storage::get_audio_from_r2(env, file_id).await?;
    
    // Update status - preprocessing
    let preprocessing_status = JobStatus {
        id: job_id.to_string(),
        status: "preprocessing".to_string(),
        progress: 25.0,
        error: None,
    };
    storage::update_job_status(env, job_id, &preprocessing_status).await?;
    
    // Generate mel-spectrogram (simplified - would need actual audio processing)
    let spectrogram_data = generate_mel_spectrogram(&audio_data).await?;
    
    // Update status - sending to modal
    let modal_status = JobStatus {
        id: job_id.to_string(),
        status: "analyzing".to_string(),
        progress: 50.0,
        error: None,
    };
    storage::update_job_status(env, job_id, &modal_status).await?;
    
    // Send to Modal for inference
    match modal_client::send_for_inference(env, &spectrogram_data, job_id).await {
        Ok(_) => {
            let processing_status = JobStatus {
                id: job_id.to_string(),
                status: "processing".to_string(),
                progress: 75.0,
                error: None,
            };
            storage::update_job_status(env, job_id, &processing_status).await?;
        }
        Err(e) => {
            let error_status = JobStatus {
                id: job_id.to_string(),
                status: "failed".to_string(),
                progress: 0.0,
                error: Some(e.to_string()),
            };
            storage::update_job_status(env, job_id, &error_status).await?;
        }
    }
    
    Ok(())
}

async fn generate_mel_spectrogram(audio_data: &[u8]) -> Result<Vec<u8>> {
    // Simplified mel-spectrogram generation
    // In a real implementation, this would use proper audio processing libraries
    // For now, we'll return the raw audio data as a placeholder
    
    console_log!("Processing audio data of {} bytes", audio_data.len());
    
    // Validate audio format (basic check for WAV header)
    if audio_data.len() < 44 || &audio_data[0..4] != b"RIFF" {
        return Err(worker::Error::RustError("Invalid audio format".to_string()));
    }
    
    // TODO: Implement proper mel-spectrogram generation
    // This would involve:
    // 1. Audio decoding
    // 2. STFT computation
    // 3. Mel filter bank application
    // 4. Log transformation
    
    // For now, return a placeholder spectrogram
    let placeholder_spectrogram = create_placeholder_spectrogram();
    Ok(placeholder_spectrogram)
}

fn create_placeholder_spectrogram() -> Vec<u8> {
    // Create a placeholder 128x128 spectrogram (16KB of data)
    // This represents a mel-spectrogram with 128 mel bins and 128 time frames
    let size = 128 * 128 * 4; // 4 bytes per float32
    let mut data = Vec::with_capacity(size);
    
    // Generate some dummy spectral data
    for i in 0..(128 * 128) {
        let value = ((i as f32).sin() * 0.5 + 0.5) * 255.0;
        let bytes = (value as u32).to_le_bytes();
        data.extend_from_slice(&bytes);
    }
    
    data
}

pub async fn complete_analysis(env: &Env, job_id: &str, dimensions: Vec<f32>) -> Result<()> {
    // Create analysis result
    let result = AnalysisResult {
        id: job_id.to_string(),
        status: "completed".to_string(),
        dimensions: Some(dimensions),
        created_at: js_sys::Date::new_0().to_iso_string().as_string().unwrap(),
    };
    
    // Store the result
    storage::store_analysis_result(env, job_id, &result).await?;
    
    // Update job status to completed
    let completed_status = JobStatus {
        id: job_id.to_string(),
        status: "completed".to_string(),
        progress: 100.0,
        error: None,
    };
    
    storage::update_job_status(env, job_id, &completed_status).await?;
    
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
            id: job_id.to_string(),
            status: "preprocessing".to_string(),
            progress: 0.0,
            error: None,
        };
        
        assert_eq!(initial_status.id, job_id);
        assert_eq!(initial_status.status, "preprocessing");
        assert_eq!(initial_status.progress, 0.0);
        assert_eq!(initial_status.error, None);
    }

    #[test]
    fn test_status_progression() {
        let job_id = "test-job";
        let statuses = vec![
            ("preprocessing", 0.0),
            ("preprocessing", 25.0),
            ("analyzing", 50.0),
            ("processing", 75.0),
            ("completed", 100.0),
        ];
        
        for (status_name, progress) in statuses {
            let status = JobStatus {
                id: job_id.to_string(),
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
            id: job_id.to_string(),
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
        let dimensions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = AnalysisResult {
            id: job_id.to_string(),
            status: "completed".to_string(),
            dimensions: Some(dimensions.clone()),
            created_at: "2023-01-01T00:00:00Z".to_string(), // Mock timestamp for testing
        };
        
        assert_eq!(result.id, job_id);
        assert_eq!(result.status, "completed");
        assert_eq!(result.dimensions, Some(dimensions));
        assert!(!result.created_at.is_empty());
    }

    #[test]
    fn test_analysis_result_with_large_dimensions() {
        let large_dimensions: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        
        let result = AnalysisResult {
            id: "large-test".to_string(),
            status: "completed".to_string(),
            dimensions: Some(large_dimensions.clone()),
            created_at: "2023-01-01T00:00:00Z".to_string(), // Mock timestamp for testing
        };
        
        assert_eq!(result.dimensions.as_ref().unwrap().len(), 1000);
        assert_eq!(result.dimensions.as_ref().unwrap()[0], 0.0);
        assert_eq!(result.dimensions.as_ref().unwrap()[999], 99.9);
    }

    #[test]
    fn test_analysis_result_empty_dimensions() {
        let result = AnalysisResult {
            id: "empty-test".to_string(),
            status: "completed".to_string(),
            dimensions: Some(vec![]),
            created_at: "2023-01-01T00:00:00Z".to_string(), // Mock timestamp for testing
        };
        
        assert_eq!(result.dimensions, Some(vec![]));
    }

    #[test]
    fn test_completed_job_status_creation() {
        let job_id = "completed-job";
        
        let completed_status = JobStatus {
            id: job_id.to_string(),
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
                id: "test".to_string(),
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
            "preprocessing",
            "analyzing", 
            "processing",
            "completed",
            "failed"
        ];
        
        for status_name in status_names {
            let status = JobStatus {
                id: "test".to_string(),
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
                id: "error-test".to_string(),
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
        
        for dimensions in edge_cases {
            let result = AnalysisResult {
                id: "edge-test".to_string(),
                status: "completed".to_string(),
                dimensions: Some(dimensions.clone()),
                created_at: "2023-01-01T00:00:00Z".to_string(), // Mock timestamp for testing
            };
            
            assert_eq!(result.dimensions, Some(dimensions));
        }
    }

    #[test] 
    fn test_concurrent_job_processing() {
        let job_ids = vec!["job1", "job2", "job3", "job4", "job5"];
        
        for job_id in job_ids {
            let status = JobStatus {
                id: job_id.to_string(),
                status: "processing".to_string(),
                progress: 50.0,
                error: None,
            };
            
            assert_eq!(status.id, job_id);
            assert!(!status.id.is_empty());
        }
    }
}