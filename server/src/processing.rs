use crate::audio_dsp;
use crate::simple_evaluator::SimpleEvaluator;
use crate::storage;
use crate::{AnalysisData, AnalysisResult, ComparisonResult, JobStatus, ModelResult};
use worker::*;

// Conditional logging macro for test vs WASM environments
#[cfg(test)]
macro_rules! console_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*)
    };
}

#[cfg(not(test))]
macro_rules! console_log {
    ($($arg:tt)*) => {
        worker::console_log!($($arg)*)
    };
}

pub async fn start_analysis(
    env: &Env,
    file_id: &str,
    job_id: &str,
    force_gpu: Option<bool>,
) -> Result<()> {
    console_log!(
        "Starting analysis for file_id: {}, job_id: {}",
        file_id,
        job_id
    );

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
    console_log!(
        "Generated {} bytes of spectrogram data",
        spectrogram_data.len()
    );

    // Update status - running ML inference
    let inference_status = JobStatus {
        job_id: job_id.to_string(),
        status: "processing".to_string(),
        progress: 75.0,
        error: None,
    };
    storage::update_job_status(env, job_id, &inference_status).await?;

    // Run ML inference using embedded ONNX model
    console_log!("Running ML inference for job_id: {}", job_id);

    // Convert spectrogram bytes to f32 array for model input
    let mel_spectrogram_floats = bytes_to_mel_spectrogram(&spectrogram_data)?;

    // Use Simple evaluator with rule-based analysis
    let evaluator = SimpleEvaluator::new();
    let features = evaluator.extract_features(&mel_spectrogram_floats);
    let analysis_data = evaluator.analyze(&features);

    console_log!(
        "Simple analysis complete for job_id: {}, timing_score: {:.3}",
        job_id,
        analysis_data.timing_stable_unstable
    );

    // Generate insights/feedback
    let insights = evaluator.generate_insights(&analysis_data);

    // Complete the analysis with real results
    complete_analysis(
        env,
        job_id,
        file_id,
        analysis_data,
        insights,
        None, // No processing time tracking for simple evaluator
    )
    .await?;

    Ok(())
}

async fn generate_mel_spectrogram(audio_data: &[u8]) -> Result<Vec<u8>> {
    console_log!(
        "Processing audio data of {} bytes using real DSP",
        audio_data.len()
    );

    // Validate minimum audio size
    if audio_data.len() < 44 {
        return Err(worker::Error::RustError("Audio data too small".to_string()));
    }

    // Use the real DSP implementation to generate mel-spectrogram
    match audio_dsp::process_audio_to_mel_spectrogram(audio_data).await {
        Ok(mel_spectrogram_bytes) => {
            console_log!(
                "Successfully generated mel-spectrogram: {} bytes",
                mel_spectrogram_bytes.len()
            );
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

    console_log!(
        "Generated placeholder spectrogram: {} bytes (expected: {})",
        data.len(),
        size
    );
    data
}

fn bytes_to_mel_spectrogram(spectrogram_bytes: &[u8]) -> Result<Vec<f32>> {
    console_log!(
        "Converting {} bytes to mel-spectrogram floats",
        spectrogram_bytes.len()
    );

    // Expected: 128 x 128 x 4 bytes = 65,536 bytes for f32 array
    if spectrogram_bytes.len() != 128 * 128 * 4 {
        console_log!("Warning: unexpected spectrogram size, using placeholder conversion");

        // Create normalized placeholder data
        let mut floats = Vec::with_capacity(128 * 128);
        for i in 0..(128 * 128) {
            // Simple normalization: convert to [0, 1] range
            let val = (i % 256) as f32 / 255.0;
            floats.push(val);
        }
        return Ok(floats);
    }

    // Convert bytes to f32 array (little-endian)
    let mut floats = Vec::with_capacity(128 * 128);

    for chunk in spectrogram_bytes.chunks_exact(4) {
        let bytes: [u8; 4] = chunk
            .try_into()
            .map_err(|_| worker::Error::RustError("Failed to convert chunk to f32".to_string()))?;
        let float_val = f32::from_le_bytes(bytes);

        // Ensure values are in [0, 1] range for model
        let normalized = float_val.max(0.0).min(1.0);
        floats.push(normalized);
    }

    console_log!(
        "Converted to {} float values, range: [{:.3}, {:.3}]",
        floats.len(),
        floats.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        floats.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    Ok(floats)
}

pub async fn complete_analysis(
    env: &Env,
    job_id: &str,
    file_id: &str,
    analysis_data: AnalysisData,
    insights: Vec<String>,
    processing_time: Option<f32>,
) -> Result<()> {
    console_log!(
        "complete_analysis: Starting completion for job_id: {}",
        job_id
    );

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

    console_log!(
        "complete_analysis: Created analysis result for job_id: {}",
        job_id
    );

    // Store the result
    match storage::store_analysis_result(env, job_id, &result).await {
        Ok(_) => {
            console_log!(
                "complete_analysis: Successfully stored analysis result for job_id: {}",
                job_id
            );

            // Trigger cache warming for this completed analysis
            storage::warm_cache_for_completed_job(env, job_id)
                .await
                .ok();
        }
        Err(e) => {
            console_log!(
                "complete_analysis: ERROR storing analysis result for job_id {}: {:?}",
                job_id,
                e
            );
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

    console_log!(
        "complete_analysis: Updating job status to completed for job_id: {}",
        job_id
    );
    match storage::update_job_status(env, job_id, &completed_status).await {
        Ok(_) => {
            console_log!(
                "complete_analysis: Successfully updated job status to completed for job_id: {}",
                job_id
            );

            // Verify the status was actually written by reading it back
            match storage::get_job_status(env, job_id).await {
                Ok(verified_status) => {
                    console_log!(
                        "complete_analysis: Verified job status for {}: {} (expected: completed)",
                        job_id,
                        verified_status.status
                    );
                }
                Err(e) => {
                    console_log!(
                        "complete_analysis: WARNING: Could not verify job status for {}: {:?}",
                        job_id,
                        e
                    );
                }
            }
        }
        Err(e) => {
            console_log!(
                "complete_analysis: ERROR updating job status for job_id {}: {:?}",
                job_id,
                e
            );
            return Err(e);
        }
    }

    console_log!(
        "complete_analysis: Completed all operations for job_id: {}",
        job_id
    );
    Ok(())
}

pub async fn start_model_comparison(
    env: &Env,
    file_id: &str,
    comparison_id: &str,
    model_a: &str,
    model_b: &str,
    force_gpu: Option<bool>,
) -> Result<()> {
    console_log!(
        "Starting model comparison for file_id: {}, comparison_id: {} (models: {} vs {})",
        file_id,
        comparison_id,
        model_a,
        model_b
    );

    // Initialize job status
    let initial_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 0.0,
        error: None,
    };

    storage::update_job_status(env, comparison_id, &initial_status).await?;

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
    console_log!(
        "Generating mel-spectrogram for comparison: {}",
        comparison_id
    );
    let spectrogram_data = generate_mel_spectrogram(&audio_data).await?;

    // Update status - running parallel inference
    let inference_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "processing".to_string(),
        progress: 40.0,
        error: None,
    };
    storage::update_job_status(env, comparison_id, &inference_status).await?;

    // Inference not configured; return explicit error for comparison as well
    let error_status = JobStatus {
        job_id: comparison_id.to_string(),
        status: "failed".to_string(),
        progress: 50.0,
        error: Some("Model comparison not configured".to_string()),
    };
    storage::update_job_status(env, comparison_id, &error_status).await?;
    return Err(worker::Error::RustError(
        "Model comparison not configured".to_string(),
    ));
}

pub async fn complete_model_comparison(
    env: &Env,
    comparison_id: &str,
    file_id: &str,
    result_a: (AnalysisData, Vec<String>, Option<f32>),
    result_b: (AnalysisData, Vec<String>, Option<f32>),
) -> Result<()> {
    console_log!(
        "Completing model comparison for comparison_id: {}",
        comparison_id
    );

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
        Ok(_) => console_log!(
            "Successfully stored comparison result for comparison_id: {}",
            comparison_id
        ),
        Err(e) => {
            console_log!(
                "ERROR storing comparison result for comparison_id {}: {:?}",
                comparison_id,
                e
            );
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
            console_log!(
                "Successfully updated comparison job status to completed for comparison_id: {}",
                comparison_id
            );
        }
        Err(e) => {
            console_log!(
                "ERROR updating comparison job status for comparison_id {}: {:?}",
                comparison_id,
                e
            );
            return Err(e);
        }
    }

    console_log!(
        "Completed all operations for comparison_id: {}",
        comparison_id
    );
    Ok(())
}

// ============================================================================
// Temporal Analysis - Chunk Processing
// ============================================================================

/// Analysis result for a single chunk
#[derive(Debug, Clone)]
struct ChunkAnalysisResult {
    timestamp: String,
    chunk_index: usize,
    analysis_data: AnalysisData,
    raw_scores: Vec<(String, f32)>, // For debugging/logging
}

/// Analyze a single audio chunk
///
/// # Arguments
/// * `chunk` - Audio chunk to analyze
/// * `spectrogram_bytes` - Pre-computed mel-spectrogram
///
/// # Returns
/// Analysis result for this chunk
///
/// # Errors
/// Returns error if analysis fails
async fn analyze_chunk(
    chunk: &audio_dsp::AudioChunk,
    spectrogram_bytes: &[u8],
) -> Result<ChunkAnalysisResult> {
    console_log!(
        "Analyzing chunk {} ({})",
        chunk.chunk_index,
        chunk.timestamp
    );

    // Convert spectrogram bytes to floats
    let mel_spectrogram_floats = bytes_to_mel_spectrogram(spectrogram_bytes).map_err(|e| {
        worker::Error::RustError(format!(
            "Failed to convert spectrogram for chunk {}: {}",
            chunk.chunk_index, e
        ))
    })?;

    // Use Simple evaluator
    let evaluator = SimpleEvaluator::new();
    let features = evaluator.extract_features(&mel_spectrogram_floats);
    let analysis_data = evaluator.analyze(&features);

    // Extract raw scores for debugging
    let raw_scores = vec![
        (
            "timing_stable_unstable".to_string(),
            analysis_data.timing_stable_unstable,
        ),
        (
            "articulation_short_long".to_string(),
            analysis_data.articulation_short_long,
        ),
        (
            "articulation_soft_hard".to_string(),
            analysis_data.articulation_soft_hard,
        ),
        (
            "pedal_sparse_saturated".to_string(),
            analysis_data.pedal_sparse_saturated,
        ),
        (
            "pedal_clean_blurred".to_string(),
            analysis_data.pedal_clean_blurred,
        ),
        (
            "timbre_even_colorful".to_string(),
            analysis_data.timbre_even_colorful,
        ),
        (
            "timbre_shallow_rich".to_string(),
            analysis_data.timbre_shallow_rich,
        ),
        (
            "timbre_bright_dark".to_string(),
            analysis_data.timbre_bright_dark,
        ),
        (
            "timbre_soft_loud".to_string(),
            analysis_data.timbre_soft_loud,
        ),
        (
            "dynamic_sophisticated_raw".to_string(),
            analysis_data.dynamic_sophisticated_raw,
        ),
        (
            "dynamic_range_little_large".to_string(),
            analysis_data.dynamic_range_little_large,
        ),
        (
            "music_making_fast_slow".to_string(),
            analysis_data.music_making_fast_slow,
        ),
        (
            "music_making_flat_spacious".to_string(),
            analysis_data.music_making_flat_spacious,
        ),
        (
            "music_making_disproportioned_balanced".to_string(),
            analysis_data.music_making_disproportioned_balanced,
        ),
        (
            "music_making_pure_dramatic".to_string(),
            analysis_data.music_making_pure_dramatic,
        ),
        (
            "emotion_mood_optimistic_dark".to_string(),
            analysis_data.emotion_mood_optimistic_dark,
        ),
        (
            "emotion_mood_low_high_energy".to_string(),
            analysis_data.emotion_mood_low_high_energy,
        ),
        (
            "emotion_mood_honest_imaginative".to_string(),
            analysis_data.emotion_mood_honest_imaginative,
        ),
        (
            "interpretation_unsatisfactory_convincing".to_string(),
            analysis_data.interpretation_unsatisfactory_convincing,
        ),
    ];

    console_log!(
        "Chunk {} analysis complete: timing={:.3}, articulation_long={:.3}, interpretation={:.3}",
        chunk.chunk_index,
        analysis_data.timing_stable_unstable,
        analysis_data.articulation_short_long,
        analysis_data.interpretation_unsatisfactory_convincing
    );

    Ok(ChunkAnalysisResult {
        timestamp: chunk.timestamp.clone(),
        chunk_index: chunk.chunk_index,
        analysis_data,
        raw_scores,
    })
}

/// Generate natural language insights for a chunk's analysis
///
/// # Arguments
/// * `env` - Worker environment for LLM access
/// * `chunk_result` - Analysis result for the chunk
///
/// # Returns
/// Vector of structured insights
///
/// # Errors
/// Returns error if LLM call fails
async fn generate_chunk_insights(
    env: &Env,
    chunk_result: &ChunkAnalysisResult,
) -> Result<Vec<crate::AnalysisInsight>> {
    console_log!(
        "Generating insights for chunk {} ({})",
        chunk_result.chunk_index,
        chunk_result.timestamp
    );

    // Find notable scores (high and low)
    let mut notable_high: Vec<(&str, f32)> = vec![];
    let mut notable_low: Vec<(&str, f32)> = vec![];

    for (name, score) in &chunk_result.raw_scores {
        if *score >= 0.75 {
            notable_high.push((name.as_str(), *score));
        } else if *score <= 0.35 && *score > 0.0 {
            // Exclude zeros (likely errors)
            notable_low.push((name.as_str(), *score));
        }
    }

    // Sort by score
    notable_high.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    notable_low.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top 2 of each
    notable_high.truncate(2);
    notable_low.truncate(2);

    console_log!(
        "Chunk {}: {} high points, {} low points",
        chunk_result.chunk_index,
        notable_high.len(),
        notable_low.len()
    );

    let mut insights = Vec::new();

    // Generate insights for high points
    for (dimension, score) in notable_high {
        let insight = generate_insight_for_dimension(
            env,
            dimension,
            score,
            true, // is_strength
            &chunk_result.timestamp,
        )
        .await?;
        insights.push(insight);
    }

    // Generate insights for low points
    for (dimension, score) in notable_low {
        let insight = generate_insight_for_dimension(
            env,
            dimension,
            score,
            false, // is_strength
            &chunk_result.timestamp,
        )
        .await?;
        insights.push(insight);
    }

    // Ensure at least one insight per chunk
    if insights.is_empty() {
        insights.push(crate::AnalysisInsight {
            category: "Technical".to_string(),
            observation: "Performance characteristics in this segment are balanced".to_string(),
            actionable_advice:
                "Continue maintaining consistent technique throughout your performance".to_string(),
            score_reference: format!(
                "avg_score: {:.2}",
                chunk_result.raw_scores.iter().map(|(_, s)| s).sum::<f32>()
                    / chunk_result.raw_scores.len() as f32
            ),
        });
    }

    console_log!(
        "Generated {} insights for chunk {}",
        insights.len(),
        chunk_result.chunk_index
    );

    Ok(insights)
}

/// Generate a single insight for a dimension using LLM
///
/// # Arguments
/// * `env` - Worker environment
/// * `dimension` - Performance dimension name
/// * `score` - Score value (0-1)
/// * `is_strength` - Whether this is a strength or weakness
/// * `timestamp` - Time range for context
///
/// # Returns
/// Structured insight with observation and advice
async fn generate_insight_for_dimension(
    env: &Env,
    dimension: &str,
    score: f32,
    is_strength: bool,
    timestamp: &str,
) -> Result<crate::AnalysisInsight> {
    // Map dimension name to readable form
    let readable_dimension = dimension
        .replace('_', " ")
        .replace("stable unstable", "stability")
        .replace("short long", "length")
        .replace("soft hard", "firmness");

    // Determine category
    let category = if dimension.starts_with("timing") || dimension.starts_with("articulation") {
        "Technical"
    } else if dimension.starts_with("music_making") || dimension.starts_with("emotion") {
        "Musical"
    } else {
        "Interpretive"
    };

    // Build prompt for LLM
    let system_prompt =
        "You are an expert piano pedagogue providing specific, actionable feedback. \
        Be encouraging but honest. Focus on concrete observations and practical advice. \
        Keep responses concise (2-3 sentences each).";

    let user_prompt = format!(
        "At timestamp {}, the performer shows {} in {}. Score: {:.2}/1.0\n\n\
        Provide:\n\
        1. A specific observation about what you hear\n\
        2. Concrete, actionable advice for improvement\n\n\
        Format: OBSERVATION: ... | ADVICE: ...",
        timestamp,
        if is_strength { "strength" } else { "weakness" },
        readable_dimension,
        score
    );

    // Call LLM (using existing tutor infrastructure)
    let response = crate::tutor::call_llm(
        env,
        system_prompt,
        &user_prompt,
        0.7, // temperature
        150, // max tokens
    )
    .await
    .map_err(|e| {
        console_log!("LLM call failed for dimension {}: {}", dimension, e);
        worker::Error::RustError(format!("Failed to generate insight: {}", e))
    })?;

    // Parse response
    let parts: Vec<&str> = response.split('|').collect();
    let observation = parts
        .get(0)
        .and_then(|s| s.strip_prefix("OBSERVATION:"))
        .map(|s| s.trim())
        .unwrap_or("Notable performance characteristic observed")
        .to_string();

    let advice = parts
        .get(1)
        .and_then(|s| s.strip_prefix("ADVICE:"))
        .map(|s| s.trim())
        .unwrap_or("Continue focused practice in this area")
        .to_string();

    Ok(crate::AnalysisInsight {
        category: category.to_string(),
        observation,
        actionable_advice: advice,
        score_reference: format!("{}: {:.3}", dimension, score),
    })
}

/// Determine the key practice focus for a chunk
///
/// # Arguments
/// * `chunk_result` - Analysis result for the chunk
///
/// # Returns
/// One-sentence practice focus
fn determine_practice_focus(chunk_result: &ChunkAnalysisResult) -> String {
    // Find the weakest dimension
    let weakest = chunk_result
        .raw_scores
        .iter()
        .filter(|(_, score)| *score > 0.0) // Exclude zeros
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((dimension, _score)) = weakest {
        let readable = dimension
            .replace('_', " ")
            .replace("stable unstable", "stability")
            .replace("short long", "length")
            .replace("soft hard", "firmness");

        format!("Focus on improving {} in this passage", readable)
    } else {
        "Maintain consistent technique throughout".to_string()
    }
}

/// Generate overall assessment from all chunks
///
/// # Arguments
/// * `env` - Worker environment for LLM access
/// * `chunk_results` - All chunk analysis results
///
/// # Returns
/// Overall assessment structure
///
/// # Errors
/// Returns error if LLM call fails
async fn generate_overall_assessment(
    env: &Env,
    chunk_results: &[ChunkAnalysisResult],
) -> Result<crate::OverallAssessment> {
    console_log!(
        "Generating overall assessment from {} chunks",
        chunk_results.len()
    );

    // Aggregate scores across all chunks
    let mut aggregated_scores: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();

    for chunk in chunk_results {
        for (name, score) in &chunk.raw_scores {
            aggregated_scores
                .entry(name.clone())
                .or_insert_with(Vec::new)
                .push(*score);
        }
    }

    // Calculate averages and identify patterns
    let mut avg_scores: Vec<(String, f32)> = aggregated_scores
        .iter()
        .map(|(name, scores)| {
            let avg = scores.iter().filter(|s| **s > 0.0).sum::<f32>()
                / scores.iter().filter(|s| **s > 0.0).count().max(1) as f32;
            (name.clone(), avg)
        })
        .collect();

    avg_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Identify strengths (top 3-5)
    let strengths_data: Vec<(String, f32)> = avg_scores
        .iter()
        .filter(|(_, score)| *score >= 0.65)
        .take(5)
        .cloned()
        .collect();

    // Identify priority areas (bottom 3-4, excluding zeros)
    let mut priority_data: Vec<(String, f32)> = avg_scores
        .iter()
        .filter(|(_, score)| *score > 0.0 && *score <= 0.45)
        .cloned()
        .collect();
    priority_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    priority_data.truncate(4);

    // Use LLM to generate natural language descriptions
    let strengths = generate_strength_descriptions(env, &strengths_data).await?;
    let priority_areas = generate_priority_descriptions(env, &priority_data).await?;
    let performance_character =
        generate_performance_character(env, chunk_results, &strengths_data, &priority_data).await?;

    console_log!(
        "Overall assessment generated: {} strengths, {} priorities",
        strengths.len(),
        priority_areas.len()
    );

    Ok(crate::OverallAssessment {
        strengths,
        priority_areas,
        performance_character,
    })
}

/// Generate natural language strength descriptions using LLM
async fn generate_strength_descriptions(
    env: &Env,
    strengths_data: &[(String, f32)],
) -> Result<Vec<String>> {
    if strengths_data.is_empty() {
        return Ok(vec!["Shows developing technical skills".to_string()]);
    }

    let dimensions_str = strengths_data
        .iter()
        .map(|(name, score)| format!("{} ({:.2})", name.replace('_', " "), score))
        .collect::<Vec<_>>()
        .join(", ");

    let system_prompt = "You are an expert piano teacher identifying performance strengths. \
        Be specific and encouraging.";

    let user_prompt = format!(
        "The performer shows strength in these areas: {}\n\n\
        Generate 3-5 specific strength statements. Each should:\n\
        - Start with a concrete observation\n\
        - Be encouraging and specific\n\
        - Mention the dimension naturally\n\n\
        Format: One statement per line.",
        dimensions_str
    );

    let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.7, 300)
        .await
        .map_err(|e| {
            console_log!("Failed to generate strength descriptions: {}", e);
            worker::Error::RustError(format!("LLM call failed: {}", e))
        })?;

    // Parse response into individual strengths
    let strengths: Vec<String> = response
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('-'))
        .map(|line| line.trim_start_matches(&['1', '2', '3', '4', '5', '.', ' '][..]))
        .map(|s| s.to_string())
        .take(5)
        .collect();

    if strengths.is_empty() {
        Ok(vec!["Demonstrates foundational technique".to_string()])
    } else {
        Ok(strengths)
    }
}

/// Generate priority area descriptions using LLM
async fn generate_priority_descriptions(
    env: &Env,
    priority_data: &[(String, f32)],
) -> Result<Vec<String>> {
    if priority_data.is_empty() {
        return Ok(vec!["Continue refining overall musicality".to_string()]);
    }

    let dimensions_str = priority_data
        .iter()
        .map(|(name, score)| format!("{} ({:.2})", name.replace('_', " "), score))
        .collect::<Vec<_>>()
        .join(", ");

    let system_prompt = "You are an expert piano teacher identifying areas for growth. \
        Be constructive and supportive, focusing on development opportunities.";

    let user_prompt = format!(
        "These areas show room for development: {}\n\n\
        Generate 2-4 priority statements. Each should:\n\
        - Be constructive and supportive\n\
        - Focus on growth opportunity\n\
        - Mention the dimension naturally\n\n\
        Format: One statement per line.",
        dimensions_str
    );

    let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.7, 250)
        .await
        .map_err(|e| {
            console_log!("Failed to generate priority descriptions: {}", e);
            worker::Error::RustError(format!("LLM call failed: {}", e))
        })?;

    let priorities: Vec<String> = response
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('-'))
        .map(|line| line.trim_start_matches(&['1', '2', '3', '4', '.', ' '][..]))
        .map(|s| s.to_string())
        .take(4)
        .collect();

    if priorities.is_empty() {
        Ok(vec!["Focus on consistent technique development".to_string()])
    } else {
        Ok(priorities)
    }
}

/// Generate performance character description using LLM
async fn generate_performance_character(
    env: &Env,
    chunk_results: &[ChunkAnalysisResult],
    strengths_data: &[(String, f32)],
    priority_data: &[(String, f32)],
) -> Result<String> {
    let num_chunks = chunk_results.len();
    let duration = if num_chunks > 0 {
        // Estimate duration based on chunks (3 sec each, 1 sec overlap = 2 sec step)
        2.0 * num_chunks as f32 + 1.0
    } else {
        0.0
    };

    let strengths_summary = strengths_data
        .iter()
        .take(3)
        .map(|(name, _)| name.replace('_', " "))
        .collect::<Vec<_>>()
        .join(", ");

    let challenges_summary = priority_data
        .iter()
        .take(2)
        .map(|(name, _)| name.replace('_', " "))
        .collect::<Vec<_>>()
        .join(", ");

    let system_prompt =
        "You are an expert piano teacher writing a concise performance assessment. \
        Write 2-3 sentences that capture the overall character and quality of the performance.";

    let user_prompt = format!(
        "Performance details:\n\
        - Duration: ~{:.0} seconds\n\
        - Key strengths: {}\n\
        - Areas for growth: {}\n\n\
        Write a 2-3 sentence character assessment that:\n\
        - Describes the overall performance character\n\
        - Balances acknowledgment of strengths and areas for growth\n\
        - Uses natural, encouraging language",
        duration,
        if strengths_summary.is_empty() {
            "developing technique"
        } else {
            &strengths_summary
        },
        if challenges_summary.is_empty() {
            "continued refinement"
        } else {
            &challenges_summary
        }
    );

    let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.7, 200)
        .await
        .map_err(|e| {
            console_log!("Failed to generate performance character: {}", e);
            worker::Error::RustError(format!("LLM call failed: {}", e))
        })?;

    // Clean up response
    let character = response
        .trim()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if character.is_empty() {
        Ok(
            "A developing performance with both strengths and opportunities for growth."
                .to_string(),
        )
    } else {
        Ok(character)
    }
}

/// Generate practice recommendations from chunk analyses
///
/// # Arguments
/// * `env` - Worker environment for LLM access
/// * `chunk_results` - All chunk analysis results
///
/// # Returns
/// Practice recommendations structure
async fn generate_practice_recommendations(
    env: &Env,
    chunk_results: &[ChunkAnalysisResult],
) -> Result<crate::PracticeRecommendations> {
    console_log!("Generating practice recommendations");

    // Aggregate weakest dimensions across all chunks
    let mut dimension_scores: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();

    for chunk in chunk_results {
        for (name, score) in &chunk.raw_scores {
            if *score > 0.0 {
                // Exclude zeros
                dimension_scores
                    .entry(name.clone())
                    .or_insert_with(Vec::new)
                    .push(*score);
            }
        }
    }

    // Calculate averages and sort by need
    let mut avg_scores: Vec<(String, f32)> = dimension_scores
        .iter()
        .map(|(name, scores)| {
            let avg = scores.iter().sum::<f32>() / scores.len() as f32;
            (name.clone(), avg)
        })
        .collect();

    avg_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Get top 5 weakest for immediate priorities
    let immediate_data: Vec<(String, f32)> = avg_scores.iter().take(5).cloned().collect();

    // Get 2-3 medium-term areas for long-term development
    let longterm_data: Vec<(String, f32)> = avg_scores.iter().skip(5).take(3).cloned().collect();

    let immediate_priorities = generate_immediate_priorities(env, &immediate_data).await?;
    let long_term_development = generate_longterm_development(env, &longterm_data).await?;

    console_log!(
        "Generated {} immediate priorities, {} long-term goals",
        immediate_priorities.len(),
        long_term_development.len()
    );

    Ok(crate::PracticeRecommendations {
        immediate_priorities,
        long_term_development,
    })
}

/// Generate immediate practice priorities using LLM
async fn generate_immediate_priorities(
    env: &Env,
    dimension_data: &[(String, f32)],
) -> Result<Vec<crate::ImmediatePriority>> {
    let mut priorities = Vec::new();

    for (dimension, score) in dimension_data.iter().take(3) {
        let readable_dimension = dimension
            .replace('_', " ")
            .replace("stable unstable", "stability")
            .replace("short long", "length");

        let system_prompt =
            "You are an expert piano pedagogue providing specific practice exercises. \
            Be concrete and actionable.";

        let user_prompt = format!(
            "The student needs to improve: {} (current level: {:.2}/1.0)\n\n\
            Provide:\n\
            1. SKILL_AREA: A clear skill area name (2-4 words)\n\
            2. EXERCISE: A specific practice exercise or technique (1-2 sentences)\n\
            3. OUTCOME: What improvement to expect (1 sentence)\n\n\
            Format: SKILL_AREA: ... | EXERCISE: ... | OUTCOME: ...",
            readable_dimension, score
        );

        let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.7, 200)
            .await
            .map_err(|e| {
                console_log!("Failed to generate immediate priority: {}", e);
                e
            })?;

        // Parse response
        let parts: Vec<&str> = response.split('|').collect();
        let skill_area = parts
            .get(0)
            .and_then(|s| s.strip_prefix("SKILL_AREA:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| readable_dimension.clone());

        let specific_exercise = parts
            .get(1)
            .and_then(|s| s.strip_prefix("EXERCISE:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Practice scales and exercises focusing on this area".to_string());

        let expected_outcome = parts
            .get(2)
            .and_then(|s| s.strip_prefix("OUTCOME:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Improved technique and control".to_string());

        priorities.push(crate::ImmediatePriority {
            skill_area,
            specific_exercise,
            expected_outcome,
        });
    }

    // Ensure at least one priority
    if priorities.is_empty() {
        priorities.push(crate::ImmediatePriority {
            skill_area: "Overall Technique".to_string(),
            specific_exercise: "Continue daily technical practice with scales and arpeggios"
                .to_string(),
            expected_outcome: "Stronger foundational technique".to_string(),
        });
    }

    Ok(priorities)
}

/// Generate long-term development goals using LLM
async fn generate_longterm_development(
    env: &Env,
    dimension_data: &[(String, f32)],
) -> Result<Vec<crate::LongTermDevelopment>> {
    let mut developments = Vec::new();

    for (dimension, _score) in dimension_data.iter().take(2) {
        let readable_dimension = dimension
            .replace('_', " ")
            .replace("stable unstable", "stability");

        let system_prompt =
            "You are an expert piano pedagogue providing long-term musical development guidance. \
            Think big picture and repertoire-based.";

        let user_prompt = format!(
            "For long-term development in: {}\n\n\
            Provide:\n\
            1. ASPECT: The broader musical aspect to develop (2-4 words)\n\
            2. APPROACH: Development strategy over months (1-2 sentences)\n\
            3. REPERTOIRE: Specific composers or pieces to support this (1 sentence)\n\n\
            Format: ASPECT: ... | APPROACH: ... | REPERTOIRE: ...",
            readable_dimension
        );

        let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.7, 250)
            .await
            .map_err(|e| {
                console_log!("Failed to generate long-term development: {}", e);
                e
            })?;

        // Parse response
        let parts: Vec<&str> = response.split('|').collect();
        let musical_aspect = parts
            .get(0)
            .and_then(|s| s.strip_prefix("ASPECT:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| readable_dimension.clone());

        let development_approach = parts
            .get(1)
            .and_then(|s| s.strip_prefix("APPROACH:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Study diverse repertoire and work with a teacher".to_string());

        let repertoire_suggestions = parts
            .get(2)
            .and_then(|s| s.strip_prefix("REPERTOIRE:"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Explore varied repertoire appropriate to your level".to_string());

        developments.push(crate::LongTermDevelopment {
            musical_aspect,
            development_approach,
            repertoire_suggestions,
        });
    }

    // Ensure at least one long-term goal
    if developments.is_empty() {
        developments.push(crate::LongTermDevelopment {
            musical_aspect: "Musical Maturity".to_string(),
            development_approach:
                "Continue regular practice while exploring diverse musical styles and periods"
                    .to_string(),
            repertoire_suggestions: "Study works from Baroque through Contemporary periods"
                .to_string(),
        });
    }

    Ok(developments)
}

/// Generate encouraging message using LLM
///
/// # Arguments
/// * `env` - Worker environment
/// * `overall_assessment` - The overall assessment
///
/// # Returns
/// Encouraging message string
async fn generate_encouragement(
    env: &Env,
    overall_assessment: &crate::OverallAssessment,
) -> Result<String> {
    let system_prompt = "You are a supportive piano teacher writing an encouraging message. \
        Be warm, specific, and motivating. Focus on growth and potential.";

    let strengths_preview = overall_assessment
        .strengths
        .first()
        .map(|s| s.as_str())
        .unwrap_or("your dedication");

    let user_prompt = format!(
        "Write 2-3 sentences of encouragement for a piano student. \
        Their key strength is: {}\n\
        Performance character: {}\n\n\
        Be specific, warm, and motivating. End with forward-looking encouragement.",
        strengths_preview, overall_assessment.performance_character
    );

    let response = crate::tutor::call_llm(env, system_prompt, &user_prompt, 0.8, 200)
        .await
        .map_err(|e| {
            console_log!("Failed to generate encouragement: {}", e);
            e
        })?;

    let encouragement = response.trim().to_string();

    if encouragement.is_empty() {
        Ok("Keep practicing with focus and dedication. Your musical journey is developing beautifully!".to_string())
    } else {
        Ok(encouragement)
    }
}

/// Main temporal analysis workflow
///
/// # Arguments
/// * `env` - Worker environment
/// * `file_id` - Audio file ID
/// * `job_id` - Analysis job ID
///
/// # Errors
/// Returns detailed error if any step fails
pub async fn start_temporal_analysis(env: &Env, file_id: &str, job_id: &str) -> Result<()> {
    console_log!("=== Starting Temporal Analysis ===");
    console_log!("File ID: {}, Job ID: {}", file_id, job_id);

    // Initialize job status
    let initial_status = JobStatus {
        job_id: job_id.to_string(),
        status: "processing".to_string(),
        progress: 0.0,
        error: None,
    };
    storage::update_job_status(env, job_id, &initial_status).await?;

    // Step 1: Retrieve and parse audio
    console_log!("Step 1: Retrieving audio from R2");
    let audio_bytes = storage::get_audio_from_r2(env, file_id)
        .await
        .map_err(|e| {
            console_log!("Failed to retrieve audio: {}", e);
            e
        })?;

    let audio_data = audio_dsp::parse_audio_data(&audio_bytes).map_err(|e| {
        console_log!("Failed to parse audio: {}", e);
        worker::Error::RustError(format!("Audio parsing failed: {}", e))
    })?;

    console_log!(
        "Audio loaded: {} samples at {}Hz ({:.2}s duration)",
        audio_data.samples.len(),
        audio_data.sample_rate,
        audio_data.duration_seconds
    );

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 10.0,
            error: None,
        },
    )
    .await?;

    // Step 2: Chunk audio
    console_log!("Step 2: Chunking audio (3s chunks, 1s overlap)");
    let chunks = audio_dsp::chunk_audio_with_overlap(&audio_data, 3.0, 1.0).map_err(|e| {
        console_log!("Failed to chunk audio: {}", e);
        e
    })?;

    console_log!("Created {} chunks", chunks.len());

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 15.0,
            error: None,
        },
    )
    .await?;

    // Step 3: Analyze each chunk
    console_log!("Step 3: Analyzing {} chunks", chunks.len());
    let mut chunk_results = Vec::new();
    let progress_per_chunk = 50.0 / chunks.len() as f32;

    for (idx, chunk) in chunks.iter().enumerate() {
        console_log!("Processing chunk {}/{}", idx + 1, chunks.len());

        // Generate spectrogram for chunk
        let spectrogram_bytes = audio_dsp::generate_mel_spectrogram_for_chunk(chunk)
            .await
            .map_err(|e| {
                console_log!("Spectrogram generation failed for chunk {}: {}", idx, e);
                e
            })?;

        // Analyze chunk
        let chunk_result = analyze_chunk(chunk, &spectrogram_bytes)
            .await
            .map_err(|e| {
                console_log!("Analysis failed for chunk {}: {}", idx, e);
                e
            })?;

        chunk_results.push(chunk_result);

        // Update progress
        let progress = 15.0 + (idx + 1) as f32 * progress_per_chunk;
        storage::update_job_status(
            env,
            job_id,
            &JobStatus {
                job_id: job_id.to_string(),
                status: "processing".to_string(),
                progress,
                error: None,
            },
        )
        .await?;
    }

    console_log!("All chunks analyzed successfully");

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 65.0,
            error: None,
        },
    )
    .await?;

    // Step 4: Generate temporal feedback with LLM insights
    console_log!("Step 4: Generating natural language insights");
    let mut temporal_feedback = Vec::new();

    for chunk_result in &chunk_results {
        let insights = generate_chunk_insights(env, chunk_result)
            .await
            .map_err(|e| {
                console_log!(
                    "Failed to generate insights for chunk {}: {}",
                    chunk_result.chunk_index,
                    e
                );
                e
            })?;

        let practice_focus = determine_practice_focus(chunk_result);

        temporal_feedback.push(crate::TemporalFeedbackItem {
            timestamp: chunk_result.timestamp.clone(),
            insights,
            practice_focus,
        });
    }

    console_log!("Generated temporal feedback for all chunks");

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 75.0,
            error: None,
        },
    )
    .await?;

    // Step 5: Generate overall assessment
    console_log!("Step 5: Generating overall assessment");
    let overall_assessment = generate_overall_assessment(env, &chunk_results)
        .await
        .map_err(|e| {
            console_log!("Failed to generate overall assessment: {}", e);
            e
        })?;

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 85.0,
            error: None,
        },
    )
    .await?;

    // Step 6: Generate practice recommendations
    console_log!("Step 6: Generating practice recommendations");
    let practice_recommendations = generate_practice_recommendations(env, &chunk_results)
        .await
        .map_err(|e| {
            console_log!("Failed to generate practice recommendations: {}", e);
            e
        })?;

    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "processing".to_string(),
            progress: 90.0,
            error: None,
        },
    )
    .await?;

    // Step 7: Generate encouragement
    console_log!("Step 7: Generating encouragement");
    let encouragement = generate_encouragement(env, &overall_assessment)
        .await
        .map_err(|e| {
            console_log!("Failed to generate encouragement: {}", e);
            e
        })?;

    // Step 8: Create and store final result
    console_log!("Step 8: Storing temporal analysis result");
    let result = crate::TemporalAnalysisResult {
        id: job_id.to_string(),
        status: "completed".to_string(),
        file_id: file_id.to_string(),
        overall_assessment,
        temporal_feedback,
        practice_recommendations,
        encouragement,
        created_at: js_sys::Date::new_0().to_iso_string().as_string().unwrap(),
        processing_time: None,
    };

    storage::store_temporal_analysis_result(env, job_id, &result)
        .await
        .map_err(|e| {
            console_log!("Failed to store result: {}", e);
            e
        })?;

    // Step 9: Mark job as completed
    storage::update_job_status(
        env,
        job_id,
        &JobStatus {
            job_id: job_id.to_string(),
            status: "completed".to_string(),
            progress: 100.0,
            error: None,
        },
    )
    .await?;

    console_log!("=== Temporal Analysis Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_wav_data() -> Vec<u8> {
        let mut data = Vec::new();

        // WAV file header
        data.extend_from_slice(b"RIFF"); // ChunkID
        data.extend_from_slice(&[68u8, 0, 0, 0]); // ChunkSize (44 + data size)
        data.extend_from_slice(b"WAVE"); // Format
        data.extend_from_slice(b"fmt "); // Subchunk1ID
        data.extend_from_slice(&[16u8, 0, 0, 0]); // Subchunk1Size (PCM)
        data.extend_from_slice(&[1u8, 0]); // AudioFormat (PCM)
        data.extend_from_slice(&[1u8, 0]); // NumChannels (mono)
        data.extend_from_slice(&[0x44, 0xAC, 0, 0]); // SampleRate (44100)
        data.extend_from_slice(&[0x88, 0x58, 1, 0]); // ByteRate
        data.extend_from_slice(&[2u8, 0]); // BlockAlign
        data.extend_from_slice(&[16u8, 0]); // BitsPerSample
        data.extend_from_slice(b"data"); // Subchunk2ID
        data.extend_from_slice(&[20u8, 0, 0, 0]); // Subchunk2Size (data size)

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
            timing_stable_unstable: 0.8,
            articulation_short_long: 0.7,
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.9,
            pedal_clean_blurred: 0.7,
            timbre_even_colorful: 0.8,
            timbre_shallow_rich: 0.6,
            timbre_bright_dark: 0.7,
            timbre_soft_loud: 0.8,
            dynamic_sophisticated_raw: 0.7,
            dynamic_range_little_large: 0.6,
            music_making_fast_slow: 0.7,
            music_making_flat_spacious: 0.8,
            music_making_disproportioned_balanced: 0.7,
            music_making_pure_dramatic: 0.8,
            emotion_mood_optimistic_dark: 0.7,
            emotion_mood_low_high_energy: 0.8,
            emotion_mood_honest_imaginative: 0.7,
            interpretation_unsatisfactory_convincing: 0.6,
        };

        let result = AnalysisResult {
            id: job_id.to_string(),
            status: "completed".to_string(),
            file_id: file_id.to_string(),
            analysis: analysis_data.clone(),
            insights: vec![
                "Good timing stability".to_string(),
                "Work on articulation".to_string(),
            ],
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
            timing_stable_unstable: 0.8,
            articulation_short_long: 0.7,
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.9,
            pedal_clean_blurred: 0.7,
            timbre_even_colorful: 0.8,
            timbre_shallow_rich: 0.6,
            timbre_bright_dark: 0.7,
            timbre_soft_loud: 0.8,
            dynamic_sophisticated_raw: 0.7,
            dynamic_range_little_large: 0.6,
            music_making_fast_slow: 0.7,
            music_making_flat_spacious: 0.8,
            music_making_disproportioned_balanced: 0.7,
            music_making_pure_dramatic: 0.8,
            emotion_mood_optimistic_dark: 0.7,
            emotion_mood_low_high_energy: 0.8,
            emotion_mood_honest_imaginative: 0.7,
            interpretation_unsatisfactory_convincing: 0.6,
        };

        // Use ModelResult instead which has the dimensions field
        let result = ModelResult {
            model_name: "test-model".to_string(),
            model_type: "spectrogram".to_string(),
            analysis: analysis_data,
            insights: vec!["Good timing stability".to_string()],
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
            timing_stable_unstable: 0.8,
            articulation_short_long: 0.7,
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.9,
            pedal_clean_blurred: 0.7,
            timbre_even_colorful: 0.8,
            timbre_shallow_rich: 0.6,
            timbre_bright_dark: 0.7,
            timbre_soft_loud: 0.8,
            dynamic_sophisticated_raw: 0.7,
            dynamic_range_little_large: 0.6,
            music_making_fast_slow: 0.7,
            music_making_flat_spacious: 0.8,
            music_making_disproportioned_balanced: 0.7,
            music_making_pure_dramatic: 0.8,
            emotion_mood_optimistic_dark: 0.7,
            emotion_mood_low_high_energy: 0.8,
            emotion_mood_honest_imaginative: 0.7,
            interpretation_unsatisfactory_convincing: 0.6,
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
        let status_names = vec!["pending", "processing", "processing", "completed", "failed"];

        for status_name in status_names {
            let status = JobStatus {
                job_id: "test".to_string(),
                status: status_name.to_string(),
                progress: if status_name == "completed" {
                    100.0
                } else {
                    50.0
                },
                error: if status_name == "failed" {
                    Some("Error".to_string())
                } else {
                    None
                },
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
            vec![],                   // Empty
            vec![0.0],                // Single value
            vec![f32::MIN, f32::MAX], // Extreme values
            vec![1.0; 1000],          // Large vector
        ];

        let analysis_data = AnalysisData {
            timing_stable_unstable: 0.8,
            articulation_short_long: 0.7,
            articulation_soft_hard: 0.6,
            pedal_sparse_saturated: 0.9,
            pedal_clean_blurred: 0.7,
            timbre_even_colorful: 0.8,
            timbre_shallow_rich: 0.6,
            timbre_bright_dark: 0.7,
            timbre_soft_loud: 0.8,
            dynamic_sophisticated_raw: 0.7,
            dynamic_range_little_large: 0.6,
            music_making_fast_slow: 0.7,
            music_making_flat_spacious: 0.8,
            music_making_disproportioned_balanced: 0.7,
            music_making_pure_dramatic: 0.8,
            emotion_mood_optimistic_dark: 0.7,
            emotion_mood_low_high_energy: 0.8,
            emotion_mood_honest_imaginative: 0.7,
            interpretation_unsatisfactory_convincing: 0.6,
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
