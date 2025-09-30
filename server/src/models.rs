use worker::*;
use serde::{Deserialize, Serialize};
use ort::{session::Session, value::Value};
use ndarray::Array4;
use once_cell::sync::OnceCell;

/// Piano performance analysis results
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceAnalysis {
    pub scores: Vec<f32>,           // 19 performance dimensions [0-1]
    pub overall_score: f32,         // Weighted average
    pub confidence: f32,            // Model confidence estimate
    pub processing_time_ms: f32,    // Inference time
}

/// ONNX model wrapper for piano performance evaluation
pub struct EvaluatorModel {
    session: Session,
}

/// Global model instance (loaded once)
static MODEL: OnceCell<EvaluatorModel> = OnceCell::new();

/// Dimension names corresponding to model outputs
const DIMENSION_NAMES: &[&str] = &[
    "timing_stability", "tempo_control", "rhythmic_accuracy",
    "articulation_length", "articulation_hardness", 
    "pedal_density", "pedal_clarity",
    "dynamic_range", "dynamic_control", "balance_melody_vs_accomp",
    "phrasing_continuity", "expressiveness_intensity", "energy_level",
    "timbre_brightness", "timbre_richness", "timbre_color_variety",
    "technique", "musicality", "overall_performance"
];

/// Dimension weights for overall score calculation
const DIMENSION_WEIGHTS: &[f32] = &[
    0.08, 0.08, 0.08,  // Timing
    0.06, 0.06,        // Articulation 
    0.05, 0.05,        // Pedaling
    0.07, 0.07, 0.06,  // Dynamics
    0.08, 0.08, 0.06,  // Musical expression
    0.05, 0.05, 0.05,  // Timbre
    0.09, 0.09, 0.12   // Technique, musicality, overall
];

impl EvaluatorModel {
    /// Initialize the ONNX model from embedded bytes
    pub fn new() -> Result<Self> {
        console_log!("Loading ONNX evaluator model...");
        
        // Model is embedded at compile time
        const MODEL_BYTES: &[u8] = include_bytes!("../models/crescend_evaluator.onnx");
        
        console_log!("Model size: {} bytes", MODEL_BYTES.len());
        
        // Create ONNX session
        let session = Session::builder()
            .map_err(|e| worker::Error::RustError(format!("ONNX session builder failed: {}", e)))?
            .with_model_from_memory(MODEL_BYTES)
            .map_err(|e| worker::Error::RustError(format!("ONNX model loading failed: {}", e)))?;
        
        console_log!("✅ ONNX model loaded successfully");
        
        // Log model info
        let inputs = session.inputs();
        let outputs = session.outputs();
        
        console_log!("Model inputs: {:?}", inputs.iter().map(|i| i.name()).collect::<Vec<_>>());
        console_log!("Model outputs: {:?}", outputs.iter().map(|o| o.name()).collect::<Vec<_>>());
        
        Ok(EvaluatorModel { session })
    }
    
    /// Get global model instance (singleton)
    pub fn get() -> Result<&'static EvaluatorModel> {
        MODEL.get_or_try_init(|| Self::new())
    }
    
    /// Analyze mel-spectrogram and return performance scores
    pub async fn analyze(&self, mel_spectrogram: &[f32]) -> Result<PerformanceAnalysis> {
        let start_time = js_sys::Date::now();
        
        console_log!("Starting model inference...");
        console_log!("Input spectrogram length: {}", mel_spectrogram.len());
        
        // Validate input size (should be 128 x 128 = 16,384)
        if mel_spectrogram.len() != 128 * 128 {
            return Err(worker::Error::RustError(format!(
                "Invalid spectrogram size: expected 16384, got {}", 
                mel_spectrogram.len()
            )));
        }
        
        // Reshape to [1, 1, 128, 128] (batch, channels, height, width)
        let input_array = Array4::from_shape_vec((1, 1, 128, 128), mel_spectrogram.to_vec())
            .map_err(|e| worker::Error::RustError(format!("Failed to reshape input: {}", e)))?;
        
        console_log!("Input array shape: {:?}", input_array.shape());
        
        // Create ONNX input
        let input_value = Value::from_array(input_array)
            .map_err(|e| worker::Error::RustError(format!("Failed to create ONNX input: {}", e)))?;
        
        // Run inference
        console_log!("Running ONNX inference...");
        let outputs = self.session.run(vec![input_value])
            .map_err(|e| worker::Error::RustError(format!("ONNX inference failed: {}", e)))?;
        
        // Extract predictions (first output)
        let predictions = outputs[0].try_extract::<f32>()
            .map_err(|e| worker::Error::RustError(format!("Failed to extract predictions: {}", e)))?;
        
        let predictions_view = predictions.view();
        let scores: Vec<f32> = predictions_view.iter().cloned().collect();
        
        console_log!("Raw predictions: {:?}", scores);
        
        // Validate output size
        if scores.len() != 19 {
            return Err(worker::Error::RustError(format!(
                "Invalid model output size: expected 19, got {}", 
                scores.len()
            )));
        }
        
        // Calculate overall score (weighted average)
        let overall_score = scores.iter()
            .zip(DIMENSION_WEIGHTS.iter())
            .map(|(score, weight)| score * weight)
            .sum::<f32>();
        
        // Estimate confidence based on variance
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f32>() / scores.len() as f32;
        let confidence = (1.0_f32 - variance.sqrt()).max(0.1).min(1.0);
        
        let processing_time_ms = (js_sys::Date::now() - start_time) as f32;
        
        console_log!("✅ Inference complete in {:.1}ms", processing_time_ms);
        console_log!("Overall score: {:.3}, Confidence: {:.3}", overall_score, confidence);
        
        Ok(PerformanceAnalysis {
            scores,
            overall_score,
            confidence,
            processing_time_ms,
        })
    }
    
    /// Convert analysis to insights/feedback
    pub fn generate_insights(&self, analysis: &PerformanceAnalysis) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Find strengths and weaknesses
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        
        for (i, &score) in analysis.scores.iter().enumerate() {
            let dimension = DIMENSION_NAMES[i];
            
            if score >= 0.75 {
                strengths.push((dimension, score));
            } else if score <= 0.4 {
                weaknesses.push((dimension, score));
            }
        }
        
        // Generate insights based on performance
        if analysis.overall_score >= 0.8 {
            insights.push("Excellent overall performance! Your technical skills are well-developed.".to_string());
        } else if analysis.overall_score >= 0.6 {
            insights.push("Good performance with room for refinement in specific areas.".to_string());
        } else {
            insights.push("Focus on fundamental techniques to improve overall performance.".to_string());
        }
        
        // Highlight strengths
        if !strengths.is_empty() {
            let strength_names: Vec<&str> = strengths.iter().map(|(name, _)| *name).collect();
            insights.push(format!("Strong areas: {}", strength_names.join(", ")));
        }
        
        // Suggest improvements
        if !weaknesses.is_empty() {
            let weakness_names: Vec<&str> = weaknesses.iter().map(|(name, _)| *name).collect();
            insights.push(format!("Areas for improvement: {}", weakness_names.join(", ")));
        }
        
        // Confidence-based insight
        if analysis.confidence < 0.5 {
            insights.push("Note: Analysis confidence is low. Consider uploading a longer or clearer recording.".to_string());
        }
        
        insights
    }
}

/// Convert PerformanceAnalysis to the AnalysisData (PercepPiano 19-d) schema
pub fn to_analysis_data(analysis: &PerformanceAnalysis) -> crate::AnalysisData {
    // Map the 19 model scores to the PP19 fields (heuristic mapping)
    let s = &analysis.scores;

    crate::AnalysisData {
        // Timing and articulation
        timing_stable_unstable: s.get(0).copied().unwrap_or(0.5),          // timing_stability
        articulation_short_long: s.get(3).copied().unwrap_or(0.5),         // articulation_length
        articulation_soft_hard: s.get(4).copied().unwrap_or(0.5),          // articulation_hardness

        // Pedal
        pedal_sparse_saturated: s.get(5).copied().unwrap_or(0.5),          // pedal_density
        pedal_clean_blurred: s.get(6).copied().unwrap_or(0.5),             // pedal_clarity

        // Timbre
        timbre_even_colorful: s.get(15).copied().unwrap_or(0.5),           // timbre_color_variety
        timbre_shallow_rich: s.get(14).copied().unwrap_or(0.5),            // timbre_richness
        timbre_bright_dark: s.get(13).copied().unwrap_or(0.5),             // timbre_brightness
        timbre_soft_loud: s.get(12).copied().unwrap_or(0.5),               // energy_level (proxy)

        // Dynamics
        dynamic_sophisticated_raw: s.get(8).copied().unwrap_or(0.5),       // dynamic_control
        dynamic_range_little_large: s.get(7).copied().unwrap_or(0.5),      // dynamic_range

        // Musical making aspects
        music_making_fast_slow: s.get(1).copied().unwrap_or(0.5),          // tempo_control
        music_making_flat_spacious: s.get(11).copied().unwrap_or(0.5),     // expressiveness_intensity (proxy)
        music_making_disproportioned_balanced: s.get(10).copied().unwrap_or(0.5), // phrasing_continuity
        music_making_pure_dramatic: s.get(11).copied().unwrap_or(0.5),     // expressiveness_intensity (proxy)

        // Emotion/musicality
        emotion_mood_optimistic_dark: s.get(13).copied().unwrap_or(0.5),   // timbre_brightness (proxy)
        emotion_mood_low_high_energy: s.get(12).copied().unwrap_or(0.5),   // energy_level
        emotion_mood_honest_imaginative: s.get(17).copied().unwrap_or(0.5),// musicality

        // Overall interpretation
        interpretation_unsatisfactory_convincing: s.get(18).copied().unwrap_or(0.5), // overall_performance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dimension_count() {
        assert_eq!(DIMENSION_NAMES.len(), 19);
        assert_eq!(DIMENSION_WEIGHTS.len(), 19);
    }
    
    #[test]
    fn test_weights_sum_to_one() {
        let sum: f32 = DIMENSION_WEIGHTS.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights should sum to ~1.0, got {}", sum);
    }
    
    #[test]
    fn test_analysis_data_mapping() {
        let analysis = PerformanceAnalysis {
            scores: vec![0.5; 19],
            overall_score: 0.6,
            confidence: 0.8,
            processing_time_ms: 50.0,
        };
        
        let data = to_analysis_data(&analysis);
        assert_eq!(data.rhythm, 0.5);
        assert_eq!(data.overall_performance, 0.5);
    }
}