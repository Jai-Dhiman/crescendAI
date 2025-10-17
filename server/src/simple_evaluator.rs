use crate::AnalysisData;
use serde::{Deserialize, Serialize};
use worker::*;

/// Simple rule-based piano performance evaluator
///
/// This provides a working system immediately while the ONNX model is being integrated.
/// It analyzes audio features and applies heuristic rules based on your research.
pub struct SimpleEvaluator;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioFeatures {
    pub duration_seconds: f32,
    pub average_amplitude: f32,
    pub amplitude_variance: f32,
    pub spectral_centroid: f32,
    pub zero_crossing_rate: f32,
    pub energy_level: f32,
}

impl SimpleEvaluator {
    pub fn new() -> Self {
        SimpleEvaluator
    }

    /// Extract basic audio features from mel-spectrogram data
    pub fn extract_features(&self, mel_data: &[f32]) -> AudioFeatures {
        console_log!("Extracting audio features from {} values", mel_data.len());

        // Calculate basic statistics
        let average_amplitude = mel_data.iter().sum::<f32>() / mel_data.len() as f32;

        let variance = mel_data
            .iter()
            .map(|&x| (x - average_amplitude).powi(2))
            .sum::<f32>()
            / mel_data.len() as f32;
        let amplitude_variance = variance.sqrt();

        // Estimate spectral features (simplified)
        let spectral_centroid = self.estimate_spectral_centroid(mel_data);
        let zero_crossing_rate = self.estimate_zero_crossings(mel_data);
        let energy_level = self.calculate_energy_level(mel_data);

        // Assume ~15 second average duration
        let duration_seconds = 15.0;

        AudioFeatures {
            duration_seconds,
            average_amplitude,
            amplitude_variance,
            spectral_centroid,
            zero_crossing_rate,
            energy_level,
        }
    }

    /// Analyze features and generate performance scores using rules
    pub fn analyze(&self, features: &AudioFeatures) -> AnalysisData {
        console_log!(
            "Analyzing features: amp={:.3}, var={:.3}, energy={:.3}",
            features.average_amplitude,
            features.amplitude_variance,
            features.energy_level
        );

        // Rule-based scoring based on your evaluation report insights

        // Rhythm & Timing (based on amplitude consistency)
        let rhythm = self.score_rhythm(features);
        let timing = self.score_timing(features);
        let tempo = self.score_tempo(features);

        // Dynamics (based on amplitude variance)
        let dynamics = self.score_dynamics(features);
        let dynamic_range = features.amplitude_variance.min(1.0);

        // Technique (based on spectral features)
        let technique = self.score_technique(features);
        let articulation = self.score_articulation(features);

        // Expression (based on energy and variance)
        let expression = self.score_expression(features);
        let phrasing = self.score_phrasing(features);

        // Pedaling (estimated from spectral content)
        let pedaling = self.score_pedaling(features);

        // Musical understanding (composite score)
        let musical_understanding = (rhythm + dynamics + expression) / 3.0;

        // Overall performance (weighted average)
        let overall_performance = (rhythm * 0.15
            + dynamics * 0.15
            + technique * 0.15
            + expression * 0.15
            + timing * 0.10
            + articulation * 0.10
            + musical_understanding * 0.20);

        AnalysisData {
            timing_stable_unstable: timing,
            articulation_short_long: articulation,
            articulation_soft_hard: dynamics,
            pedal_sparse_saturated: pedaling,
            pedal_clean_blurred: pedaling * 0.8,
            timbre_even_colorful: expression,
            timbre_shallow_rich: technique,
            timbre_bright_dark: technique * 0.9,
            timbre_soft_loud: dynamics,
            dynamic_sophisticated_raw: expression,
            dynamic_range_little_large: dynamics,
            music_making_fast_slow: tempo,
            music_making_flat_spacious: expression,
            music_making_disproportioned_balanced: musical_understanding,
            music_making_pure_dramatic: expression * 0.8,
            emotion_mood_optimistic_dark: expression * 0.7,
            emotion_mood_low_high_energy: rhythm,
            emotion_mood_honest_imaginative: musical_understanding * 0.9,
            interpretation_unsatisfactory_convincing: overall_performance,
        }
    }

    /// Generate insights based on analysis
    pub fn generate_insights(&self, analysis: &AnalysisData) -> Vec<String> {
        let mut insights = Vec::new();

        // Overall assessment based on interpretation score
        if analysis.interpretation_unsatisfactory_convincing >= 0.8 {
            insights.push("Excellent performance! Strong musical interpretation.".to_string());
        } else if analysis.interpretation_unsatisfactory_convincing >= 0.6 {
            insights.push("Good performance with room for improvement.".to_string());
        } else {
            insights.push("Focus on fundamentals to strengthen your playing.".to_string());
        }

        // Specific feedback based on scores
        if analysis.timing_stable_unstable < 0.5 {
            insights.push("Work on timing stability - practice with a metronome.".to_string());
        }

        if analysis.dynamic_range_little_large < 0.5 {
            insights
                .push("Develop dynamic control - practice crescendos and diminuendos.".to_string());
        }

        if analysis.timbre_shallow_rich < 0.5 {
            insights.push("Focus on technical exercises to improve tone quality.".to_string());
        }

        if analysis.emotion_mood_honest_imaginative > 0.7 {
            insights.push("Great expressive playing - you convey emotion well!".to_string());
        }

        if analysis.timing_stable_unstable < 0.5 {
            insights.push("Work on timing precision - practice slowly at first.".to_string());
        }

        // Strengths
        let mut strengths = Vec::new();
        if analysis.emotion_mood_low_high_energy > 0.7 {
            strengths.push("energy");
        }
        if analysis.dynamic_range_little_large > 0.7 {
            strengths.push("dynamics");
        }
        if analysis.timbre_shallow_rich > 0.7 {
            strengths.push("technique");
        }
        if analysis.emotion_mood_honest_imaginative > 0.7 {
            strengths.push("expression");
        }

        if !strengths.is_empty() {
            insights.push(format!("Strengths: {}", strengths.join(", ")));
        }

        insights
    }

    // Feature scoring methods
    fn score_rhythm(&self, features: &AudioFeatures) -> f32 {
        // Good rhythm = consistent amplitude with moderate variance
        let consistency = 1.0 - (features.amplitude_variance * 2.0).min(1.0);
        (consistency * 0.6 + features.energy_level * 0.4).min(1.0)
    }

    fn score_timing(&self, features: &AudioFeatures) -> f32 {
        // Similar to rhythm but emphasizes zero-crossing regularity
        let regularity = (1.0 - features.zero_crossing_rate.abs() * 2.0).max(0.0);
        (regularity * 0.7 + self.score_rhythm(features) * 0.3).min(1.0)
    }

    fn score_tempo(&self, features: &AudioFeatures) -> f32 {
        // Stable tempo correlates with consistent energy
        let stability = 1.0 - (features.energy_level - 0.5).abs();
        stability.max(0.3).min(1.0)
    }

    fn score_dynamics(&self, features: &AudioFeatures) -> f32 {
        // Good dynamics = appropriate amplitude variance (not too flat, not too chaotic)
        let optimal_variance = 0.3; // Target variance
        let variance_score = 1.0 - (features.amplitude_variance - optimal_variance).abs() * 2.0;
        variance_score.max(0.2).min(1.0)
    }

    fn score_technique(&self, features: &AudioFeatures) -> f32 {
        // Technical proficiency correlates with spectral clarity and energy
        let clarity = features.spectral_centroid.min(1.0);
        let control = 1.0 - (features.amplitude_variance * 1.5).min(1.0);
        (clarity * 0.6 + control * 0.4).min(1.0)
    }

    fn score_articulation(&self, features: &AudioFeatures) -> f32 {
        // Clear articulation = higher zero-crossing rate and good spectral content
        let clarity = features.zero_crossing_rate.min(1.0);
        let definition = features.spectral_centroid.min(1.0);
        (clarity * 0.5 + definition * 0.5).min(1.0)
    }

    fn score_expression(&self, features: &AudioFeatures) -> f32 {
        // Expression correlates with appropriate dynamic range and energy variation
        let dynamic_range = features.amplitude_variance.min(1.0);
        let energy_presence = features.energy_level.min(1.0);
        (dynamic_range * 0.6 + energy_presence * 0.4).min(1.0)
    }

    fn score_phrasing(&self, features: &AudioFeatures) -> f32 {
        // Musical phrasing benefits from moderate variance and good energy
        let musical_flow = self.score_expression(features) * 0.8;
        let structural_clarity = (1.0 - (features.amplitude_variance - 0.4).abs()).max(0.0);
        (musical_flow * 0.7 + structural_clarity * 0.3).min(1.0)
    }

    fn score_pedaling(&self, features: &AudioFeatures) -> f32 {
        // Pedaling affects spectral content and sustain
        let spectral_richness = features.spectral_centroid.min(1.0);
        let sustain_quality = features.average_amplitude.min(1.0);
        (spectral_richness * 0.6 + sustain_quality * 0.4).min(1.0)
    }

    // Feature extraction helpers
    fn estimate_spectral_centroid(&self, mel_data: &[f32]) -> f32 {
        // Estimate spectral centroid from mel-spectrogram
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in mel_data.iter().enumerate() {
            let frequency_weight = (i as f32) / (mel_data.len() as f32);
            weighted_sum += frequency_weight * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            (weighted_sum / magnitude_sum).min(1.0)
        } else {
            0.5 // Default middle value
        }
    }

    fn estimate_zero_crossings(&self, mel_data: &[f32]) -> f32 {
        // Estimate zero-crossing rate from spectral data
        let mean_value = mel_data.iter().sum::<f32>() / mel_data.len() as f32;
        let mut crossings = 0;

        for i in 1..mel_data.len() {
            let prev_centered = mel_data[i - 1] - mean_value;
            let curr_centered = mel_data[i] - mean_value;

            if prev_centered * curr_centered < 0.0 {
                crossings += 1;
            }
        }

        (crossings as f32 / mel_data.len() as f32).min(1.0)
    }

    fn calculate_energy_level(&self, mel_data: &[f32]) -> f32 {
        // Calculate normalized energy level
        let energy = mel_data.iter().map(|&x| x * x).sum::<f32>();
        (energy / mel_data.len() as f32).sqrt().min(1.0)
    }
}
