use serde::{Deserialize, Serialize};
use worker::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MertAnalysisResult {
    pub recording_id: String,
    pub overall_scores: DimensionScores,
    pub temporal_segments: Vec<TemporalSegment>,
    pub uncertainty: UncertaintyEstimates,
    pub metadata: AnalysisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScores {
    // Technical dimensions (6)
    pub note_accuracy: f64,
    pub rhythmic_precision: f64,
    pub dynamics_control: f64,
    pub articulation_quality: f64,
    pub pedaling_technique: f64,
    pub tone_quality: f64,

    // Interpretive dimensions (6)
    pub phrasing: f64,
    pub expressiveness: f64,
    pub musical_interpretation: f64,
    pub stylistic_appropriateness: f64,
    pub overall_musicality: f64,
    pub overall_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub bar_range: Option<String>,
    pub scores: DimensionScores,
    pub attention_weights: Vec<f64>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyEstimates {
    // Per-dimension uncertainty
    pub note_accuracy_std: f64,
    pub rhythmic_precision_std: f64,
    pub dynamics_control_std: f64,
    pub articulation_quality_std: f64,
    pub pedaling_technique_std: f64,
    pub tone_quality_std: f64,
    pub phrasing_std: f64,
    pub expressiveness_std: f64,
    pub musical_interpretation_std: f64,
    pub stylistic_appropriateness_std: f64,
    pub overall_musicality_std: f64,
    pub overall_quality_std: f64,

    // Decomposed uncertainty
    pub aleatoric_uncertainty: f64,
    pub epistemic_uncertainty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub model_version: String,
    pub processing_time_ms: u64,
    pub audio_duration_seconds: f64,
    pub num_segments: usize,
    pub confidence_score: f64,
}

impl DimensionScores {
    pub fn new_mock(base_quality: f64, variation: f64) -> Self {
        // base_quality: 0.0-1.0 representing overall performance level
        // variation: amount of randomness to add

        // Technical dimensions tend to be more objective and vary less
        let note_accuracy = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 0.8));
        let rhythmic_precision = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 0.8));
        let dynamics_control = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.0));
        let articulation_quality = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.0));
        let pedaling_technique = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.2));
        let tone_quality = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.0));

        // Interpretive dimensions are more subjective and vary more
        let phrasing = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.5));
        let expressiveness = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.5));
        let musical_interpretation = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.8));
        let stylistic_appropriateness = Self::clamp_score(base_quality * 100.0 + Self::random_variation(variation * 1.5));

        // Overall scores are weighted averages with some adjustment
        let overall_musicality = Self::clamp_score(
            (phrasing + expressiveness + musical_interpretation) / 3.0 + Self::random_variation(variation * 0.5)
        );
        let overall_quality = Self::clamp_score(
            (note_accuracy * 0.35 +
             (dynamics_control + tone_quality) / 2.0 * 0.25 +
             musical_interpretation * 0.25 +
             (phrasing + expressiveness) / 2.0 * 0.15) +
            Self::random_variation(variation * 0.5)
        );

        Self {
            note_accuracy,
            rhythmic_precision,
            dynamics_control,
            articulation_quality,
            pedaling_technique,
            tone_quality,
            phrasing,
            expressiveness,
            musical_interpretation,
            stylistic_appropriateness,
            overall_musicality,
            overall_quality,
        }
    }

    fn random_variation(scale: f64) -> f64 {
        // Simple pseudo-random using timestamp
        let timestamp = Date::now().as_millis() as f64;
        let noise = ((timestamp * 9301.0 + 49297.0) % 233280.0) / 233280.0;
        (noise - 0.5) * scale * 20.0
    }

    fn clamp_score(score: f64) -> f64 {
        score.max(0.0).min(100.0)
    }
}

impl UncertaintyEstimates {
    pub fn new_mock(dimension_scores: &DimensionScores) -> Self {
        // Technical dimensions have lower uncertainty (more objective)
        let note_accuracy_std = Self::calculate_uncertainty(dimension_scores.note_accuracy, 3.0, 7.0);
        let rhythmic_precision_std = Self::calculate_uncertainty(dimension_scores.rhythmic_precision, 3.5, 7.5);
        let dynamics_control_std = Self::calculate_uncertainty(dimension_scores.dynamics_control, 4.0, 8.0);
        let articulation_quality_std = Self::calculate_uncertainty(dimension_scores.articulation_quality, 4.5, 8.5);
        let pedaling_technique_std = Self::calculate_uncertainty(dimension_scores.pedaling_technique, 4.5, 9.0);
        let tone_quality_std = Self::calculate_uncertainty(dimension_scores.tone_quality, 5.0, 9.0);

        // Interpretive dimensions have higher uncertainty (more subjective)
        let phrasing_std = Self::calculate_uncertainty(dimension_scores.phrasing, 6.0, 12.0);
        let expressiveness_std = Self::calculate_uncertainty(dimension_scores.expressiveness, 6.5, 13.0);
        let musical_interpretation_std = Self::calculate_uncertainty(dimension_scores.musical_interpretation, 7.0, 14.0);
        let stylistic_appropriateness_std = Self::calculate_uncertainty(dimension_scores.stylistic_appropriateness, 6.0, 12.0);
        let overall_musicality_std = Self::calculate_uncertainty(dimension_scores.overall_musicality, 6.0, 13.0);
        let overall_quality_std = Self::calculate_uncertainty(dimension_scores.overall_quality, 5.5, 12.0);

        // Average uncertainties for decomposition
        let avg_uncertainty = (note_accuracy_std + rhythmic_precision_std + dynamics_control_std +
                              articulation_quality_std + pedaling_technique_std + tone_quality_std +
                              phrasing_std + expressiveness_std + musical_interpretation_std +
                              stylistic_appropriateness_std + overall_musicality_std + overall_quality_std) / 12.0;

        // Aleatoric (inherent subjectivity) vs Epistemic (model uncertainty)
        // Ratio is roughly 60% aleatoric, 40% epistemic
        let aleatoric_uncertainty = avg_uncertainty * 0.6;
        let epistemic_uncertainty = avg_uncertainty * 0.4;

        Self {
            note_accuracy_std,
            rhythmic_precision_std,
            dynamics_control_std,
            articulation_quality_std,
            pedaling_technique_std,
            tone_quality_std,
            phrasing_std,
            expressiveness_std,
            musical_interpretation_std,
            stylistic_appropriateness_std,
            overall_musicality_std,
            overall_quality_std,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        }
    }

    fn calculate_uncertainty(score: f64, min_std: f64, max_std: f64) -> f64 {
        // Uncertainty is higher at extremes (very low or very high scores)
        // and lower in the middle range
        let distance_from_middle = ((score - 50.0).abs() / 50.0).min(1.0);
        min_std + (max_std - min_std) * distance_from_middle
    }
}

pub fn mock_mert_analysis(recording_id: &str, audio_duration_seconds: f64) -> Result<MertAnalysisResult> {
    // Generate mock analysis based on recording characteristics

    // Use recording_id to seed "random" generation for consistency
    let id_hash = recording_id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let base_quality = 0.4 + ((id_hash % 50) as f64 / 100.0); // Range: 0.4 to 0.9

    // Calculate number of segments (roughly 20-30 seconds each)
    let num_segments = ((audio_duration_seconds / 25.0).ceil() as usize).max(1).min(10);
    let segment_duration = audio_duration_seconds / num_segments as f64;

    // Generate overall scores
    let overall_scores = DimensionScores::new_mock(base_quality, 1.0);

    // Generate temporal segments with variation
    let mut temporal_segments = Vec::new();
    for i in 0..num_segments {
        let start_time = i as f64 * segment_duration;
        let end_time = ((i + 1) as f64 * segment_duration).min(audio_duration_seconds);

        // Each segment varies slightly from overall quality
        let segment_variation = ((i as f64 * 17.0 + id_hash as f64) % 100.0) / 500.0 - 0.1;
        let segment_quality = (base_quality + segment_variation).max(0.3).min(1.0);

        let segment_scores = DimensionScores::new_mock(segment_quality, 0.8);

        // Generate mock attention weights (10 bars per segment)
        let attention_weights: Vec<f64> = (0..10)
            .map(|j| {
                let base_attention = 0.3 + ((i * 10 + j) as f64 % 7.0) / 10.0;
                base_attention.max(0.1).min(1.0)
            })
            .collect();

        // Identify potential issues based on low scores
        let mut issues = Vec::new();
        if segment_scores.note_accuracy < 60.0 {
            issues.push("Note accuracy issues detected".to_string());
        }
        if segment_scores.rhythmic_precision < 60.0 {
            issues.push("Rhythmic inconsistencies".to_string());
        }
        if segment_scores.dynamics_control < 60.0 {
            issues.push("Dynamics could be more controlled".to_string());
        }
        if segment_scores.pedaling_technique < 60.0 {
            issues.push("Pedaling technique needs attention".to_string());
        }

        let bar_start = i * 10 + 1;
        let bar_end = (i + 1) * 10;

        temporal_segments.push(TemporalSegment {
            start_time,
            end_time,
            bar_range: Some(format!("{}-{}", bar_start, bar_end)),
            scores: segment_scores,
            attention_weights,
            issues,
        });
    }

    // Generate uncertainty estimates
    let uncertainty = UncertaintyEstimates::new_mock(&overall_scores);

    // Create metadata
    let metadata = AnalysisMetadata {
        model_version: "MERT-330M-v1.0-mock".to_string(),
        processing_time_ms: 1500, // Mock processing time
        audio_duration_seconds,
        num_segments,
        confidence_score: 0.75 + (id_hash % 20) as f64 / 100.0, // 0.75-0.95
    };

    Ok(MertAnalysisResult {
        recording_id: recording_id.to_string(),
        overall_scores,
        temporal_segments,
        uncertainty,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_scores_in_range() {
        let scores = DimensionScores::new_mock(0.7, 1.0);

        assert!(scores.note_accuracy >= 0.0 && scores.note_accuracy <= 100.0);
        assert!(scores.rhythmic_precision >= 0.0 && scores.rhythmic_precision <= 100.0);
        assert!(scores.overall_quality >= 0.0 && scores.overall_quality <= 100.0);
    }

    #[test]
    fn test_mock_analysis_generation() {
        let result = mock_mert_analysis("test-recording-123", 120.0).unwrap();

        assert_eq!(result.recording_id, "test-recording-123");
        assert!(result.temporal_segments.len() > 0);
        assert_eq!(result.metadata.audio_duration_seconds, 120.0);
        assert!(result.metadata.confidence_score >= 0.75 && result.metadata.confidence_score <= 0.95);
    }

    #[test]
    fn test_uncertainty_estimates() {
        let scores = DimensionScores::new_mock(0.7, 1.0);
        let uncertainty = UncertaintyEstimates::new_mock(&scores);

        // Technical dimensions should have lower uncertainty
        assert!(uncertainty.note_accuracy_std < uncertainty.phrasing_std);
        assert!(uncertainty.rhythmic_precision_std < uncertainty.expressiveness_std);

        // Total uncertainty should be sum of aleatoric and epistemic
        assert!(uncertainty.aleatoric_uncertainty > 0.0);
        assert!(uncertainty.epistemic_uncertainty > 0.0);
    }

    #[test]
    fn test_consistent_results_same_id() {
        let result1 = mock_mert_analysis("consistent-test", 90.0).unwrap();
        let result2 = mock_mert_analysis("consistent-test", 90.0).unwrap();

        // Same recording ID should produce same base quality
        assert!((result1.overall_scores.note_accuracy - result2.overall_scores.note_accuracy).abs() < 0.1);
    }
}
