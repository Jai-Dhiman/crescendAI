use leptos::prelude::*;

use crate::models::{AnalysisResult, ModelResult, Performance, PerformanceDimensions};
use crate::services::{generate_teacher_feedback, get_performance_dimensions, get_practice_tips};

fn generate_model_variants(base: &PerformanceDimensions) -> Vec<ModelResult> {
    // Generate variations for each model type
    // In production, these would come from actual model inference
    let symbolic = PerformanceDimensions {
        timing: (base.timing * 1.05).min(1.0),
        articulation_length: (base.articulation_length * 0.95).min(1.0),
        articulation_touch: (base.articulation_touch * 0.98).min(1.0),
        pedal_amount: (base.pedal_amount * 0.85).min(1.0),
        pedal_clarity: (base.pedal_clarity * 0.88).min(1.0),
        timbre_variety: (base.timbre_variety * 0.75).min(1.0),
        timbre_depth: (base.timbre_depth * 0.78).min(1.0),
        timbre_brightness: (base.timbre_brightness * 0.80).min(1.0),
        timbre_loudness: (base.timbre_loudness * 0.82).min(1.0),
        dynamics_range: base.dynamics_range,
        tempo: (base.tempo * 1.02).min(1.0),
        space: base.space,
        balance: base.balance,
        drama: (base.drama * 0.95).min(1.0),
        mood_valence: base.mood_valence,
        mood_energy: base.mood_energy,
        mood_imagination: (base.mood_imagination * 0.92).min(1.0),
        interpretation_sophistication: base.interpretation_sophistication,
        interpretation_overall: (base.interpretation_overall * 0.96).min(1.0),
    };

    let audio = PerformanceDimensions {
        timing: (base.timing * 0.97).min(1.0),
        articulation_length: base.articulation_length,
        articulation_touch: (base.articulation_touch * 1.02).min(1.0),
        pedal_amount: (base.pedal_amount * 1.08).min(1.0),
        pedal_clarity: (base.pedal_clarity * 1.05).min(1.0),
        timbre_variety: (base.timbre_variety * 1.12).min(1.0),
        timbre_depth: (base.timbre_depth * 1.10).min(1.0),
        timbre_brightness: (base.timbre_brightness * 1.08).min(1.0),
        timbre_loudness: (base.timbre_loudness * 1.05).min(1.0),
        dynamics_range: (base.dynamics_range * 1.03).min(1.0),
        tempo: (base.tempo * 0.98).min(1.0),
        space: (base.space * 1.02).min(1.0),
        balance: (base.balance * 1.01).min(1.0),
        drama: (base.drama * 1.04).min(1.0),
        mood_valence: (base.mood_valence * 1.02).min(1.0),
        mood_energy: (base.mood_energy * 1.03).min(1.0),
        mood_imagination: (base.mood_imagination * 1.05).min(1.0),
        interpretation_sophistication: (base.interpretation_sophistication * 1.02).min(1.0),
        interpretation_overall: (base.interpretation_overall * 1.03).min(1.0),
    };

    vec![
        ModelResult {
            model_name: "PercePiano".to_string(),
            model_type: "Symbolic".to_string(),
            r_squared: 0.395,
            dimensions: symbolic,
        },
        ModelResult {
            model_name: "MERT-330M".to_string(),
            model_type: "Audio".to_string(),
            r_squared: 0.433,
            dimensions: audio,
        },
        ModelResult {
            model_name: "Late Fusion".to_string(),
            model_type: "Fusion".to_string(),
            r_squared: 0.510,
            dimensions: base.clone(),
        },
    ]
}

#[server(AnalyzePerformance, "/api")]
pub async fn analyze_performance(id: String) -> Result<AnalysisResult, ServerFnError> {
    let performance = Performance::find_by_id(&id)
        .ok_or_else(|| ServerFnError::new("Performance not found"))?;

    let dimensions = get_performance_dimensions(&id).await;
    let models = generate_model_variants(&dimensions);
    let practice_tips = get_practice_tips(&performance, &dimensions).await;
    let teacher_feedback = generate_teacher_feedback(&performance, &dimensions).await;

    Ok(AnalysisResult {
        performance_id: id,
        dimensions,
        models,
        teacher_feedback,
        practice_tips,
    })
}
