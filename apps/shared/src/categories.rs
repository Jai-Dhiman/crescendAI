use crate::models::{CategoryScore, PerformanceDimensions};

/// Aggregate 19 PercePiano dimensions into 4 provisional product categories.
/// Uses equal weights (interim). Weights will be updated to MLP probing R-squared
/// values once the teacher-grounded taxonomy work is complete.
#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn to_category_scores(dimensions: PerformanceDimensions) -> Vec<CategoryScore> {
    let sq_dims = vec![
        ("dynamic range", dimensions.dynamics_range),
        ("tonal depth", dimensions.timbre_depth),
        ("tonal variety", dimensions.timbre_variety),
        ("projection", dimensions.timbre_loudness),
        ("brightness", dimensions.timbre_brightness),
    ];
    let ms_dims = vec![
        ("rhythmic timing", dimensions.timing),
        ("tempo control", dimensions.tempo),
        ("use of space", dimensions.space),
        ("dramatic arc", dimensions.drama),
    ];
    let tc_dims = vec![
        ("pedal use", dimensions.pedal_amount),
        ("pedal clarity", dimensions.pedal_clarity),
        ("note articulation", dimensions.articulation_length),
        ("touch sensitivity", dimensions.articulation_touch),
    ];
    let ic_dims = vec![
        ("emotional expression", dimensions.mood_valence),
        ("musical energy", dimensions.mood_energy),
        ("creative imagination", dimensions.mood_imagination),
        ("interpretive depth", dimensions.interpretation_sophistication),
        ("overall interpretation", dimensions.interpretation_overall),
    ];

    vec![
        build_category("Sound Quality", &sq_dims),
        build_category("Musical Shaping", &ms_dims),
        build_category("Technical Control", &tc_dims),
        build_category("Interpretive Choices", &ic_dims),
    ]
}

#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn score_to_label(score: f64) -> String {
    if score >= 0.7 {
        "Strong".to_string()
    } else if score >= 0.5 {
        "Good".to_string()
    } else if score >= 0.3 {
        "Developing".to_string()
    } else {
        "Needs focus".to_string()
    }
}

fn build_category(name: &str, dims: &[(&str, f64)]) -> CategoryScore {
    let score = dims.iter().map(|(_, v)| v).sum::<f64>() / dims.len() as f64;
    let label = score_to_label(score);
    let summary = generate_summary(dims);
    let weakest = dims.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let practice_tip = dimension_practice_tip(weakest.0).to_string();

    CategoryScore {
        name: name.to_string(),
        score,
        label,
        summary,
        practice_tip,
    }
}

fn generate_summary(dims: &[(&str, f64)]) -> String {
    let best = dims.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let worst = dims.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    if (best.1 - worst.1).abs() < 0.05 {
        let avg = dims.iter().map(|(_, v)| v).sum::<f64>() / dims.len() as f64;
        if avg >= 0.6 {
            "Consistently strong across all areas.".to_string()
        } else if avg >= 0.4 {
            "Showing solid foundations across all areas.".to_string()
        } else {
            "This area has room for growth across the board.".to_string()
        }
    } else {
        format!(
            "Your {} stands out as a strength. {} has the most room to develop.",
            best.0, worst.0
        )
    }
}

fn dimension_practice_tip(dim: &str) -> &'static str {
    match dim {
        "dynamic range" => "Practice the same phrase at five dynamic levels (pp through ff), exaggerating contrasts before dialing back to musical levels.",
        "tonal depth" => "Play slow passages focusing on arm weight into the keys. Listen for full resonance before moving to the next note.",
        "tonal variety" => "Try the same melody with fingertip, flat finger, and arm weight touches. Notice how each changes the tonal color.",
        "projection" => "Voice the melody above accompaniment by giving top notes slightly more weight while lightening inner voices.",
        "brightness" => "Use faster key descent with firm fingertips in upper register passages. Listen for clarity and shimmer.",
        "rhythmic timing" => "Practice with a metronome at half tempo, locking each beat precisely. Gradually increase to performance tempo.",
        "tempo control" => "Record yourself and compare to a reference recording. Note where you rush or drag, then target those passages.",
        "use of space" => "Experiment with slight pauses between phrases. Let the music breathe at phrase endings before continuing.",
        "dramatic arc" => "Map the emotional shape of each section before playing. Mark climax points and plan your dynamic trajectory.",
        "pedal use" => "Practice the passage without pedal first to ensure clean technique, then add pedal gradually, listening for blur.",
        "pedal clarity" => "Try half-pedaling through chromatic passages. Change pedal precisely on harmonic changes, not rhythmically.",
        "note articulation" => "Practice staccato and legato versions of the same passage. Focus on consistent, intentional note releases.",
        "touch sensitivity" => "Play scales with each finger producing equal volume. Strengthen fingers 4 and 5 with targeted exercises.",
        "emotional expression" => "Identify the emotional character of each section and play through focusing solely on conveying that emotion.",
        "musical energy" => "Contrast energy levels between sections. Let active passages drive forward and lyrical passages settle back.",
        "creative imagination" => "Listen to three different recordings of the same piece. Note interpretive choices you find compelling and try them.",
        "interpretive depth" => "Research the composer's markings and historical context. Let this knowledge inform your musical decisions.",
        "overall interpretation" => "Record yourself, listen back without the score, and note what you would change. Then address each point.",
        _ => "Focus on this area in your next practice session, starting slowly and building confidence.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_dimensions(val: f64) -> PerformanceDimensions {
        PerformanceDimensions {
            timing: val,
            articulation_length: val,
            articulation_touch: val,
            pedal_amount: val,
            pedal_clarity: val,
            timbre_variety: val,
            timbre_depth: val,
            timbre_brightness: val,
            timbre_loudness: val,
            dynamics_range: val,
            tempo: val,
            space: val,
            balance: val,
            drama: val,
            mood_valence: val,
            mood_energy: val,
            mood_imagination: val,
            interpretation_sophistication: val,
            interpretation_overall: val,
        }
    }

    #[test]
    fn test_category_scores_returns_four_categories() {
        let dims = uniform_dimensions(0.5);
        let scores = to_category_scores(dims);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_category_names() {
        let dims = uniform_dimensions(0.5);
        let scores = to_category_scores(dims);
        assert_eq!(scores[0].name, "Sound Quality");
        assert_eq!(scores[1].name, "Musical Shaping");
        assert_eq!(scores[2].name, "Technical Control");
        assert_eq!(scores[3].name, "Interpretive Choices");
    }

    #[test]
    fn test_uniform_high_scores_labeled_strong() {
        let dims = uniform_dimensions(0.8);
        let scores = to_category_scores(dims);
        for score in &scores {
            assert_eq!(score.label, "Strong");
        }
    }

    #[test]
    fn test_uniform_mid_scores_labeled_good() {
        let dims = uniform_dimensions(0.55);
        let scores = to_category_scores(dims);
        for score in &scores {
            assert_eq!(score.label, "Good");
        }
    }

    #[test]
    fn test_score_to_label_thresholds() {
        assert_eq!(score_to_label(0.8), "Strong");
        assert_eq!(score_to_label(0.7), "Strong");
        assert_eq!(score_to_label(0.55), "Good");
        assert_eq!(score_to_label(0.35), "Developing");
        assert_eq!(score_to_label(0.15), "Needs focus");
    }
}
