use super::CitedFeedback;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PerformanceDimensions {
    // Timing
    pub timing: f64,

    // Articulation
    pub articulation_length: f64,
    pub articulation_touch: f64,

    // Pedal
    pub pedal_amount: f64,
    pub pedal_clarity: f64,

    // Timbre
    pub timbre_variety: f64,
    pub timbre_depth: f64,
    pub timbre_brightness: f64,
    pub timbre_loudness: f64,

    // Dynamics
    pub dynamics_range: f64,

    // Performance qualities
    pub tempo: f64,
    pub space: f64,
    pub balance: f64,
    pub drama: f64,

    // Mood
    pub mood_valence: f64,
    pub mood_energy: f64,
    pub mood_imagination: f64,

    // Interpretation
    pub interpretation_sophistication: f64,
    pub interpretation_overall: f64,
}

impl PerformanceDimensions {
    pub fn to_labeled_vec(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("Timing", self.timing),
            ("Art. Length", self.articulation_length),
            ("Art. Touch", self.articulation_touch),
            ("Pedal Amt", self.pedal_amount),
            ("Pedal Clarity", self.pedal_clarity),
            ("Timbre Var.", self.timbre_variety),
            ("Timbre Depth", self.timbre_depth),
            ("Brightness", self.timbre_brightness),
            ("Loudness", self.timbre_loudness),
            ("Dyn. Range", self.dynamics_range),
            ("Tempo", self.tempo),
            ("Space", self.space),
            ("Balance", self.balance),
            ("Drama", self.drama),
            ("Valence", self.mood_valence),
            ("Energy", self.mood_energy),
            ("Imagination", self.mood_imagination),
            ("Sophistication", self.interpretation_sophistication),
            ("Interpretation", self.interpretation_overall),
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PracticeTip {
    pub title: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelResult {
    pub model_name: String,
    pub model_type: String, // "Symbolic" or "Audio"
    pub r_squared: f64,
    pub dimensions: PerformanceDimensions,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisResult {
    pub performance_id: String,
    /// Raw model predictions (0-1 scale from model output)
    pub dimensions: PerformanceDimensions,
    /// Calibrated predictions relative to MAESTRO professional benchmarks
    /// ~0.5 = average professional level, can exceed [0,1]
    pub calibrated_dimensions: PerformanceDimensions,
    /// Context explaining how to interpret calibrated scores
    pub calibration_context: Option<String>,
    pub models: Vec<ModelResult>,
    pub teacher_feedback: CitedFeedback,
    pub practice_tips: Vec<PracticeTip>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisState {
    Idle,
    Loading { message: String, progress: u8 },
    Complete(AnalysisResult),
    Error(String),
}

/// Composite score for a product category aggregating multiple PercePiano dimensions.
/// Used for the 4-category feedback display on the analyze page.
#[derive(Clone, Debug)]
pub struct CategoryScore {
    pub name: String,
    pub score: f64,
    pub label: String,
    pub summary: String,
    pub practice_tip: String,
}

impl PerformanceDimensions {
    /// Aggregate 19 PercePiano dimensions into 4 provisional product categories.
    /// Uses equal weights (interim). Weights will be updated to MLP probing R-squared
    /// values once the teacher-grounded taxonomy work is complete.
    pub fn to_category_scores(&self) -> Vec<CategoryScore> {
        let sq_dims = vec![
            ("dynamic range", self.dynamics_range),
            ("tonal depth", self.timbre_depth),
            ("tonal variety", self.timbre_variety),
            ("projection", self.timbre_loudness),
            ("brightness", self.timbre_brightness),
        ];
        let ms_dims = vec![
            ("rhythmic timing", self.timing),
            ("tempo control", self.tempo),
            ("use of space", self.space),
            ("dramatic arc", self.drama),
        ];
        let tc_dims = vec![
            ("pedal use", self.pedal_amount),
            ("pedal clarity", self.pedal_clarity),
            ("note articulation", self.articulation_length),
            ("touch sensitivity", self.articulation_touch),
        ];
        let ic_dims = vec![
            ("emotional expression", self.mood_valence),
            ("musical energy", self.mood_energy),
            ("creative imagination", self.mood_imagination),
            ("interpretive depth", self.interpretation_sophistication),
            ("overall interpretation", self.interpretation_overall),
        ];

        vec![
            build_category("Sound Quality", &sq_dims),
            build_category("Musical Shaping", &ms_dims),
            build_category("Technical Control", &tc_dims),
            build_category("Interpretive Choices", &ic_dims),
        ]
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
        label: label.to_string(),
        summary,
        practice_tip,
    }
}

fn score_to_label(score: f64) -> &'static str {
    if score >= 0.7 {
        "Strong"
    } else if score >= 0.5 {
        "Good"
    } else if score >= 0.3 {
        "Developing"
    } else {
        "Needs focus"
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
        let scores = dims.to_category_scores();
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_category_names() {
        let dims = uniform_dimensions(0.5);
        let scores = dims.to_category_scores();
        assert_eq!(scores[0].name, "Sound Quality");
        assert_eq!(scores[1].name, "Musical Shaping");
        assert_eq!(scores[2].name, "Technical Control");
        assert_eq!(scores[3].name, "Interpretive Choices");
    }

    #[test]
    fn test_uniform_high_scores_labeled_strong() {
        let dims = uniform_dimensions(0.8);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Strong");
        }
    }

    #[test]
    fn test_uniform_mid_scores_labeled_good() {
        let dims = uniform_dimensions(0.55);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Good");
        }
    }

    #[test]
    fn test_uniform_low_scores_labeled_developing() {
        let dims = uniform_dimensions(0.35);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Developing");
        }
    }

    #[test]
    fn test_very_low_scores_labeled_needs_focus() {
        let dims = uniform_dimensions(0.15);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Needs focus");
        }
    }

    #[test]
    fn test_summary_not_empty() {
        let dims = uniform_dimensions(0.6);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert!(!score.summary.is_empty());
        }
    }

    #[test]
    fn test_practice_tip_not_empty() {
        let dims = uniform_dimensions(0.4);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert!(!score.practice_tip.is_empty());
        }
    }

    #[test]
    fn test_mixed_dimensions_category_average() {
        let mut dims = uniform_dimensions(0.5);
        // Make sound quality dimensions high
        dims.dynamics_range = 0.9;
        dims.timbre_depth = 0.9;
        dims.timbre_variety = 0.9;
        dims.timbre_loudness = 0.9;
        dims.timbre_brightness = 0.9;
        let scores = dims.to_category_scores();
        assert_eq!(scores[0].label, "Strong"); // Sound Quality should be strong
        assert_eq!(scores[1].label, "Good");   // Musical Shaping still at 0.5
    }
}
