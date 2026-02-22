use serde::{Deserialize, Serialize};

/// 19-dimensional piano performance evaluation from PercePiano.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
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

/// Composite score for a product category aggregating multiple PercePiano dimensions.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct CategoryScore {
    pub name: String,
    pub score: f64,
    pub label: String,
    pub summary: String,
    pub practice_tip: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct PracticeTip {
    pub title: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ModelResult {
    pub model_name: String,
    pub model_type: String,
    pub r_squared: f64,
    pub dimensions: PerformanceDimensions,
}

/// Response from audio upload endpoint.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct UploadedPerformance {
    pub id: String,
    pub audio_url: String,
    pub r2_key: String,
    pub title: String,
    pub file_size_bytes: u64,
    pub content_type: String,
}
