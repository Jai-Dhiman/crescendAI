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
