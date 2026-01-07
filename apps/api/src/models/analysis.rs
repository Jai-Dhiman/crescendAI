use serde::{Deserialize, Serialize};

/// The 19 perceptual dimensions evaluated by the PercePiano model.
/// Each dimension is a score from 0.0 to 1.0.
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

/// A practice tip retrieved from the RAG knowledge base.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PracticeTip {
    pub title: String,
    pub description: String,
}

/// The complete analysis result returned to the frontend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisResult {
    pub performance_id: String,
    pub dimensions: PerformanceDimensions,
    pub teacher_feedback: String,
    pub practice_tips: Vec<PracticeTip>,
}
