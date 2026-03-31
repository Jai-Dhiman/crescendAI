//! Shared types that cross the WASM boundary.

use serde::{Deserialize, Serialize};

// --- Score context types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreNote {
    pub pitch: u8,
    pub pitch_name: String,
    pub velocity: u8,
    pub onset_tick: u32,
    pub onset_seconds: f64,
    pub duration_ticks: u32,
    pub duration_seconds: f64,
    pub track: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScorePedalEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub tick: u32,
    pub seconds: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreBar {
    pub bar_number: u32,
    pub start_tick: u32,
    pub start_seconds: f64,
    pub time_signature: String,
    pub notes: Vec<ScoreNote>,
    pub pedal_events: Vec<ScorePedalEvent>,
    pub note_count: u32,
    pub pitch_range: Vec<u8>,
    pub mean_velocity: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreData {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub key_signature: Option<String>,
    pub time_signatures: Vec<serde_json::Value>,
    pub tempo_markings: Vec<serde_json::Value>,
    pub total_bars: u32,
    pub bars: Vec<ScoreBar>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceBar {
    pub bar_number: u32,
    pub velocity_mean: f64,
    pub velocity_std: f64,
    pub onset_deviation_mean_ms: f64,
    pub onset_deviation_std_ms: f64,
    pub pedal_duration_mean_beats: Option<f64>,
    pub pedal_changes: Option<u32>,
    pub note_duration_ratio_mean: f64,
    pub performer_count: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceProfile {
    pub piece_id: String,
    pub performer_count: u32,
    pub bars: Vec<ReferenceBar>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreContext {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub score: ScoreData,
    pub reference: Option<ReferenceProfile>,
    pub match_confidence: f64,
}

// --- Performance note types ---

/// A performance note from AMT (subset of HF response).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerfNote {
    pub pitch: u8,
    pub onset: f64,
    pub offset: f64,
    pub velocity: u8,
}

/// A pedal event from AMT.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerfPedalEvent {
    pub time: f64,
    pub value: u8,
}

// --- Score follower types ---

/// Alignment of a single performance note to a score note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteAlignment {
    pub perf_onset: f64,
    pub perf_pitch: u8,
    pub perf_velocity: u8,
    pub score_bar: u32,
    pub score_beat: f64,
    pub score_pitch: u8,
    pub onset_deviation_ms: f64,
}

/// Bar map for a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarMap {
    pub chunk_index: usize,
    pub bar_start: u32,
    pub bar_end: u32,
    pub alignments: Vec<NoteAlignment>,
    pub confidence: f64,
    pub is_reanchored: bool,
}

/// Persistent state across chunks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FollowerState {
    pub last_known_bar: Option<u32>,
}

// --- Teaching moment types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk_index: usize,
    pub scores: [f64; 6],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentBaselines {
    pub dynamics: f64,
    pub timing: f64,
    pub pedaling: f64,
    pub articulation: f64,
    pub phrasing: f64,
    pub interpretation: f64,
}

impl StudentBaselines {
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.dynamics,
            self.timing,
            self.pedaling,
            self.articulation,
            self.phrasing,
            self.interpretation,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentObservation {
    pub dimension: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachingMoment {
    pub chunk_index: usize,
    pub dimension: String,
    pub score: f64,
    pub baseline: f64,
    pub deviation: f64,
    pub stop_probability: f64,
    pub reasoning: String,
    pub is_positive: bool,
}

// --- STOP result ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopResult {
    pub probability: f64,
    pub triggered: bool,
    pub top_dimension: String,
    pub top_deviation: f64,
}

// --- Bar analysis types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionAnalysis {
    pub dimension: String,
    pub analysis: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_marking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_comparison: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkAnalysis {
    pub tier: u8,
    pub bar_range: Option<String>,
    pub dimensions: Vec<DimensionAnalysis>,
}

// --- align_chunk return type ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignChunkResult {
    pub bar_map: Option<BarMap>,
    pub state: FollowerState,
}
