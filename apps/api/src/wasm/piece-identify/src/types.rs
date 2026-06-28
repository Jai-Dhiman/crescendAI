//! Types that cross the WASM boundary. All boundary types implement Serialize + Deserialize.

use serde::{Deserialize, Serialize};

/// A performance note from AMT output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfNote {
    pub pitch: u8,
    pub onset: f64,
    pub offset: f64,
    pub velocity: u8,
}

/// A single text match result from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextMatchResult {
    pub piece_id: String,
    pub confidence: f64,
}

/// A catalog entry used for text query matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
}

/// One catalog piece in the v2 artifact: chroma recall vector + chord-event masks +
/// optional per-piece note-window chroma vectors for the hybrid shortlist (#96).
#[derive(Serialize, Deserialize)]
pub struct PieceArtifact {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub chroma: [f64; 12],
    pub events: Vec<u16>,
    /// Overlapping note-window chroma vectors (400 notes, hop 200) used for the
    /// windowed recall pass. Missing in old artifacts deserialized as empty (back-compat).
    #[serde(default)]
    pub windows: Vec<[f64; 12]>,
}

/// The v2 piece-ID artifact loaded from R2 (`fingerprint/v2/piece_index.json`).
#[derive(Serialize, Deserialize)]
pub struct PieceIndex {
    pub onset_tol_ms: f64,
    pub pieces: Vec<PieceArtifact>,
}

/// Result of identify_piece (marshaled to JS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifyResult {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub margin: f64,
    pub locked: bool,
}
