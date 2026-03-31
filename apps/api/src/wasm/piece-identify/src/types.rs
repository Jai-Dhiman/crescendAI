//! Types that cross the WASM boundary. All boundary types implement Serialize + Deserialize.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Inverted index: trigram key ("p1,p2,p3") -> Vec<(piece_id, bar_number)>.
/// Loaded from `ngram_index.json` by the TS layer and passed as a JS value.
pub type NgramIndex = HashMap<String, Vec<(String, u32)>>;

/// Per-piece 128-dim feature vectors.
/// Loaded from `rerank_features.json` by the TS layer and passed as a JS value.
pub type RerankFeatures = HashMap<String, Vec<f64>>;

/// A performance note from AMT output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfNote {
    pub pitch: u8,
    pub onset: f64,
    pub offset: f64,
    pub velocity: u8,
}

/// A single N-gram recall candidate: piece_id + hit count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramCandidate {
    pub piece_id: String,
    pub hit_count: usize,
}

/// A single rerank result: piece_id + cosine similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub piece_id: String,
    pub similarity: f64,
}

/// Result of DTW confirmation (Stage 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtwConfirmResult {
    /// Whether the DTW cost is below the confirmation threshold.
    pub confirmed: bool,
    /// Raw normalized DTW cost (lower is better).
    pub cost: f64,
    /// Confidence derived as 1 / (1 + cost).
    pub confidence: f64,
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
