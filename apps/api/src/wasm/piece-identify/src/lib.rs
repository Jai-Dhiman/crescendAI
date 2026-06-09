//! piece-identify: standalone WASM crate exposing the three-stage piece identification
//! pipeline (N-gram recall, cosine rerank, DTW confirmation) and text query matching
//! for use from TypeScript workers.
//!
//! The TS layer is responsible for loading the N-gram index and rerank feature vectors
//! from R2 and passing them in as JS values. No async I/O happens inside this crate.

use wasm_bindgen::prelude::*;

mod chroma;
mod gate;
mod identify;
mod dtw_confirm;
mod ngram;
mod rerank;
mod text_match;
mod types;

#[cfg(test)]
mod real_recording_test;

#[cfg(test)]
mod parity_test;

// ---------------------------------------------------------------------------
// Stage 1: N-gram recall
// ---------------------------------------------------------------------------

/// Stage 1: N-gram recall.
///
/// Extracts pitch trigrams from `notes_js` (array of `PerfNote`), looks them up in
/// `index_js` (the inverted index loaded from R2), and returns top-10 candidates
/// sorted by hit count.
///
/// `notes_js`: `Array<{ pitch: number, onset: number, offset: number, velocity: number }>`
/// `index_js`: `Record<string, Array<[string, number]>>` (trigram -> [(piece_id, bar)])
///
/// Returns `Array<{ piece_id: string, hit_count: number }>`.
#[wasm_bindgen]
pub fn ngram_recall(notes_js: JsValue, index_js: JsValue) -> Result<JsValue, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let index: types::NgramIndex =
        serde_wasm_bindgen::from_value(index_js).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let candidates = ngram::ngram_recall(&notes, &index);
    serde_wasm_bindgen::to_value(&candidates).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Stage 2: Rerank
// ---------------------------------------------------------------------------

/// Stage 2a: Compute 128-dim rerank feature vector from performance notes.
///
/// `notes_js`: `Array<PerfNote>`
///
/// Returns `Float64Array` (128 elements).
#[wasm_bindgen]
pub fn compute_rerank_features(notes_js: JsValue) -> Result<Vec<f64>, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(rerank::compute_rerank_features(&notes))
}

/// Stage 2b: Rerank candidates by cosine similarity.
///
/// `notes_js`: `Array<PerfNote>` (the same window used for N-gram recall)
/// `candidates_js`: `Array<{ piece_id: string, hit_count: number }>` (from `ngram_recall`)
/// `features_js`: `Record<string, number[]>` (per-piece 128-dim vectors loaded from R2)
///
/// Returns `Array<{ piece_id: string, similarity: number }>` (top-2, descending).
#[wasm_bindgen]
pub fn rerank_candidates(
    notes_js: JsValue,
    candidates_js: JsValue,
    features_js: JsValue,
) -> Result<JsValue, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let candidates: Vec<types::NgramCandidate> = serde_wasm_bindgen::from_value(candidates_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let features: types::RerankFeatures =
        serde_wasm_bindgen::from_value(features_js).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let results = rerank::rerank_candidates(&notes, &candidates, &features);
    serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Stage 3: DTW confirmation
// ---------------------------------------------------------------------------

/// Stage 3: DTW confirmation.
///
/// Runs subsequence DTW alignment between the performance window and the score
/// to confirm or reject the top rerank candidate.
///
/// `perf_notes_js`: `Array<PerfNote>` (the identification window)
/// `score_notes_js`: `Array<{ onset: number, pitch: number }>` (flattened score notes
///                   from the candidate's score JSON, provided by the TS layer)
/// `threshold`: normalized DTW cost below which the candidate is confirmed (use 0.3)
///
/// Returns `{ confirmed: boolean, cost: number, confidence: number }`.
#[wasm_bindgen]
pub fn dtw_confirm(
    perf_notes_js: JsValue,
    score_notes_js: JsValue,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let perf_notes: Vec<types::PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // score_notes_js: Array<{ onset: number, pitch: number }>
    #[derive(serde::Deserialize)]
    struct ScoreNoteFlat {
        onset: f64,
        pitch: u8,
    }
    let score_flat: Vec<ScoreNoteFlat> = serde_wasm_bindgen::from_value(score_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let score_notes: Vec<(f64, u8)> = score_flat.into_iter().map(|n| (n.onset, n.pitch)).collect();

    let result = dtw_confirm::dtw_confirm(&perf_notes, &score_notes, threshold);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Text query matching
// ---------------------------------------------------------------------------

/// Match a free-text query against the piece catalog using Dice similarity on bigrams.
///
/// `query`: user-supplied free-text query (e.g. "chopin ballade 1")
/// `catalog_js`: `Array<{ piece_id: string, composer: string, title: string }>`
///
/// Returns `{ piece_id: string, confidence: number } | null`.
#[wasm_bindgen]
pub fn match_piece_text(query: &str, catalog_js: JsValue) -> Result<JsValue, JsValue> {
    let catalog: Vec<types::CatalogEntry> = serde_wasm_bindgen::from_value(catalog_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = text_match::match_piece_text(query, &catalog);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Certified 2-stage piece identification (chroma recall -> elastic-DTW margin gate)
// ---------------------------------------------------------------------------

/// Identify the piece for the accumulated performance notes against the v2
/// catalog artifact. Locks only when the margin gate clears `margin_threshold`.
///
/// `notes_js`: `Array<{ pitch, onset, offset, velocity }>` (the accumulated buffer)
/// `artifact_json`: the raw `fingerprint/v2/piece_index.json` text
/// `margin_threshold`: certified value 0.0935
///
/// Returns `{ piece_id, composer, title, margin, locked } | null`.
#[wasm_bindgen]
pub fn identify_piece(
    notes_js: JsValue,
    artifact_json: &str,
    margin_threshold: f64,
) -> Result<JsValue, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let index: types::PieceIndex =
        serde_json::from_str(artifact_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    match identify::run_identify(&notes, &index, margin_threshold) {
        Some(result) => {
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        }
        None => Ok(JsValue::NULL),
    }
}

#[cfg(test)]
mod lib_tests {
    use crate::identify::run_identify;
    use crate::types::{PerfNote, PieceArtifact, PieceIndex};

    #[test]
    fn crate_exposes_identify_pipeline() {
        let notes = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 },
            PerfNote { pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 },
            PerfNote { pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 },
        ];
        let p = |id: &str, events: Vec<u16>, chroma: [f64; 12]| PieceArtifact {
            piece_id: id.into(), composer: "C".into(), title: id.into(), chroma, events,
        };
        let index = PieceIndex {
            onset_tol_ms: 50.0,
            pieces: vec![p("a", vec![1, 2, 4], [0.1; 12]), p("b", vec![16, 32, 64], [0.0; 12])],
        };
        // Smoke: the orchestrator is reachable from the crate root and returns a decision.
        assert!(run_identify(&notes, &index, 0.0935).is_some());
    }
}
