//! score-analysis: standalone WASM crate exposing chroma-DTW score follower,
//! bar analysis, and teaching moment selection for use from TypeScript workers.

use wasm_bindgen::prelude::*;

mod bar_analysis;
mod chroma_dtw;
mod dims;
mod note_align;
mod teaching_moments;
pub mod types;

#[cfg(test)]
mod real_recording_test;

// Re-export pure Rust core for integration tests (not wasm_bindgen — no JsValue in tests)
pub use chroma_dtw::chroma_dtw_native;
pub use note_align::{align_chunk_notes_native, ChunkNoteResult};

/// Select the top-1 teaching moment from a session's scored chunks.
///
/// `chunks_js`: array of `{ chunk_index: number, scores: number[] }`
/// `baselines_js`: `{ dynamics, timing, pedaling, articulation, phrasing, interpretation }`
/// `recent_observations_js`: array of `{ dimension: string }`
///
/// Returns a serialized `TeachingMoment` or `null` if fewer than 2 chunks are provided.
#[wasm_bindgen]
pub fn select_teaching_moment(
    chunks_js: JsValue,
    baselines_js: JsValue,
    recent_observations_js: JsValue,
) -> Result<JsValue, JsValue> {
    let chunks: Vec<types::ScoredChunk> =
        serde_wasm_bindgen::from_value(chunks_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let baselines: types::StudentBaselines = serde_wasm_bindgen::from_value(baselines_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let recent: Vec<types::RecentObservation> =
        serde_wasm_bindgen::from_value(recent_observations_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = teaching_moments::select_teaching_moment(&chunks, &baselines, &recent);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Select up to `max` within-session teaching moments from a session's scored chunks.
///
/// `chunks_js`: array of `{ chunk_index: number, scores: number[] }`
/// `reference_js`: `{ dynamics, timing, pedaling, articulation, phrasing, interpretation }`
///   (typically the per-dimension session mean)
/// `max`: maximum number of moments to return
///
/// Returns a serialized `Vec<TeachingMoment>` (empty array if fewer than 2 chunks).
#[wasm_bindgen]
pub fn select_session_moments(
    chunks_js: JsValue,
    reference_js: JsValue,
    max: usize,
) -> Result<JsValue, JsValue> {
    let chunks: Vec<types::ScoredChunk> =
        serde_wasm_bindgen::from_value(chunks_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let reference: types::StudentBaselines = serde_wasm_bindgen::from_value(reference_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result =
        teaching_moments::select_session_moments(&chunks, &reference.as_array(), max);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Tier 1 bar-aligned analysis: full analysis with score and reference comparison.
///
/// `bar_map_js`: `BarMap` from a prior `align_chunk` call
/// `perf_notes_js`: array of `PerfNote`
/// `perf_pedal_js`: array of `PerfPedalEvent`
/// `scores`: 6-element Float64Array [dynamics, timing, pedaling, articulation, phrasing, interpretation]
/// `score_context_js`: `ScoreContext` (includes `.score` and optional `.reference`)
///
/// Returns a `ChunkAnalysis` with `tier: 1`.
#[wasm_bindgen]
pub fn analyze_tier1(
    bar_map_js: JsValue,
    perf_notes_js: JsValue,
    perf_pedal_js: JsValue,
    scores: &[f64],
    score_context_js: JsValue,
) -> Result<String, JsValue> {
    if scores.len() != 6 {
        return Err(JsValue::from_str(&format!(
            "analyze_tier1: expected 6 scores, got {}",
            scores.len()
        )));
    }
    let scores_arr: [f64; 6] = scores.try_into().expect("length checked above");

    let bar_map: types::BarMap = serde_wasm_bindgen::from_value(bar_map_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let perf_notes: Vec<types::PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let perf_pedal: Vec<types::PerfPedalEvent> = serde_wasm_bindgen::from_value(perf_pedal_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let score_ctx: types::ScoreContext = serde_wasm_bindgen::from_value(score_context_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = bar_analysis::analyze_tier1(&bar_map, &perf_notes, &perf_pedal, &scores_arr, &score_ctx);
    // Return JSON string (parsed JS-side) instead of serde_wasm_bindgen::to_value: under
    // real workerd the latter mismarshals the `Option<String>` bar_range field (a None
    // aliases a reclaimed input externref), so JS reads bar_range as the perf_notes array.
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Tier 2 absolute MIDI analysis: no score context required.
///
/// `perf_notes_js`: array of `PerfNote`
/// `perf_pedal_js`: array of `PerfPedalEvent`
/// `scores`: 6-element Float64Array [dynamics, timing, pedaling, articulation, phrasing, interpretation]
///
/// Returns a `ChunkAnalysis` with `tier: 2`.
#[wasm_bindgen]
pub fn analyze_tier2(
    perf_notes_js: JsValue,
    perf_pedal_js: JsValue,
    scores: &[f64],
) -> Result<String, JsValue> {
    if scores.len() != 6 {
        return Err(JsValue::from_str(&format!(
            "analyze_tier2: expected 6 scores, got {}",
            scores.len()
        )));
    }
    let scores_arr: [f64; 6] = scores.try_into().expect("length checked above");

    let perf_notes: Vec<types::PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let perf_pedal: Vec<types::PerfPedalEvent> = serde_wasm_bindgen::from_value(perf_pedal_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = bar_analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_arr);
    // See analyze_tier1: return JSON string to avoid the workerd serde_wasm_bindgen
    // bar_range mismarshal; JS-side JSON.parse yields a correct plain object.
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
