//! score-analysis: standalone WASM crate exposing STOP classifier, DTW score follower,
//! bar analysis, and teaching moment selection for use from TypeScript workers.

use wasm_bindgen::prelude::*;

mod bar_analysis;
mod dims;
mod score_follower;
mod stop;
mod teaching_moments;
mod types;

/// Classify a scored chunk with the STOP logistic regression classifier.
///
/// `scores` must be a 6-element array in order:
/// [dynamics, timing, pedaling, articulation, phrasing, interpretation]
///
/// Returns a serialized `StopResult` with `probability`, `triggered`,
/// `top_dimension`, and `top_deviation`.
#[wasm_bindgen]
pub fn classify_stop(scores: &[f64], threshold: f64) -> Result<JsValue, JsValue> {
    if scores.len() != 6 {
        return Err(JsValue::from_str(&format!(
            "classify_stop: expected 6 scores, got {}",
            scores.len()
        )));
    }
    let arr: [f64; 6] = scores.try_into().expect("length checked above");
    let mut result = stop::classify(&arr);
    // Override triggered based on caller-supplied threshold (default is 0.5 built-in;
    // callers may pass a different threshold for tuning without recompiling).
    result.triggered = result.probability >= threshold;
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

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

/// Align a chunk of performance notes to a score using subsequence DTW.
///
/// `chunk_index`: integer index of this chunk in the session
/// `perf_notes_js`: array of `{ pitch, onset, offset, velocity }`
/// `score_bars_js`: array of score bars (ScoreBar) from the loaded score JSON
/// `follower_state_js`: `{ last_known_bar: number | null }`
///
/// Returns `{ bar_map: BarMap | null, state: FollowerState }`.
#[wasm_bindgen]
pub fn align_chunk(
    chunk_index: usize,
    perf_notes_js: JsValue,
    score_bars_js: JsValue,
    follower_state_js: JsValue,
) -> Result<JsValue, JsValue> {
    let perf_notes: Vec<types::PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let score_bars: Vec<types::ScoreBar> = serde_wasm_bindgen::from_value(score_bars_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut state: types::FollowerState = serde_wasm_bindgen::from_value(follower_state_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let bar_map = score_follower::align_chunk(chunk_index, &perf_notes, &score_bars, &mut state);

    let result = types::AlignChunkResult { bar_map, state };
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
) -> Result<JsValue, JsValue> {
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
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
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
) -> Result<JsValue, JsValue> {
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
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
