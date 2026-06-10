//! piece-identify: standalone WASM crate exposing the certified piece identification
//! gate (chroma recall -> elastic-DTW margin gate) and text query matching for use
//! from TypeScript workers.
//!
//! The TS layer loads the v2 catalog artifact from R2 and passes it in as a string.
//! No async I/O happens inside this crate.

use wasm_bindgen::prelude::*;

mod chroma;
mod gate;
mod identify;
mod text_match;
mod types;

#[cfg(test)]
mod parity_test;

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
) -> Result<Option<String>, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let index: types::PieceIndex =
        serde_json::from_str(artifact_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    // Return the result as a JSON string (parsed JS-side), `None` -> `undefined`.
    // serde-wasm-bindgen's to_value mismarshals the bool field when the call also
    // deserializes a JS array and takes a &str arg (externref-table aliasing); the
    // JSON round-trip avoids it.
    match identify::run_identify(&notes, &index, margin_threshold) {
        Some(result) => {
            let json = serde_json::to_string(&result)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(Some(json))
        }
        None => Ok(None),
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
