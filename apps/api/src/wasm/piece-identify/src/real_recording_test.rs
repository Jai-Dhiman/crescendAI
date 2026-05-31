//! Isolated piece-ID test driven by a real AMT-transcribed recording.
//! Inputs are passed via env vars so no fixtures are committed:
//!   NOTES_JSON, NGRAM_INDEX, RERANK_FEATURES, SCORE_JSON, EXPECTED_PIECE
//! Run: cargo test -p piece-identify identify_real_recording -- --nocapture

use crate::types::{NgramIndex, PerfNote, RerankFeatures};
use crate::{dtw_confirm, ngram, rerank};
use std::fs;

fn read(key: &str) -> String {
    let path = std::env::var(key).unwrap_or_else(|_| panic!("env {key} not set"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[derive(serde::Deserialize)]
struct AmtNotes {
    notes: Vec<PerfNote>,
}

#[derive(serde::Deserialize)]
struct ScoreNoteJson {
    pitch: u8,
    onset_seconds: f64,
}
#[derive(serde::Deserialize)]
struct ScoreBarJson {
    notes: Vec<ScoreNoteJson>,
}
#[derive(serde::Deserialize)]
struct ScoreJson {
    bars: Vec<ScoreBarJson>,
}

#[test]
fn identify_real_recording() {
    let perf: Vec<PerfNote> = serde_json::from_str::<AmtNotes>(&read("NOTES_JSON"))
        .expect("parse notes")
        .notes;
    let index: NgramIndex = serde_json::from_str(&read("NGRAM_INDEX")).expect("parse index");
    let feats: RerankFeatures =
        serde_json::from_str(&read("RERANK_FEATURES")).expect("parse features");
    let score: ScoreJson = serde_json::from_str(&read("SCORE_JSON")).expect("parse score");
    let expected = std::env::var("EXPECTED_PIECE").unwrap_or_default();

    eprintln!("--- piece-ID on {} perf notes ---", perf.len());

    let candidates = ngram::ngram_recall(&perf, &index);
    eprintln!("Stage 1 ngram candidates (top 5):");
    for c in candidates.iter().take(5) {
        eprintln!("  {:<40} hits={}", c.piece_id, c.hit_count);
    }

    let reranked = rerank::rerank_candidates(&perf, &candidates, &feats);
    eprintln!("Stage 2 reranked (top 5):");
    for r in reranked.iter().take(5) {
        eprintln!("  {:<40} sim={:.3}", r.piece_id, r.similarity);
    }

    let score_notes: Vec<(f64, u8)> = score
        .bars
        .iter()
        .flat_map(|b| b.notes.iter().map(|n| (n.onset_seconds, n.pitch)))
        .collect();

    let top = reranked.first().expect("no candidates after rerank");
    let dtw = dtw_confirm::dtw_confirm(&perf, &score_notes, 0.3);
    eprintln!(
        "Stage 3 DTW: top={} sim={:.3} threshold=0.50  dtw_confirmed={} cost={:.3} (thr 0.30)",
        top.piece_id, top.similarity, dtw.confirmed, dtw.cost
    );
    eprintln!(
        "VERDICT: passes_rerank_gate={} dtw_confirmed={}",
        top.similarity >= 0.5,
        dtw.confirmed
    );

    if !expected.is_empty() {
        assert_eq!(top.piece_id, expected, "top candidate mismatch");
    }
}
