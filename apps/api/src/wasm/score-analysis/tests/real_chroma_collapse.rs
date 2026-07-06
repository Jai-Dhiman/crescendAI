// apps/api/src/wasm/score-analysis/tests/real_chroma_collapse.rs
//
// Regression for issue #64: the tier-1 chroma-DTW warp must span the bars the
// audio actually covers on REAL MuQ chroma, not collapse the whole 15s chunk
// onto score bar 1.
//
// The fixture is REAL, captured from the live MuQ (:8000) + AMT (:8001) servers:
//   bach_inv1_chunk0_chroma.f32  751 x 12 f32 chroma of the first 15s of
//                                bach_invention_1 (7zVlDxBO5q4.wav)
//   bach_inv1_chunk0.json        {frames, frame_rate_hz, perf_notes, score_bars}
//                                82 AMT perf notes + the 22-bar bach.inventions.1 score
//
// Synthetic chroma is exactly what hid this bug: the 7/7 in-crate note_align/
// chroma_dtw tests feed clean synthetic warps (one even asserts bar_min==bar_max==1
// for a single-bar fixture). A two-voice C-major invention is diatonically uniform,
// so its per-bar chroma is nearly identical -> a flat DTW cost surface with no
// gradient to hold the path on the true diagonal. Only a structural warp constraint
// prevents the degenerate collapse; this test locks that in against real data.
//
// Run with: cargo test --test real_chroma_collapse

use score_analysis::align_chunk_notes_native;
use score_analysis::types::ScoreBar;
use std::path::Path;

#[derive(serde::Deserialize)]
struct PerfNoteJson {
    pitch: u8,
    onset: f64,
    offset: f64,
    velocity: u8,
}

#[derive(serde::Deserialize)]
struct Fixture {
    frames: u32,
    frame_rate_hz: f32,
    perf_notes: Vec<PerfNoteJson>,
    score_bars: Vec<ScoreBar>,
}

fn load() -> (Vec<f32>, Fixture) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let raw = std::fs::read(base.join("bach_inv1_chunk0_chroma.f32"))
        .expect("missing bach_inv1_chunk0_chroma.f32");
    assert_eq!(raw.len() % 4, 0, "chroma bytes not f32-aligned");
    let chroma: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let json = std::fs::read_to_string(base.join("bach_inv1_chunk0.json"))
        .expect("missing bach_inv1_chunk0.json");
    let fx: Fixture = serde_json::from_str(&json).expect("fixture json parse error");
    assert_eq!(
        chroma.len(),
        12 * fx.frames as usize,
        "chroma len {} != 12 * frames {}",
        chroma.len(),
        fx.frames
    );
    (chroma, fx)
}

#[test]
fn tier1_warp_spans_multiple_bars_on_real_bach_chroma() {
    let (chroma, fx) = load();
    let perf: Vec<score_analysis::types::PerfNote> = fx
        .perf_notes
        .iter()
        .map(|p| score_analysis::types::PerfNote {
            pitch: p.pitch,
            onset: p.onset,
            offset: p.offset,
            velocity: p.velocity,
        })
        .collect();

    // Production seam: prior=-1 (unconstrained full-score search), decim 5Hz,
    // onset window 0.15s, chunk 0 -- exactly what session-brain.ts passes.
    let res = align_chunk_notes_native(
        &chroma,
        fx.frames,
        &perf,
        &fx.score_bars,
        fx.frame_rate_hz,
        5.0,
        -1,
        0,
        0,
        0,
        0.15,
    )
    .expect("aligner should run end-to-end on real chroma");

    let bm = &res.bar_map;
    let distinct_bars: std::collections::BTreeSet<u32> =
        bm.alignments.iter().map(|a| a.score_bar).collect();
    let devs: Vec<f64> = bm.alignments.iter().map(|a| a.onset_deviation_ms).collect();
    let mean_dev = if devs.is_empty() {
        0.0
    } else {
        devs.iter().sum::<f64>() / devs.len() as f64
    };
    eprintln!(
        "[#64 real-chroma] bar_start={} bar_end={} alignments={} distinct_bars={:?} mean_dev_ms={:.1}",
        bm.bar_start,
        bm.bar_end,
        bm.alignments.len(),
        distinct_bars,
        mean_dev
    );

    // (1) The warp must span the bars the 15s audio covers, not collapse to 1..1.
    assert!(
        bm.bar_end > bm.bar_start,
        "warp collapsed: bar_start={} bar_end={} (expected a multi-bar span)",
        bm.bar_start,
        bm.bar_end
    );
    // Chunk 0 starts at the piece opening (bar 1) and 15s of a Bach invention
    // covers roughly 5-6 bars; require a non-trivial span. On the frozen fixture
    // the constrained warp yields bars 1..4; the slope constraint guarantees the
    // score advances >= n_audio/2 frames, so a multi-bar span is structural.
    assert!(
        bm.bar_end - bm.bar_start >= 2,
        "warp span too small: bars {}..{} (expected a multi-bar span for 15s)",
        bm.bar_start,
        bm.bar_end
    );

    // (2) Non-empty alignments reaching the tier-1 analyzer.
    assert!(
        !bm.alignments.is_empty(),
        "no note alignments produced -> tier-1 silently falls back to tier-2"
    );
    // Alignments must actually span more than one bar (not all pooled in bar 1).
    assert!(
        distinct_bars.len() >= 2,
        "alignments cover only bars {:?}; expected notes across multiple bars",
        distinct_bars
    );

    // (3) onset_deviation_ms must be finite and bounded (the affine tempo fit
    // removes global tempo, so residual rush/drag is small), and carry real
    // directional signal -- not all identically zero.
    assert!(
        devs.iter().all(|d| d.is_finite() && d.abs() < 500.0),
        "onset deviations must be finite and bounded (<500ms residual)"
    );
    assert!(
        devs.iter().any(|d| d.abs() > 1e-6),
        "onset deviations are all ~0; no rush/drag signal reaches the teacher"
    );
}
