// apps/api/src/wasm/score-analysis/tests/chroma_dtw_roundtrip.rs
//
// Cargo integration test: loads committed binary fixtures and asserts
// align_chunk_chroma_native returns sane bar ranges.
//
// Run with: cargo test chroma_dtw_roundtrip
// (in apps/api/src/wasm/score-analysis/)

use score_analysis::chroma_dtw_native;
use score_analysis::types::{BarMapChroma, ScoreBar};
use std::path::Path;

#[derive(serde::Deserialize)]
struct ExpectedFixture {
    bar_min_lo: u32,
    bar_min_hi: u32,
    bar_max_lo: u32,
    bar_max_hi: u32,
    decim_n: usize,
    cost_hi: f32,
    frame_rate_hz: f32,
}

fn load_fixture(slug: &str) -> (Vec<f32>, u32, Vec<ScoreBar>, ExpectedFixture) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(slug);

    // audio_chroma.bin: raw LE f32 bytes, row-major 12 x n_frames
    let bin = std::fs::read(base.join("audio_chroma.bin"))
        .unwrap_or_else(|_| panic!("missing audio_chroma.bin for {slug}"));
    let floats: Vec<f32> = bin
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n_frames = (floats.len() / 12) as u32;

    let bars_json = std::fs::read_to_string(base.join("score_bars.json"))
        .unwrap_or_else(|_| panic!("missing score_bars.json for {slug}"));
    let score_bars: Vec<ScoreBar> = serde_json::from_str(&bars_json)
        .unwrap_or_else(|e| panic!("score_bars.json parse error for {slug}: {e}"));

    let expected_json = std::fs::read_to_string(base.join("expected.json"))
        .unwrap_or_else(|_| panic!("missing expected.json for {slug}"));
    let expected: ExpectedFixture = serde_json::from_str(&expected_json)
        .unwrap_or_else(|e| panic!("expected.json parse error for {slug}: {e}"));

    (floats, n_frames, score_bars, expected)
}

#[test]
fn chroma_dtw_roundtrip_coldstart() {
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_coldstart_111s");

    let result: BarMapChroma =
        chroma_dtw_native(&audio_f32, n_audio, &score_bars, expected.frame_rate_hz, 5.0, -1, 0, 0)
            .expect("align_chunk_chroma_native should not fail on valid fixture");

    assert!(
        result.bar_min >= expected.bar_min_lo && result.bar_min <= expected.bar_min_hi,
        "bar_min={} not in [{}, {}]",
        result.bar_min,
        expected.bar_min_lo,
        expected.bar_min_hi,
    );
    assert!(
        result.bar_max >= expected.bar_max_lo && result.bar_max <= expected.bar_max_hi,
        "bar_max={} not in [{}, {}]",
        result.bar_max,
        expected.bar_max_lo,
        expected.bar_max_hi,
    );
    assert_eq!(
        result.bar_per_frame.len(),
        expected.decim_n,
        "bar_per_frame length mismatch"
    );
    assert!(
        result.cost < expected.cost_hi,
        "cost={} not below {}",
        result.cost,
        expected.cost_hi,
    );
    // Monotone non-decreasing check
    for w in result.bar_per_frame.windows(2) {
        assert!(
            w[1] >= w[0],
            "bar_per_frame is not non-decreasing at {:?}",
            w
        );
    }
}

#[test]
fn chroma_dtw_roundtrip_forward() {
    let (audio_f32, n_audio, score_bars, expected) = load_fixture("ballade1_forward_2min");

    let result: BarMapChroma =
        chroma_dtw_native(&audio_f32, n_audio, &score_bars, expected.frame_rate_hz, 5.0, -1, 0, 0)
            .expect("align_chunk_chroma_native should not fail on valid fixture");

    assert!(
        result.bar_min >= expected.bar_min_lo && result.bar_min <= expected.bar_min_hi,
        "bar_min={} not in [{}, {}]",
        result.bar_min,
        expected.bar_min_lo,
        expected.bar_min_hi,
    );
    assert!(
        result.bar_max >= expected.bar_max_lo && result.bar_max <= expected.bar_max_hi,
        "bar_max={} not in [{}, {}]",
        result.bar_max,
        expected.bar_max_lo,
        expected.bar_max_hi,
    );
    assert_eq!(result.bar_per_frame.len(), expected.decim_n);
    assert!(
        result.cost < expected.cost_hi,
        "cost={} not below {}",
        result.cost,
        expected.cost_hi,
    );
}
