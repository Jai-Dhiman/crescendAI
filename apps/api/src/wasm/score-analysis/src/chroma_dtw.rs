// apps/api/src/wasm/score-analysis/src/chroma_dtw.rs
//
// Stateless per-chunk chroma subsequence DTW.
// Entry point: align_chunk_chroma (wasm_bindgen) + chroma_dtw_native (Rust tests).

use crate::types::{BarMapChroma, ScoreBar};

/// Map a score-side frame index to its bar number using bar start_seconds.
fn frame_to_bar(frame_idx: usize, frame_rate: f32, bars: &[ScoreBar]) -> u32 {
    let t = frame_idx as f32 / frame_rate;
    let mut last = bars.first().map(|b| b.bar_number).unwrap_or(1);
    for bar in bars {
        if bar.start_seconds as f32 <= t {
            last = bar.bar_number;
        } else {
            break;
        }
    }
    last
}

/// Build a 12 x N score chroma matrix from the bar note list.
/// Each note contributes its pitch class to every frame it spans.
/// Adds 1e-3 floor, then L2-normalizes per column.
fn build_score_chroma(bars: &[ScoreBar], frame_rate: f32) -> Vec<f32> {
    // Find total duration from last note end
    let mut end_time: f32 = 0.0;
    for bar in bars {
        for note in &bar.notes {
            let note_end = note.onset_seconds as f32 + (note.duration_seconds as f32).max(0.05);
            if note_end > end_time {
                end_time = note_end;
            }
        }
    }
    if end_time <= 0.0 {
        return Vec::new();
    }
    let n_frames = (end_time * frame_rate).ceil() as usize + 1;
    let mut chroma = vec![0.0_f32; 12 * n_frames];

    for bar in bars {
        for note in &bar.notes {
            let onset = note.onset_seconds as f32;
            let dur = (note.duration_seconds as f32).max(0.05);
            let pc = (note.pitch % 12) as usize;
            let f0 = (onset * frame_rate).floor() as usize;
            let f1 = ((onset + dur) * frame_rate).ceil() as usize;
            let f0 = f0.min(n_frames);
            let f1 = f1.min(n_frames);
            for f in f0..f1 {
                chroma[pc * n_frames + f] += 1.0;
            }
        }
    }

    // Floor + L2-normalize per column
    for f in 0..n_frames {
        let mut norm_sq: f32 = 0.0;
        for pc in 0..12 {
            chroma[pc * n_frames + f] += 1e-3;
            norm_sq += chroma[pc * n_frames + f] * chroma[pc * n_frames + f];
        }
        let norm = norm_sq.sqrt() + 1e-9;
        for pc in 0..12 {
            chroma[pc * n_frames + f] /= norm;
        }
    }

    chroma
}

/// Cosine distance between two column vectors of length 12, stored row-major.
/// `a`: row-major 12 x n_a, column j.  `b`: row-major 12 x n_b, column i.
#[inline]
fn cosine_dist(
    a: &[f32], n_a: usize, j: usize,
    b: &[f32], n_b: usize, i: usize,
) -> f32 {
    let mut dot = 0.0_f32;
    for pc in 0..12 {
        dot += a[pc * n_a + j] * b[pc * n_b + i];
    }
    (1.0 - dot).max(0.0)
}

/// Subsequence DTW: finds best contiguous region of `score` (long) matching `audio` (short).
/// Step pattern: (1,0), (0,1), (1,1) — standard monotonic.
/// Returns (warping_path_vec, mean_cost).
/// `warping_path_vec`: Vec of (score_frame, audio_frame) pairs, forward order.
fn subseq_dtw(
    audio: &[f32], n_audio: usize,  // row-major 12 x n_audio
    score: &[f32], n_score: usize,  // row-major 12 x n_score
) -> (Vec<(usize, usize)>, f32) {
    let n_a = n_audio;
    let n_s = n_score;

    let mut d = vec![f32::INFINITY; n_a * n_s];
    let mut p: Vec<(i32, i32)> = vec![(0, 0); n_a * n_s]; // predecessor (da, ds)

    // idx helper: audio row i, score col j -> flat index
    let idx = |i: usize, j: usize| i * n_s + j;

    // Initialize first audio row — subsequence: free start anywhere on score
    for j in 0..n_s {
        d[idx(0, j)] = cosine_dist(audio, n_a, 0, score, n_s, j);
    }

    // Fill
    for i in 1..n_a {
        for j in 0..n_s {
            let c = cosine_dist(audio, n_a, i, score, n_s, j);
            // Predecessors: (i-1, j), (i, j-1), (i-1, j-1)
            let from_up = if i > 0 { d[idx(i - 1, j)] } else { f32::INFINITY };
            let from_left = if j > 0 { d[idx(i, j - 1)] } else { f32::INFINITY };
            let from_diag = if i > 0 && j > 0 { d[idx(i - 1, j - 1)] } else { f32::INFINITY };

            let (best, pred) = if from_diag <= from_up && from_diag <= from_left {
                (from_diag, (-1i32, -1i32))
            } else if from_up <= from_left {
                (from_up, (-1, 0))
            } else {
                (from_left, (0, -1))
            };

            d[idx(i, j)] = c + best;
            p[idx(i, j)] = pred;
        }
    }

    // Find best end on last audio row (minimum over all score positions)
    let last_row = n_a - 1;
    let j_end = (0..n_s)
        .min_by(|&ja, &jb| d[idx(last_row, ja)].partial_cmp(&d[idx(last_row, jb)]).unwrap())
        .unwrap_or(0);

    // Backtrack
    let mut path = Vec::with_capacity(n_a + n_s);
    let mut i = last_row as i32;
    let mut j = j_end as i32;
    while i >= 0 {
        path.push((j as usize, i as usize)); // (score_frame, audio_frame)
        let (di, dj) = p[idx(i as usize, j as usize)];
        if di == 0 && dj == 0 {
            // First row reached — (0,0) signals termination (first row was never overwritten)
            break;
        }
        i += di;
        j += dj;
    }
    path.reverse();

    // Mean cost along path
    let mean_cost = if path.is_empty() {
        1.0
    } else {
        let total: f32 = path
            .iter()
            .map(|&(sf, af)| cosine_dist(audio, n_a, af, score, n_s, sf))
            .sum();
        total / path.len() as f32
    };

    (path, mean_cost)
}

/// Pure Rust core, callable from cargo tests without WASM boundary.
pub fn chroma_dtw_native(
    audio_f32: &[f32],   // row-major float32, 12 x n_audio
    n_audio: u32,
    score_bars: &[ScoreBar],
    frame_rate_hz: f32,
    decim_hz: f32,
) -> Result<BarMapChroma, String> {
    let n_a = n_audio as usize;
    if audio_f32.len() != 12 * n_a {
        return Err(format!(
            "audio_f32 length {} != 12 * n_audio {}",
            audio_f32.len(),
            12 * n_a
        ));
    }
    if score_bars.is_empty() {
        return Err("score_bars is empty".to_string());
    }
    if n_a == 0 {
        return Err("n_audio is zero".to_string());
    }

    let score_chroma = build_score_chroma(score_bars, frame_rate_hz);
    let n_s = score_chroma.len() / 12;
    if n_s == 0 {
        return Err("score has no notes".to_string());
    }

    let (path, mean_cost) = subseq_dtw(audio_f32, n_a, &score_chroma, n_s);

    // Build bar_per_frame at decim_hz by sampling the warping path
    let decim_step = (frame_rate_hz / decim_hz).round() as usize;
    let decim_step = decim_step.max(1);
    let decim_n = (n_a + decim_step - 1) / decim_step;

    // path is (score_frame, audio_frame) pairs; build audio_frame -> score_frame map
    let mut audio_to_score = vec![0usize; n_a];
    let mut frame_mapped = vec![false; n_a];
    for &(sf, af) in &path {
        if af < n_a {
            audio_to_score[af] = sf;
            frame_mapped[af] = true;
        }
    }
    // Fill gaps (path may not cover every audio frame); use frame_mapped to avoid
    // treating valid score frame 0 as unmapped.
    let mut last_sf = path.first().map(|&(sf, _)| sf).unwrap_or(0);
    for af in 0..n_a {
        if frame_mapped[af] {
            last_sf = audio_to_score[af];
        } else {
            audio_to_score[af] = last_sf;
        }
    }

    let score_frame_per_audio_frame: Vec<u32> =
        audio_to_score.iter().map(|&sf| sf as u32).collect();

    // Defensive assertion: length must equal n_audio regardless of decim ratio.
    assert_eq!(
        score_frame_per_audio_frame.len(),
        n_a,
        "score_frame_per_audio_frame length {} != n_audio {}",
        score_frame_per_audio_frame.len(),
        n_a
    );

    let bar_per_frame: Vec<u32> = (0..decim_n)
        .map(|d| {
            let af = (d * decim_step).min(n_a - 1);
            let sf = audio_to_score[af];
            frame_to_bar(sf, frame_rate_hz, score_bars)
        })
        .collect();

    let bar_min = bar_per_frame.iter().copied().min().unwrap_or(1);
    let bar_max = bar_per_frame.iter().copied().max().unwrap_or(1);

    Ok(BarMapChroma {
        bar_min,
        bar_max,
        cost: mean_cost,
        bar_per_frame,
        score_frame_per_audio_frame,
    })
}

// --- WASM entry point ---

use wasm_bindgen::prelude::*;

/// Align a 15s audio chunk to a score using chroma-based subsequence DTW.
///
/// `audio_bytes`: raw LE float32 bytes, row-major 12 x n_audio
/// `n_audio`: number of audio frames (columns)
/// `score_bars_js`: JS array of ScoreBar objects
/// `frame_rate_hz`: frame rate of audio chroma (typically 50.0)
/// `decim_hz`: output frame rate for bar_per_frame (typically 5.0)
///
/// Returns `BarMapChroma` on success, error string on failure.
#[wasm_bindgen]
pub fn align_chunk_chroma(
    audio_bytes: &[u8],
    n_audio: u32,
    score_bars_js: JsValue,
    frame_rate_hz: f32,
    decim_hz: f32,
) -> Result<JsValue, JsValue> {
    // Reinterpret bytes as f32 LE
    if audio_bytes.len() % 4 != 0 {
        return Err(JsValue::from_str("audio_bytes length not a multiple of 4"));
    }
    let audio_f32: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let score_bars: Vec<crate::types::ScoreBar> =
        serde_wasm_bindgen::from_value(score_bars_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = chroma_dtw_native(&audio_f32, n_audio, &score_bars, frame_rate_hz, decim_hz)
        .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScoreBar;

    fn make_score_bar() -> ScoreBar {
        ScoreBar {
            bar_number: 1,
            start_tick: 0,
            start_seconds: 0.0,
            time_signature: "4/4".to_string(),
            notes: vec![],
            pedal_events: vec![],
            note_count: 0,
            pitch_range: vec![],
            mean_velocity: 64,
        }
    }

    #[test]
    fn n_audio_zero_returns_error() {
        let score_bars = vec![make_score_bar()];
        // n_audio = 0 => audio_f32 len = 12 * 0 = 0, passes length check, must be caught by guard
        let result = chroma_dtw_native(&[], 0, &score_bars, 10.0, 2.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "n_audio is zero");
    }

    // Regression: gap-fill must not overwrite audio frames that legitimately map to score
    // frame 0 (sf=0).  The old code used `audio_to_score[af] == 0` as a sentinel for
    // "unmapped", which silently replaced valid sf=0 mappings with the previous non-zero
    // score frame.  The fix uses a parallel `frame_mapped` Vec.
    //
    // Fixture: single-frame score (one C note at onset 0s), two audio frames both strongly
    // matching C.  Subsequence DTW assigns both af=0 and af=1 to sf=0.  After gap-fill,
    // bar_per_frame must reflect bar 1 (the only bar) for every decimated frame — not some
    // stale non-zero value that would indicate sf=0 was wrongly overwritten.
    #[test]
    fn valid_score_frame_zero_not_overwritten_by_gap_fill() {
        use crate::types::ScoreNote;

        // Build a score bar with a single C4 note at onset 0s, duration 0.2s.
        // At frame_rate=10 Hz this produces exactly 1 score chroma column (frame 0).
        let c_note = ScoreNote {
            pitch: 60, // C4, pitch class 0
            pitch_name: "C4".to_string(),
            velocity: 80,
            onset_tick: 0,
            onset_seconds: 0.0,
            duration_ticks: 480,
            duration_seconds: 0.2,
            track: 0,
        };
        let bar = ScoreBar {
            bar_number: 1,
            start_tick: 0,
            start_seconds: 0.0,
            time_signature: "4/4".to_string(),
            notes: vec![c_note],
            pedal_events: vec![],
            note_count: 1,
            pitch_range: vec![60],
            mean_velocity: 80,
        };

        // Audio chroma: 2 frames, both strongly C (pitch class 0).
        // Row-major 12 x 2: pc=0 is row 0; all other rows near zero.
        // Each column is L2-normalised so we set pc=0 to 1.0 and rest to 0.0.
        let mut audio_f32 = vec![0.0_f32; 12 * 2];
        audio_f32[0 * 2 + 0] = 1.0; // pc=0, af=0
        audio_f32[0 * 2 + 1] = 1.0; // pc=0, af=1

        // frame_rate=10 Hz, decim_hz=10 Hz (1:1 so bar_per_frame has 2 entries).
        let result = chroma_dtw_native(&audio_f32, 2, &[bar], 10.0, 10.0);
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
        let bar_map = result.unwrap();

        // Both audio frames map to sf=0 which belongs to bar 1.
        // If the sentinel bug were present, the second frame could be overwritten to bar 0
        // or an arbitrary stale value; bar 1 is the only valid answer here.
        for (i, &b) in bar_map.bar_per_frame.iter().enumerate() {
            assert_eq!(
                b, 1,
                "bar_per_frame[{}] = {} but expected 1 (sf=0 belongs to bar 1)",
                i, b
            );
        }
    }
}
