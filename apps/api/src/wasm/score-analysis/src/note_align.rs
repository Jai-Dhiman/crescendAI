//! Tier-1 note aligner: turns a chroma frame-warp + AMT performance notes + score
//! into per-note `NoteAlignment`s carrying a score-relative onset deviation (rush/drag).
//!
//! This is the deep module behind `align_chunk_notes`. The chroma DTW
//! (`chroma_dtw.rs`) already follows global tempo and the performer's rubato, so a
//! per-note deviation measured against the raw warp is ~0 by construction. To
//! recover a *directional* rush/drag signal we fit a smooth affine tempo
//! (`audio_time ~= a*score_time + b`) over the matched notes and take each note's
//! residual against that smooth line -- the local push/pull the DTW absorbed.
//!
//! Design note (#64): timing here is COARSE/directional by design. The downstream
//! `analyze_timing_tier1` reports the bar-MEAN of `onset_deviation_ms` (+-30ms
//! rush/drag band), so per-note matching noise averages down by ~sqrt(N) per bar.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::chroma_dtw::chroma_dtw_native;
use crate::types::{BarMap, NoteAlignment, PerfNote, ScoreBar};

/// Result of `align_chunk_notes`: the tier-1 `BarMap` plus the `bar_per_frame`
/// map. Both come from the SAME single chroma pass; `bar_per_frame` is retained
/// for client cursor following (the `chunk_bar_map` message) that previously
/// rode on `BarMapChroma`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkNoteResult {
    pub bar_map: BarMap,
    pub bar_per_frame: Vec<u32>,
}

/// A score note flattened for matching.
struct ScoreNoteRef {
    onset_seconds: f64,
    pitch: u8,
    bar_number: u32,
    beat_in_bar: f64,
}

/// Flatten score notes within the bar range, sorted by onset.
fn flatten_score_notes(score_bars: &[ScoreBar], bar_start: u32, bar_end: u32) -> Vec<ScoreNoteRef> {
    let mut out: Vec<ScoreNoteRef> = Vec::new();
    for bar in score_bars
        .iter()
        .filter(|b| b.bar_number >= bar_start && b.bar_number <= bar_end)
    {
        for n in &bar.notes {
            out.push(ScoreNoteRef {
                onset_seconds: n.onset_seconds,
                pitch: n.pitch,
                bar_number: bar.bar_number,
                beat_in_bar: (n.onset_seconds - bar.start_seconds).max(0.0),
            });
        }
    }
    out.sort_by(|a, b| a.onset_seconds.partial_cmp(&b.onset_seconds).unwrap());
    out
}

/// Map a performance audio time (chunk-relative seconds) to score time via the warp.
fn warp_audio_to_score_time(
    score_frame_per_audio_frame: &[u32],
    frame_rate_hz: f32,
    audio_t: f64,
) -> f64 {
    if score_frame_per_audio_frame.is_empty() || frame_rate_hz <= 0.0 {
        return audio_t;
    }
    let af = (audio_t * frame_rate_hz as f64).round() as i64;
    let af = af.clamp(0, score_frame_per_audio_frame.len() as i64 - 1) as usize;
    score_frame_per_audio_frame[af] as f64 / frame_rate_hz as f64
}

/// Least-squares affine fit `perf ~= a*score + b` over matched (score_onset, perf_onset).
/// Falls back to a unit slope with a constant offset when degenerate.
fn affine_fit(pairs: &[(f64, f64)]) -> (f64, f64) {
    let n = pairs.len() as f64;
    if pairs.len() < 2 {
        let b = if pairs.is_empty() {
            0.0
        } else {
            pairs[0].1 - pairs[0].0
        };
        return (1.0, b);
    }
    let sx: f64 = pairs.iter().map(|p| p.0).sum();
    let sy: f64 = pairs.iter().map(|p| p.1).sum();
    let sxx: f64 = pairs.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pairs.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return (1.0, (sy - sx) / n);
    }
    let a = (n * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n;
    (a, b)
}

/// Pure core: produce per-note alignments from a precomputed warp + notes.
/// `onset_window_s` is the score-time matching tolerance (pitch-exact, nearest, unique).
pub fn align_notes_from_warp(
    score_frame_per_audio_frame: &[u32],
    frame_rate_hz: f32,
    perf_notes: &[PerfNote],
    score_bars: &[ScoreBar],
    bar_start: u32,
    bar_end: u32,
    onset_window_s: f64,
) -> Vec<NoteAlignment> {
    let score_refs = flatten_score_notes(score_bars, bar_start, bar_end);
    if score_refs.is_empty() {
        return Vec::new();
    }
    let mut used = vec![false; score_refs.len()];

    // Greedy match: each perf note (in onset order) takes the nearest unused
    // same-pitch score note within the window, comparing in score time.
    let mut perf_sorted: Vec<&PerfNote> = perf_notes.iter().collect();
    perf_sorted.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());

    // (perf note, matched score index)
    let mut matches: Vec<(&PerfNote, usize)> = Vec::new();
    for p in &perf_sorted {
        let mapped =
            warp_audio_to_score_time(score_frame_per_audio_frame, frame_rate_hz, p.onset);
        let mut best: Option<(usize, f64)> = None;
        for (i, s) in score_refs.iter().enumerate() {
            if used[i] || s.pitch != p.pitch {
                continue;
            }
            let d = (s.onset_seconds - mapped).abs();
            if d <= onset_window_s && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        if let Some((i, _)) = best {
            used[i] = true;
            matches.push((p, i));
        }
    }

    // Smooth affine tempo fit, then per-note residual = directional rush/drag.
    let pairs: Vec<(f64, f64)> = matches
        .iter()
        .map(|(p, i)| (score_refs[*i].onset_seconds, p.onset))
        .collect();
    let (a, b) = affine_fit(&pairs);

    let mut alignments: Vec<NoteAlignment> = matches
        .iter()
        .map(|(p, i)| {
            let s = &score_refs[*i];
            let predicted = a * s.onset_seconds + b;
            NoteAlignment {
                perf_onset: p.onset,
                perf_pitch: p.pitch,
                perf_velocity: p.velocity,
                score_bar: s.bar_number,
                score_beat: s.beat_in_bar,
                score_pitch: s.pitch,
                onset_deviation_ms: (p.onset - predicted) * 1000.0,
            }
        })
        .collect();
    alignments.sort_by(|x, y| x.perf_onset.partial_cmp(&y.perf_onset).unwrap());
    alignments
}

/// Full producer: run chroma DTW, then align notes against the resulting warp.
/// Returns a `BarMap` ready for `analyze_tier1`.
#[allow(clippy::too_many_arguments)]
pub fn align_chunk_notes_native(
    audio_f32: &[f32],
    n_audio: u32,
    perf_notes: &[PerfNote],
    score_bars: &[ScoreBar],
    frame_rate_hz: f32,
    decim_hz: f32,
    prior_score_frame: i64,
    band_back_frames: u32,
    band_fwd_frames: u32,
    chunk_index: usize,
    onset_window_s: f64,
) -> Result<ChunkNoteResult, String> {
    let chroma = chroma_dtw_native(
        audio_f32,
        n_audio,
        score_bars,
        frame_rate_hz,
        decim_hz,
        prior_score_frame,
        band_back_frames,
        band_fwd_frames,
    )?;
    let alignments = align_notes_from_warp(
        &chroma.score_frame_per_audio_frame,
        frame_rate_hz,
        perf_notes,
        score_bars,
        chroma.bar_min,
        chroma.bar_max,
        onset_window_s,
    );
    Ok(ChunkNoteResult {
        bar_map: BarMap {
            chunk_index,
            bar_start: chroma.bar_min,
            bar_end: chroma.bar_max,
            alignments,
            confidence: (1.0 - chroma.cost as f64).clamp(0.0, 1.0),
            is_reanchored: false,
        },
        bar_per_frame: chroma.bar_per_frame,
    })
}

// --- WASM entry point ---

/// Align a 15s audio chunk to a score and produce per-note `NoteAlignment`s.
///
/// Mirrors `align_chunk_chroma` but additionally takes the AMT performance notes
/// and returns a full `BarMap` (with `alignments`) instead of a bar-level
/// `BarMapChroma`. Production WASM keeps the unconstrained search (prior < 0).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn align_chunk_notes(
    audio_bytes: &[u8],
    n_audio: u32,
    perf_notes_js: JsValue,
    score_bars_js: JsValue,
    frame_rate_hz: f32,
    decim_hz: f32,
    chunk_index: u32,
    onset_window_s: f64,
) -> Result<JsValue, JsValue> {
    if audio_bytes.len() % 4 != 0 {
        return Err(JsValue::from_str("audio_bytes length not a multiple of 4"));
    }
    let audio_f32: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let perf_notes: Vec<PerfNote> = serde_wasm_bindgen::from_value(perf_notes_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let score_bars: Vec<ScoreBar> = serde_wasm_bindgen::from_value(score_bars_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = align_chunk_notes_native(
        &audio_f32,
        n_audio,
        &perf_notes,
        &score_bars,
        frame_rate_hz,
        decim_hz,
        -1,
        0,
        0,
        chunk_index as usize,
        onset_window_s,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScoreNote;

    fn snote(pitch: u8, onset: f64) -> ScoreNote {
        ScoreNote {
            pitch,
            pitch_name: String::new(),
            velocity: 80,
            onset_tick: 0,
            onset_seconds: onset,
            duration_ticks: 240,
            duration_seconds: 0.25,
            track: 0,
        }
    }

    fn sbar(bar_number: u32, start_seconds: f64, notes: Vec<ScoreNote>) -> ScoreBar {
        ScoreBar {
            bar_number,
            start_tick: 0,
            start_seconds,
            time_signature: "4/4".to_string(),
            notes,
            pedal_events: vec![],
            note_count: 0,
            pitch_range: vec![],
            mean_velocity: 80,
        }
    }

    fn pnote(pitch: u8, onset: f64) -> PerfNote {
        PerfNote {
            pitch,
            onset,
            offset: onset + 0.25,
            velocity: 80,
        }
    }

    /// Two bars, four notes each at a steady 0.5s spacing.
    fn two_bar_score() -> Vec<ScoreBar> {
        vec![
            sbar(
                1,
                0.0,
                vec![snote(60, 0.0), snote(62, 0.5), snote(64, 1.0), snote(65, 1.5)],
            ),
            sbar(
                2,
                2.0,
                vec![snote(67, 2.0), snote(69, 2.5), snote(71, 3.0), snote(72, 3.5)],
            ),
        ]
    }

    /// Identity warp: audio frame i maps to score frame i (audio time == score time).
    fn identity_warp(n_frames: usize) -> Vec<u32> {
        (0..n_frames as u32).collect()
    }

    #[test]
    fn matches_all_notes_pitch_and_bar_correct() {
        let score = two_bar_score();
        let warp = identity_warp(250); // 5s at 50Hz
        let perf: Vec<PerfNote> = vec![
            pnote(60, 0.0), pnote(62, 0.5), pnote(64, 1.0), pnote(65, 1.5),
            pnote(67, 2.0), pnote(69, 2.5), pnote(71, 3.0), pnote(72, 3.5),
        ];
        let al = align_notes_from_warp(&warp, 50.0, &perf, &score, 1, 2, 0.25);
        assert_eq!(al.len(), 8, "all eight notes should match");
        assert!(al.iter().filter(|a| a.score_bar == 1).count() == 4);
        assert!(al.iter().filter(|a| a.score_bar == 2).count() == 4);
        assert!(al.iter().all(|a| a.perf_pitch == a.score_pitch));
    }

    #[test]
    fn uniform_shift_yields_near_zero_deviation() {
        // Whole performance shifted +0.2s (constant): affine `b` absorbs it -> ~0 residual.
        let score = two_bar_score();
        let warp = identity_warp(300);
        let perf: Vec<PerfNote> = (0..8)
            .map(|i| {
                let (pitch, onset) = [
                    (60, 0.0), (62, 0.5), (64, 1.0), (65, 1.5),
                    (67, 2.0), (69, 2.5), (71, 3.0), (72, 3.5),
                ][i];
                pnote(pitch, onset + 0.2)
            })
            .collect();
        let al = align_notes_from_warp(&warp, 50.0, &perf, &score, 1, 2, 0.3);
        assert_eq!(al.len(), 8);
        for a in &al {
            assert!(
                a.onset_deviation_ms.abs() < 5.0,
                "uniform shift should leave ~0 deviation, got {}",
                a.onset_deviation_ms
            );
        }
    }

    #[test]
    fn local_rush_makes_bar_mean_negative_relative_to_steady() {
        // Bar 1 steady; bar 2 played 0.12s EARLY (rushing). The affine tempo fit absorbs
        // PART of a whole-bar step shift as a slope change (a sustained regional speed-up
        // is interpretation, not a timing error), so the residual is a fraction of 120ms.
        // The directional contract still holds: bar-2 mean is clearly negative and well
        // below bar-1 mean.
        let score = two_bar_score();
        let warp = identity_warp(300);
        let perf: Vec<PerfNote> = vec![
            pnote(60, 0.0), pnote(62, 0.5), pnote(64, 1.0), pnote(65, 1.5),
            pnote(67, 2.0 - 0.12), pnote(69, 2.5 - 0.12),
            pnote(71, 3.0 - 0.12), pnote(72, 3.5 - 0.12),
        ];
        let al = align_notes_from_warp(&warp, 50.0, &perf, &score, 1, 2, 0.3);
        assert_eq!(al.len(), 8);
        let mean = |bar: u32| {
            let v: Vec<f64> = al
                .iter()
                .filter(|a| a.score_bar == bar)
                .map(|a| a.onset_deviation_ms)
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(
            mean(2) < mean(1) - 20.0,
            "bar2 (rushed) mean {} should be clearly below bar1 mean {}",
            mean(2),
            mean(1)
        );
        assert!(mean(2) < 0.0, "bar2 rush should read negative, got {}", mean(2));
    }

    #[test]
    fn wrong_pitch_and_far_notes_do_not_match() {
        let score = two_bar_score();
        let warp = identity_warp(300);
        let perf: Vec<PerfNote> = vec![
            pnote(61, 0.0),   // wrong pitch (no C#4 in score)
            pnote(62, 0.5),   // good
            pnote(64, 4.9),   // right pitch, far outside any score onset window
        ];
        let al = align_notes_from_warp(&warp, 50.0, &perf, &score, 1, 2, 0.2);
        assert_eq!(al.len(), 1, "only the in-window same-pitch note matches");
        assert_eq!(al[0].score_pitch, 62);
    }

    #[test]
    fn dropped_note_reduces_alignment_count() {
        let score = two_bar_score();
        let warp = identity_warp(300);
        // Performer omits the third note of bar 1 (pitch 64).
        let perf: Vec<PerfNote> = vec![
            pnote(60, 0.0), pnote(62, 0.5), pnote(65, 1.5),
            pnote(67, 2.0), pnote(69, 2.5), pnote(71, 3.0), pnote(72, 3.5),
        ];
        let al = align_notes_from_warp(&warp, 50.0, &perf, &score, 1, 2, 0.25);
        assert_eq!(al.len(), 7, "seven played notes -> seven alignments");
        assert!(al.iter().all(|a| a.score_pitch != 64));
    }

    #[test]
    fn affine_fit_recovers_slope_and_intercept() {
        // perf = 2*score + 1 exactly -> slope 2, intercept 1.
        let pairs = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
        let (a, b) = affine_fit(&pairs);
        assert!((a - 2.0).abs() < 1e-9, "slope {}", a);
        assert!((b - 1.0).abs() < 1e-9, "intercept {}", b);
    }
}
