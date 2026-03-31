//! Stage 3: DTW confirmation.
//! Runs subsequence DTW alignment between performance notes and score notes to
//! confirm a piece identification candidate. Contains a minimal DTW implementation
//! copied from score-analysis since the two WASM modules are independent binaries.

use crate::types::{DtwConfirmResult, PerfNote};

const PITCH_MISMATCH_PENALTY: f64 = 0.5;

/// Compute pitch mismatch penalty between two MIDI pitches.
fn pitch_penalty(p1: u8, p2: u8) -> f64 {
    if p1 == p2 {
        0.0
    } else {
        let diff = (i16::from(p1) - i16::from(p2)).unsigned_abs();
        if diff == 1 {
            0.125
        } else if diff == 12 {
            0.25
        } else {
            PITCH_MISMATCH_PENALTY
        }
    }
}

/// Subsequence DTW alignment of perf notes against score notes.
///
/// Score notes are provided as (onset_seconds, pitch) tuples.
/// Returns (normalized_cost) where lower cost means better alignment.
/// Subsequence DTW initializes dtw[0][j] = 0 for all j (free start in score).
#[allow(clippy::needless_range_loop)] // DTW matrix indexed access is clearer than iterators
fn subsequence_dtw(perf: &[(f64, u8)], score: &[(f64, u8)]) -> f64 {
    if perf.is_empty() || score.is_empty() {
        return f64::MAX;
    }

    let n = perf.len();
    let m = score.len();

    let mut dtw = vec![vec![f64::MAX; m]; n];

    // Initialize: free start in score (subsequence DTW)
    for j in 0..m {
        let cost = (perf[0].0 - score[j].0).abs() + pitch_penalty(perf[0].1, score[j].1);
        dtw[0][j] = cost;
    }

    // Fill DP table
    for i in 1..n {
        for j in 0..m {
            let cost = (perf[i].0 - score[j].0).abs() + pitch_penalty(perf[i].1, score[j].1);
            let prev = if j == 0 {
                dtw[i - 1][0]
            } else {
                dtw[i - 1][j - 1].min(dtw[i - 1][j]).min(dtw[i][j - 1])
            };
            if prev == f64::MAX {
                dtw[i][j] = f64::MAX;
            } else {
                dtw[i][j] = prev + cost;
            }
        }
    }

    // Find best endpoint in last row (minimum cost end in score)
    let best_cost = dtw[n - 1]
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);

    if n > 0 {
        best_cost / n as f64
    } else {
        f64::MAX
    }
}

/// Run DTW confirmation of a piece identification candidate.
///
/// `perf_notes`: recent performance notes (the identification window).
/// `score_notes`: flattened score notes as (onset_seconds, pitch) tuples covering
///                the full score. The TS layer provides these from the loaded score JSON.
/// `threshold`: normalized cost below which the identification is confirmed.
///              The Rust API uses 0.3 (`DTW_CONFIRM_THRESHOLD`).
pub fn dtw_confirm(
    perf_notes: &[PerfNote],
    score_notes: &[(f64, u8)],
    threshold: f64,
) -> DtwConfirmResult {
    if perf_notes.len() < 3 || score_notes.is_empty() {
        return DtwConfirmResult { confirmed: false, cost: f64::MAX, confidence: 0.0 };
    }

    // Normalize perf sequence to start at 0 (same as score_follower.rs)
    let perf_start = perf_notes[0].onset;
    let perf_seq: Vec<(f64, u8)> = perf_notes
        .iter()
        .map(|n| (n.onset - perf_start, n.pitch))
        .collect();

    let cost = subsequence_dtw(&perf_seq, score_notes);
    let confirmed = cost < threshold;
    let confidence = 1.0 / (1.0 + cost);

    DtwConfirmResult { confirmed, cost, confidence }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_perf_note(pitch: u8, onset: f64) -> PerfNote {
        PerfNote { pitch, onset, offset: onset + 0.4, velocity: 80 }
    }

    #[test]
    fn confirms_exact_match() {
        // Perfect match: perf notes == score notes
        let perf = vec![
            make_perf_note(60, 0.0),
            make_perf_note(62, 0.5),
            make_perf_note(64, 1.0),
            make_perf_note(65, 1.5),
        ];
        let score: Vec<(f64, u8)> = vec![(0.0, 60), (0.5, 62), (1.0, 64), (1.5, 65)];

        let result = dtw_confirm(&perf, &score, 0.3);
        assert!(result.confirmed, "Expected confirmation for exact match");
        assert!(result.cost < 0.3, "Expected low DTW cost, got {}", result.cost);
        assert!(result.confidence > 0.5, "Expected confidence > 0.5, got {}", result.confidence);
    }

    #[test]
    fn rejects_wrong_piece() {
        // Perf notes completely mismatch score
        let perf = vec![
            make_perf_note(60, 0.0),
            make_perf_note(62, 0.5),
            make_perf_note(64, 1.0),
        ];
        // Score is a different pitch sequence far away
        let score: Vec<(f64, u8)> = vec![(0.0, 90), (0.5, 91), (1.0, 92)];

        let result = dtw_confirm(&perf, &score, 0.3);
        assert!(!result.confirmed, "Expected rejection for completely wrong piece");
    }

    #[test]
    fn too_few_notes_returns_unconfirmed() {
        let perf = vec![make_perf_note(60, 0.0), make_perf_note(62, 0.5)];
        let score: Vec<(f64, u8)> = vec![(0.0, 60), (0.5, 62)];

        let result = dtw_confirm(&perf, &score, 0.3);
        assert!(!result.confirmed, "Expected unconfirmed for < 3 notes");
        assert_eq!(result.cost, f64::MAX);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn pitch_penalty_exact() {
        assert_eq!(pitch_penalty(60, 60), 0.0);
    }

    #[test]
    fn pitch_penalty_semitone() {
        assert!((pitch_penalty(60, 61) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn pitch_penalty_octave() {
        assert!((pitch_penalty(60, 72) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn pitch_penalty_large_mismatch() {
        assert!((pitch_penalty(60, 70) - PITCH_MISMATCH_PENALTY).abs() < 1e-10);
    }
}
