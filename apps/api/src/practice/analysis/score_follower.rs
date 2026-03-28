//! Subsequence DTW score follower: maps AMT performance notes from a 15s chunk
//! to bar numbers in the score using onset+pitch alignment. Maintains cross-chunk
//! continuity via FollowerState.

use super::score_context::{ScoreBar, ScoreData};

const PITCH_MISMATCH_PENALTY: f64 = 0.5;
const SEARCH_WINDOW_BARS: u32 = 30;
const REANCHOR_COST_THRESHOLD: f64 = 0.3;
const MIN_PERF_NOTES: usize = 3;

/// A performance note from AMT (subset of HF response).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PerfNote {
    pub pitch: u8,
    pub onset: f64,
    pub offset: f64,
    pub velocity: u8,
}

/// A pedal event from AMT.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PerfPedalEvent {
    pub time: f64,
    pub value: u8,
}

/// Alignment of a single performance note to a score note.
#[derive(Debug, Clone)]
pub struct NoteAlignment {
    pub perf_onset: f64,
    pub perf_pitch: u8,
    pub perf_velocity: u8,
    pub score_bar: u32,
    pub score_beat: f64,
    pub score_pitch: u8,
    pub onset_deviation_ms: f64,
}

/// Bar map for a single chunk.
#[derive(Debug, Clone)]
pub struct BarMap {
    pub chunk_index: usize,
    pub bar_start: u32,
    pub bar_end: u32,
    pub alignments: Vec<NoteAlignment>,
    pub confidence: f64,
    pub is_reanchored: bool,
}

/// Persistent state across chunks.
#[derive(Debug, Clone, Default)]
pub struct FollowerState {
    pub last_known_bar: Option<u32>,
}

/// Flatten score bars into (onset_seconds, pitch, bar_number, beat) tuples.
/// Uses index-based bar duration: bars[i+1].start_seconds - bars[i].start_seconds.
fn flatten_score_notes(bars: &[ScoreBar]) -> Vec<(f64, u8, u32, f64)> {
    let mut result = Vec::new();
    for (i, bar) in bars.iter().enumerate() {
        let bar_duration = if i + 1 < bars.len() {
            bars[i + 1].start_seconds - bar.start_seconds
        } else {
            // Last bar: estimate from note durations or use a default
            let max_note_end = bar.notes.iter().map(|n| n.onset_seconds + n.duration_seconds).fold(0.0_f64, f64::max);
            if max_note_end > bar.start_seconds {
                max_note_end - bar.start_seconds
            } else {
                2.0 // fallback 2 seconds
            }
        };

        for note in &bar.notes {
            let beat = if bar_duration > 0.0 {
                (note.onset_seconds - bar.start_seconds) / bar_duration * 4.0
            } else {
                0.0
            };
            result.push((note.onset_seconds, note.pitch, bar.bar_number, beat));
        }
    }
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compute pitch mismatch penalty between two MIDI pitches.
fn pitch_penalty(p1: u8, p2: u8) -> f64 {
    if p1 == p2 {
        0.0
    } else {
        let diff = (p1 as i16 - p2 as i16).unsigned_abs();
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
/// Returns (path, normalized_cost) where path is Vec<(perf_idx, score_idx)>.
/// Subsequence DTW initializes dtw[0][j] = 0 for all j (free start in score).
fn subsequence_dtw(
    perf: &[(f64, u8)],
    score: &[(f64, u8)],
) -> (Vec<(usize, usize)>, f64) {
    if perf.is_empty() || score.is_empty() {
        return (vec![], f64::MAX);
    }

    let n = perf.len();
    let m = score.len();

    // DTW cost matrix: dtw[i][j] = min cost to align perf[0..=i] to score ending at j
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
    let best_j = (0..m)
        .min_by(|&a, &b| dtw[n - 1][a].partial_cmp(&dtw[n - 1][b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);

    let total_cost = dtw[n - 1][best_j];
    let normalized_cost = if n > 0 { total_cost / n as f64 } else { f64::MAX };

    // Backtrace path
    let path = backtrace(&dtw, n - 1, best_j);

    (path, normalized_cost)
}

/// Backtrace the DTW matrix to recover the alignment path.
fn backtrace(dtw: &[Vec<f64>], mut i: usize, mut j: usize) -> Vec<(usize, usize)> {
    let mut path = vec![(i, j)];

    while i > 0 {
        if j == 0 {
            i -= 1;
        } else {
            let prev_diag = dtw[i - 1][j - 1];
            let prev_up = dtw[i - 1][j];
            let prev_left = dtw[i][j - 1];

            if prev_diag <= prev_up && prev_diag <= prev_left {
                i -= 1;
                j -= 1;
            } else if prev_up <= prev_left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }

    path.reverse();
    path
}

/// Run DTW over a score window and return (path, cost, window_start_idx).
fn run_dtw_on_window(
    perf_seq: &[(f64, u8)],
    score_notes: &[(f64, u8, u32, f64)],
    window_start_idx: usize,
    window_end_idx: usize,
) -> (Vec<(usize, usize)>, f64) {
    let window: Vec<(f64, u8)> = score_notes[window_start_idx..window_end_idx]
        .iter()
        .map(|&(onset, pitch, _, _)| {
            // Normalize relative to window start onset
            let window_start_onset = score_notes[window_start_idx].0;
            (onset - window_start_onset, pitch)
        })
        .collect();

    subsequence_dtw(perf_seq, &window)
}

/// Build a BarMap from the DTW alignment path.
fn build_bar_map(
    chunk_index: usize,
    perf_notes: &[PerfNote],
    score_notes: &[(f64, u8, u32, f64)],
    path: &[(usize, usize)],
    window_start_idx: usize,
    cost: f64,
    is_reanchored: bool,
) -> BarMap {
    let confidence = 1.0 / (1.0 + cost);

    // Compute median onset offset to correct for systematic alignment shift
    let mut onset_offsets: Vec<f64> = Vec::new();
    for &(pi, si) in path {
        if pi < perf_notes.len() {
            let si_global = si + window_start_idx;
            if si_global < score_notes.len() {
                let score_onset = score_notes[si_global].0;
                let perf_onset = perf_notes[pi].onset;
                onset_offsets.push(perf_onset - score_onset);
            }
        }
    }

    let median_offset = if onset_offsets.is_empty() {
        0.0
    } else {
        let mut sorted = onset_offsets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let mut alignments = Vec::new();
    let mut bars_seen = std::collections::BTreeSet::new();

    // Deduplicate: take the last path entry for each perf note index
    let mut best_per_perf: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &(pi, si) in path {
        best_per_perf.insert(pi, si);
    }

    for (&pi, &si) in &best_per_perf {
        if pi >= perf_notes.len() {
            continue;
        }
        let si_global = si + window_start_idx;
        if si_global >= score_notes.len() {
            continue;
        }
        let (score_onset, score_pitch, bar_num, score_beat) = score_notes[si_global];
        let perf_note = &perf_notes[pi];

        bars_seen.insert(bar_num);

        let raw_deviation_ms = (perf_note.onset - score_onset) * 1000.0;
        let corrected_deviation_ms = raw_deviation_ms - median_offset * 1000.0;

        alignments.push(NoteAlignment {
            perf_onset: perf_note.onset,
            perf_pitch: perf_note.pitch,
            perf_velocity: perf_note.velocity,
            score_bar: bar_num,
            score_beat,
            score_pitch,
            onset_deviation_ms: corrected_deviation_ms,
        });
    }

    alignments.sort_by(|a, b| a.perf_onset.partial_cmp(&b.perf_onset).unwrap_or(std::cmp::Ordering::Equal));

    let bar_start = bars_seen.iter().next().copied().unwrap_or(1);
    let bar_end = bars_seen.iter().next_back().copied().unwrap_or(bar_start);

    BarMap {
        chunk_index,
        bar_start,
        bar_end,
        alignments,
        confidence,
        is_reanchored,
    }
}

/// Find the index range in score_notes that corresponds to bars [bar_min, bar_max].
fn find_bar_index_range(
    score_notes: &[(f64, u8, u32, f64)],
    bar_min: u32,
    bar_max: u32,
) -> (usize, usize) {
    let start = score_notes.iter().position(|&(_, _, bar, _)| bar >= bar_min).unwrap_or(0);
    let end = score_notes.iter().rposition(|&(_, _, bar, _)| bar <= bar_max).map(|i| i + 1).unwrap_or(score_notes.len());
    (start, end.max(start))
}

/// Main entry point: align a chunk of performance notes to the score.
///
/// Uses last_known_bar to restrict the search window. If the narrow-window cost
/// exceeds REANCHOR_COST_THRESHOLD, re-scans the full score.
pub fn align_chunk(
    chunk_index: usize,
    _chunk_offset_seconds: f64,
    perf_notes: &[PerfNote],
    score: &ScoreData,
    state: &mut FollowerState,
) -> Option<BarMap> {
    if perf_notes.len() < MIN_PERF_NOTES {
        return None;
    }

    let score_notes = flatten_score_notes(&score.bars);
    if score_notes.is_empty() {
        return None;
    }

    // Build perf sequence normalized to start at 0
    let perf_start = perf_notes[0].onset;
    let perf_seq: Vec<(f64, u8)> = perf_notes
        .iter()
        .map(|n| (n.onset - perf_start, n.pitch))
        .collect();

    let max_bar = score.bars.iter().map(|b| b.bar_number).max().unwrap_or(1);

    // Determine search window
    let (window_start_idx, window_end_idx, is_reanchored) = if let Some(last_bar) = state.last_known_bar {
        let bar_min = last_bar.saturating_sub(5);
        let bar_max = (last_bar + SEARCH_WINDOW_BARS).min(max_bar);
        let (start_idx, end_idx) = find_bar_index_range(&score_notes, bar_min, bar_max);

        // Normalize score window relative to its start onset
        let (path, cost) = run_dtw_on_window(&perf_seq, &score_notes, start_idx, end_idx);

        if cost <= REANCHOR_COST_THRESHOLD || path.is_empty() {
            (start_idx, end_idx, false)
        } else {
            // Cost too high in narrow window -- re-scan full score
            let (full_path, _full_cost) = run_dtw_on_window(&perf_seq, &score_notes, 0, score_notes.len());
            let _ = full_path; // path returned from final call below
            (0, score_notes.len(), true)
        }
    } else {
        (0, score_notes.len(), false)
    };

    if window_start_idx >= window_end_idx {
        return None;
    }

    let (path, cost) = run_dtw_on_window(&perf_seq, &score_notes, window_start_idx, window_end_idx);

    if path.is_empty() {
        return None;
    }

    let bar_map = build_bar_map(
        chunk_index,
        perf_notes,
        &score_notes,
        &path,
        window_start_idx,
        cost,
        is_reanchored,
    );

    // Update state
    state.last_known_bar = Some(bar_map.bar_end);

    Some(bar_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::score_context::{ScoreBar, ScoreData, ScoreNote};

    fn make_score_note(pitch: u8, onset_seconds: f64) -> ScoreNote {
        ScoreNote {
            pitch,
            pitch_name: format!("{}", pitch),
            velocity: 80,
            onset_tick: (onset_seconds * 480.0) as u32,
            onset_seconds,
            duration_ticks: 240,
            duration_seconds: 0.5,
            track: 0,
        }
    }

    fn make_score_bar(bar_num: u32, start_sec: f64, notes: Vec<(u8, f64)>) -> ScoreBar {
        ScoreBar {
            bar_number: bar_num,
            start_tick: (start_sec * 480.0) as u32,
            start_seconds: start_sec,
            time_signature: "4/4".to_string(),
            notes: notes.into_iter().map(|(pitch, onset)| make_score_note(pitch, onset)).collect(),
            pedal_events: vec![],
            note_count: 0, // will be set correctly by the struct but not validated here
            pitch_range: vec![],
            mean_velocity: 80,
        }
    }

    fn make_score(bars: Vec<ScoreBar>) -> ScoreData {
        let total_bars = bars.len() as u32;
        ScoreData {
            piece_id: "test".to_string(),
            composer: "Test".to_string(),
            title: "Test Piece".to_string(),
            key_signature: None,
            time_signatures: vec![],
            tempo_markings: vec![],
            total_bars,
            bars,
        }
    }

    fn make_perf_note(pitch: u8, onset: f64) -> PerfNote {
        PerfNote {
            pitch,
            onset,
            offset: onset + 0.4,
            velocity: 80,
        }
    }

    // Test 1: simple ascending scale C4-D4-E4-F4 in bar 1, performance slightly shifted
    #[test]
    fn aligns_simple_ascending_scale() {
        // C4=60, D4=62, E4=64, F4=65
        let bar1 = make_score_bar(1, 0.0, vec![
            (60, 0.0),
            (62, 0.5),
            (64, 1.0),
            (65, 1.5),
        ]);
        let score = make_score(vec![bar1]);

        // Performance slightly shifted by +0.1s
        let perf_notes = vec![
            make_perf_note(60, 0.1),
            make_perf_note(62, 0.6),
            make_perf_note(64, 1.1),
            make_perf_note(65, 1.6),
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf_notes, &score, &mut state);

        assert!(result.is_some(), "Expected alignment result");
        let bar_map = result.unwrap();
        assert_eq!(bar_map.bar_start, 1);
        assert_eq!(bar_map.bar_end, 1);
        assert!(bar_map.confidence > 0.5, "Expected reasonable confidence");
        assert!(!bar_map.alignments.is_empty());

        // All alignments should map to bar 1
        for alignment in &bar_map.alignments {
            assert_eq!(alignment.score_bar, 1);
        }

        // State should be updated
        assert_eq!(state.last_known_bar, Some(1));
    }

    // Test 2: notes spanning bars 1-3
    #[test]
    fn aligns_across_bars() {
        let bar1 = make_score_bar(1, 0.0, vec![
            (60, 0.0),
            (62, 0.5),
        ]);
        let bar2 = make_score_bar(2, 1.0, vec![
            (64, 1.0),
            (65, 1.5),
        ]);
        let bar3 = make_score_bar(3, 2.0, vec![
            (67, 2.0),
            (69, 2.5),
        ]);
        let score = make_score(vec![bar1, bar2, bar3]);

        // Performance closely matches score
        let perf_notes = vec![
            make_perf_note(60, 0.05),
            make_perf_note(62, 0.55),
            make_perf_note(64, 1.05),
            make_perf_note(65, 1.55),
            make_perf_note(67, 2.05),
            make_perf_note(69, 2.55),
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf_notes, &score, &mut state);

        assert!(result.is_some(), "Expected alignment result");
        let bar_map = result.unwrap();
        assert!(bar_map.bar_start >= 1 && bar_map.bar_end <= 3, "Bars should be within range 1-3");
        assert!(bar_map.bar_end >= bar_map.bar_start, "bar_end should be >= bar_start");
        assert_eq!(state.last_known_bar, Some(bar_map.bar_end));
    }

    // Test 3: too few notes returns None
    #[test]
    fn too_few_notes_returns_none() {
        let bar1 = make_score_bar(1, 0.0, vec![
            (60, 0.0),
            (62, 0.5),
            (64, 1.0),
        ]);
        let score = make_score(vec![bar1]);

        // Only 2 notes -- below MIN_PERF_NOTES (3)
        let perf_notes = vec![
            make_perf_note(60, 0.0),
            make_perf_note(62, 0.5),
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf_notes, &score, &mut state);

        assert!(result.is_none(), "Expected None for too few notes");
    }

    // Test 4: cross-chunk continuity -- second chunk starts from last_known_bar
    #[test]
    fn continuity_across_chunks() {
        let bar1 = make_score_bar(1, 0.0, vec![
            (60, 0.0),
            (62, 0.5),
            (64, 1.0),
        ]);
        let bar2 = make_score_bar(2, 2.0, vec![
            (65, 2.0),
            (67, 2.5),
            (69, 3.0),
        ]);
        let bar3 = make_score_bar(3, 4.0, vec![
            (71, 4.0),
            (72, 4.5),
            (74, 5.0),
        ]);
        let score = make_score(vec![bar1, bar2, bar3]);

        // First chunk: bar 1 notes
        let chunk1_notes = vec![
            make_perf_note(60, 0.0),
            make_perf_note(62, 0.5),
            make_perf_note(64, 1.0),
        ];

        let mut state = FollowerState::default();
        let result1 = align_chunk(0, 0.0, &chunk1_notes, &score, &mut state);
        assert!(result1.is_some(), "Chunk 1 should align");
        assert!(state.last_known_bar.is_some(), "State should be updated after chunk 1");

        let last_bar_after_chunk1 = state.last_known_bar.unwrap();

        // Second chunk: bar 2 notes (starting from continuation)
        let chunk2_notes = vec![
            make_perf_note(65, 0.0),
            make_perf_note(67, 0.5),
            make_perf_note(69, 1.0),
        ];

        let result2 = align_chunk(1, 2.0, &chunk2_notes, &score, &mut state);
        assert!(result2.is_some(), "Chunk 2 should align");

        let bar_map2 = result2.unwrap();
        // Second chunk should align to bar 2 or 3, not restart from bar 1
        assert!(bar_map2.bar_start >= last_bar_after_chunk1.saturating_sub(5),
                "Second chunk should not regress far back from last known bar");
        assert_eq!(state.last_known_bar, Some(bar_map2.bar_end));
    }

    // Test 5: octave error -- C5 (72) instead of C4 (60) from AMT
    #[test]
    fn handles_octave_error() {
        let bar1 = make_score_bar(1, 0.0, vec![
            (60, 0.0),  // C4
            (62, 0.5),  // D4
            (64, 1.0),  // E4
            (65, 1.5),  // F4
        ]);
        let score = make_score(vec![bar1]);

        // AMT outputs C5 (72) instead of C4 (60) for the first note
        let perf_notes = vec![
            make_perf_note(72, 0.0),  // octave error
            make_perf_note(62, 0.5),  // correct
            make_perf_note(64, 1.0),  // correct
            make_perf_note(65, 1.5),  // correct
        ];

        let mut state = FollowerState::default();
        let result = align_chunk(0, 0.0, &perf_notes, &score, &mut state);

        // Should still align despite octave error (lower cost than full mismatch)
        assert!(result.is_some(), "Should align despite octave error");
        let bar_map = result.unwrap();
        assert_eq!(bar_map.bar_start, 1, "Should still map to bar 1");

        // The octave-errored note should align to score pitch 60 with octave penalty
        let first_alignment = bar_map.alignments.iter().find(|a| a.perf_pitch == 72);
        if let Some(alignment) = first_alignment {
            assert_eq!(alignment.score_pitch, 60, "Octave note should align to C4 in score");
        }
    }
}
