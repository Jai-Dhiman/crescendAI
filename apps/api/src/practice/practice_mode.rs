use std::collections::{HashSet, VecDeque};

// --- Types ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PracticeMode {
    Warming,
    Drilling,
    Running,
    Winding,
    Regular,
}

#[derive(Clone)]
pub struct ChunkSignal {
    pub chunk_index: usize,
    pub timestamp_ms: u64,
    pub pitch_bigrams: Vec<(u8, u8)>,
    pub bar_range: Option<(u32, u32)>,
    pub has_piece_match: bool,
    pub scores: [f64; 6],
}

pub struct ObservationPolicy {
    pub suppress: bool,
    pub min_interval_ms: u64,
    pub comparative: bool,
}

pub struct DrillingPassage {
    pub bar_range: Option<(u32, u32)>,
    pub repetition_count: usize,
    pub first_scores: [f64; 6],
}

pub struct ModeTransition {
    pub mode: PracticeMode,
    pub chunk_index: usize,
}

pub struct ModeDetector {
    pub mode: PracticeMode,
    entered_at_ms: u64,
    chunk_count: usize,
    recent_signals: VecDeque<ChunkSignal>,
    last_chunk_at_ms: u64,
    pub drilling_passage: Option<DrillingPassage>,
}

// --- Constants ---

pub(crate) const RECENT_WINDOW: usize = 4;
pub(crate) const DICE_THRESHOLD: f64 = 0.6;
pub(crate) const BAR_OVERLAP_THRESHOLD: f64 = 0.5;
pub(crate) const SILENCE_GAP_MS: u64 = 60_000;
pub(crate) const WARMING_CHUNK_LIMIT: usize = 4;
pub(crate) const DRILLING_DWELL_MS: u64 = 30_000;
pub(crate) const RUNNING_DWELL_MS: u64 = 30_000;
pub(crate) const REGULAR_DWELL_MS: u64 = 15_000;

// --- Repetition detection helpers ---

/// Extract consecutive pitch pairs from performance notes.
pub fn pitch_bigrams_from_notes(pitches: &[u8]) -> Vec<(u8, u8)> {
    pitches.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Dice coefficient on pitch bigram sets.
pub(crate) fn bigram_dice(a: &[(u8, u8)], b: &[(u8, u8)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<(u8, u8)> = a.iter().copied().collect();
    let set_b: HashSet<(u8, u8)> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    2.0 * intersection as f64 / (set_a.len() + set_b.len()) as f64
}

/// Bar range overlap fraction (min of overlap/span_a, overlap/span_b).
pub(crate) fn bar_overlap_fraction(a: (u32, u32), b: (u32, u32)) -> f64 {
    let overlap_start = a.0.max(b.0);
    let overlap_end = a.1.min(b.1);
    if overlap_start > overlap_end {
        return 0.0;
    }
    let overlap = (overlap_end - overlap_start + 1) as f64;
    let span_a = (a.1 - a.0 + 1) as f64;
    let span_b = (b.1 - b.0 + 1) as f64;
    (overlap / span_a).min(overlap / span_b)
}

/// Check if recent signals indicate repetition (drilling).
/// All consecutive pairs in the last 3 chunks must show repetition.
/// With 3 chunks [A, B, C], both (A,B) and (B,C) must match.
/// With 2 chunks [A, B], the single pair must match.
pub(crate) fn detect_repetition(signals: &VecDeque<ChunkSignal>) -> bool {
    if signals.len() < 2 {
        return false;
    }
    let recent: Vec<&ChunkSignal> = signals.iter().rev().take(3).collect();
    let mut repeat_count = 0;
    let total_pairs = recent.len() - 1;

    for pair in recent.windows(2) {
        let a = pair[0];
        let b = pair[1];

        // Layer 1: bar range overlap (preferred when available)
        if let (Some(br_a), Some(br_b)) = (a.bar_range, b.bar_range) {
            if bar_overlap_fraction(br_a, br_b) >= BAR_OVERLAP_THRESHOLD {
                repeat_count += 1;
                continue;
            }
        }

        // Layer 2: pitch bigram Dice (fallback)
        if bigram_dice(&a.pitch_bigrams, &b.pitch_bigrams) >= DICE_THRESHOLD {
            repeat_count += 1;
        }
    }

    // All pairs must match: 2 of 2 for 3 chunks, 1 of 1 for 2 chunks
    repeat_count >= total_pairs
}

/// Check if the two most recent chunks show NO repetition (for exiting drilling).
pub(crate) fn no_recent_repetition(signals: &VecDeque<ChunkSignal>) -> bool {
    if signals.len() < 2 {
        return true;
    }
    let latest = &signals[signals.len() - 1];
    let prev = &signals[signals.len() - 2];

    if let (Some(br_a), Some(br_b)) = (latest.bar_range, prev.bar_range) {
        if bar_overlap_fraction(br_a, br_b) >= BAR_OVERLAP_THRESHOLD {
            return false;
        }
    }

    if bigram_dice(&latest.pitch_bigrams, &prev.pitch_bigrams) >= DICE_THRESHOLD {
        return false;
    }

    true
}

/// Check if bars are progressing forward.
pub(crate) fn bars_progressing(signals: &VecDeque<ChunkSignal>) -> bool {
    if signals.len() < 2 {
        return false;
    }
    let latest = &signals[signals.len() - 1];
    let prev = &signals[signals.len() - 2];
    match (latest.bar_range, prev.bar_range) {
        (Some(a), Some(b)) => a.0 > b.0 || a.1 > b.1,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(
        index: usize,
        ts: u64,
        bigrams: Vec<(u8, u8)>,
        bar_range: Option<(u32, u32)>,
        piece: bool,
    ) -> ChunkSignal {
        ChunkSignal {
            chunk_index: index,
            timestamp_ms: ts,
            pitch_bigrams: bigrams,
            bar_range,
            has_piece_match: piece,
            scores: [0.5; 6],
        }
    }

    #[test]
    fn pitch_bigrams_extraction() {
        let pitches = vec![60, 62, 64, 65, 67];
        let bigrams = pitch_bigrams_from_notes(&pitches);
        assert_eq!(bigrams, vec![(60, 62), (62, 64), (64, 65), (65, 67)]);
    }

    #[test]
    fn pitch_bigrams_empty() {
        assert_eq!(pitch_bigrams_from_notes(&[]), Vec::<(u8, u8)>::new());
        assert_eq!(pitch_bigrams_from_notes(&[60]), Vec::<(u8, u8)>::new());
    }

    #[test]
    fn bigram_dice_identical() {
        let a = vec![(60, 62), (62, 64), (64, 65)];
        assert!((bigram_dice(&a, &a) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bigram_dice_disjoint() {
        let a = vec![(60, 62), (62, 64)];
        let b = vec![(70, 72), (72, 74)];
        assert!((bigram_dice(&a, &b)).abs() < f64::EPSILON);
    }

    #[test]
    fn bigram_dice_partial_overlap() {
        let a = vec![(60, 62), (62, 64), (64, 65), (65, 67)];
        let b = vec![(60, 62), (62, 64), (64, 66), (66, 68)];
        let dice = bigram_dice(&a, &b);
        assert!((dice - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn bigram_dice_empty() {
        assert!((bigram_dice(&[], &[(60, 62)])).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_overlap_full() {
        assert!((bar_overlap_fraction((5, 10), (5, 10)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_overlap_none() {
        assert!((bar_overlap_fraction((1, 5), (10, 15))).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_overlap_partial() {
        let frac = bar_overlap_fraction((5, 10), (8, 14));
        assert!(frac > 0.42 && frac < 0.44);
    }

    #[test]
    fn detect_repetition_with_bar_overlap() {
        let mut signals = VecDeque::new();
        signals.push_back(make_signal(0, 0, vec![], Some((5, 10)), true));
        signals.push_back(make_signal(1, 15000, vec![], Some((5, 10)), true));
        signals.push_back(make_signal(2, 30000, vec![], Some((6, 11)), true));
        assert!(detect_repetition(&signals));
    }

    #[test]
    fn detect_repetition_with_bigrams() {
        let bg = vec![(60, 62), (62, 64), (64, 65), (65, 67)];
        let mut signals = VecDeque::new();
        signals.push_back(make_signal(0, 0, bg.clone(), None, false));
        signals.push_back(make_signal(1, 15000, bg.clone(), None, false));
        assert!(detect_repetition(&signals));
    }

    #[test]
    fn no_repetition_different_material() {
        let mut signals = VecDeque::new();
        signals.push_back(make_signal(0, 0, vec![(60, 62)], Some((1, 5)), true));
        signals.push_back(make_signal(1, 15000, vec![(70, 72)], Some((10, 15)), true));
        assert!(!detect_repetition(&signals));
    }

    #[test]
    fn detect_repetition_single_chunk() {
        let mut signals = VecDeque::new();
        signals.push_back(make_signal(0, 0, vec![(60, 62)], None, false));
        assert!(!detect_repetition(&signals));
    }
}
