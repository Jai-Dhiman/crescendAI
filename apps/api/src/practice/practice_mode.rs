use std::collections::{HashSet, VecDeque};

// --- Types ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

pub struct ModeContext {
    pub mode: PracticeMode,
    pub comparative: bool,
    pub entered_at_ms: u64,
    pub chunk_count: usize,
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

// --- ModeDetector implementation ---

impl ModeDetector {
    pub fn new() -> Self {
        ModeDetector {
            mode: PracticeMode::Warming,
            entered_at_ms: 0,
            chunk_count: 0,
            recent_signals: VecDeque::new(),
            last_chunk_at_ms: 0,
            drilling_passage: None,
        }
    }

    pub fn update(&mut self, signal: &ChunkSignal) -> Vec<ModeTransition> {
        let mut transitions: Vec<ModeTransition> = Vec::new();

        // Two-step silence detection: if gap > 60s and not already Winding,
        // transition to Winding first, then evaluate resume from Winding.
        let gap_ms = if self.last_chunk_at_ms == 0 {
            0
        } else {
            signal.timestamp_ms.saturating_sub(self.last_chunk_at_ms)
        };

        if gap_ms > SILENCE_GAP_MS && self.mode != PracticeMode::Winding {
            self.set_mode(PracticeMode::Winding, signal.timestamp_ms);
            transitions.push(ModeTransition {
                mode: PracticeMode::Winding,
                chunk_index: signal.chunk_index,
            });
            // Now evaluate from Winding using the current signal directly
            let resume_mode = self.eval_from_winding(signal);
            if let Some(next) = resume_mode {
                self.set_mode(next, signal.timestamp_ms);
                transitions.push(ModeTransition {
                    mode: next,
                    chunk_index: signal.chunk_index,
                });
            }
            self.last_chunk_at_ms = signal.timestamp_ms;
            self.chunk_count += 1;
            self.recent_signals.push_back(signal.clone());
            if self.recent_signals.len() > RECENT_WINDOW {
                self.recent_signals.pop_front();
            }
            return transitions;
        }

        // Push signal into sliding window before evaluating transitions
        self.recent_signals.push_back(signal.clone());
        if self.recent_signals.len() > RECENT_WINDOW {
            self.recent_signals.pop_front();
        }
        self.last_chunk_at_ms = signal.timestamp_ms;
        self.chunk_count += 1;

        // Evaluate transitions
        if let Some(next) = self.evaluate_transitions(signal) {
            // Handle drilling entry vs stay
            if next == PracticeMode::Drilling {
                if self.mode == PracticeMode::Drilling {
                    // Already drilling: increment repetition count
                    if let Some(ref mut dp) = self.drilling_passage {
                        dp.repetition_count += 1;
                    }
                } else {
                    // Entering drilling: initialize passage
                    self.drilling_passage = Some(DrillingPassage {
                        bar_range: signal.bar_range,
                        repetition_count: 1,
                        first_scores: signal.scores,
                    });
                    self.set_mode(PracticeMode::Drilling, signal.timestamp_ms);
                    transitions.push(ModeTransition {
                        mode: PracticeMode::Drilling,
                        chunk_index: signal.chunk_index,
                    });
                }
            } else {
                self.set_mode(next, signal.timestamp_ms);
                transitions.push(ModeTransition {
                    mode: next,
                    chunk_index: signal.chunk_index,
                });
            }
        }

        transitions
    }

    pub fn observation_policy(&self) -> ObservationPolicy {
        match self.mode {
            PracticeMode::Warming => ObservationPolicy {
                suppress: false,
                min_interval_ms: 30_000,  // Allow observations after 30s, sparse during warm-up
                comparative: false,
            },
            PracticeMode::Drilling => ObservationPolicy {
                suppress: false,
                min_interval_ms: 90_000,
                comparative: true,
            },
            PracticeMode::Running => ObservationPolicy {
                suppress: false,
                min_interval_ms: 150_000,
                comparative: false,
            },
            PracticeMode::Regular => ObservationPolicy {
                suppress: false,
                min_interval_ms: 180_000,
                comparative: false,
            },
            PracticeMode::Winding => ObservationPolicy {
                suppress: true,
                min_interval_ms: 0,
                comparative: false,
            },
        }
    }

    /// Return mode metadata for synthesis context.
    pub fn mode_context(&self) -> ModeContext {
        ModeContext {
            mode: self.mode,
            comparative: matches!(self.mode, PracticeMode::Drilling),
            entered_at_ms: self.entered_at_ms,
            chunk_count: self.chunk_count,
        }
    }

    /// If drilling mode was active, take the DrillingPassage and convert it to a
    /// DrillingRecord with final_scores from the current chunk.
    pub fn take_completed_drilling(&mut self, current_scores: [f64; 6], current_chunk: usize) -> Option<crate::practice::accumulator::DrillingRecord> {
        self.drilling_passage.take().map(|dp| {
            crate::practice::accumulator::DrillingRecord {
                bar_range: dp.bar_range,
                repetition_count: dp.repetition_count,
                first_scores: dp.first_scores,
                final_scores: current_scores,
                started_at_chunk: current_chunk.saturating_sub(dp.repetition_count * 15),
                ended_at_chunk: current_chunk,
            }
        })
    }

    fn evaluate_transitions(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        match self.mode {
            PracticeMode::Warming => self.eval_from_warming(signal),
            PracticeMode::Running => self.eval_from_running(signal),
            PracticeMode::Drilling => self.eval_from_drilling(signal),
            PracticeMode::Regular => self.eval_from_regular(signal),
            PracticeMode::Winding => self.eval_from_winding(signal),
        }
    }

    fn eval_from_warming(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        // Transition to Running: piece match + bar progress
        if signal.has_piece_match && bars_progressing(&self.recent_signals) {
            return Some(PracticeMode::Running);
        }

        // Transition to Drilling: repetition detected across at least 3 signals
        // with a substantive passage (>= 3 bigrams per chunk) to avoid false triggers
        // on trivial single-bigram patterns.
        if self.recent_signals.len() >= 3 && self.passage_has_substance() && detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }

        // Fallback to Regular after WARMING_CHUNK_LIMIT ambiguous chunks
        if self.chunk_count >= WARMING_CHUNK_LIMIT {
            return Some(PracticeMode::Regular);
        }

        None
    }

    /// Returns true if recent signals contain substantive passages (>= 3 bigrams)
    /// to avoid false drilling triggers on short/trivial patterns.
    fn passage_has_substance(&self) -> bool {
        const MIN_BIGRAMS: usize = 3;
        self.recent_signals
            .iter()
            .rev()
            .take(3)
            .all(|s| s.pitch_bigrams.len() >= MIN_BIGRAMS)
    }

    fn eval_from_running(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(RUNNING_DWELL_MS, signal.timestamp_ms) {
            return None;
        }

        // Transition to Drilling on repetition
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }

        None
    }

    fn eval_from_drilling(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(DRILLING_DWELL_MS, signal.timestamp_ms) {
            // Still increment repetition count during dwell
            if detect_repetition(&self.recent_signals) {
                return Some(PracticeMode::Drilling); // signals "stay + increment"
            }
            return None;
        }

        // Transition to Running when new material appears (no recent repetition + piece match + bar progress)
        if no_recent_repetition(&self.recent_signals)
            && signal.has_piece_match
            && bars_progressing(&self.recent_signals)
        {
            return Some(PracticeMode::Running);
        }

        // Stay in drilling if still repeating
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }

        None
    }

    fn eval_from_regular(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(REGULAR_DWELL_MS, signal.timestamp_ms) {
            return None;
        }

        // Piece match + forward progress -> Running
        if signal.has_piece_match && bars_progressing(&self.recent_signals) {
            return Some(PracticeMode::Running);
        }

        // Repetition -> Drilling
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }

        None
    }

    fn eval_from_winding(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        // Resume: piece match -> Running, else -> Regular
        if signal.has_piece_match {
            Some(PracticeMode::Running)
        } else {
            Some(PracticeMode::Regular)
        }
    }

    fn set_mode(&mut self, mode: PracticeMode, timestamp_ms: u64) {
        self.mode = mode;
        self.entered_at_ms = timestamp_ms;
    }

    fn dwell_elapsed(&self, dwell_ms: u64, current_ts_ms: u64) -> bool {
        current_ts_ms.saturating_sub(self.entered_at_ms) >= dwell_ms
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

    // --- ModeDetector transition tests ---

    fn make_detector() -> ModeDetector {
        ModeDetector::new()
    }

    fn signal_at(
        index: usize,
        ts: u64,
        bigrams: Vec<(u8, u8)>,
        bar_range: Option<(u32, u32)>,
        piece: bool,
    ) -> ChunkSignal {
        make_signal(index, ts, bigrams, bar_range, piece)
    }

    #[test]
    fn starts_in_warming() {
        let det = make_detector();
        assert_eq!(det.mode, PracticeMode::Warming);
    }

    #[test]
    fn warming_to_running_on_piece_match_with_progress() {
        let mut det = make_detector();
        let s1 = signal_at(0, 1000, vec![], Some((1, 4)), true);
        det.update(&s1);
        assert_eq!(det.mode, PracticeMode::Warming);

        let s2 = signal_at(1, 16000, vec![], Some((5, 8)), true);
        let transitions = det.update(&s2);
        assert_eq!(det.mode, PracticeMode::Running);
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].mode, PracticeMode::Running);
    }

    #[test]
    fn warming_to_drilling_on_repetition() {
        let mut det = make_detector();
        let bg = vec![(60, 62), (62, 64), (64, 65), (65, 67)];
        det.update(&signal_at(0, 1000, bg.clone(), None, false));
        det.update(&signal_at(1, 16000, bg.clone(), None, false));
        let t = det.update(&signal_at(2, 31000, bg.clone(), None, false));
        assert_eq!(det.mode, PracticeMode::Drilling);
        assert!(t.iter().any(|t| t.mode == PracticeMode::Drilling));
    }

    #[test]
    fn warming_to_regular_after_4_ambiguous_chunks() {
        let mut det = make_detector();
        for i in 0..4 {
            det.update(&signal_at(i, (i as u64) * 15000 + 1000, vec![(60, 62)], None, false));
        }
        assert_eq!(det.mode, PracticeMode::Regular);
    }

    #[test]
    fn running_to_drilling_on_repetition() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        assert_eq!(det.mode, PracticeMode::Running);

        let bg = vec![(60, 62), (62, 64), (64, 65)];
        det.update(&signal_at(2, 50000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(3, 65000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(4, 80000, bg.clone(), Some((10, 14)), true));
        assert_eq!(det.mode, PracticeMode::Drilling);
    }

    #[test]
    fn running_to_winding_on_silence() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        assert_eq!(det.mode, PracticeMode::Running);

        let t = det.update(&signal_at(2, 81000, vec![], Some((9, 12)), true));
        assert!(t.iter().any(|tr| tr.mode == PracticeMode::Winding));
        assert_eq!(det.mode, PracticeMode::Running); // resumed via two-step
    }

    #[test]
    fn drilling_to_running_on_new_material() {
        let mut det = make_detector();
        let bg = vec![(60, 62), (62, 64), (64, 65)];
        det.update(&signal_at(0, 1000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(1, 16000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(2, 31000, bg.clone(), Some((10, 14)), true));
        assert_eq!(det.mode, PracticeMode::Drilling);

        let new_bg = vec![(72, 74), (74, 76), (76, 77)];
        det.update(&signal_at(3, 65000, new_bg.clone(), Some((15, 18)), true));
        det.update(&signal_at(4, 80000, vec![(77, 79), (79, 81)], Some((19, 22)), true));
        assert_eq!(det.mode, PracticeMode::Running);
    }

    #[test]
    fn winding_to_regular_on_resume_without_piece() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![(60, 62)], None, false));
        det.update(&signal_at(1, 16000, vec![(60, 62)], None, false));
        det.update(&signal_at(2, 31000, vec![(60, 62)], None, false));
        det.update(&signal_at(3, 46000, vec![(60, 62)], None, false));
        let t = det.update(&signal_at(4, 120000, vec![(70, 72)], None, false));
        assert!(t.iter().any(|tr| tr.mode == PracticeMode::Winding));
        assert_eq!(det.mode, PracticeMode::Regular);
    }

    #[test]
    fn min_dwell_prevents_early_exit_from_drilling() {
        let mut det = make_detector();
        let bg = vec![(60, 62), (62, 64), (64, 65)];
        det.update(&signal_at(0, 1000, bg.clone(), None, false));
        det.update(&signal_at(1, 16000, bg.clone(), None, false));
        det.update(&signal_at(2, 31000, bg.clone(), None, false));
        assert_eq!(det.mode, PracticeMode::Drilling);

        let new_bg = vec![(72, 74), (74, 76)];
        det.update(&signal_at(3, 32000, new_bg.clone(), Some((20, 25)), true));
        assert_eq!(det.mode, PracticeMode::Drilling);
    }

    #[test]
    fn two_step_silence_transition_returns_two_events() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        assert_eq!(det.mode, PracticeMode::Running);

        let t = det.update(&signal_at(2, 81000, vec![], Some((9, 12)), true));
        assert!(t.len() >= 2);
        assert_eq!(t[0].mode, PracticeMode::Winding);
    }

    #[test]
    fn empty_midi_no_crash() {
        let mut det = make_detector();
        let t = det.update(&signal_at(0, 1000, vec![], None, false));
        assert_eq!(det.mode, PracticeMode::Warming);
        assert!(t.is_empty());
    }

    // --- Pacing tests ---

    #[test]
    fn warming_allows_sparse_observations() {
        let det = make_detector();
        let policy = det.observation_policy();
        assert!(!policy.suppress);
        assert_eq!(policy.min_interval_ms, 30_000);
    }

    #[test]
    fn drilling_comparative_90s() {
        let mut det = make_detector();
        let bg = vec![(60, 62), (62, 64), (64, 65)];
        det.update(&signal_at(0, 1000, bg.clone(), None, false));
        det.update(&signal_at(1, 16000, bg.clone(), None, false));
        det.update(&signal_at(2, 31000, bg.clone(), None, false));
        let policy = det.observation_policy();
        assert!(!policy.suppress);
        assert_eq!(policy.min_interval_ms, 90_000);
        assert!(policy.comparative);
    }

    #[test]
    fn running_150s() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        let policy = det.observation_policy();
        assert_eq!(policy.min_interval_ms, 150_000);
        assert!(!policy.comparative);
    }

    #[test]
    fn regular_180s() {
        let mut det = make_detector();
        for i in 0..4 {
            det.update(&signal_at(i, (i as u64) * 15000 + 1000, vec![(60, 62)], None, false));
        }
        let policy = det.observation_policy();
        assert_eq!(policy.min_interval_ms, 180_000);
    }
}
