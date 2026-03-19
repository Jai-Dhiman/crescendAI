# Practice Mode State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a practice mode state machine to the Durable Object that detects what the student is doing (warming up, drilling, running through, winding down) and adapts observation pacing accordingly.

**Architecture:** New `practice_mode.rs` module with pure state machine logic (no IO). `session.rs` builds `ChunkSignal` from inference results, feeds it to `ModeDetector`, uses the returned `ObservationPolicy` for pacing decisions. Client-side throttle simplified to passthrough. Mode changes broadcast over WebSocket.

**Tech Stack:** Rust (Cloudflare Workers WASM), TypeScript (React hooks)

**Spec:** `docs/superpowers/specs/2026-03-19-practice-mode-state-machine-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `apps/api/src/practice/practice_mode.rs` | CREATE | PracticeMode enum, ChunkSignal, ModeDetector, ObservationPolicy, transition logic, repetition detection, unit tests |
| `apps/api/src/practice/mod.rs` | MODIFY | Add `pub mod practice_mode;` |
| `apps/api/src/practice/session.rs` | MODIFY | Add ModeDetector to SessionState, build ChunkSignal, call update(), mode-aware throttle, WS broadcast, drilling context |
| `apps/web/src/lib/practice-api.ts` | MODIFY | Add ModeChangeEvent type to PracticeWsEvent union |
| `apps/web/src/lib/observation-throttle.ts` | MODIFY | Remove window-based throttle, keep queue/drain |
| `apps/web/src/hooks/usePracticeSession.ts` | MODIFY | Handle mode_change WS message, remove tick() throttle, simplify observation delivery |

---

## Task 1: Core Data Types and Repetition Detection

**Files:**
- Create: `apps/api/src/practice/practice_mode.rs`
- Modify: `apps/api/src/practice/mod.rs:1-9`

This task creates the module with all types and the repetition detection functions (the lowest-level building blocks). No state machine yet.

- [ ] **Step 1: Create practice_mode.rs with types and pitch_bigrams_from_notes()**

```rust
// apps/api/src/practice/practice_mode.rs

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

// --- Repetition detection helpers ---

const RECENT_WINDOW: usize = 4;
const DICE_THRESHOLD: f64 = 0.6;
const BAR_OVERLAP_THRESHOLD: f64 = 0.5;
const SILENCE_GAP_MS: u64 = 60_000;
const WARMING_CHUNK_LIMIT: usize = 4;
const DRILLING_DWELL_MS: u64 = 30_000;
const RUNNING_DWELL_MS: u64 = 30_000;
const REGULAR_DWELL_MS: u64 = 15_000;

/// Extract consecutive pitch pairs from performance notes.
pub fn pitch_bigrams_from_notes(pitches: &[u8]) -> Vec<(u8, u8)> {
    pitches.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Dice coefficient on pitch bigram sets.
fn bigram_dice(a: &[(u8, u8)], b: &[(u8, u8)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<(u8, u8)> = a.iter().copied().collect();
    let set_b: HashSet<(u8, u8)> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    2.0 * intersection as f64 / (set_a.len() + set_b.len()) as f64
}

/// Bar range overlap fraction (min of overlap/span_a, overlap/span_b).
fn bar_overlap_fraction(a: (u32, u32), b: (u32, u32)) -> f64 {
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
/// Returns true if at least 2 consecutive pairs in the last 3 chunks show repetition.
/// With 3 chunks [A, B, C], both (A,B) and (B,C) must match.
/// With 2 chunks [A, B], the single pair must match (repeat_count >= 1 from 1 pair).
fn detect_repetition(signals: &VecDeque<ChunkSignal>) -> bool {
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
fn no_recent_repetition(signals: &VecDeque<ChunkSignal>) -> bool {
    if signals.len() < 2 {
        return true;
    }
    let latest = &signals[signals.len() - 1];
    let prev = &signals[signals.len() - 2];

    // Check bar overlap
    if let (Some(br_a), Some(br_b)) = (latest.bar_range, prev.bar_range) {
        if bar_overlap_fraction(br_a, br_b) >= BAR_OVERLAP_THRESHOLD {
            return false;
        }
    }

    // Check pitch bigram Dice
    if bigram_dice(&latest.pitch_bigrams, &prev.pitch_bigrams) >= DICE_THRESHOLD {
        return false;
    }

    true
}

/// Check if bars are progressing forward.
fn bars_progressing(signals: &VecDeque<ChunkSignal>) -> bool {
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
```

- [ ] **Step 2: Write unit tests for repetition detection helpers**

Add at the bottom of `practice_mode.rs`:

```rust
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
        // intersection: {(60,62), (62,64)} = 2, |a|=4, |b|=4
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
        // (5,10) and (8,14): overlap = 8-10 = 3 bars, span_a = 6, span_b = 7
        // fraction = min(3/6, 3/7) = 3/7 = 0.428...
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
```

- [ ] **Step 3: Register module in mod.rs**

Add `pub mod practice_mode;` to `apps/api/src/practice/mod.rs`.

- [ ] **Step 4: Run tests to verify**

Run: `cd apps/api && cargo test practice_mode -- --nocapture`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/practice_mode.rs apps/api/src/practice/mod.rs
git commit -m "feat: add practice mode types and repetition detection helpers"
```

---

## Task 2: ModeDetector State Machine

**Files:**
- Modify: `apps/api/src/practice/practice_mode.rs`

Implement the `ModeDetector` with `update()` and `observation_policy()`. This is the core state machine.

- [ ] **Step 1: Write failing tests for state transitions**

Add to the `#[cfg(test)]` module in `practice_mode.rs`:

```rust
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
        assert_eq!(det.mode, PracticeMode::Warming); // need 2 chunks for progress

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
        // Get to Running first
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        assert_eq!(det.mode, PracticeMode::Running);

        // Wait out dwell time then start repeating
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

        // 65s gap triggers two-step: Winding then resume
        let t = det.update(&signal_at(2, 81000, vec![], Some((9, 12)), true));
        // Should have passed through Winding and resumed to Running (piece match + progress)
        assert!(t.iter().any(|tr| tr.mode == PracticeMode::Winding));
        assert_eq!(det.mode, PracticeMode::Running); // resumed
    }

    #[test]
    fn drilling_to_running_on_new_material() {
        let mut det = make_detector();
        let bg = vec![(60, 62), (62, 64), (64, 65)];
        // Get to Drilling
        det.update(&signal_at(0, 1000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(1, 16000, bg.clone(), Some((10, 14)), true));
        det.update(&signal_at(2, 31000, bg.clone(), Some((10, 14)), true));
        assert_eq!(det.mode, PracticeMode::Drilling);

        // Wait out dwell, then play new material forward
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
        // Should be in Regular or Drilling at this point; force to Winding via gap
        let t = det.update(&signal_at(4, 120000, vec![(70, 72)], None, false));
        assert!(t.iter().any(|tr| tr.mode == PracticeMode::Winding));
        // Should resume to Regular (no piece match)
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

        // Immediately send different material -- should stay Drilling (dwell not elapsed)
        let new_bg = vec![(72, 74), (74, 76)];
        det.update(&signal_at(3, 32000, new_bg.clone(), Some((20, 25)), true));
        assert_eq!(det.mode, PracticeMode::Drilling); // dwell = 30s not reached
    }

    #[test]
    fn two_step_silence_transition_returns_two_events() {
        let mut det = make_detector();
        det.update(&signal_at(0, 1000, vec![], Some((1, 4)), true));
        det.update(&signal_at(1, 16000, vec![], Some((5, 8)), true));
        assert_eq!(det.mode, PracticeMode::Running);

        // 65s gap
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
    fn warming_suppresses() {
        let det = make_detector();
        let policy = det.observation_policy();
        assert!(policy.suppress);
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/api && cargo test practice_mode -- --nocapture`
Expected: FAIL -- `ModeDetector::new()`, `update()`, `observation_policy()` don't exist yet.

- [ ] **Step 3: Implement ModeDetector**

Add to `practice_mode.rs`, above the `#[cfg(test)]` module:

```rust
impl ModeDetector {
    pub fn new() -> Self {
        Self {
            mode: PracticeMode::Warming,
            entered_at_ms: 0,
            chunk_count: 0,
            recent_signals: VecDeque::with_capacity(RECENT_WINDOW + 1),
            last_chunk_at_ms: 0,
            drilling_passage: None,
        }
    }

    /// Feed a new chunk signal. Returns transitions that occurred (0, 1, or 2 for silence gap).
    pub fn update(&mut self, signal: &ChunkSignal) -> Vec<ModeTransition> {
        let mut transitions = Vec::new();
        self.chunk_count += 1;

        // Two-step silence gap detection: if gap > 60s, enter Winding first
        if self.last_chunk_at_ms > 0
            && signal.timestamp_ms.saturating_sub(self.last_chunk_at_ms) > SILENCE_GAP_MS
            && self.mode != PracticeMode::Winding
        {
            self.set_mode(PracticeMode::Winding, signal.timestamp_ms);
            self.drilling_passage = None;
            transitions.push(ModeTransition {
                mode: PracticeMode::Winding,
                chunk_index: signal.chunk_index,
            });
        }

        // Push signal into sliding window
        self.recent_signals.push_back(signal.clone());
        if self.recent_signals.len() > RECENT_WINDOW {
            self.recent_signals.pop_front();
        }

        // Evaluate transitions from current mode
        if let Some(new_mode) = self.evaluate_transitions(signal) {
            if new_mode == PracticeMode::Drilling && self.mode != PracticeMode::Drilling {
                // Initialize drilling passage
                self.drilling_passage = Some(DrillingPassage {
                    bar_range: signal.bar_range,
                    repetition_count: 1,
                    first_scores: signal.scores,
                });
            } else if new_mode != PracticeMode::Drilling {
                self.drilling_passage = None;
            }

            if new_mode == PracticeMode::Drilling && self.mode == PracticeMode::Drilling {
                // Staying in drilling -- increment repetition count
                if let Some(ref mut dp) = self.drilling_passage {
                    dp.repetition_count += 1;
                    if signal.bar_range.is_some() {
                        dp.bar_range = signal.bar_range;
                    }
                }
            } else {
                self.set_mode(new_mode, signal.timestamp_ms);
                transitions.push(ModeTransition {
                    mode: new_mode,
                    chunk_index: signal.chunk_index,
                });
            }
        }

        self.last_chunk_at_ms = signal.timestamp_ms;
        transitions
    }

    pub fn observation_policy(&self) -> ObservationPolicy {
        match self.mode {
            PracticeMode::Warming => ObservationPolicy {
                suppress: true,
                min_interval_ms: 0,
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
            PracticeMode::Winding => ObservationPolicy {
                suppress: true,
                min_interval_ms: 0,
                comparative: false,
            },
            PracticeMode::Regular => ObservationPolicy {
                suppress: false,
                min_interval_ms: 180_000,
                comparative: false,
            },
        }
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
        // Piece match + forward progress -> Running
        if signal.has_piece_match && bars_progressing(&self.recent_signals) {
            return Some(PracticeMode::Running);
        }
        // Repetition -> Drilling
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }
        // 4+ chunks without clear signal -> Regular
        if self.chunk_count >= WARMING_CHUNK_LIMIT {
            return Some(PracticeMode::Regular);
        }
        None
    }

    fn eval_from_running(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(signal.timestamp_ms, RUNNING_DWELL_MS) {
            return None;
        }
        // Repetition -> Drilling
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }
        None // stays Running
    }

    fn eval_from_drilling(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(signal.timestamp_ms, DRILLING_DWELL_MS) {
            // Stay drilling, but still increment repetition
            if detect_repetition(&self.recent_signals) {
                return Some(PracticeMode::Drilling); // signals "stay + increment"
            }
            return None;
        }
        // No repetition + forward progress -> Running
        if no_recent_repetition(&self.recent_signals) {
            if signal.has_piece_match && bars_progressing(&self.recent_signals) {
                return Some(PracticeMode::Running);
            }
        }
        // Still repeating -> stay Drilling (increment)
        if detect_repetition(&self.recent_signals) {
            return Some(PracticeMode::Drilling);
        }
        None
    }

    fn eval_from_regular(&self, signal: &ChunkSignal) -> Option<PracticeMode> {
        if !self.dwell_elapsed(signal.timestamp_ms, REGULAR_DWELL_MS) {
            return None;
        }
        // Piece match + progress -> Running
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
        // A new chunk arriving means the student resumed
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

    fn dwell_elapsed(&self, now_ms: u64, min_dwell_ms: u64) -> bool {
        now_ms.saturating_sub(self.entered_at_ms) >= min_dwell_ms
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/api && cargo test practice_mode -- --nocapture`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/practice_mode.rs
git commit -m "feat: implement ModeDetector state machine with transitions and pacing"
```

---

## Task 3: Integrate ModeDetector into Session DO

**Files:**
- Modify: `apps/api/src/practice/session.rs` (SessionState, process_inference_result, throttle, generate_observation)

This task wires ModeDetector into the DO pipeline. It combines the bar_range extraction and all session.rs changes into a single coherent edit pass.

- [ ] **Step 1: Add imports and ModeDetector to SessionState**

In `session.rs`, add the import at top (after existing `use` statements):

```rust
use crate::practice::practice_mode::{
    ChunkSignal, ModeDetector, ModeTransition, ObservationPolicy, PracticeMode,
    pitch_bigrams_from_notes,
};
```

Add field to `SessionState` struct (after `is_eval_session`):

```rust
    mode_detector: ModeDetector,
```

And in `Default` impl (after `is_eval_session: false,`):

```rust
    mode_detector: ModeDetector::new(),
```

- [ ] **Step 2: Refactor score following block to extract bar_range from BarMap**

The current block at `process_inference_result` (the `// 9. Run score following + analysis` block starting around line 414) returns `Option<analysis::ChunkAnalysis>`. Refactor it to also return the raw bar range from the `BarMap`, since `ChunkAnalysis.bar_range` is a formatted `String` but we need `(u32, u32)`.

Replace the entire block (lines ~414-460) with:

```rust
        // 9. Run score following + analysis (also extract raw bar range for mode detector)
        let (chunk_analysis, chunk_bar_range): (Option<analysis::ChunkAnalysis>, Option<(u32, u32)>) = {
            let (score_data_clone, follower_state_clone) = {
                let s = self.inner.borrow();
                (
                    s.score_context.as_ref().map(|ctx| ctx.score.clone()),
                    s.follower_state.clone(),
                )
            };

            if !perf_notes.is_empty() {
                if let Some(score_data) = score_data_clone {
                    let mut fs = follower_state_clone;
                    let bar_map = crate::practice::score_follower::align_chunk(
                        index,
                        0.0,
                        &perf_notes,
                        &score_data,
                        &mut fs,
                    );
                    self.inner.borrow_mut().follower_state = fs;

                    if let Some(ref bm) = bar_map {
                        // Extract raw (u32, u32) from BarMap before analysis consumes it
                        let bar_range = (bm.bar_start, bm.bar_end);
                        let score_ctx = self.inner.borrow().score_context.clone().unwrap();
                        let analysis = analysis::analyze_tier1(
                            bm,
                            &perf_notes,
                            &perf_pedal,
                            &scores_array,
                            &score_ctx,
                        );
                        (Some(analysis), Some(bar_range))
                    } else {
                        (Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array)), None)
                    }
                } else {
                    (Some(analysis::analyze_tier2(&perf_notes, &perf_pedal, &scores_array)), None)
                }
            } else {
                (None, None)
            }
        };
```

- [ ] **Step 3: Build ChunkSignal and call update() after analysis**

Add after the score following block, before the STOP classifier (before `// 10. Run STOP classifier`):

```rust
        // 9b. Build ChunkSignal and update practice mode
        let perf_pitches: Vec<u8> = perf_notes.iter().map(|n| n.pitch).collect();
        let chunk_signal = ChunkSignal {
            chunk_index: index,
            timestamp_ms: js_sys::Date::now() as u64,
            pitch_bigrams: pitch_bigrams_from_notes(&perf_pitches),
            bar_range: chunk_bar_range,
            has_piece_match: self.inner.borrow().score_context.is_some(),
            scores: scores_array,
        };

        let mode_transitions = self.inner.borrow_mut().mode_detector.update(&chunk_signal);

        // Broadcast mode changes over WebSocket
        for transition in &mode_transitions {
            let context = self.build_mode_context(transition);
            let msg = serde_json::json!({
                "type": "mode_change",
                "mode": transition.mode,
                "chunkIndex": transition.chunk_index,
                "context": context,
            });
            let _ = ws.send_with_str(&msg.to_string());
        }
```

- [ ] **Step 4: Replace throttle_allows with mode-aware pacing**

Delete the `OBSERVATION_THROTTLE_MS` constant (line 29) and the `throttle_allows` method (lines 483-491).

Add the new method to the `impl PracticeSession` block:

```rust
    fn mode_throttle_allows(&self, s: &SessionState, policy: &ObservationPolicy) -> bool {
        match s.last_observation_at {
            None => true,
            Some(last) => {
                let now = js_sys::Date::now() as u64;
                now - last >= policy.min_interval_ms
            }
        }
    }
```

Replace the `should_generate` check (currently lines ~466-475):

```rust
        // 11. Check if we should generate an observation (mode-aware)
        let policy = self.inner.borrow().mode_detector.observation_policy();
        let should_generate = {
            let s = self.inner.borrow();
            !policy.suppress
                && stop_result.triggered
                && s.baselines.is_some()
                && self.mode_throttle_allows(&s, &policy)
        };
```

- [ ] **Step 5: Inject drilling context into generate_observation**

In `generate_observation` (starts at line ~507), after building `piece_context` (the `let piece_context = { ... };` block around line 547-563), inject drilling context into it:

```rust
        // Inject drilling comparison context if in drilling mode
        let piece_context = {
            let mut pc = piece_context; // move the existing piece_context
            let s = self.inner.borrow();
            if let Some(ref dp) = s.mode_detector.drilling_passage {
                let current_scores = s.scored_chunks.last().map(|c| c.scores).unwrap_or([0.0; 6]);
                let drilling_ctx = serde_json::json!({
                    "repetition_count": dp.repetition_count,
                    "first_attempt_scores": {
                        "dynamics": dp.first_scores[0],
                        "timing": dp.first_scores[1],
                        "pedaling": dp.first_scores[2],
                        "articulation": dp.first_scores[3],
                        "phrasing": dp.first_scores[4],
                        "interpretation": dp.first_scores[5],
                    },
                    "current_scores": {
                        "dynamics": current_scores[0],
                        "timing": current_scores[1],
                        "pedaling": current_scores[2],
                        "articulation": current_scores[3],
                        "phrasing": current_scores[4],
                        "interpretation": current_scores[5],
                    },
                    "bar_range": dp.bar_range,
                });
                match &mut pc {
                    Some(serde_json::Value::Object(ref mut ctx)) => {
                        ctx.insert("drilling_context".into(), drilling_ctx);
                    }
                    None => {
                        let mut ctx = serde_json::Map::new();
                        ctx.insert("drilling_context".into(), drilling_ctx);
                        pc = Some(serde_json::Value::Object(ctx));
                    }
                    _ => {}
                }
            }
            pc
        };
```

This injects drilling context into the existing `piece_context` JSON that is already passed to `AskInnerRequest`. No struct changes needed.

- [ ] **Step 6: Add build_mode_context helper**

Add to the `impl PracticeSession` block:

```rust
    fn build_mode_context(&self, transition: &ModeTransition) -> serde_json::Value {
        let s = self.inner.borrow();
        match transition.mode {
            PracticeMode::Drilling => {
                let mut ctx = serde_json::json!({});
                if let Some(ref dp) = s.mode_detector.drilling_passage {
                    if let Some(br) = dp.bar_range {
                        ctx["bars"] = serde_json::json!([br.0, br.1]);
                    }
                    ctx["repetition"] = serde_json::json!(dp.repetition_count);
                }
                ctx
            }
            PracticeMode::Running => {
                let mut ctx = serde_json::json!({});
                if let Some(ref sc) = s.score_context {
                    ctx["piece"] = serde_json::json!(format!("{} - {}", sc.composer, sc.title));
                }
                ctx
            }
            _ => serde_json::json!({}),
        }
    }
```

- [ ] **Step 7: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: Compiles without errors.

- [ ] **Step 8: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: integrate ModeDetector into session DO with mode-aware pacing"
```

---

## Task 4: Client-Side Changes

**Files:**
- Modify: `apps/web/src/lib/practice-api.ts:40-58`
- Modify: `apps/web/src/lib/observation-throttle.ts`
- Modify: `apps/web/src/hooks/usePracticeSession.ts`

- [ ] **Step 1: Add ModeChangeEvent type to practice-api.ts**

Add the types and extend the union in `apps/web/src/lib/practice-api.ts`:

```typescript
export type PracticeMode =
	| "warming"
	| "drilling"
	| "running"
	| "winding"
	| "regular";

// ModeChangeContext is discriminated by the parent ModeChangeEvent.mode field
export type ModeChangeContext =
	| { bars?: [number, number]; repetition: number }
	| { piece?: string }
	| Record<string, never>;

export interface ModeChangeEvent {
	type: "mode_change";
	mode: PracticeMode;
	chunkIndex: number;
	context: ModeChangeContext;
}
```

Add to the `PracticeWsEvent` union:

```typescript
export type PracticeWsEvent =
	| { type: "connected" }
	| { type: "chunk_processed"; index: number; scores: DimScores }
	| {
			type: "observation";
			text: string;
			dimension: string;
			framing: string;
			barRange?: string;
	  }
	| {
			type: "session_summary";
			observations: ObservationEvent[];
			summary: string;
			inference_failures?: number;
			total_chunks?: number;
	  }
	| { type: "piece_set"; query: string }
	| ModeChangeEvent
	| { type: "error"; message: string };
```

- [ ] **Step 2: Simplify observation-throttle.ts**

Replace the contents of `apps/web/src/lib/observation-throttle.ts`:

```typescript
import type { ObservationEvent } from "./practice-api";

/**
 * Simplified observation queue. The DO owns pacing decisions;
 * the client just queues for reconnection resilience and drains on session end.
 */
export class ObservationThrottle {
	private queued: ObservationEvent | null = null;
	private chunksReceived = 0;

	enqueue(obs: ObservationEvent): ObservationEvent {
		// DO controls pacing -- deliver immediately
		this.chunksReceived++;
		return obs;
	}

	onChunkProcessed(): null {
		this.chunksReceived++;
		return null;
	}

	drain(): ObservationEvent[] {
		if (this.queued) {
			const obs = this.queued;
			this.queued = null;
			return [obs];
		}
		return [];
	}

	reset(): void {
		this.queued = null;
		this.chunksReceived = 0;
	}

	getChunksReceived(): number {
		return this.chunksReceived;
	}
}
```

- [ ] **Step 3: Handle mode_change in usePracticeSession.ts**

Add `practiceMode` state and handle the new WS message.

**Add `PracticeMode` to the import** (line 3-7):

```typescript
import type {
	DimScores,
	ObservationEvent,
	PracticeMode,
	PracticeWsEvent,
} from "../lib/practice-api";
```

**Add state** to the hook's state declarations (around line 55-66):

```typescript
const [practiceMode, setPracticeMode] = useState<PracticeMode | null>(null);
```

**Add to the return type interface** `UsePracticeSessionReturn` (around line 36-52):

```typescript
practiceMode: PracticeMode | null;
```

**Add to the actual return object** at the end of the hook (around line 614-631):

```typescript
practiceMode,
```

**In the WebSocket message handler** switch (around line 202), add case:

```typescript
				case "mode_change": {
					setPracticeMode(data.mode);
					break;
				}
```

**Simplify the `chunk_processed` handler** (around line 203-210). The `onChunkProcessed` now always returns `null`, so remove the dead released-observation check:

```typescript
				case "chunk_processed": {
					setLatestScores(data.scores);
					setChunksProcessed((prev) => prev + 1);
					break;
				}
```

**Remove the `tick()` call** from the elapsed timer interval (around line 459-466). The timer should just update elapsed seconds:

```typescript
		timerRef.current = setInterval(() => {
			setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
		}, 1000);
```

The `throttleRef.current.enqueue()` call in the observation handler (line 219) now always returns the observation immediately, so the existing code works without changes (the truthy check still passes).

- [ ] **Step 4: Verify web build**

Run: `cd apps/web && bun run build`
Expected: Builds without errors.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/practice-api.ts apps/web/src/lib/observation-throttle.ts apps/web/src/hooks/usePracticeSession.ts
git commit -m "feat: simplify client throttle, add mode_change WS handling"
```

---

## Task 5: Verify Full Build and Manual Smoke Test

**Files:** None (verification only)

- [ ] **Step 1: Run Rust tests**

Run: `cd apps/api && cargo test -- --nocapture`
Expected: All existing tests pass + new practice_mode tests pass.

- [ ] **Step 2: Run Rust compilation check**

Run: `cd apps/api && cargo check`
Expected: No errors.

- [ ] **Step 3: Run web build**

Run: `cd apps/web && bun run build`
Expected: No errors.

- [ ] **Step 4: Verify no type errors**

Run: `cd apps/web && bun run typecheck` (or `bunx tsc --noEmit`)
Expected: No type errors.

- [ ] **Step 5: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address build/type issues from practice mode integration"
```
