//! The certified open-set gate: 50ms chord-events, Jaccard subsequence-DTW, margin.
use crate::types::PerfNote;

/// Collapse onsets within `onset_tol_s` into chord-events (12-bit pc-set masks).
/// Mirrors piece_id_eval.stage0c_elastic_dtwgate._notes_to_events (pitch-only).
pub fn notes_to_events(notes: &[PerfNote], onset_tol_s: f64) -> Vec<u16> {
    if notes.is_empty() {
        return Vec::new();
    }
    let mut ordered: Vec<&PerfNote> = notes.iter().collect();
    ordered.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap_or(std::cmp::Ordering::Equal));
    let mut events: Vec<u16> = Vec::new();
    let mut anchor = ordered[0].onset;
    let mut cur: u16 = 0;
    for n in &ordered {
        if n.onset - anchor > onset_tol_s {
            events.push(cur);
            anchor = n.onset;
            cur = 0;
        }
        cur |= 1u16 << (n.pitch % 12);
    }
    events.push(cur);
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pitch: u8, onset: f64) -> PerfNote {
        PerfNote { pitch, onset, offset: onset + 0.4, velocity: 80 }
    }

    #[test]
    fn notes_to_events_collapses_within_tolerance() {
        // C and E within 50ms -> one event {0,4}; D at 1.0s -> event {2}.
        let notes = vec![note(60, 0.0), note(64, 0.02), note(62, 1.0)];
        let ev = notes_to_events(&notes, 0.05);
        assert_eq!(ev, vec![(1u16 << 0) | (1u16 << 4), 1u16 << 2]); // [17, 4]
    }

    #[test]
    fn notes_to_events_sorts_unordered_input() {
        let notes = vec![note(62, 1.0), note(60, 0.0)];
        let ev = notes_to_events(&notes, 0.05);
        assert_eq!(ev, vec![1u16 << 0, 1u16 << 2]);
    }
}
