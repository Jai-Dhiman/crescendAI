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

/// Jaccard distance between two 12-bit pitch-class-set masks: 1 - |A∩B|/|A∪B|.
fn jaccard_dist(a: u16, b: u16) -> f64 {
    let union = (a | b).count_ones();
    if union == 0 {
        return 1.0;
    }
    let inter = (a & b).count_ones();
    1.0 - f64::from(inter) / f64::from(union)
}

/// Subsequence-DTW cost: local cost = Jaccard distance over pc-set masks.
/// Embeds the SHORTER sequence as rows in the longer (free start/end on the
/// longer), normalizes by the shorter length. Reproduces librosa
/// dtw(subseq=True) as used in stage0c._elastic_cost with w_time=0.
/// Returns +inf if either side has < 2 events.
pub fn elastic_cost(q: &[u16], r: &[u16]) -> f64 {
    if q.len() < 2 || r.len() < 2 {
        return f64::INFINITY;
    }
    // shorter on rows so the subsequence embedding is well-posed
    let (rows, cols) = if q.len() <= r.len() { (q, r) } else { (r, q) };
    let nr = rows.len();
    let nc = cols.len();

    // subseq DTW: free start across columns -> row 0 = local cost only.
    let mut prev: Vec<f64> = (0..nc).map(|j| jaccard_dist(rows[0], cols[j])).collect();
    let mut curr = vec![0.0_f64; nc];
    for i in 1..nr {
        for j in 0..nc {
            let c = jaccard_dist(rows[i], cols[j]);
            let p = if j == 0 {
                prev[0] // column 0 is NOT free: must accumulate downward
            } else {
                prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
            curr[j] = p + c;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    // free end across columns -> min of the last row, normalized by shorter length.
    let best = prev.iter().copied().fold(f64::INFINITY, f64::min);
    best / nr as f64
}

/// Outcome of the open-set margin gate over chroma top-K candidates.
pub struct GateDecision {
    pub best_index: usize, // index into the candidates slice
    pub margin: f64,
    pub locked: bool,
}

/// Cost each candidate, take the best and 2nd-best; lock iff the margin
/// (2nd-best − best) ≥ threshold. Returns None if fewer than two candidates
/// produce a finite cost. Mirrors stage0f._score_candidate + the certified
/// margin operating point.
pub fn margin_gate(query: &[u16], candidates: &[&[u16]], threshold: f64) -> Option<GateDecision> {
    let mut costs: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, ev)| (i, elastic_cost(query, ev)))
        .filter(|(_, c)| c.is_finite())
        .collect();
    if costs.len() < 2 {
        return None;
    }
    costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let margin = costs[1].1 - costs[0].1;
    Some(GateDecision { best_index: costs[0].0, margin, locked: margin >= threshold })
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

    #[test]
    fn elastic_cost_zero_for_identical_subsequence() {
        // Identical sequences -> every aligned event Jaccard distance 0 -> cost 0.
        let a: Vec<u16> = vec![1, 2, 4, 8];
        assert!(elastic_cost(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn elastic_cost_free_start_embeds_short_query() {
        // Query {2,4} is an exact contiguous subsequence of {1,2,4,8}; subseq DTW
        // finds it at zero cost (free start skips the leading {1}, free end the trailing {8}).
        let q: Vec<u16> = vec![2, 4];
        let r: Vec<u16> = vec![1, 2, 4, 8];
        assert!(elastic_cost(&q, &r).abs() < 1e-12);
        // symmetric: shorter is always placed on rows
        assert!(elastic_cost(&r, &q).abs() < 1e-12);
    }

    #[test]
    fn elastic_cost_jaccard_penalizes_mismatch() {
        // Disjoint pc-sets -> Jaccard distance 1 at every cell -> normalized cost 1.0.
        let q: Vec<u16> = vec![1, 2];      // {0}, {1}
        let r: Vec<u16> = vec![4, 8, 16];  // {2}, {3}, {4}
        let c = elastic_cost(&q, &r);
        assert!((c - 1.0).abs() < 1e-12, "expected 1.0, got {c}");
    }

    #[test]
    fn elastic_cost_too_short_is_infinite() {
        assert!(elastic_cost(&[1], &[1, 2, 3]).is_infinite());
    }

    #[test]
    fn margin_gate_locks_on_clear_winner() {
        let query: Vec<u16> = vec![1, 2, 4, 8];
        let exact: Vec<u16> = vec![1, 2, 4, 8];       // cost ~0 (best)
        let wrong: Vec<u16> = vec![16, 32, 64, 128];  // cost ~1 (far)
        let cands: Vec<&[u16]> = vec![&exact, &wrong];
        let d = margin_gate(&query, &cands, 0.0935).expect("two finite candidates");
        assert_eq!(d.best_index, 0);
        assert!(d.margin > 0.9, "margin {} should be large", d.margin);
        assert!(d.locked);
    }

    #[test]
    fn margin_gate_stays_unknown_on_ambiguous() {
        let query: Vec<u16> = vec![1, 2, 4, 8];
        let a: Vec<u16> = vec![1, 2, 4, 8];
        let b: Vec<u16> = vec![1, 2, 4, 8]; // identical -> margin 0 < threshold
        let cands: Vec<&[u16]> = vec![&a, &b];
        let d = margin_gate(&query, &cands, 0.0935).unwrap();
        assert!(d.margin.abs() < 1e-12);
        assert!(!d.locked);
    }

    #[test]
    fn margin_gate_needs_two_finite_candidates() {
        let query: Vec<u16> = vec![1, 2, 4, 8];
        let only: Vec<u16> = vec![1, 2, 4, 8];
        let cands: Vec<&[u16]> = vec![&only];
        assert!(margin_gate(&query, &cands, 0.0935).is_none());
    }
}
