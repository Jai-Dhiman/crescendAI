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
}
