//! C1 recall: key-dependent velocity-weighted pitch-class chroma + cosine top-k.
use crate::types::PerfNote;

/// 12-bin key-dependent velocity-weighted pitch-class histogram, L2-normalized.
/// Mirrors piece_id_eval.note_chroma.chroma_vector.
pub fn chroma_vector(notes: &[PerfNote]) -> [f64; 12] {
    let mut cv = [0.0_f64; 12];
    for n in notes {
        cv[(n.pitch % 12) as usize] += f64::from(n.velocity);
    }
    let norm: f64 = cv.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in cv.iter_mut() {
            *x /= norm;
        }
    }
    cv
}

fn dot12(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let mut s = 0.0;
    for i in 0..12 {
        s += a[i] * b[i];
    }
    s
}

/// Rank catalog chroma vectors by cosine to the query (dot, since both are
/// L2-normalized) and return the top-k catalog indices, descending. Stable on ties.
pub fn rank_top_k(query: &[f64; 12], catalog: &[[f64; 12]], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = catalog
        .iter()
        .enumerate()
        .map(|(i, v)| (i, dot12(query, v)))
        .collect();
    // stable sort by descending score (ties keep catalog order)
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pitch: u8, velocity: u8) -> PerfNote {
        PerfNote { pitch, onset: 0.0, offset: 0.4, velocity }
    }

    #[test]
    fn chroma_is_velocity_weighted_and_l2_normalized() {
        // C (pc0) velocity 30, G (pc7) velocity 40.
        let notes = vec![note(60, 30), note(67, 40)];
        let cv = chroma_vector(&notes);
        let norm: f64 = cv.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "expected unit norm, got {norm}");
        let expected_c = 30.0 / (30.0_f64 * 30.0 + 40.0 * 40.0).sqrt();
        assert!((cv[0] - expected_c).abs() < 1e-12);
        assert!((cv[7] - 40.0 / (30.0_f64 * 30.0 + 40.0 * 40.0).sqrt()).abs() < 1e-12);
        assert!(cv[1].abs() < 1e-12);
    }

    #[test]
    fn rank_top_k_orders_by_cosine_desc() {
        let q = chroma_vector(&[note(60, 100), note(64, 100)]); // C + E
        let exact = chroma_vector(&[note(60, 100), note(64, 100)]); // identical -> cosine 1
        let close = chroma_vector(&[note(60, 100), note(64, 100), note(67, 5)]); // C+E+tiny G
        let far = chroma_vector(&[note(61, 100), note(66, 100)]); // C#+F# -> orthogonal-ish
        let catalog = vec![far, close, exact]; // indices 0,1,2
        let top = rank_top_k(&q, &catalog, 2);
        assert_eq!(top, vec![2, 1], "expected exact (2) then close (1)");
    }
}
