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
}
