//! Stage 2: Feature computation and cosine rerank.
//! Computes a 128-dim feature vector from performance notes and reranks
//! N-gram candidates by cosine similarity to pre-built score feature vectors.

use crate::types::{NgramCandidate, PerfNote, RerankFeatures, RerankResult};

/// Compute a 128-dim feature vector from performance notes.
///
/// Layout (matches Python fingerprint.py exactly):
/// - [0:12]    pitch class histogram (normalized)
/// - [12:37]   interval histogram, intervals clamped to [-12, 12] (25 bins)
/// - [37:41]   pitch stats: min/127, max/127, mean/127, std/127
/// - [41:66]   IOI histogram: 25 bins at 50ms each, last bin = 1200ms+
/// - [66:78]   velocity histogram: 12 bins (v/11)
/// - [78:82]   velocity stats: min/127, max/127, mean/127, std/127
/// - [82:128]  reserved zeros
pub fn compute_rerank_features(notes: &[PerfNote]) -> Vec<f64> {
    let mut features = vec![0.0_f64; 128];
    let n = notes.len();
    if n == 0 {
        return features;
    }

    // [0:12] pitch class histogram
    let mut pitch_class = [0.0_f64; 12];
    for note in notes {
        pitch_class[(note.pitch % 12) as usize] += 1.0;
    }
    for i in 0..12 {
        features[i] = pitch_class[i] / n as f64;
    }

    // [12:37] interval histogram: intervals clamped to [-12, 12] -> bins [0..25]
    if n >= 2 {
        let mut interval_hist = [0.0_f64; 25];
        let interval_count = n - 1;
        for i in 1..n {
            let interval = i32::from(notes[i].pitch) - i32::from(notes[i - 1].pitch);
            let clamped = interval.clamp(-12, 12);
            let bin = (clamped + 12) as usize; // -12 -> 0, 0 -> 12, +12 -> 24
            interval_hist[bin] += 1.0;
        }
        for i in 0..25 {
            features[12 + i] = interval_hist[i] / interval_count as f64;
        }
    }

    // [37:41] pitch stats: min/127, max/127, mean/127, std/127
    {
        let pitches: Vec<f64> = notes.iter().map(|n| f64::from(n.pitch)).collect();
        let min_p = pitches.iter().copied().fold(f64::INFINITY, f64::min);
        let max_p = pitches.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_p = pitches.iter().sum::<f64>() / n as f64;
        let var_p = pitches.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / n as f64;
        let std_p = var_p.sqrt();
        features[37] = min_p / 127.0;
        features[38] = max_p / 127.0;
        features[39] = mean_p / 127.0;
        features[40] = std_p / 127.0;
    }

    // [41:66] IOI histogram: 25 bins at 50ms each, last bin = 1200ms+
    if n >= 2 {
        let mut ioi_hist = [0.0_f64; 25];
        let ioi_count = n - 1;
        for i in 1..n {
            let ioi_sec = notes[i].onset - notes[i - 1].onset;
            let ioi_ms = ioi_sec * 1000.0;
            let bin = (ioi_ms / 50.0) as usize;
            let bin = bin.min(24);
            ioi_hist[bin] += 1.0;
        }
        for i in 0..25 {
            features[41 + i] = ioi_hist[i] / ioi_count as f64;
        }
    }

    // [66:78] velocity histogram: 12 bins, bin = v / 11, capped at 11
    {
        let mut vel_hist = [0.0_f64; 12];
        for note in notes {
            let bin = (note.velocity / 11).min(11) as usize;
            vel_hist[bin] += 1.0;
        }
        for i in 0..12 {
            features[66 + i] = vel_hist[i] / n as f64;
        }
    }

    // [78:82] velocity stats: min/127, max/127, mean/127, std/127
    {
        let vels: Vec<f64> = notes.iter().map(|n| f64::from(n.velocity)).collect();
        let min_v = vels.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = vels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_v = vels.iter().sum::<f64>() / n as f64;
        let var_v = vels.iter().map(|v| (v - mean_v).powi(2)).sum::<f64>() / n as f64;
        let std_v = var_v.sqrt();
        features[78] = min_v / 127.0;
        features[79] = max_v / 127.0;
        features[80] = mean_v / 127.0;
        features[81] = std_v / 127.0;
    }

    // [82:128] zeros (reserved) -- already zero from initialization

    features
}

/// Standard cosine similarity: dot(a,b) / (|a| * |b|).
/// Returns 0.0 if either vector is zero-magnitude.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Rerank N-gram candidates by cosine similarity of feature vectors.
/// Returns top-2 candidates sorted by descending similarity.
pub fn rerank_candidates(
    notes: &[PerfNote],
    candidates: &[NgramCandidate],
    features: &RerankFeatures,
) -> Vec<RerankResult> {
    let perf_features = compute_rerank_features(notes);

    let mut scored: Vec<RerankResult> = candidates
        .iter()
        .filter_map(|c| {
            features.get(&c.piece_id).map(|score_features| {
                let similarity = cosine_similarity(&perf_features, score_features);
                RerankResult { piece_id: c.piece_id.clone(), similarity }
            })
        })
        .collect();

    scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(2);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(pitch: u8, onset: f64, velocity: u8) -> PerfNote {
        PerfNote { pitch, onset, offset: onset + 0.5, velocity }
    }

    #[test]
    fn feature_vector_is_128_elements() {
        let notes = vec![
            make_note(60, 0.0, 80),
            make_note(64, 0.5, 90),
            make_note(67, 1.0, 70),
            make_note(72, 1.5, 85),
        ];
        let f = compute_rerank_features(&notes);
        assert_eq!(f.len(), 128);
    }

    #[test]
    fn pitch_class_histogram_all_c() {
        // All C4 notes -> pitch class 0 should be 1.0, rest 0.0
        let notes: Vec<PerfNote> = (0..4).map(|i| make_note(60, i as f64 * 0.5, 80)).collect();
        let f = compute_rerank_features(&notes);
        assert!((f[0] - 1.0).abs() < 1e-10, "pitch class 0 should be 1.0");
        for i in 1..12 {
            assert!(f[i].abs() < 1e-10, "pitch class {} should be 0.0", i);
        }
    }

    #[test]
    fn reserved_zeros() {
        let notes = vec![make_note(60, 0.0, 80), make_note(64, 0.5, 90)];
        let f = compute_rerank_features(&notes);
        for i in 82..128 {
            assert!(f[i].abs() < 1e-10, "features[{}] should be 0.0", i);
        }
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10, "identical vectors: expected 1.0, got {}", sim);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "orthogonal vectors: expected 0.0, got {}", sim);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "zero vector: expected 0.0, got {}", sim);
    }
}
