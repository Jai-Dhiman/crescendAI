//! Multi-signal piece identification: N-gram recall (Stage 1) + rerank (Stage 2).
//! Stage 3 (DTW confirmation) happens in session.rs because it requires async R2 access.

use std::collections::HashMap;

use serde::Deserialize;

use super::score_follower::PerfNote;

/// DTW confirmation threshold (Stage 3, used by session.rs).
pub const DTW_CONFIRM_THRESHOLD: f64 = 0.3;

/// Maximum candidates returned from N-gram recall.
const MAX_CANDIDATES: usize = 10;

/// Minimum notes required to attempt identification.
const MIN_NOTES: usize = 10;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Inverted index: trigram key ("p1,p2,p3") -> Vec<(`piece_id`, `bar_number`)>.
/// Deserialized from `ngram_index.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct NgramIndex(pub HashMap<String, Vec<(String, u32)>>);

/// Pre-computed 128-dim feature vectors per piece.
/// Deserialized from `rerank_features.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RerankFeatures(pub HashMap<String, Vec<f64>>);

/// Result of piece identification.
#[derive(Debug, Clone)]
pub struct PieceIdentification {
    pub piece_id: String,
    pub confidence: f64,
    pub method: String,
}

// ---------------------------------------------------------------------------
// Stage 1: N-gram recall
// ---------------------------------------------------------------------------

/// Extract pitch trigrams from notes, look up in inverted index,
/// count hits per piece, return top candidates by hit count.
pub fn ngram_recall(notes: &[PerfNote], index: &NgramIndex) -> Vec<(String, usize)> {
    if notes.len() < 3 {
        return vec![];
    }

    let mut hits: HashMap<String, usize> = HashMap::new();

    for window in notes.windows(3) {
        let key = format!(
            "{},{},{}",
            window[0].pitch, window[1].pitch, window[2].pitch
        );
        if let Some(entries) = index.0.get(&key) {
            for (piece_id, _bar) in entries {
                *hits.entry(piece_id.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<(String, usize)> = hits.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.truncate(MAX_CANDIDATES);
    ranked
}

// ---------------------------------------------------------------------------
// Stage 2: Rerank
// ---------------------------------------------------------------------------

/// Compute 128-dim feature vector from performance notes.
/// Layout matches Python fingerprint.py exactly.
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

    // [82:128] zeros (reserved) -- already zero

    features
}

/// Standard cosine similarity: dot(a,b) / (|a| * |b|).
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

/// Rerank N-gram candidates using cosine similarity of feature vectors.
/// Returns top-2 candidates sorted by similarity.
pub fn rerank_candidates(
    notes: &[PerfNote],
    candidates: &[(String, usize)],
    features: &RerankFeatures,
) -> Vec<(String, f64)> {
    let perf_features = compute_rerank_features(notes);

    let mut scored: Vec<(String, f64)> = candidates
        .iter()
        .filter_map(|(piece_id, _hits)| {
            features.0.get(piece_id).map(|score_features| {
                let sim = cosine_similarity(&perf_features, score_features);
                (piece_id.clone(), sim)
            })
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(2);
    scored
}

// ---------------------------------------------------------------------------
// Top-level identification
// ---------------------------------------------------------------------------

/// Run Stage 1 (N-gram recall) + Stage 2 (rerank). Returns top candidate.
/// DTW confirmation (Stage 3) happens in session.rs.
pub fn identify_piece(
    notes: &[PerfNote],
    index: &NgramIndex,
    features: &RerankFeatures,
) -> Option<PieceIdentification> {
    if notes.len() < MIN_NOTES {
        return None;
    }

    let candidates = ngram_recall(notes, index);
    if candidates.is_empty() {
        return None;
    }

    let reranked = rerank_candidates(notes, &candidates, features);
    if reranked.is_empty() {
        // Fall back to top N-gram candidate if no rerank features available
        let (piece_id, _hits) = &candidates[0];
        return Some(PieceIdentification {
            piece_id: piece_id.clone(),
            confidence: 0.0,
            method: "ngram_only".to_string(),
        });
    }

    let (piece_id, similarity) = &reranked[0];
    Some(PieceIdentification {
        piece_id: piece_id.clone(),
        confidence: *similarity,
        method: "ngram_rerank".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(pitch: u8, onset: f64, velocity: u8) -> PerfNote {
        PerfNote {
            pitch,
            onset,
            offset: onset + 0.5,
            velocity,
        }
    }

    fn make_small_index() -> NgramIndex {
        let mut map = HashMap::new();
        map.insert(
            "60,64,67".to_string(),
            vec![
                ("chopin.ballades.1".to_string(), 5),
                ("beethoven.sonatas.14".to_string(), 12),
            ],
        );
        map.insert(
            "64,67,72".to_string(),
            vec![("chopin.ballades.1".to_string(), 6)],
        );
        map.insert(
            "67,72,76".to_string(),
            vec![("bach.wtc_i.prelude.1".to_string(), 3)],
        );
        NgramIndex(map)
    }

    #[test]
    fn test_ngram_recall_finds_matching_piece() {
        let index = make_small_index();
        // Notes: 60, 64, 67, 72 -> trigrams: (60,64,67), (64,67,72)
        // chopin.ballades.1 should get 2 hits, beethoven 1 hit
        let notes = vec![
            make_note(60, 0.0, 80),
            make_note(64, 0.5, 80),
            make_note(67, 1.0, 80),
            make_note(72, 1.5, 80),
        ];
        let results = ngram_recall(&notes, &index);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "chopin.ballades.1");
        assert_eq!(results[0].1, 2); // 2 trigram hits
    }

    #[test]
    fn test_ngram_recall_empty_on_few_notes() {
        let index = make_small_index();
        let notes = vec![make_note(60, 0.0, 80), make_note(64, 0.5, 80)];
        let results = ngram_recall(&notes, &index);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Expected 0.0, got {}", sim);
    }

    #[test]
    fn test_compute_rerank_features_length() {
        let notes = vec![
            make_note(60, 0.0, 80),
            make_note(64, 0.5, 90),
            make_note(67, 1.0, 70),
            make_note(72, 1.5, 85),
        ];
        let features = compute_rerank_features(&notes);
        assert_eq!(
            features.len(),
            128,
            "Feature vector must be exactly 128 elements"
        );
    }

    #[test]
    fn test_identify_piece_returns_none_on_insufficient_notes() {
        let index = make_small_index();
        let features = RerankFeatures(HashMap::new());
        // Only 5 notes, below MIN_NOTES (10)
        let notes: Vec<PerfNote> = (0..5)
            .map(|i| make_note(60 + i as u8, i as f64 * 0.5, 80))
            .collect();
        let result = identify_piece(&notes, &index, &features);
        assert!(result.is_none(), "Expected None for < 10 notes");
    }

    #[test]
    fn test_compute_rerank_features_pitch_class_histogram() {
        // All C4 notes -> pitch class 0 should be 1.0, rest 0.0
        let notes: Vec<PerfNote> = (0..4).map(|i| make_note(60, i as f64 * 0.5, 80)).collect();
        let features = compute_rerank_features(&notes);
        assert!(
            (features[0] - 1.0).abs() < 1e-10,
            "pitch class 0 should be 1.0"
        );
        for i in 1..12 {
            assert!(features[i].abs() < 1e-10, "pitch class {} should be 0.0", i);
        }
    }

    #[test]
    fn test_compute_rerank_features_reserved_zeros() {
        let notes = vec![make_note(60, 0.0, 80), make_note(64, 0.5, 90)];
        let features = compute_rerank_features(&notes);
        for i in 82..128 {
            assert!(features[i].abs() < 1e-10, "features[{}] should be 0.0", i);
        }
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "Expected 0.0 for zero vector, got {}",
            sim
        );
    }
}
