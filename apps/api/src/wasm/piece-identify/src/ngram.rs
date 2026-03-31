//! Stage 1: N-gram recall.
//! Extracts pitch trigrams from performance notes, looks them up in the inverted
//! index, and returns top candidates ranked by hit count.

use std::collections::HashMap;

use crate::types::{NgramCandidate, NgramIndex, PerfNote};

/// Maximum candidates returned from N-gram recall.
const MAX_CANDIDATES: usize = 10;

/// Extract pitch trigrams from notes, look up in inverted index,
/// count hits per piece, return top candidates by hit count.
pub fn ngram_recall(notes: &[PerfNote], index: &NgramIndex) -> Vec<NgramCandidate> {
    if notes.len() < 3 {
        return vec![];
    }

    let mut hits: HashMap<String, usize> = HashMap::new();

    for window in notes.windows(3) {
        let key = format!(
            "{},{},{}",
            window[0].pitch, window[1].pitch, window[2].pitch
        );
        if let Some(entries) = index.get(&key) {
            for (piece_id, _bar) in entries {
                *hits.entry(piece_id.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<NgramCandidate> = hits
        .into_iter()
        .map(|(piece_id, hit_count)| NgramCandidate { piece_id, hit_count })
        .collect();
    ranked.sort_by(|a, b| b.hit_count.cmp(&a.hit_count));
    ranked.truncate(MAX_CANDIDATES);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(pitch: u8, onset: f64) -> PerfNote {
        PerfNote { pitch, onset, offset: onset + 0.5, velocity: 80 }
    }

    fn make_index() -> NgramIndex {
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
        map
    }

    #[test]
    fn finds_matching_piece() {
        let index = make_index();
        // Trigrams: (60,64,67), (64,67,72) -> chopin gets 2 hits, beethoven gets 1
        let notes = vec![
            make_note(60, 0.0),
            make_note(64, 0.5),
            make_note(67, 1.0),
            make_note(72, 1.5),
        ];
        let results = ngram_recall(&notes, &index);
        assert!(!results.is_empty());
        assert_eq!(results[0].piece_id, "chopin.ballades.1");
        assert_eq!(results[0].hit_count, 2);
    }

    #[test]
    fn empty_on_fewer_than_3_notes() {
        let index = make_index();
        let notes = vec![make_note(60, 0.0), make_note(64, 0.5)];
        let results = ngram_recall(&notes, &index);
        assert!(results.is_empty());
    }
}
