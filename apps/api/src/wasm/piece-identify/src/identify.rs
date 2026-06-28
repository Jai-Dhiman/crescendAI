//! Orchestrates the certified pipeline: hybrid chroma recall -> margin gate.
use crate::chroma;
use crate::gate;
use crate::types::{IdentifyResult, PieceIndex};
use std::collections::HashSet;

// Hybrid shortlist sizing (#96). The shortlist is the UNION of whole-piece
// top-TOP_K and windowed top-WINDOWED_K. Windowing recovers partial/mid-piece
// queries that the whole-piece chroma drops on full-length fingerprints. These
// are a latency-vs-recall tradeoff (each extra candidate is one more elastic-DTW
// alignment in the gate); larger WINDOWED_K raises recall on full pieces but
// costs alignments. Tuned on the production-faithful (uncapped) re-tune.
const TOP_K: usize = 20;
const WINDOWED_K: usize = 16;

/// Run the certified pipeline over the artifact. Returns None when the query has
/// < 2 chord-events or the artifact has < 2 pieces (cannot form a margin).
///
/// The recall stage is an ADDITIVE hybrid: whole-piece chroma top-K UNION the
/// per-piece windowed chroma top-WINDOWED_K. The elastic-DTW margin gate then
/// re-ranks the union UNCHANGED. An artifact without `windows` (old v2) yields an
/// empty windowed pass, so the shortlist degrades to whole-piece-only (back-compat).
pub fn run_identify(
    notes: &[crate::types::PerfNote],
    index: &PieceIndex,
    margin_threshold: f64,
) -> Option<IdentifyResult> {
    if index.pieces.len() < 2 {
        return None;
    }
    let onset_tol_s = index.onset_tol_ms / 1000.0;
    let q_events = gate::notes_to_events(notes, onset_tol_s);
    if q_events.len() < 2 {
        return None;
    }
    let q_chroma = chroma::chroma_vector(notes);

    // Whole-piece recall (the baseline shortlist; preserved untruncated).
    let catalog_chroma: Vec<[f64; 12]> = index.pieces.iter().map(|p| p.chroma).collect();
    let mut shortlist = chroma::rank_top_k(&q_chroma, &catalog_chroma, TOP_K);

    // Windowed recall: flatten every piece's window vectors with parallel owner
    // indices, then take the best-window-per-piece top-WINDOWED_K and append the
    // ones not already surfaced by the whole-piece pass (whole-piece first).
    let mut windows: Vec<[f64; 12]> = Vec::new();
    let mut owners: Vec<usize> = Vec::new();
    for (i, p) in index.pieces.iter().enumerate() {
        for w in &p.windows {
            windows.push(*w);
            owners.push(i);
        }
    }
    if !windows.is_empty() {
        let mut seen: HashSet<usize> = shortlist.iter().copied().collect();
        for i in chroma::windowed_top_k(&q_chroma, &windows, &owners, WINDOWED_K) {
            if seen.insert(i) {
                shortlist.push(i);
            }
        }
    }

    let cand_events: Vec<&[u16]> =
        shortlist.iter().map(|&i| index.pieces[i].events.as_slice()).collect();
    let decision = gate::margin_gate(&q_events, &cand_events, margin_threshold)?;
    let piece = &index.pieces[shortlist[decision.best_index]];
    Some(IdentifyResult {
        piece_id: piece.piece_id.clone(),
        composer: piece.composer.clone(),
        title: piece.title.clone(),
        margin: decision.margin,
        locked: decision.locked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PerfNote, PieceArtifact};

    fn note(pitch: u8, onset: f64) -> PerfNote {
        PerfNote { pitch, onset, offset: onset + 0.4, velocity: 100 }
    }

    fn piece(id: &str, events: Vec<u16>, chroma: [f64; 12]) -> PieceArtifact {
        PieceArtifact {
            piece_id: id.into(), composer: "C".into(), title: id.into(),
            chroma, events, windows: Vec::new(),
        }
    }

    #[test]
    fn run_identify_locks_to_clear_match() {
        // Query: C,E,G,C arpeggio across 4 distinct onsets.
        let notes = vec![note(60, 0.0), note(64, 0.5), note(67, 1.0), note(72, 1.5)];
        let q_events = gate::notes_to_events(&notes, 0.05);
        let q_chroma = chroma::chroma_vector(&notes);
        // exact piece shares query's events + chroma; decoy is disjoint.
        let exact = piece("exact", q_events.clone(), q_chroma);
        let decoy = piece("decoy", vec![1, 2, 4, 8], [0.0; 12]);
        let index = PieceIndex { onset_tol_ms: 50.0, pieces: vec![decoy, exact] };

        let r = run_identify(&notes, &index, 0.0935).expect("a decision");
        assert_eq!(r.piece_id, "exact");
        assert!(r.locked);
        assert!(r.margin > 0.0935);
    }

    #[test]
    fn run_identify_none_when_too_few_pieces() {
        let notes = vec![note(60, 0.0), note(64, 0.5)];
        let index = PieceIndex { onset_tol_ms: 50.0, pieces: vec![piece("only", vec![1, 2], [0.0; 12])] };
        assert!(run_identify(&notes, &index, 0.0935).is_none());
    }

    #[test]
    fn windowed_pass_recovers_a_piece_the_whole_chroma_misses() {
        // Query = C,E,G,C. The true piece's WHOLE-piece chroma is dominated by a
        // long unrelated tail (disjoint pitch classes), so it loses the whole-piece
        // cosine to a decoy -- but it carries a window chroma equal to the query's,
        // so the windowed pass surfaces it and the gate (events) locks it.
        let notes = vec![note(60, 0.0), note(64, 0.5), note(67, 1.0), note(72, 1.5)];
        let q_events = gate::notes_to_events(&notes, 0.05);
        let q_chroma = chroma::chroma_vector(&notes);

        let mut true_piece = piece("true", q_events.clone(), [0.0; 12]);
        // whole-piece chroma points entirely away from the query (pc 1,6 mass).
        true_piece.chroma[1] = 0.8;
        true_piece.chroma[6] = 0.6;
        true_piece.windows = vec![q_chroma]; // a local window matches the query exactly
        // decoy: whole-piece chroma equals the query (wins whole-piece recall) but
        // its events are disjoint, so the gate rejects it on alignment cost.
        let decoy = piece("decoy", vec![1, 2, 4, 8], q_chroma);

        let index = PieceIndex { onset_tol_ms: 50.0, pieces: vec![decoy, true_piece] };
        let r = run_identify(&notes, &index, 0.0935).expect("a decision");
        assert_eq!(r.piece_id, "true", "windowed pass must surface the true piece");
        assert!(r.locked);
    }

    #[test]
    fn empty_windows_reproduces_whole_piece_only_shortlist() {
        // Back-compat: an artifact with NO windows behaves as the whole-piece path
        // (the windowed pass contributes nothing). Locks on the clear chroma+events match.
        let notes = vec![note(60, 0.0), note(64, 0.5), note(67, 1.0), note(72, 1.5)];
        let q_events = gate::notes_to_events(&notes, 0.05);
        let q_chroma = chroma::chroma_vector(&notes);
        let exact = piece("exact", q_events.clone(), q_chroma); // windows = empty
        let decoy = piece("decoy", vec![1, 2, 4, 8], [0.0; 12]);
        let index = PieceIndex { onset_tol_ms: 50.0, pieces: vec![decoy, exact] };
        let r = run_identify(&notes, &index, 0.0935).expect("a decision");
        assert_eq!(r.piece_id, "exact");
        assert!(r.locked);
    }
}
