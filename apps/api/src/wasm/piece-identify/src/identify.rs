//! Orchestrates the certified pipeline: chroma recall (top-5) -> margin gate.
use crate::chroma;
use crate::gate;
use crate::types::{IdentifyResult, PieceIndex};

/// Run the certified pipeline over the artifact. Returns None when the query has
/// < 2 chord-events or the artifact has < 2 pieces (cannot form a margin).
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
    let catalog_chroma: Vec<[f64; 12]> = index.pieces.iter().map(|p| p.chroma).collect();
    let topk = chroma::rank_top_k(&q_chroma, &catalog_chroma, 5);
    let cand_events: Vec<&[u16]> = topk.iter().map(|&i| index.pieces[i].events.as_slice()).collect();
    let decision = gate::margin_gate(&q_events, &cand_events, margin_threshold)?;
    let piece = &index.pieces[topk[decision.best_index]];
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
        PieceArtifact { piece_id: id.into(), composer: "C".into(), title: id.into(), chroma, events }
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
}
