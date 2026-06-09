//! Port-fidelity: the Rust gate must reproduce the FROZEN Python reference
//! (Stage-0c/0f) decisions + margins on the committed golden fixtures.
#![cfg(test)]

use crate::gate::{elastic_cost, margin_gate};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct ParityCandidate {
    piece_id: String,
    events: Vec<u16>,
    expected_cost: f64,
}
#[derive(serde::Deserialize)]
struct ParityQuery {
    query_id: String,
    in_catalog: bool,
    query_events: Vec<u16>,
    candidates: Vec<ParityCandidate>,
    expected_best_piece_id: String,
    expected_margin: f64,
    expected_locked: bool,
}
#[derive(serde::Deserialize)]
struct ParityFixtures {
    margin_threshold: f64,
    queries: Vec<ParityQuery>,
}

fn fixtures_path() -> PathBuf {
    // crate dir: apps/api/src/wasm/piece-identify -> repo root is 5 levels up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../model/data/evals/piece_id/parity_fixtures.json")
}

fn load() -> ParityFixtures {
    let path = fixtures_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("parse parity fixtures")
}

#[test]
fn rust_elastic_cost_matches_python_per_candidate() {
    let fx = load();
    for q in &fx.queries {
        for c in &q.candidates {
            let got = elastic_cost(&q.query_events, &c.events);
            assert!(
                (got - c.expected_cost).abs() < 1e-4,
                "{} / {}: rust cost {got} vs python {} (Δ {})",
                q.query_id, c.piece_id, c.expected_cost, (got - c.expected_cost).abs()
            );
        }
    }
}

#[test]
fn rust_margin_gate_matches_python_decision() {
    let fx = load();
    for q in &fx.queries {
        let cand_events: Vec<&[u16]> = q.candidates.iter().map(|c| c.events.as_slice()).collect();
        let d = margin_gate(&q.query_events, &cand_events, fx.margin_threshold)
            .unwrap_or_else(|| panic!("{}: gate returned None", q.query_id));
        let best_id = &q.candidates[d.best_index].piece_id;
        assert_eq!(best_id, &q.expected_best_piece_id, "{}: best piece mismatch", q.query_id);
        assert!(
            (d.margin - q.expected_margin).abs() < 1e-4,
            "{}: rust margin {} vs python {}", q.query_id, d.margin, q.expected_margin
        );
        assert_eq!(d.locked, q.expected_locked, "{}: lock decision mismatch", q.query_id);
    }
}

#[test]
fn certified_operating_point_holds() {
    let fx = load();
    let in_cat: Vec<&ParityQuery> = fx.queries.iter().filter(|q| q.in_catalog).collect();
    let ood: Vec<&ParityQuery> = fx.queries.iter().filter(|q| !q.in_catalog).collect();
    let in_locked = in_cat.iter().filter(|q| q.expected_locked).count();
    assert!(in_locked >= 14, "in-catalog locks {in_locked}/16 below certified TA");
    assert_eq!(ood.iter().filter(|q| q.expected_locked).count(), 0, "an OOD query locked");
}
