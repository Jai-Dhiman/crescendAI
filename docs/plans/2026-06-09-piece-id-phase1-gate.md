# Piece-ID Phase-1 Certified Gate Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task) where the group is marked parallel. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the legacy 3-stage piece-ID pipeline with the Phase-0-CERTIFIED 2-stage gate (chroma recall → pitch-only chord-Jaccard elastic-DTW margin gate) in production Rust/WASM + the catalog artifact, locking to a piece only when the margin clears 0.0935 and otherwise staying `unknown`.
**Spec:** docs/specs/2026-06-09-piece-id-phase1-gate-design.md
**Style:** Follow CLAUDE.md + apps/api/TS_STYLE.md. Python: `uv`, explicit exceptions (no silent fallback), no emojis. Rust: match existing crate idioms.

## Frozen constants (used verbatim across tasks)
- `PIECE_ID_MARGIN_THRESHOLD = 0.0935`
- `ONSET_TOL = 0.05` s (50 ms)
- `TOP_K = 5`
- Parity tolerance: `1e-4` on margin and per-candidate cost.

## Task Groups
- **Group 0 (harness, FIRST, sequential):** Task 1
- **Group A (Python generator, parallel with Group B):** Task 2 → Task 3 → Task 4 (sequential within A; share `fingerprint.py`/`cli.py`)
- **Group B (Rust WASM core, parallel with Group A; sequential within B — shared `lib.rs`/`types.rs`):** Task 5 → 6 → 7 → 8 → 9 → 10 → 11
- **Group C (port-fidelity gate, depends on Task 1 + Task 11):** Task 12
- **Group D (TS bridge, depends on Task 11):** Task 13
- **Group E (local R2 seed, depends on Task 3 + Task 11):** Task 14

`[SHIPS INDEPENDENTLY]` — Groups 0+A+B+C together (the WASM gate + artifact + proven parity) are a self-contained, shippable PR. Group D makes the bridge call the new export; Group E enables `wrangler dev`. The `session-brain.ts` rewire is the DEFERRED slice (BLOCKED-ON-#28, spec §"DEFERRED slice"), NOT in this plan.

---

### Task 1: Freeze the Python-reference decisions into golden parity fixtures
**Group:** 0 (FIRST — the harness everything else is verified against)

**Behavior being verified:** the committed `parity_fixtures.json` reproduces the certified operating point — in-catalog full-piece queries lock at threshold 0.0935 at the certified TA rate (≥14/16) and out-of-catalog queries never lock.
**Interface under test:** the golden fixture file `model/data/evals/piece_id/parity_fixtures.json` (produced by `python -m piece_id_eval.export_parity_fixtures`).

**Files:**
- Create: `model/src/piece_id_eval/export_parity_fixtures.py`
- Create (generated, committed): `model/data/evals/piece_id/parity_fixtures.json`
- Test: `model/src/piece_id_eval/test_parity_fixtures.py`

- [ ] **Step 1: Write the failing test**
```python
# model/src/piece_id_eval/test_parity_fixtures.py
import json
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[2] / "data/evals/piece_id/parity_fixtures.json"


def test_parity_fixtures_reproduce_certified_operating_point():
    data = json.loads(_FIXTURES.read_text())
    assert data["margin_threshold"] == 0.0935
    queries = data["queries"]

    in_cat = [q for q in queries if q["in_catalog"]]
    ood = [q for q in queries if not q["in_catalog"]]
    assert len(in_cat) == 16, f"expected 16 in-catalog queries, got {len(in_cat)}"
    assert len(ood) >= 10, f"expected >=10 OOD queries, got {len(ood)}"

    # Certified TA point estimate is 0.875 -> >=14/16 in-catalog queries lock.
    assert sum(q["expected_locked"] for q in in_cat) >= 14
    # Certified false-accept axis: OOD queries must not lock.
    assert sum(q["expected_locked"] for q in ood) == 0

    for q in queries:
        assert len(q["query_events"]) >= 2
        assert 2 <= len(q["candidates"]) <= 5
        for c in q["candidates"]:
            assert all(0 <= m <= 0x0FFF for m in c["events"])  # 12-bit pc masks
        # margin is (2nd-best - best) over candidate costs
        costs = sorted(c["expected_cost"] for c in q["candidates"])
        assert abs(q["expected_margin"] - (costs[1] - costs[0])) < 1e-9
        assert q["expected_locked"] == (q["expected_margin"] >= 0.0935)
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd model && uv run pytest src/piece_id_eval/test_parity_fixtures.py -q
```
Expected: FAIL — `FileNotFoundError`/`No such file ... parity_fixtures.json` (the fixture has not been generated yet).

- [ ] **Step 3: Implement the exporter, then generate + commit the fixture**
```python
# model/src/piece_id_eval/export_parity_fixtures.py
"""Freeze the FROZEN Phase-0 gate (Stage-0c/0f, pitch-only chord-Jaccard elastic
margin) into a golden fixture the Rust/WASM port is asserted against.

For each in-catalog recording (full piece) and a deterministic OOD sample, emit:
  - query_events: u16 12-bit pitch-class-set masks (onsets collapsed within 50ms)
  - candidates: chroma top-5, each with its u16 events + Python elastic cost
  - expected_best_piece_id / expected_margin / expected_locked at threshold 0.0935

Requires ASAP + MAESTRO present (data/raw); raises if missing (no silent fallback).
Run: cd model && uv run python -m piece_id_eval.export_parity_fixtures
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.stage0c_elastic_dtwgate import (
    _TOP_K,
    _W_PITCH,
    ElasticGate,
    _notes_to_events,
    load_data,
)
from piece_id_eval.stage0f_hard_ood_certify import (
    _MAESTRO_DIR,
    _MAX_OOD_EVENTS,
    _candidates,
    _load_perf_midi,
)

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_OUT = _MODEL_ROOT / "data/evals/piece_id/parity_fixtures.json"
_THRESHOLD = 0.0935
_N_OOD = 12
_SEED = 42


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pc_mat_to_masks(pc_mat: np.ndarray) -> list[int]:
    """Each event row (E,12) binary -> a 12-bit pitch-class-set mask (bit pc set)."""
    masks: list[int] = []
    for i in range(pc_mat.shape[0]):
        m = 0
        for pc in range(12):
            if pc_mat[i, pc] > 0:
                m |= 1 << pc
        masks.append(m)
    return masks


def _query_record(query_id: str, in_catalog: bool, notes, full_chroma: NoteChromaMatcher, gate: ElasticGate) -> dict | None:
    q_pc, q_li = _notes_to_events(notes)
    if q_pc.shape[0] < 2:
        return None
    topk = [r.piece_id for r in full_chroma.rank(notes)[:_TOP_K]]
    cands: list[dict] = []
    for cid in topk:
        c = gate.cost(q_pc, q_li, cid, _W_PITCH, 0.0)  # pitch-only (w_time=0)
        if c is None or not np.isfinite(c):
            continue
        r_pc, _ = gate._events[cid]
        cands.append({"piece_id": cid, "events": _pc_mat_to_masks(r_pc), "expected_cost": round(float(c), 6)})
    if len(cands) < 2:
        return None
    by_cost = sorted(cands, key=lambda x: x["expected_cost"])
    margin = by_cost[1]["expected_cost"] - by_cost[0]["expected_cost"]
    return {
        "query_id": query_id,
        "in_catalog": in_catalog,
        "query_events": _pc_mat_to_masks(q_pc),
        "candidates": cands,  # chroma-rank order — matches the order the WASM receives
        "expected_best_piece_id": by_cost[0]["piece_id"],
        "expected_margin": round(float(margin), 6),
        "expected_locked": bool(margin >= _THRESHOLD),
    }


def main() -> None:
    catalog, recordings = load_data()
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    queries: list[dict] = []
    for true_id, notes in recordings.items():
        rec = _query_record(f"in:{true_id}", True, notes, full_chroma, gate)
        if rec is None:
            raise RuntimeError(f"in-catalog query produced no record: {true_id}")
        queries.append(rec)

    if not _MAESTRO_DIR.exists():
        raise FileNotFoundError(f"MAESTRO required for OOD fixtures, missing: {_MAESTRO_DIR}")
    cands = _candidates()
    rng = random.Random(_SEED)
    rng.shuffle(cands)
    n_ood = 0
    for c in cands:
        if n_ood >= _N_OOD:
            break
        midi = _MAESTRO_DIR / c["row"]["midi_filename"]
        if not midi.exists():
            continue
        notes = _load_perf_midi(midi, _MAX_OOD_EVENTS)
        rec = _query_record(f"ood:{c['row']['canonical_title']}", False, notes, full_chroma, gate)
        if rec is None:
            continue
        queries.append(rec)
        n_ood += 1
    if n_ood < 10:
        raise RuntimeError(f"only {n_ood} OOD queries materialized; need >=10")

    out = {"margin_threshold": _THRESHOLD, "onset_tol_ms": 50, "top_k": _TOP_K, "queries": queries}
    _OUT.write_text(json.dumps(out))
    _log(f"Wrote {_OUT}: {len(queries)} queries ({n_ood} OOD)")


if __name__ == "__main__":
    main()
```
Then generate and stage the golden fixture:
```bash
cd model && uv run python -m piece_id_eval.export_parity_fixtures
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd model && uv run pytest src/piece_id_eval/test_parity_fixtures.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add model/src/piece_id_eval/export_parity_fixtures.py model/src/piece_id_eval/test_parity_fixtures.py model/data/evals/piece_id/parity_fixtures.json
git commit -m "test(#26): freeze certified gate decisions into golden parity fixtures"
```

---

### Task 2: `build_piece_index` emits the v2 chroma+events artifact
**Group:** A (parallel with Group B; first in A)

**Behavior being verified:** the generator turns a scores directory into the v2 artifact — per piece a L2-normalized 12-bin velocity-weighted chroma vector and a sequence of u16 pitch-class-set masks (onsets collapsed within 50 ms), sorted by `piece_id`.
**Interface under test:** `build_piece_index(scores_dir: Path, onset_tol_s: float) -> dict`

**Files:**
- Modify: `model/src/score_library/fingerprint.py`
- Test: `model/src/score_library/test_fingerprint_v2.py`

- [ ] **Step 1: Write the failing test**
```python
# model/src/score_library/test_fingerprint_v2.py
import json
import math
from pathlib import Path

from score_library.fingerprint import build_piece_index


def _write_score(dirpath: Path, piece_id: str, composer: str, title: str, notes):
    bars = [{"bar_number": 1, "notes": [
        {"pitch": p, "onset_seconds": o, "velocity": v, "duration_seconds": 0.4} for (p, o, v) in notes
    ]}]
    (dirpath / f"{piece_id}.json").write_text(json.dumps(
        {"piece_id": piece_id, "composer": composer, "title": title, "bars": bars}))


def test_build_piece_index_chroma_and_events(tmp_path):
    # Two C-major notes at the same onset (chord) + one D a beat later.
    _write_score(tmp_path, "b.piece", "B", "Beta", [(60, 0.0, 100), (64, 0.02, 100), (62, 1.0, 100)])
    _write_score(tmp_path, "a.piece", "A", "Alpha", [(60, 0.0, 80), (60, 0.5, 80)])

    index = build_piece_index(tmp_path, onset_tol_s=0.05)
    assert index["version"] == "v2"
    assert index["onset_tol_ms"] == 50
    ids = [p["piece_id"] for p in index["pieces"]]
    assert ids == ["a.piece", "b.piece"]  # sorted by piece_id

    beta = next(p for p in index["pieces"] if p["piece_id"] == "b.piece")
    assert beta["composer"] == "B" and beta["title"] == "Beta"
    # chroma is L2-normalized
    assert abs(math.sqrt(sum(x * x for x in beta["chroma"])) - 1.0) < 1e-9
    assert len(beta["chroma"]) == 12
    # events: onset 0.0 (C,E within 50ms -> {60,64}) then 1.0 (D -> {62})
    assert beta["events"] == [(1 << 0) | (1 << 4), (1 << 2)]  # [17, 4]
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_v2.py -q
```
Expected: FAIL — `ImportError: cannot import name 'build_piece_index'`

- [ ] **Step 3: Implement the minimum to make the test pass**
Add to `model/src/score_library/fingerprint.py` (keep `_collect_all_notes`; the legacy `build_ngram_index`/`build_rerank_features`/`compute_rerank_features`/`extract_pitch_trigrams` are removed in Task 3):
```python
import math


def _piece_chroma(notes: list[dict]) -> list[float]:
    """12-bin key-dependent velocity-weighted pitch-class histogram, L2-normalized.

    Mirrors piece_id_eval.note_chroma.chroma_vector (the certified recall feature).
    """
    cv = [0.0] * 12
    for n in notes:
        cv[int(n["pitch"]) % 12] += float(n.get("velocity", 80))
    norm = math.sqrt(sum(x * x for x in cv))
    if norm > 0:
        cv = [x / norm for x in cv]
    return cv


def _piece_events(notes: list[dict], onset_tol_s: float) -> list[int]:
    """Collapse onsets within onset_tol_s into chord-events; each = a 12-bit pc-set mask.

    Mirrors piece_id_eval.stage0c_elastic_dtwgate._notes_to_events (pitch-only).
    """
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: float(n["onset_seconds"]))
    events: list[int] = []
    anchor = float(ordered[0]["onset_seconds"])
    cur = 0
    for n in ordered:
        onset = float(n["onset_seconds"])
        if onset - anchor > onset_tol_s:
            events.append(cur)
            anchor = onset
            cur = 0
        cur |= 1 << (int(n["pitch"]) % 12)
    events.append(cur)
    return events


def build_piece_index(scores_dir: Path, onset_tol_s: float = 0.05) -> dict:
    """Build the v2 piece-ID artifact (chroma vector + chord-event masks per piece)."""
    json_files = sorted(
        f for f in scores_dir.glob("*.json") if f.name not in ("titles.json", "seed.sql")
    )
    pieces: list[dict] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        notes = _collect_all_notes(data)
        pieces.append({
            "piece_id": data["piece_id"],
            "composer": data["composer"],
            "title": data["title"],
            "chroma": _piece_chroma(notes),
            "events": _piece_events(notes, onset_tol_s),
        })
    pieces.sort(key=lambda p: p["piece_id"])
    return {"version": "v2", "onset_tol_ms": int(round(onset_tol_s * 1000)), "pieces": pieces}
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_v2.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add model/src/score_library/fingerprint.py model/src/score_library/test_fingerprint_v2.py
git commit -m "feat(#26): build_piece_index emits v2 chroma+events artifact"
```

---

### Task 3: `fingerprint` CLI + `just fingerprint` emit `piece_index.json`; delete legacy builders
**Group:** A (depends on Task 2)

**Behavior being verified:** running the fingerprint command over a scores dir writes a single `piece_index.json` v2 artifact and no longer writes the legacy `ngram_index.json`/`rerank_features.json`.
**Interface under test:** `cmd_fingerprint(args)` (via the CLI argparse path).

**Files:**
- Modify: `model/src/score_library/cli.py`
- Modify: `model/src/score_library/fingerprint.py` (remove legacy builders)
- Modify: `Justfile`
- Test: `model/src/score_library/test_fingerprint_cli.py`

- [ ] **Step 1: Write the failing test**
```python
# model/src/score_library/test_fingerprint_cli.py
import json
from argparse import Namespace
from pathlib import Path

from score_library.cli import cmd_fingerprint


def test_cmd_fingerprint_writes_only_piece_index(tmp_path):
    scores = tmp_path / "scores"
    scores.mkdir()
    (scores / "x.piece.json").write_text(json.dumps({
        "piece_id": "x.piece", "composer": "X", "title": "Ex",
        "bars": [{"bar_number": 1, "notes": [
            {"pitch": 60, "onset_seconds": 0.0, "velocity": 90, "duration_seconds": 0.4},
            {"pitch": 67, "onset_seconds": 0.5, "velocity": 90, "duration_seconds": 0.4},
        ]}],
    }))
    out = tmp_path / "fp"
    cmd_fingerprint(Namespace(scores_dir=str(scores), output_dir=str(out)))

    artifact = json.loads((out / "piece_index.json").read_text())
    assert artifact["version"] == "v2"
    assert artifact["pieces"][0]["piece_id"] == "x.piece"
    assert not (out / "ngram_index.json").exists()
    assert not (out / "rerank_features.json").exists()
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_cli.py -q
```
Expected: FAIL — `AssertionError` on `piece_index.json` missing (cmd_fingerprint still writes ngram_index.json / rerank_features.json).

- [ ] **Step 3: Implement the minimum to make the test pass**
Replace `cmd_fingerprint` in `model/src/score_library/cli.py`:
```python
def cmd_fingerprint(args):
    from score_library.fingerprint import build_piece_index

    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = build_piece_index(scores_dir)
    out_path = output_dir / "piece_index.json"
    with open(out_path, "w") as f:
        json.dump(index, f)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Piece index: {len(index['pieces'])} pieces -> {out_path} ({size_kb:.1f} KB)")
```
Remove the now-unused `--max-freq` argument from the `p_fingerprint` subparser in `main()` (delete the three lines adding `--max-freq`). In `model/src/score_library/fingerprint.py` delete `extract_pitch_trigrams`, `compute_rerank_features`, `_collect_bar_pitches`, `build_ngram_index`, `build_rerank_features` (the legacy builders; `_collect_all_notes` stays). Update the `Justfile` `fingerprint` recipe comment only if it mentions ngram (the command line is unchanged):
```
# Justfile (recipe body unchanged; it already passes --scores-dir/--output-dir)
fingerprint:
    cd model && uv run python -m score_library.cli fingerprint --scores-dir data/scores --output-dir data/fingerprints
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_cli.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add model/src/score_library/cli.py model/src/score_library/fingerprint.py Justfile model/src/score_library/test_fingerprint_cli.py
git commit -m "feat(#26): fingerprint CLI emits v2 piece_index; drop legacy ngram/rerank builders"
```

---

### Task 4: Generator events match the Stage-0c reference on a real catalog piece
**Group:** A (depends on Task 2)

**Behavior being verified:** the v2 generator's per-piece `events` are byte-identical to the Stage-0c reference `_notes_to_events` masks for a real catalog score — proving the artifact the WASM consumes equals the reference the parity fixtures were frozen from.
**Interface under test:** `build_piece_index` vs `piece_id_eval.stage0c_elastic_dtwgate._notes_to_events` + `piece_id_eval.notes.load_score_notes`.

**Files:**
- Test: `model/src/score_library/test_fingerprint_reference_parity.py`

- [ ] **Step 1: Write the failing test**
```python
# model/src/score_library/test_fingerprint_reference_parity.py
from pathlib import Path

import numpy as np
import pytest

from score_library.fingerprint import build_piece_index, _piece_events
from piece_id_eval.notes import load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import _notes_to_events

_SCORES = Path(__file__).resolve().parents[2] / "data/scores"


def _ref_masks(notes) -> list[int]:
    pc_mat, _ = _notes_to_events(notes)
    out = []
    for i in range(pc_mat.shape[0]):
        m = 0
        for pc in range(12):
            if pc_mat[i, pc] > 0:
                m |= 1 << pc
        out.append(m)
    return out


@pytest.mark.skipif(not _SCORES.exists(), reason="catalog scores not present")
def test_generator_events_match_stage0c_reference():
    sample = sorted(p for p in _SCORES.glob("*.json") if p.name not in ("titles.json", "seed.sql"))[:3]
    assert sample, "no catalog scores found"
    for jf in sample:
        ref = _ref_masks(load_score_notes(jf))
        idx = build_piece_index(jf.parent)
        gen = next(p["events"] for p in idx["pieces"] if p["piece_id"] == jf.stem)
        assert gen == ref, f"event mismatch for {jf.stem}"
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES-for-the-right-reason**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_reference_parity.py -q
```
Expected: if it FAILS, the failure is a real event-encoding divergence (e.g. unsorted notes, off-by-one in the 50 ms boundary) that must be fixed in `_piece_events` until masks match `_notes_to_events`. If it PASSES immediately, confirm by temporarily perturbing `onset_tol_s` (e.g. 0.0) and re-running to see it FAIL — then revert. (This guards against a test that passes vacuously.)

- [ ] **Step 3: Implement / fix until parity holds**
If masks diverge, the cause is the onset-collapse boundary or note ordering. Align `_piece_events` to `_notes_to_events`: both sort ascending by onset, reset the anchor to the first note that exceeds `anchor + tol` (strict `>`), and OR each note's `pitch % 12` bit into the current event. No code change is expected if Task 2 was faithful; if a divergence is found, the fix lives in `_piece_events` only.

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd model && uv run pytest src/score_library/test_fingerprint_reference_parity.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add model/src/score_library/test_fingerprint_reference_parity.py model/src/score_library/fingerprint.py
git commit -m "test(#26): generator events match Stage-0c reference on real scores"
```

---

### Task 5: Rust `chroma_vector` reproduces the velocity-weighted L2 histogram
**Group:** B (first in B; parallel with Group A)

**Behavior being verified:** `chroma_vector` produces a 12-bin velocity-weighted pitch-class histogram, L2-normalized, matching `note_chroma.chroma_vector`.
**Interface under test:** `chroma::chroma_vector(&[PerfNote]) -> [f64; 12]`

**Files:**
- Create: `apps/api/src/wasm/piece-identify/src/chroma.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/lib.rs` (add `mod chroma;`)
- Test: in `chroma.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (in `chroma.rs`)
```rust
//! C1 recall: key-dependent velocity-weighted pitch-class chroma + cosine top-k.
use crate::types::PerfNote;

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
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test chroma_is_velocity_weighted
```
Expected: FAIL — `cannot find function chroma_vector in this scope`.

- [ ] **Step 3: Implement** (prepend to `chroma.rs`, above the test module; add `mod chroma;` after `mod text_match;` in `lib.rs`)
```rust
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
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test chroma_is_velocity_weighted
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/chroma.rs apps/api/src/wasm/piece-identify/src/lib.rs
git commit -m "feat(#26): rust chroma_vector (velocity-weighted, L2-normalized)"
```

---

### Task 6: Rust `rank_top_k` returns catalog indices by descending cosine
**Group:** B (depends on Task 5)

**Behavior being verified:** `rank_top_k` ranks catalog chroma vectors by cosine (dot, since L2-normalized) to the query and returns the top-k catalog indices, descending, stable on ties.
**Interface under test:** `chroma::rank_top_k(&[f64; 12], &[[f64; 12]], usize) -> Vec<usize>`

**Files:**
- Modify: `apps/api/src/wasm/piece-identify/src/chroma.rs`
- Test: in `chroma.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (add to `chroma.rs` tests)
```rust
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
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test rank_top_k_orders
```
Expected: FAIL — `cannot find function rank_top_k in this scope`.

- [ ] **Step 3: Implement** (add to `chroma.rs`)
```rust
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
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test rank_top_k_orders
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/chroma.rs
git commit -m "feat(#26): rust rank_top_k cosine recall"
```

---

### Task 7: Rust `notes_to_events` collapses onsets into 12-bit pc-set masks
**Group:** B (depends on Task 6)

**Behavior being verified:** `notes_to_events` sorts notes by onset, collapses notes within `onset_tol_s` (strict `>` boundary) into chord-events, and encodes each as a 12-bit pitch-class-set mask — matching `stage0c._notes_to_events`.
**Interface under test:** `gate::notes_to_events(&[PerfNote], f64) -> Vec<u16>`

**Files:**
- Create: `apps/api/src/wasm/piece-identify/src/gate.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/lib.rs` (add `mod gate;`)
- Test: in `gate.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (in `gate.rs`)
```rust
//! The certified open-set gate: 50ms chord-events, Jaccard subsequence-DTW, margin.
use crate::types::PerfNote;

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
}
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test notes_to_events
```
Expected: FAIL — `cannot find function notes_to_events in this scope`.

- [ ] **Step 3: Implement** (prepend to `gate.rs`; add `mod gate;` after `mod chroma;` in `lib.rs`)
```rust
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
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test notes_to_events
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/gate.rs apps/api/src/wasm/piece-identify/src/lib.rs
git commit -m "feat(#26): rust notes_to_events (50ms chord-event masks)"
```

---

### Task 8: Rust `elastic_cost` — Jaccard subsequence-DTW (shorter on rows, normalize by shorter)
**Group:** B (depends on Task 7)

**Behavior being verified:** `elastic_cost` computes the subsequence-DTW cost with local cost = Jaccard distance over pc-set masks, embedding the SHORTER event sequence as rows (free start/end on the longer) and normalizing by the shorter length — reproducing librosa `dtw(subseq=True)` as used in `stage0c._elastic_cost` with `w_time=0`.
**Interface under test:** `gate::elastic_cost(&[u16], &[u16]) -> f64`

**Files:**
- Modify: `apps/api/src/wasm/piece-identify/src/gate.rs`
- Test: in `gate.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (add to `gate.rs` tests)
```rust
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
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test elastic_cost
```
Expected: FAIL — `cannot find function elastic_cost in this scope`.

- [ ] **Step 3: Implement** (add to `gate.rs`)
```rust
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
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test elastic_cost
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/gate.rs
git commit -m "feat(#26): rust Jaccard subsequence-DTW elastic_cost"
```

---

### Task 9: Rust `margin_gate` — lock iff (2nd-best − best) ≥ threshold
**Group:** B (depends on Task 8)

**Behavior being verified:** `margin_gate` computes each candidate's elastic cost, the margin between the best and 2nd-best, the index of the best candidate, and whether `margin ≥ threshold`.
**Interface under test:** `gate::margin_gate(&[u16], &[&[u16]], f64) -> Option<GateDecision>`

**Files:**
- Modify: `apps/api/src/wasm/piece-identify/src/gate.rs`
- Test: in `gate.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (add to `gate.rs` tests)
```rust
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
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test margin_gate
```
Expected: FAIL — `cannot find function margin_gate` / `cannot find type GateDecision`.

- [ ] **Step 3: Implement** (add to `gate.rs`)
```rust
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
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test margin_gate
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/gate.rs
git commit -m "feat(#26): rust margin_gate open-set decision"
```

---

### Task 10: `run_identify` orchestrates recall → gate over the artifact
**Group:** B (depends on Task 9)

**Behavior being verified:** `run_identify` parses the artifact, runs chroma recall (top-5) and the margin gate over the query notes, and returns the best piece's identity + margin + lock decision, or `None` when the query has < 2 events or the artifact has < 2 pieces.
**Interface under test:** `identify::run_identify(&[PerfNote], &PieceIndex, f64) -> Option<IdentifyResult>`

**Files:**
- Create: `apps/api/src/wasm/piece-identify/src/identify.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/types.rs` (add `PieceArtifact`, `PieceIndex`, `IdentifyResult`)
- Modify: `apps/api/src/wasm/piece-identify/src/lib.rs` (add `mod identify;`)
- Test: in `identify.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (in `identify.rs`)
```rust
//! Orchestrates the certified pipeline: chroma recall (top-5) -> margin gate.
use crate::chroma;
use crate::gate;
use crate::types::{IdentifyResult, PieceArtifact, PieceIndex};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PerfNote;

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
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test run_identify
```
Expected: FAIL — `cannot find function run_identify` / unresolved `PieceIndex` import.

- [ ] **Step 3: Implement**
Add to `apps/api/src/wasm/piece-identify/src/types.rs`:
```rust
/// One catalog piece in the v2 artifact: chroma recall vector + chord-event masks.
#[derive(Serialize, Deserialize)]
pub struct PieceArtifact {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub chroma: [f64; 12],
    pub events: Vec<u16>,
}

/// The v2 piece-ID artifact loaded from R2 (`fingerprint/v2/piece_index.json`).
#[derive(Serialize, Deserialize)]
pub struct PieceIndex {
    pub onset_tol_ms: f64,
    pub pieces: Vec<PieceArtifact>,
}

/// Result of identify_piece (marshaled to JS).
#[derive(Serialize, Deserialize)]
pub struct IdentifyResult {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub margin: f64,
    pub locked: bool,
}
```
Create `apps/api/src/wasm/piece-identify/src/identify.rs` (below the `use` lines from Step 1):
```rust
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
```
Add `mod identify;` after `mod gate;` in `lib.rs`. Ensure `types.rs` has `use serde::{Serialize, Deserialize};` at the top (it already derives Serialize/Deserialize for existing types — reuse the existing import).

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test run_identify
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/identify.rs apps/api/src/wasm/piece-identify/src/types.rs apps/api/src/wasm/piece-identify/src/lib.rs
git commit -m "feat(#26): run_identify orchestrates chroma recall + margin gate"
```

---

### Task 11: `identify_piece` WASM export; delete the legacy pipeline; wire `just test-piece-id`
**Group:** B (depends on Task 10; last in B)

**Behavior being verified:** the crate exposes a single `identify_piece(notes_js, artifact_json, margin_threshold)` WASM export, the legacy 3-stage exports/modules are gone, `text_match` is retained, and the crate builds for both native test and the `bundler` WASM target.
**Interface under test:** `#[wasm_bindgen] identify_piece` (compiled via `cargo build` + `wasm-pack build`); native `lib`-level smoke through `run_identify`.

**Files:**
- Modify: `apps/api/src/wasm/piece-identify/src/lib.rs` (add export; remove `ngram_recall`/`compute_rerank_features`/`rerank_candidates`/`dtw_confirm` exports + `mod ngram`/`mod rerank`/`mod dtw_confirm`/`mod real_recording_test`)
- Delete: `apps/api/src/wasm/piece-identify/src/ngram.rs`, `rerank.rs`, `dtw_confirm.rs`, `real_recording_test.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/types.rs` (remove `NgramIndex`, `RerankFeatures`, `NgramCandidate`, `RerankResult`, `DtwConfirmResult`; keep `PerfNote`, `CatalogEntry`, `TextMatchResult`, plus the new Task-10 types)
- Modify: `Justfile` (add `test-piece-id`)
- Test: in `lib.rs` `#[cfg(test)]`

- [ ] **Step 1: Write the failing test** (add a `#[cfg(test)]` block at the bottom of `lib.rs`)
```rust
#[cfg(test)]
mod lib_tests {
    use crate::identify::run_identify;
    use crate::types::{PerfNote, PieceArtifact, PieceIndex};

    #[test]
    fn crate_exposes_identify_pipeline() {
        let notes = vec![
            PerfNote { pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 },
            PerfNote { pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 },
            PerfNote { pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 },
        ];
        let p = |id: &str, events: Vec<u16>, chroma: [f64; 12]| PieceArtifact {
            piece_id: id.into(), composer: "C".into(), title: id.into(), chroma, events,
        };
        let index = PieceIndex {
            onset_tol_ms: 50.0,
            pieces: vec![p("a", vec![1, 2, 4], [0.1; 12]), p("b", vec![16, 32, 64], [0.0; 12])],
        };
        // Smoke: the orchestrator is reachable from the crate root and returns a decision.
        assert!(run_identify(&notes, &index, 0.0935).is_some());
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api/src/wasm/piece-identify && cargo test crate_exposes_identify_pipeline
```
Expected: FAIL — compile error: `lib.rs` still declares `mod ngram; mod rerank; mod dtw_confirm; mod real_recording_test;` whose files reference deleted types, and `identify_piece` is not yet exported (or the legacy modules still reference removed `types`). The test cannot compile until the legacy surface is removed.

- [ ] **Step 3: Implement**
Rewrite `apps/api/src/wasm/piece-identify/src/lib.rs` module section and exports so the file reads:
```rust
//! piece-identify: the certified open-set piece identification gate
//! (chroma recall -> pitch-only chord-Jaccard elastic-DTW margin gate) plus
//! free-text query matching, for use from TypeScript workers.

use wasm_bindgen::prelude::*;

mod chroma;
mod gate;
mod identify;
mod text_match;
mod types;

#[cfg(test)]
mod parity_test;

/// Identify the piece for the accumulated performance notes against the v2
/// catalog artifact. Locks only when the margin gate clears `margin_threshold`.
///
/// `notes_js`: `Array<{ pitch, onset, offset, velocity }>` (the accumulated buffer)
/// `artifact_json`: the raw `fingerprint/v2/piece_index.json` text
/// `margin_threshold`: certified value 0.0935
///
/// Returns `{ piece_id, composer, title, margin, locked } | null`.
#[wasm_bindgen]
pub fn identify_piece(
    notes_js: JsValue,
    artifact_json: &str,
    margin_threshold: f64,
) -> Result<JsValue, JsValue> {
    let notes: Vec<types::PerfNote> =
        serde_wasm_bindgen::from_value(notes_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let index: types::PieceIndex =
        serde_json::from_str(artifact_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    match identify::run_identify(&notes, &index, margin_threshold) {
        Some(result) => {
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        }
        None => Ok(JsValue::NULL),
    }
}

/// Match a free-text query against the piece catalog using Dice similarity on bigrams.
/// `catalog_js`: `Array<{ piece_id, composer, title }>`. Returns `{ piece_id, confidence } | null`.
#[wasm_bindgen]
pub fn match_piece_text(query: &str, catalog_js: JsValue) -> Result<JsValue, JsValue> {
    let catalog: Vec<types::CatalogEntry> = serde_wasm_bindgen::from_value(catalog_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = text_match::match_piece_text(query, &catalog);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
```
Then append the `#[cfg(test)] mod lib_tests` block from Step 1. Delete the four legacy files. In `types.rs`, delete `NgramIndex`, `RerankFeatures`, `NgramCandidate`, `RerankResult`, `DtwConfirmResult` (keep `PerfNote`, `CatalogEntry`, `TextMatchResult`, and the Task-10 types). Add the `test-piece-id` recipe to the `Justfile`:
```
# Run the piece-identify Rust unit + parity tests (native target).
test-piece-id:
    cd apps/api/src/wasm/piece-identify && cargo test
```
NOTE: `parity_test.rs` is referenced by `mod parity_test;` but created in Task 12. Until Task 12 lands, temporarily guard it: create an empty `apps/api/src/wasm/piece-identify/src/parity_test.rs` containing only `// populated in Task 12` so the crate compiles. (Task 12 overwrites it.)

- [ ] **Step 4: Run test — verify it PASSES + crate builds for WASM**
```bash
cd apps/api/src/wasm/piece-identify && cargo test crate_exposes_identify_pipeline
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run build:wasm
```
Expected: PASS, and `build:wasm` regenerates `apps/api/src/wasm/piece-identify/pkg/` with `identify_piece` + `match_piece_text` exports (and no `ngram_recall`).

- [ ] **Step 5: Commit**
```bash
cd /Users/jdhiman/Documents/crescendai
git add apps/api/src/wasm/piece-identify/src/lib.rs apps/api/src/wasm/piece-identify/src/types.rs apps/api/src/wasm/piece-identify/src/parity_test.rs Justfile
git rm apps/api/src/wasm/piece-identify/src/ngram.rs apps/api/src/wasm/piece-identify/src/rerank.rs apps/api/src/wasm/piece-identify/src/dtw_confirm.rs apps/api/src/wasm/piece-identify/src/real_recording_test.rs
git add apps/api/src/wasm/piece-identify/pkg
git commit -m "feat(#26): identify_piece WASM export; delete legacy 3-stage pipeline"
```

---

### Task 12: PORT-FIDELITY — Rust gate reproduces the Python reference on the golden fixtures
**Group:** C (depends on Task 1 + Task 11) — **the primary correctness gate**

**Behavior being verified:** for every golden fixture query, the Rust `elastic_cost` reproduces each candidate's Python cost within 1e-4, and `margin_gate` reproduces the best piece, the margin (within 1e-4), and the lock decision; in aggregate, in-catalog queries lock at the certified rate (≥14/16) and OOD queries never lock.
**Interface under test:** `gate::elastic_cost`, `gate::margin_gate` against `model/data/evals/piece_id/parity_fixtures.json`.

**Files:**
- Modify (overwrite the Task-11 placeholder): `apps/api/src/wasm/piece-identify/src/parity_test.rs`

- [ ] **Step 1: Write the failing test** (overwrite `parity_test.rs`)
```rust
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
```

- [ ] **Step 2: Run test — verify it FAILS (for the right reason if at all)**
```bash
cd apps/api/src/wasm/piece-identify && cargo test --test-threads=1 rust_
```
Expected: the test compiles and runs. If `rust_elastic_cost_matches_python_per_candidate` FAILS, it is a genuine port-fidelity gap (DTW transpose/normalize/boundary divergence) that must be fixed in `gate.rs` until costs match within 1e-4. If all three PASS immediately, the port is faithful — confirm by temporarily breaking `jaccard_dist` (e.g. return `inter/union` instead of `1 - inter/union`) and re-running to see the cost test FAIL, then revert.

- [ ] **Step 3: Implement / fix until parity holds**
No new feature code is expected if Tasks 7–9 were faithful. If the per-candidate cost test fails, the divergence is in `elastic_cost` — check (a) shorter-on-rows transpose, (b) normalize by shorter length, (c) row-0 free-start vs column-0 accumulation, (d) the 3-direction `min`. Adjust `gate::elastic_cost` only; re-run until `rust_elastic_cost_matches_python_per_candidate` passes within 1e-4.

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api/src/wasm/piece-identify && cargo test
```
Expected: PASS (all crate tests, including the three parity tests).

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/wasm/piece-identify/src/parity_test.rs apps/api/src/wasm/piece-identify/src/gate.rs
git commit -m "test(#26): port-fidelity parity gate (rust reproduces python reference)"
```

---

### Task 13: `wasm-bridge.ts` exposes `identifyPiece`; remove legacy wrappers
**Group:** D (depends on Task 11)

**Behavior being verified:** through the real WASM in workerd, `identifyPiece` locks to the correct piece on an in-catalog fixture and returns `locked: false` (or a non-locking result) on an out-of-catalog query, via the typed bridge.
**Interface under test:** `wasm-bridge.identifyPiece(notes, artifactJson, threshold)`

**Files:**
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Modify: `apps/api/src/services/wasm-bridge.workerd.test.ts`
- Modify: `apps/api/src/services/wasm-bridge.test.ts`

- [ ] **Step 1: Write the failing test** (replace the `ngramRecall` block in `wasm-bridge.workerd.test.ts`)
```typescript
// apps/api/src/services/wasm-bridge.workerd.test.ts
import { describe, it, expect } from "vitest";
import { identifyPiece, type PerfNote } from "./wasm-bridge";

describe("identifyPiece (real WASM)", () => {
  // Minimal v2 artifact: two pieces. "exact" shares the query's chord-events; "decoy" is disjoint.
  const artifact = JSON.stringify({
    version: "v2",
    onset_tol_ms: 50,
    pieces: [
      { piece_id: "decoy", composer: "X", title: "Decoy", chroma: new Array(12).fill(0), events: [16, 32, 64, 128] },
      { piece_id: "exact", composer: "Y", title: "Exact", chroma: (() => { const a = new Array(12).fill(0); a[0] = 0.5; a[4] = 0.5; a[7] = 0.5; return a; })(), events: [1, 16, 128, 1] },
    ],
  });
  const notes: PerfNote[] = [
    { pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 }, // C  -> bit 0
    { pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 }, // E  -> bit 4
    { pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 }, // G  -> bit 7
    { pitch: 72, onset: 1.5, offset: 1.9, velocity: 100 }, // C  -> bit 0 (+12 octave) and a 2nd C event
  ];

  it("locks to the matching piece", () => {
    const r = identifyPiece(notes, artifact, 0.0935);
    expect(r).not.toBeNull();
    expect(r!.piece_id).toBe("exact");
    expect(r!.locked).toBe(true);
  });

  it("returns null when the artifact has fewer than two pieces", () => {
    const tiny = JSON.stringify({ version: "v2", onset_tol_ms: 50, pieces: [{ piece_id: "only", composer: "X", title: "Only", chroma: new Array(12).fill(0), events: [1, 2] }] });
    expect(identifyPiece(notes, tiny, 0.0935)).toBeNull();
  });
});
```
(The `events` values in the artifact must be the masks `notes_to_events` produces for the query so `exact` aligns at ~0 cost; the build agent computes them from the query: C={1}, E={16}, G={128}, C={1} → after 50ms collapse the four distinct onsets give `[1,16,128,1]`. Use exactly the masks the gate yields for `notes`; if the arpeggio onsets differ, recompute. The decoy is disjoint → margin large → lock to `exact`.)

- [ ] **Step 2: Run test — verify it FAILS**
```bash
cd apps/api && bun run build:wasm && bun run test -- --run wasm-bridge.workerd
```
Expected: FAIL — `identifyPiece is not exported from ./wasm-bridge`.

- [ ] **Step 3: Implement** — in `apps/api/src/services/wasm-bridge.ts`:
Remove the `ngramRecall`, `rerankCandidates`, `dtwConfirm` wrappers and the `NgramCandidate`/`RerankResult`/`DtwConfirmResult`/`NgramIndex`/`RerankFeatures` interfaces. Add:
```typescript
/** Result of identify_piece. null when no confident identification is possible. */
export interface IdentifyResult {
  piece_id: string;
  composer: string;
  title: string;
  margin: number;
  locked: boolean;
}

/**
 * Identify the piece for the accumulated performance notes against the v2 catalog
 * artifact (raw `fingerprint/v2/piece_index.json` text). Locks only when the margin
 * gate clears `marginThreshold` (certified 0.0935); otherwise returns a non-locking
 * result or null. Runs chroma recall + the elastic-DTW margin gate inside WASM.
 */
export function identifyPiece(
  notes: PerfNote[],
  artifactJson: string,
  marginThreshold = 0.0935,
): IdentifyResult | null {
  return pieceIdentifyModule.identify_piece(
    notes,
    artifactJson,
    marginThreshold,
  ) as IdentifyResult | null;
}
```
In `wasm-bridge.test.ts` (node, mocked): replace the `vi.mock` of `piece_identify` to export `identify_piece: mockIdentifyPiece` (drop the three legacy mocks) and replace the `ngramRecall` describe block with:
```typescript
describe("identifyPiece", () => {
  it("forwards notes, artifact JSON, and threshold to identify_piece", async () => {
    const { identifyPiece } = await import("./wasm-bridge");
    mockIdentifyPiece.mockReturnValue({ piece_id: "p", composer: "c", title: "t", margin: 0.2, locked: true });
    const notes = [{ pitch: 60, onset: 0, offset: 0.5, velocity: 80 }];
    identifyPiece(notes, '{"version":"v2","onset_tol_ms":50,"pieces":[]}', 0.0935);
    expect(mockIdentifyPiece).toHaveBeenCalledWith(notes, '{"version":"v2","onset_tol_ms":50,"pieces":[]}', 0.0935);
  });
});
```

- [ ] **Step 4: Run test — verify it PASSES**
```bash
cd apps/api && bun run test -- --run wasm-bridge.workerd && bun run test:scripts -- --run wasm-bridge.test
```
Expected: PASS (both the real-WASM workerd test and the mocked node test).

- [ ] **Step 5: Commit**
```bash
git add apps/api/src/services/wasm-bridge.ts apps/api/src/services/wasm-bridge.workerd.test.ts apps/api/src/services/wasm-bridge.test.ts
git commit -m "feat(#26): wasm-bridge identifyPiece; drop legacy ngram/rerank/dtw wrappers"
```

---

### Task 14: `just seed-fingerprint` lands the v2 artifact in local R2
**Group:** E (depends on Task 3 + Task 11)

**Behavior being verified:** running the generator then `just seed-fingerprint` puts `fingerprint/v2/piece_index.json` into the local `wrangler` R2 state so `wrangler dev` can serve it.
**Interface under test:** the `Justfile` `seed-fingerprint` recipe + local R2 (`wrangler r2 object get ... --local`).

**Files:**
- Modify: `Justfile`

- [ ] **Step 1: Write the failing check** (the verification is a shell assertion; record it as the test)
```bash
# Verification command (run after Step 3). Must succeed and print the object.
cd apps/api && wrangler r2 object get "crescendai-bucket/fingerprint/v2/piece_index.json" --local --pipe | head -c 32
```
Expected before Step 3: FAIL — `The specified key does not exist` (no `seed-fingerprint` recipe has run; the object is absent from local R2).

- [ ] **Step 2: Generate the artifact**
```bash
just fingerprint   # writes model/data/fingerprints/piece_index.json (Task 3)
```

- [ ] **Step 3: Implement the recipe** — add to `Justfile`:
```
# Seed the v2 piece-ID artifact into LOCAL wrangler R2 for `wrangler dev`.
# Run `just fingerprint` first to produce model/data/fingerprints/piece_index.json.
seed-fingerprint:
    cd apps/api && wrangler r2 object put "crescendai-bucket/fingerprint/v2/piece_index.json" \
        --file="../../model/data/fingerprints/piece_index.json" --local
```
Then run it:
```bash
just seed-fingerprint
```

- [ ] **Step 4: Run the verification — confirm it SUCCEEDS**
```bash
cd apps/api && wrangler r2 object get "crescendai-bucket/fingerprint/v2/piece_index.json" --local --pipe | head -c 32
```
Expected: prints the leading bytes of the artifact JSON (e.g. `{"version": "v2", "onset_tol_ms"`), confirming the object is in local R2.
Manual click-through (the local-first "done" bar): `just dev-light`, start a practice session in the web app, and confirm no console/Sentry error from the piece-ID load path. (The DEFERRED slice wires the DO call; until it lands, this confirms only that the artifact is servable.)

- [ ] **Step 5: Commit**
```bash
git add Justfile
git commit -m "chore(#26): seed-fingerprint recipe lands v2 artifact in local R2"
```

---

## Plan self-review notes
- **Spec coverage:** chroma recall (T5,6), events (T7), Jaccard DTW (T8), margin gate (T9), orchestration (T10), WASM export + legacy deletion (T11), artifact generator (T2,3), generator↔reference parity (T4), port-fidelity (T12), bridge (T13), local R2 (T14), harness (T1). The DEFERRED `session-brain.ts` rewire is intentionally excluded (BLOCKED-ON-#28) and documented in the spec.
- **Type consistency:** `PerfNote`/`PieceArtifact`/`PieceIndex`/`IdentifyResult`/`GateDecision` names are identical across Rust tasks; `identifyPiece`/`IdentifyResult` identical across TS tasks; `build_piece_index`/`_piece_events`/`_piece_chroma` identical across Python tasks; artifact keys (`version`,`onset_tol_ms`,`pieces`,`piece_id`,`composer`,`title`,`chroma`,`events`) identical across generator, Rust, bridge, fixtures.
- **Parallel-group file safety:** Group A touches `model/**` only; Group B touches `apps/api/src/wasm/**` only — disjoint, safe to run concurrently. Within B, tasks share `lib.rs`/`types.rs` so they are sequential. Group C/D/E each depend on B completing.
- **Vertical-slice check:** every task = one failing test → one implementation → one passing run → one commit. Tasks 4 and 12 add an explicit anti-vacuous-pass guard (perturb-and-revert) because they assert parity against a reference that a faithful prior task may already satisfy.
- **Behavior-through-public-interface check:** all tests exercise public functions (`chroma_vector`, `rank_top_k`, `notes_to_events`, `elastic_cost`, `margin_gate`, `run_identify`, `identify_piece`, `identifyPiece`, `build_piece_index`, `cmd_fingerprint`) or committed artifacts; none mock internal collaborators or assert on private state. The node-mocked `wasm-bridge.test.ts` forwarding test is the established bridge pattern (the real behavior is covered by the workerd test in the same task).
