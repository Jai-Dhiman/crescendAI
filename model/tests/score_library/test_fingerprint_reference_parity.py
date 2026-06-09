import json
from pathlib import Path

import pytest

from score_library.fingerprint import build_piece_index
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
        piece_id = json.loads(jf.read_text())["piece_id"]
        ref = _ref_masks(load_score_notes(jf))
        idx = build_piece_index(jf.parent)
        gen = next(p["events"] for p in idx["pieces"] if p["piece_id"] == piece_id)
        assert gen == ref, f"event mismatch for {piece_id}"
