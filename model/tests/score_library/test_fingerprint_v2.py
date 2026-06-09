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
