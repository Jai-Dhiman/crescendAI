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
