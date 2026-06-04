"""Smoke test: CLI on toy fixture data exits 0 and prints a VERDICT line."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[2]


def _make_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal fixture: one piece, one recording, score JSON, piece map."""
    # Score JSON
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    score = {
        "piece_id": "test.piece.1",
        "bars": [
            {
                "bar_number": 1,
                "start_seconds": 0.0,
                "notes": [
                    {"pitch": 60, "onset_seconds": float(i) * 0.25, "duration_seconds": 0.25}
                    for i in range(16)
                ],
            }
        ],
    }
    (scores_dir / "test.piece.1.json").write_text(json.dumps(score))

    # Piece map
    evals_dir = tmp_path / "evals" / "piece_id"
    evals_dir.mkdir(parents=True)
    piece_map = {"test_slug": "test.piece.1"}
    (evals_dir / "eval_piece_map.json").write_text(json.dumps(piece_map))

    # candidates.yaml
    eval_root = tmp_path / "practice_eval"
    piece_dir = eval_root / "test_slug"
    piece_dir.mkdir(parents=True)
    (piece_dir / "candidates.yaml").write_text("""\
piece: test_slug
title: Test Piece
composer: Test
recordings:
- video_id: testvid001
  title: Test Recording
  channel: Test Channel
  duration_seconds: 6
  view_count: 100
  url: https://youtube.com/watch?v=testvid001
  query_source: test
  approved: true
  review_notes: ''
""")

    # Audio WAV
    audio_dir = piece_dir / "audio"
    audio_dir.mkdir()
    sr = 16000
    t = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
    y = (np.sin(2 * np.pi * 261.63 * t) * 0.5).astype(np.float32)
    sf.write(audio_dir / "testvid001.wav", y, sr)

    return tmp_path, eval_root


def test_cli_smoke_exits_zero_and_prints_verdict(tmp_path: Path) -> None:
    fixture_root, eval_root = _make_fixture_tree(tmp_path)
    sidecar = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "piece_id_eval.cli",
            "--slugs", "test_slug",
            "--eval-root", str(eval_root),
            "--scores-dir", str(fixture_root / "scores"),
            "--piece-map", str(fixture_root / "evals" / "piece_id" / "eval_piece_map.json"),
            "--sidecar", str(sidecar),
            "--no-track",
            "--window-seconds", "2.0",
            "--hop-seconds", "1.0",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    assert "VERDICT:" in result.stdout, f"no VERDICT in stdout: {result.stdout!r}"
    assert sidecar.exists(), "sidecar JSON not written"
    sidecar_data = json.loads(sidecar.read_text())
    assert "verdict" in sidecar_data, f"sidecar missing 'verdict' key: {sidecar_data}"
    assert sidecar_data["verdict"] in ("KILL", "TUNE", "PROCEED")


def test_cli_smoke_sidecar_has_matcher_results(tmp_path: Path) -> None:
    fixture_root, eval_root = _make_fixture_tree(tmp_path)
    sidecar = tmp_path / "result2.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "piece_id_eval.cli",
            "--slugs", "test_slug",
            "--eval-root", str(eval_root),
            "--scores-dir", str(fixture_root / "scores"),
            "--piece-map", str(fixture_root / "evals" / "piece_id" / "eval_piece_map.json"),
            "--sidecar", str(sidecar),
            "--no-track",
            "--window-seconds", "2.0",
            "--hop-seconds", "1.0",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    sidecar_data = json.loads(sidecar.read_text())
    assert "matchers" in sidecar_data
    assert len(sidecar_data["matchers"]) >= 1
    for m in sidecar_data["matchers"]:
        assert "name" in m
        assert "recall_at_10" in m
        assert "mrr" in m
