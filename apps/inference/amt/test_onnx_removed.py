"""Run: cd apps/inference/amt && uv run --with pytest pytest test_onnx_removed.py"""
from __future__ import annotations

from pathlib import Path

AMT = Path(__file__).resolve().parent
REPO = AMT.parents[2]


def test_export_onnx_deleted():
    assert not (AMT / "scripts/export_onnx.py").exists()


def test_dockerfile_has_no_onnx():
    df = (AMT / "Dockerfile").read_text()
    assert "export_onnx" not in df
    assert "onnxruntime" not in df


def test_audio_chunker_preserved():
    # audio_chunker has live importers (MuQ path) and must NOT be deleted.
    assert (REPO / "apps/inference/audio_chunker.py").exists()
