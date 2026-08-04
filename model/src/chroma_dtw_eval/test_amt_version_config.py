"""Run: cd model && uv run --no-project --with pytest pytest \
src/chroma_dtw_eval/test_amt_version_config.py"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config/amt_version.json"


def test_amt_version_names_transkun():
    body = json.loads(CONFIG.read_text())
    assert body["model_name"] == "transkun"
    assert body["regen_source_default"] == "local:transkun"
    assert "transkun" in body["checkpoint_hash"]
