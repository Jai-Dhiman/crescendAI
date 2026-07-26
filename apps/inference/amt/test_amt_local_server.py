"""Run: cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy \
        --with soundfile --with pretty_midi --with pytest pytest test_amt_local_server.py"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt_local_server


def test_health_reports_transkun_before_model_load():
    client = TestClient(amt_local_server.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "transkun"
    assert body["loaded"] is False  # _handler not initialized in-process
