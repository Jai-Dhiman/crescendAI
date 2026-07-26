"""Run: cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy \
        --with soundfile --with pretty_midi --with pytest pytest test_server.py"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server


def test_health_shape_no_onnx():
    # Do not trigger lifespan model load; hit the route function directly.
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "inference_count" in body
    # ONNX globals must be gone from the module surface.
    assert not hasattr(server, "_encoder_onnx")
