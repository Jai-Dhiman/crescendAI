"""Verifies ModelRunner adapter contract: HTTP POST + parse response."""
from pathlib import Path
import httpx
import pytest
from content_engine.adapters.model_runner import ModelRunner, InferenceError


def test_run_posts_audio_and_returns_parsed_output(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["content_len"] = len(kwargs.get("content", b""))
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "scores": {
                    "dynamics": [0.4, 0.5],
                    "timing": [0.6, 0.55],
                    "pedaling": [0.5, 0.5],
                    "articulation": [0.5, 0.5],
                    "phrasing": [0.4, 0.45],
                    "interpretation": [0.5, 0.5],
                },
                "duration_sec": 15.0,
            },
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    runner = ModelRunner(url="https://infer.example/", token="hf_xyz")
    output = runner.run(audio)

    assert captured["url"] == "https://infer.example/"
    assert captured["headers"]["Authorization"] == "Bearer hf_xyz"
    assert output.duration_sec == 15.0
    assert "phrasing" in output.scores
    assert output.scores["phrasing"] == [0.4, 0.45]


def test_run_raises_inference_error_on_5xx(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    def fake_post(url, **kwargs):
        return httpx.Response(503, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    runner = ModelRunner(url="https://infer.example/", token="t")
    with pytest.raises(InferenceError):
        runner.run(audio)
