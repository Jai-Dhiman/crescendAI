"""RecordedResponseClient: offline replay through the ProbeClient contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_teacher.client import RecordedResponseClient
from audio_teacher.manifest import ContrastPair


def _pair(pair_id: str) -> ContrastPair:
    return ContrastPair(
        pair_id=pair_id,
        axis="pedaling",
        population="real",
        clip_a=Path("clips/a.wav"),
        clip_b=Path("clips/b.wav"),
        degraded="a",
        description="",
    )


def test_replays_recorded_response_and_errors_on_missing_pair(tmp_path):
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(json.dumps({"pair_id": "p1", "text": "ANSWER: A"}) + "\n")
    client = RecordedResponseClient(recorded)

    assert client.estimate_cost_usd(_pair("p1")) == 0.0
    resp = client.ask(_pair("p1"))
    assert resp.pair_id == "p1"
    assert resp.text == "ANSWER: A"
    assert resp.cost_usd == 0.0

    with pytest.raises(KeyError) as excinfo:
        client.ask(_pair("p9"))
    assert "p9" in str(excinfo.value)


def test_duplicate_pair_id_in_recording_fails_loudly(tmp_path):
    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(
        json.dumps({"pair_id": "p1", "text": "ANSWER: A"}) + "\n"
        + json.dumps({"pair_id": "p1", "text": "ANSWER: B"}) + "\n"
    )
    with pytest.raises(ValueError) as excinfo:
        RecordedResponseClient(recorded)
    assert "p1" in str(excinfo.value)
