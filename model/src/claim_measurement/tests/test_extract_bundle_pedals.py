"""Unit test (no live AMT server): extract_bundle populates bundle['pedal_events'].

Mocks the AMT/score seams so the test never touches the network or a real score,
and asserts the pedal events returned by the transcription path land in the bundle
(previously hardcoded to []).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from chroma_dtw_eval.amt_regen import AmtRegenError
from claim_measurement import extractor
from claim_measurement.extractor import BundleExtractionError


def test_extract_bundle_populates_pedal_events(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "vid.wav"
    audio.write_bytes(b"stub")
    score = tmp_path / "score.json"
    score.write_text("{}")

    fake_notes = [{"onset": 0.0, "offset": 0.1, "pitch": 60, "velocity": 80}]
    fake_pedals = [{"time": 1.0, "value": 127}, {"time": 2.0, "value": 0}]
    score_na = np.zeros(1, dtype=[("onset_sec", float)])
    measure_table = [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}]

    monkeypatch.setattr(extractor, "_read_wav_16k_mono",
                        lambda p: np.zeros(10, dtype=np.float32))
    monkeypatch.setattr(extractor, "_transcribe_clip_with_pedals",
                        lambda a, u: (fake_notes, fake_pedals))
    monkeypatch.setattr(extractor, "_dedup_amt_notes", lambda notes: notes)
    monkeypatch.setattr(extractor, "_load_bach_json_score",
                        lambda p: (score_na, measure_table, "deadbeef", 0.5))
    monkeypatch.setattr(extractor, "_amt_to_perf_na", lambda notes, bs: np.zeros(1))
    monkeypatch.setattr(extractor, "_match", lambda s, p: [])
    monkeypatch.setattr(extractor, "_build_pairs",
                        lambda s, p, m: (np.array([0.0]), np.array([0.0])))

    out_path = extractor.extract_bundle(
        "piece_x", "vid",
        audio_path=audio, score_path=score,
        cache_root=tmp_path, bundle_root=tmp_path / "bundles",
        force=True,
    )
    bundle = json.loads(out_path.read_text())
    assert bundle["pedal_events"] == fake_pedals


def test_amt_failure_becomes_bundle_extraction_error(tmp_path, monkeypatch) -> None:
    """An AmtRegenError from transcription/alignment (e.g. AMT server down) must
    surface as BundleExtractionError so a batch runner records a per-clip failure
    instead of crashing the whole run."""
    audio = tmp_path / "vid.wav"
    audio.write_bytes(b"stub")
    score = tmp_path / "score.json"
    score.write_text("{}")

    monkeypatch.setattr(extractor, "_read_wav_16k_mono",
                        lambda p: np.zeros(10, dtype=np.float32))

    def _boom(a, u):
        raise AmtRegenError("AMT POST failed after 3 attempts: Connection refused")

    monkeypatch.setattr(extractor, "_transcribe_clip_with_pedals", _boom)

    with pytest.raises(BundleExtractionError):
        extractor.extract_bundle(
            "piece_x", "vid",
            audio_path=audio, score_path=score,
            cache_root=tmp_path, bundle_root=tmp_path / "bundles",
            force=True,
        )
