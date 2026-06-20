"""Unit tests (no live AMT server) for pedal-event threading.

The AMT server emits pedal_events alongside midi_notes (server.py:485-487), but
chroma_dtw_eval._transcribe_clip historically read only body["midi_notes"]. These
tests pin the sibling that also threads pedal_events, with the same per-chunk time
offset that notes receive, and verify the extractor populates bundle["pedal_events"].
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from chroma_dtw_eval import amt_regen


def _two_chunk_audio() -> np.ndarray:
    # 1.5 chunks -> n_chunks = ceil(1.5) = 2, so the second chunk gets a nonzero offset.
    n = int(amt_regen.AMT_CHUNK_S * amt_regen.TARGET_SR * 1.5)
    return np.zeros(n, dtype=np.float32)


def _mock_bodies() -> list[dict]:
    return [
        {
            "midi_notes": [{"onset": 1.0, "offset": 1.5, "pitch": 60, "velocity": 80}],
            "pedal_events": [{"time": 2.0, "value": 127}, {"time": 3.0, "value": 0}],
        },
        {
            "midi_notes": [{"onset": 0.5, "offset": 0.9, "pitch": 62, "velocity": 70}],
            "pedal_events": [{"time": 1.0, "value": 127}],
        },
    ]


def test_transcribe_clip_with_pedals_threads_and_offsets() -> None:
    audio = _two_chunk_audio()
    with patch.object(amt_regen, "_post_chunk", side_effect=_mock_bodies()):
        notes, pedals = amt_regen._transcribe_clip_with_pedals(audio, "http://x")

    # Chunk 0 offset = 0; chunk 1 offset = AMT_CHUNK_S (27.0s).
    assert pedals == [
        {"time": 2.0, "value": 127},
        {"time": 3.0, "value": 0},
        {"time": amt_regen.AMT_CHUNK_S + 1.0, "value": 127},
    ]
    assert [n["pitch"] for n in notes] == [60, 62]


def test_transcribe_clip_notes_only_backward_compatible() -> None:
    audio = _two_chunk_audio()
    with patch.object(amt_regen, "_post_chunk", side_effect=_mock_bodies()):
        notes = amt_regen._transcribe_clip(audio, "http://x")
    # Existing callers expect a flat list of note dicts; signature unchanged.
    assert [n["pitch"] for n in notes] == [60, 62]
    assert all(set(n.keys()) == {"onset", "offset", "pitch", "velocity"} for n in notes)


def test_transcribe_clip_with_pedals_tolerates_missing_pedal_key() -> None:
    # A chunk body without pedal_events (older server / error path) yields no pedals,
    # not a KeyError.
    audio = np.zeros(int(amt_regen.AMT_CHUNK_S * amt_regen.TARGET_SR), dtype=np.float32)
    body = [{"midi_notes": [{"onset": 0.1, "offset": 0.2, "pitch": 64, "velocity": 90}]}]
    with patch.object(amt_regen, "_post_chunk", side_effect=body):
        notes, pedals = amt_regen._transcribe_clip_with_pedals(audio, "http://x")
    assert pedals == []
    assert [n["pitch"] for n in notes] == [64]
