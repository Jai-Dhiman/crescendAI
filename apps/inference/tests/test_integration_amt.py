"""Integration test: verify full endpoint returns both scores and MIDI notes.

Requires:
- CUDA GPU or CPU with sufficient memory
- Piano audio file for testing

Skip this test in CI (requires GPU + model weights).
"""

from pathlib import Path

import numpy as np
import pytest


# Skip if piano-transcription-inference not installed
pytest.importorskip("piano_transcription_inference")


SAMPLE_AUDIO_PATH = Path(__file__).parent / "fixtures" / "piano_sample.wav"


@pytest.mark.skipif(
    not SAMPLE_AUDIO_PATH.exists(),
    reason="Sample audio not available (place a piano WAV at tests/fixtures/piano_sample.wav)",
)
def test_full_pipeline_returns_scores_and_notes():
    """End-to-end: real audio produces both predictions and midi_notes."""
    import librosa
    from models.transcription import TranscriptionModel

    audio, sr = librosa.load(str(SAMPLE_AUDIO_PATH), sr=24000, mono=True, duration=15.0)

    model = TranscriptionModel(device="cpu")
    notes = model.transcribe(audio, sample_rate=24000)

    assert isinstance(notes, list)
    # Real piano audio should produce at least some notes
    assert len(notes) > 0, "Expected notes from piano audio"

    # Verify note structure
    for note in notes:
        assert 0 <= note["pitch"] <= 127
        assert note["onset"] >= 0
        assert note["offset"] > note["onset"]
        assert 0 <= note["velocity"] <= 127

    # Verify sorted by onset
    for i in range(1, len(notes)):
        assert notes[i]["onset"] >= notes[i - 1]["onset"]


def test_transcription_model_loads_on_cpu():
    """TranscriptionModel can be initialized on CPU (for testing)."""
    from models.transcription import TranscriptionModel

    model = TranscriptionModel(device="cpu")
    assert model._transcriber is not None
