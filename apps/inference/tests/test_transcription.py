"""Tests for the ByteDance AMT transcription module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_transcribe_returns_sorted_note_list():
    """Transcription of audio returns notes sorted by onset."""
    from models.transcription import TranscriptionModel

    # Create a mock that simulates ByteDance writing a MIDI file
    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        # Mock the transcribe method to write a simple MIDI
        def fake_transcribe(audio, midi_path):
            # Write a minimal MIDI file using pretty_midi
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=0.9))
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.1, end=0.4))
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=0.3, end=0.7))
            pm.instruments.append(inst)
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        audio = np.zeros(24000, dtype=np.float32)  # 1s silence at 24kHz
        notes = model.transcribe(audio, sample_rate=24000)

        assert len(notes) == 3
        # Sorted by onset
        assert notes[0]["onset"] == pytest.approx(0.1, abs=0.01)
        assert notes[1]["onset"] == pytest.approx(0.3, abs=0.01)
        assert notes[2]["onset"] == pytest.approx(0.5, abs=0.01)
        # Check all fields present
        for note in notes:
            assert "pitch" in note
            assert "onset" in note
            assert "offset" in note
            assert "velocity" in note


def test_transcribe_empty_audio_returns_empty_list():
    """Silent audio produces an empty note list (not null)."""
    from models.transcription import TranscriptionModel

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        def fake_transcribe(audio, midi_path):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        audio = np.zeros(24000 * 15, dtype=np.float32)
        notes = model.transcribe(audio, sample_rate=24000)

        assert notes == []
        assert isinstance(notes, list)


def test_transcribe_error_raises_transcription_error():
    """If ByteDance transcriber fails, TranscriptionError is raised."""
    from models.transcription import TranscriptionModel, TranscriptionError

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")
        model._transcriber.transcribe = MagicMock(side_effect=RuntimeError("GPU OOM"))

        audio = np.zeros(24000, dtype=np.float32)
        with pytest.raises(TranscriptionError, match="GPU OOM"):
            model.transcribe(audio, sample_rate=24000)


def test_transcribe_cleans_up_temp_files():
    """Temp directory is cleaned up even on success."""
    from models.transcription import TranscriptionModel

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        temp_dirs_created = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            temp_dirs_created.append(d)
            return d

        def fake_transcribe(audio, midi_path):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        with patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            audio = np.zeros(24000, dtype=np.float32)
            model.transcribe(audio, sample_rate=24000)

        # Temp dir should have been cleaned up
        for d in temp_dirs_created:
            assert not Path(d).exists(), f"Temp dir not cleaned up: {d}"
