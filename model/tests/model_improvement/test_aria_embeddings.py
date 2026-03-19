"""Tests for Aria embedding extraction from MIDI files."""

import mido
import pytest
import torch
from pathlib import Path

from model_improvement.aria_embeddings import (
    extract_all_embeddings,
    extract_embedding,
    tokenize_midi,
)


@pytest.fixture
def midi_file(tmp_path: Path) -> Path:
    """Create a minimal valid MIDI file for testing."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=90, time=0))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=480))
    track.append(mido.Message("note_on", channel=0, note=67, velocity=70, time=0))
    track.append(mido.Message("note_off", channel=0, note=67, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    path = tmp_path / "test_piece.mid"
    mid.save(str(path))
    return path


@pytest.fixture
def midi_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple MIDI files and a non-MIDI file."""
    midi_subdir = tmp_path / "midi_collection"
    midi_subdir.mkdir()

    for name, note in [("piece_a", 60), ("piece_b", 72)]:
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))
        track.append(
            mido.Message("note_on", channel=0, note=note, velocity=80, time=0)
        )
        track.append(
            mido.Message("note_off", channel=0, note=note, velocity=0, time=480)
        )
        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.save(str(midi_subdir / f"{name}.mid"))

    # Non-MIDI file that should be skipped
    (midi_subdir / "notes.txt").write_text("not a midi file")

    return midi_subdir


class TestTokenizeMidi:
    def test_returns_token_list(self, midi_file: Path):
        tokens = tokenize_midi(midi_file)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            tokenize_midi(tmp_path / "nonexistent.mid")


class TestExtractEmbedding:
    def test_embedding_variant_shape(self, midi_file: Path):
        emb = extract_embedding(midi_file, variant="embedding")
        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (512,)
        assert emb.dtype == torch.float32

    def test_base_variant_shape(self, midi_file: Path):
        emb = extract_embedding(midi_file, variant="base")
        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (1536,)
        assert emb.dtype == torch.float32

    def test_invalid_variant_raises(self, midi_file: Path):
        with pytest.raises(ValueError, match="Unknown variant"):
            extract_embedding(midi_file, variant="invalid")


class TestExtractAllEmbeddings:
    def test_returns_correct_keys_and_shapes(self, midi_dir: Path):
        result = extract_all_embeddings(midi_dir, variant="embedding")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"piece_a", "piece_b"}
        for key, emb in result.items():
            assert emb.shape == (512,)
            assert emb.dtype == torch.float32
