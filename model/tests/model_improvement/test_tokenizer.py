import pytest
import numpy as np
from pathlib import Path
from model_improvement.tokenizer import PianoTokenizer, extract_continuous_features


@pytest.fixture
def sample_midi(tmp_path):
    """Create a minimal MIDI file for testing."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    piano.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
    piano.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=1.0, end=1.5))
    pm.instruments.append(piano)
    path = tmp_path / "test.mid"
    pm.write(str(path))
    return path


def test_tokenizer_encodes_midi(sample_midi):
    tok = PianoTokenizer()
    tokens = tok.encode(sample_midi)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_tokenizer_vocab_size():
    tok = PianoTokenizer()
    assert tok.vocab_size > 0
    assert tok.vocab_size < 1000


def test_tokenizer_roundtrip(sample_midi, tmp_path):
    tok = PianoTokenizer()
    tokens = tok.encode(sample_midi)
    output_path = tmp_path / "roundtrip.mid"
    tok.decode(tokens, output_path)
    assert output_path.exists()


def test_tokenizer_special_tokens():
    tok = PianoTokenizer()
    assert hasattr(tok, 'pad_token_id')
    assert hasattr(tok, 'mask_token_id')


def test_continuous_features(sample_midi):
    features = extract_continuous_features(sample_midi, frame_rate=50)
    assert features.ndim == 2  # [T, D]
    assert features.shape[1] >= 3  # pitch, velocity, timing at minimum
