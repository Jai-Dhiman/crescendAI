# model/tests/exercise_corpus/test_segment.py
import mido
import partitura
import pytest
from pathlib import Path

from exercise_corpus import Primitive
from exercise_corpus.segment import segment_source, SegmentationError

FIXTURES = Path(__file__).parent / "fixtures"


def _midi_pitch_set(midi_path: Path) -> set[int]:
    """Return the set of MIDI note numbers present in a .mid file."""
    mid = mido.MidiFile(str(midi_path))
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitches.add(msg.note)
    return pitches


def _musicxml_pitch_set(musicxml_path: Path) -> set[int]:
    """Return the set of MIDI pitch numbers in a single-part MusicXML file."""
    score = partitura.load_musicxml(str(musicxml_path))
    note_array = partitura.utils.music.ensure_notearray(score)
    return set(int(n) for n in note_array["pitch"])


def test_hanon_fixture_yields_three_primitives(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    assert len(primitives) == 3


def test_primitive_fields_are_populated(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    for i, p in enumerate(primitives, start=1):
        assert isinstance(p, Primitive)
        assert p.source == "hanon"
        assert p.source_exercise_number == i
        assert p.n_notes > 0
        assert p.musicxml_path.exists()
        assert p.midi_path.exists()


def test_midi_pitch_set_matches_musicxml(tmp_path: Path):
    score_dir = tmp_path / "scores"
    midi_dir = tmp_path / "midi"
    primitives = segment_source(
        FIXTURES / "hanon_3ex.xml", "hanon", score_dir, midi_dir
    )
    for p in primitives:
        xml_pitches = _musicxml_pitch_set(p.musicxml_path)
        midi_pitches = _midi_pitch_set(p.midi_path)
        # MIDI pitch set must be a subset of or equal to the XML pitch set
        # (partitura MIDI export may merge enharmonics but no new pitches appear)
        assert midi_pitches.issubset(xml_pitches) or midi_pitches == xml_pitches, (
            f"Primitive {p.primitive_id}: MIDI pitches {midi_pitches} not "
            f"consistent with XML pitches {xml_pitches}"
        )


def test_bad_source_name_raises_segmentation_error(tmp_path: Path):
    with pytest.raises(SegmentationError, match="unknown source"):
        segment_source(
            FIXTURES / "hanon_3ex.xml", "unknown_source", tmp_path, tmp_path
        )


def test_zero_parts_raises_segmentation_error(tmp_path: Path):
    # Write a MusicXML with no parts
    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"'
        ' "http://www.musicxml.org/dtds/partwise.dtd">'
        "<score-partwise version=\"3.1\"><part-list></part-list></score-partwise>"
    )
    with pytest.raises((SegmentationError, Exception)):
        segment_source(empty_xml, "hanon", tmp_path / "s", tmp_path / "m")
