"""Tests for transforms.py -- deterministic partitura transforms.

A "valid variant" is checkable structurally in the score domain: transpose
preserves note count and shifts every MIDI pitch by N; tempo scaling produces a
Tempo with the scaled bpm; excerpt subsets the requested bar span. These run
against a real corpus primitive (hanon_001) so they exercise the same Part shape
the briefing loop will see.
"""

from pathlib import Path

import partitura
import pytest
from partitura.score import Note, Tempo

from exercise_corpus.transforms import (
    Variant,
    excerpt,
    load_primitive,
    materialize,
    scale_tempo,
    transpose,
)

PRIMITIVE = Path("data/midi/exercise_primitives/hanon_001.mid")


def _midi_pitches(part) -> list[int]:
    return sorted(n.midi_pitch for n in part.iter_all(Note))


def test_transpose_shifts_every_pitch_and_preserves_count():
    part = load_primitive(PRIMITIVE)
    before = _midi_pitches(part)
    variant = transpose(part, 5)

    assert isinstance(variant, Variant)
    assert variant.transform == "transpose"
    assert variant.params == {"semitones": 5}
    after = _midi_pitches(variant.part)
    assert len(after) == len(before)
    assert after == [p + 5 for p in before]


def test_transpose_out_of_piano_range_raises():
    part = load_primitive(PRIMITIVE)
    # +60 semitones pushes the top of Hanon (>C5) past C8.
    with pytest.raises(ValueError, match="piano range"):
        transpose(part, 60)


def test_scale_tempo_sets_scaled_bpm():
    part = load_primitive(PRIMITIVE)
    base = next(part.iter_all(Tempo)).bpm  # 120
    variant = scale_tempo(part, 0.5)

    assert variant.transform == "tempo"
    assert variant.params["factor"] == 0.5
    tempos = list(variant.part.iter_all(Tempo))
    assert len(tempos) == 1
    assert tempos[0].bpm == pytest.approx(base * 0.5)
    # notes are untouched by a tempo change
    assert _midi_pitches(variant.part) == _midi_pitches(part)


def test_scale_tempo_rejects_nonpositive_factor():
    part = load_primitive(PRIMITIVE)
    with pytest.raises(ValueError, match="factor"):
        scale_tempo(part, 0.0)


def test_excerpt_subsets_requested_bars():
    part = load_primitive(PRIMITIVE)
    variant = excerpt(part, 1, 4)

    assert variant.transform == "excerpt"
    assert variant.params == {"start_bar": 1, "end_bar": 4}
    # fewer notes than the whole 29-bar exercise, and at least one note kept
    n_variant = len(list(variant.part.iter_all(Note)))
    n_full = len(list(part.iter_all(Note)))
    assert 0 < n_variant < n_full
    # rebased: the excerpt starts at time 0
    first_t = min(n.start.t for n in variant.part.iter_all(Note))
    assert first_t == 0


def test_excerpt_rejects_inverted_range():
    part = load_primitive(PRIMITIVE)
    with pytest.raises(ValueError, match="start_bar"):
        excerpt(part, 5, 2)


def test_excerpt_rejects_out_of_range_bar():
    part = load_primitive(PRIMITIVE)
    with pytest.raises(ValueError, match="out of range"):
        excerpt(part, 1, 999)


def test_materialize_writes_nonempty_roundtrippable_midi(tmp_path: Path):
    part = load_primitive(PRIMITIVE)
    variant = transpose(part, 2)
    out = tmp_path / "variant.mid"
    materialize(variant, out)

    assert out.exists() and out.stat().st_size > 0
    back = partitura.load_score(str(out))
    assert len(list(back.parts[0].iter_all(Note))) == len(
        list(variant.part.iter_all(Note))
    )
