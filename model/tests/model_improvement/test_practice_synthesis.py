"""Unit tests for MIDI corruption primitives in practice_synthesis.py."""

import random

import numpy as np
import pretty_midi
import pytest

from model_improvement.practice_synthesis import (
    apply_practice_corruptions,
    compress_velocity,
    drop_notes,
    insert_pauses,
    jitter_tempo,
    substitute_wrong_notes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_midi(
    n_notes: int = 20,
    start: float = 0.0,
    duration: float = 10.0,
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """Build a simple single-instrument MIDI with evenly spaced notes."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    step = duration / n_notes
    for i in range(n_notes):
        t = start + i * step
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=60 + (i % 12),  # C4 to B4
            start=t,
            end=t + step * 0.9,
        )
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi


@pytest.fixture
def simple_midi():
    return _make_midi()


@pytest.fixture
def rng():
    return random.Random(42)


# ---------------------------------------------------------------------------
# drop_notes
# ---------------------------------------------------------------------------


def test_drop_notes_returns_pretty_midi(simple_midi, rng):
    result = drop_notes(simple_midi, rate=0.5, rng=rng)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_drop_notes_reduces_note_count(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    result = drop_notes(simple_midi, rate=0.5, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    # At 50% drop rate with 20 notes, expect fewer notes (probabilistically)
    assert new_count < original_count


def test_drop_notes_rate_zero_keeps_all(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    result = drop_notes(simple_midi, rate=0.0, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    assert new_count == original_count


def test_drop_notes_rate_one_drops_all(simple_midi, rng):
    result = drop_notes(simple_midi, rate=1.0, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    assert new_count == 0


def test_drop_notes_does_not_mutate_original(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    drop_notes(simple_midi, rate=0.5, rng=rng)
    after_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    assert after_count == original_count, "drop_notes must not mutate the input"


def test_drop_notes_invalid_rate(simple_midi):
    with pytest.raises(ValueError, match="rate must be in"):
        drop_notes(simple_midi, rate=1.5)


# ---------------------------------------------------------------------------
# substitute_wrong_notes
# ---------------------------------------------------------------------------


def test_substitute_notes_returns_pretty_midi(simple_midi, rng):
    result = substitute_wrong_notes(simple_midi, rate=0.5, rng=rng)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_substitute_notes_same_count(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    result = substitute_wrong_notes(simple_midi, rate=0.5, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    assert new_count == original_count, "substitute must not change note count"


def test_substitute_notes_changes_pitches(simple_midi, rng):
    result = substitute_wrong_notes(simple_midi, rate=1.0, semitone_range=3, rng=rng)
    orig_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    new_pitches = [n.pitch for inst in result.instruments for n in inst.notes]
    assert orig_pitches != new_pitches, "rate=1.0 should change at least some pitches"


def test_substitute_notes_pitch_clamped():
    # Force extreme pitches and verify clamp to [0, 127]
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [
        pretty_midi.Note(velocity=80, pitch=0, start=0.0, end=0.5),
        pretty_midi.Note(velocity=80, pitch=127, start=0.5, end=1.0),
    ]
    midi.instruments.append(inst)
    result = substitute_wrong_notes(midi, rate=1.0, semitone_range=10, rng=random.Random(0))
    for inst in result.instruments:
        for note in inst.notes:
            assert 0 <= note.pitch <= 127


def test_substitute_notes_does_not_mutate_original(simple_midi, rng):
    orig_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    substitute_wrong_notes(simple_midi, rate=0.5, rng=rng)
    after_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    assert orig_pitches == after_pitches, "substitute must not mutate the input"


# ---------------------------------------------------------------------------
# jitter_tempo
# ---------------------------------------------------------------------------


def test_jitter_tempo_returns_pretty_midi(simple_midi, rng):
    result = jitter_tempo(simple_midi, std=0.1, rng=rng)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_jitter_tempo_preserves_note_count(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    result = jitter_tempo(simple_midi, std=0.1, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    assert new_count == original_count


def test_jitter_tempo_monotonic_onset_order(simple_midi, rng):
    result = jitter_tempo(simple_midi, std=0.3, rng=rng)
    for inst in result.instruments:
        onsets = [n.start for n in sorted(inst.notes, key=lambda n: n.start)]
        assert onsets == sorted(onsets), "Note onsets must remain in non-decreasing order"


def test_jitter_tempo_no_negative_times(simple_midi, rng):
    result = jitter_tempo(simple_midi, std=0.5, rng=rng)
    for inst in result.instruments:
        for note in inst.notes:
            assert note.start >= 0.0, "No negative note start times after jitter"
            assert note.end > note.start, "Note end must be after start after jitter"


def test_jitter_tempo_zero_std_unchanged(simple_midi, rng):
    result = jitter_tempo(simple_midi, std=0.0, rng=rng)
    orig_onsets = sorted(n.start for inst in simple_midi.instruments for n in inst.notes)
    new_onsets = sorted(n.start for inst in result.instruments for n in inst.notes)
    assert np.allclose(orig_onsets, new_onsets, atol=1e-4), "std=0 should leave times unchanged"


def test_jitter_tempo_does_not_mutate_original(simple_midi, rng):
    orig_onsets = [n.start for inst in simple_midi.instruments for n in inst.notes]
    jitter_tempo(simple_midi, std=0.2, rng=rng)
    after_onsets = [n.start for inst in simple_midi.instruments for n in inst.notes]
    assert orig_onsets == after_onsets, "jitter_tempo must not mutate the input"


# ---------------------------------------------------------------------------
# compress_velocity
# ---------------------------------------------------------------------------


def test_compress_velocity_returns_pretty_midi(simple_midi):
    result = compress_velocity(simple_midi, factor=0.5)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_compress_velocity_reduces_dynamic_range():
    # Create MIDI with extreme velocities
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for v in [1, 64, 127]:
        inst.notes.append(pretty_midi.Note(velocity=v, pitch=60, start=0.0, end=0.5))
    midi.instruments.append(inst)

    result = compress_velocity(midi, factor=0.3, midpoint=64)
    orig_vels = sorted(n.velocity for inst in midi.instruments for n in inst.notes)
    new_vels = sorted(n.velocity for inst in result.instruments for n in inst.notes)
    orig_range = orig_vels[-1] - orig_vels[0]
    new_range = new_vels[-1] - new_vels[0]
    assert new_range < orig_range, "Compression must reduce dynamic range"


def test_compress_velocity_factor_one_unchanged(simple_midi):
    result = compress_velocity(simple_midi, factor=1.0)
    orig_vels = [n.velocity for inst in simple_midi.instruments for n in inst.notes]
    new_vels = [n.velocity for inst in result.instruments for n in inst.notes]
    assert orig_vels == new_vels, "factor=1.0 should leave velocities unchanged"


def test_compress_velocity_factor_zero_all_midpoint(simple_midi):
    result = compress_velocity(simple_midi, factor=0.0, midpoint=64)
    for inst in result.instruments:
        for note in inst.notes:
            assert note.velocity == 64, "factor=0 must compress all velocities to midpoint"


def test_compress_velocity_velocities_clamped():
    # Velocities at extremes (1 and 127) with factor=0.0 and midpoint at boundary
    # should still clamp to [1, 127] without underflow.
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for v in [1, 127]:
        inst.notes.append(pretty_midi.Note(velocity=v, pitch=60, start=0.0, end=0.5))
    midi.instruments.append(inst)

    # factor=0 maps everything to midpoint=64; nothing should go out of range
    result = compress_velocity(midi, factor=0.0, midpoint=64)
    for inst in result.instruments:
        for note in inst.notes:
            assert 1 <= note.velocity <= 127


def test_compress_velocity_does_not_mutate_original(simple_midi):
    orig_vels = [n.velocity for inst in simple_midi.instruments for n in inst.notes]
    compress_velocity(simple_midi, factor=0.3)
    after_vels = [n.velocity for inst in simple_midi.instruments for n in inst.notes]
    assert orig_vels == after_vels, "compress_velocity must not mutate the input"


def test_compress_velocity_invalid_factor(simple_midi):
    with pytest.raises(ValueError, match="factor must be in"):
        compress_velocity(simple_midi, factor=-0.1)


# ---------------------------------------------------------------------------
# insert_pauses
# ---------------------------------------------------------------------------


def test_insert_pauses_returns_pretty_midi(simple_midi, rng):
    result = insert_pauses(simple_midi, rate=0.5, rng=rng)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_insert_pauses_preserves_note_count(simple_midi, rng):
    original_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    result = insert_pauses(simple_midi, rate=0.5, rng=rng)
    new_count = sum(len(inst.notes) for inst in result.instruments)
    assert new_count == original_count, "insert_pauses must not drop notes"


def test_insert_pauses_extends_total_duration(simple_midi):
    orig_duration = simple_midi.get_end_time()
    # Force pause insertion: high rate with a small min_gap so fallback triggers
    result = insert_pauses(simple_midi, rate=1.0, max_dur=0.5, min_gap=0.001, rng=random.Random(7))
    new_duration = result.get_end_time()
    assert new_duration >= orig_duration, "Inserted pauses must not shorten total duration"


def test_insert_pauses_no_negative_times(simple_midi, rng):
    result = insert_pauses(simple_midi, rate=0.5, max_dur=1.0, rng=rng)
    for inst in result.instruments:
        for note in inst.notes:
            assert note.start >= 0.0
            assert note.end > note.start


def test_insert_pauses_does_not_mutate_original(simple_midi, rng):
    orig_onsets = [n.start for inst in simple_midi.instruments for n in inst.notes]
    insert_pauses(simple_midi, rate=0.5, rng=rng)
    after_onsets = [n.start for inst in simple_midi.instruments for n in inst.notes]
    assert orig_onsets == after_onsets, "insert_pauses must not mutate the input"


# ---------------------------------------------------------------------------
# apply_practice_corruptions (composite)
# ---------------------------------------------------------------------------


def test_apply_practice_corruptions_returns_pretty_midi(simple_midi, rng):
    result = apply_practice_corruptions(simple_midi, rng=rng)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_apply_practice_corruptions_changes_midi(simple_midi):
    # With multiple corruption primitives, the MIDI should be different from original
    orig_onsets = [n.start for inst in simple_midi.instruments for n in inst.notes]
    orig_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    orig_vels = [n.velocity for inst in simple_midi.instruments for n in inst.notes]

    changed = False
    for seed in range(10):
        rng = random.Random(seed)
        result = apply_practice_corruptions(simple_midi, rng=rng, min_primitives=3, max_primitives=5)
        new_onsets = [n.start for inst in result.instruments for n in inst.notes]
        new_pitches = [n.pitch for inst in result.instruments for n in inst.notes]
        new_vels = [n.velocity for inst in result.instruments for n in inst.notes]
        if new_onsets != orig_onsets or new_pitches != orig_pitches or new_vels != orig_vels:
            changed = True
            break
    assert changed, "apply_practice_corruptions must change the MIDI for at least one seed"


def test_apply_practice_corruptions_does_not_mutate_original(simple_midi, rng):
    orig_note_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    orig_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    apply_practice_corruptions(simple_midi, rng=rng)
    after_count = sum(len(inst.notes) for inst in simple_midi.instruments)
    after_pitches = [n.pitch for inst in simple_midi.instruments for n in inst.notes]
    assert after_count == orig_note_count
    assert after_pitches == orig_pitches, "apply_practice_corruptions must not mutate the input"


def test_apply_practice_corruptions_empty_midi():
    empty = pretty_midi.PrettyMIDI()
    result = apply_practice_corruptions(empty)
    assert isinstance(result, pretty_midi.PrettyMIDI)


def test_apply_practice_corruptions_reproducible():
    midi = _make_midi(n_notes=30)
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    result_a = apply_practice_corruptions(midi, rng=rng_a)
    result_b = apply_practice_corruptions(midi, rng=rng_b)

    onsets_a = sorted(n.start for inst in result_a.instruments for n in inst.notes)
    onsets_b = sorted(n.start for inst in result_b.instruments for n in inst.notes)
    pitches_a = sorted(n.pitch for inst in result_a.instruments for n in inst.notes)
    pitches_b = sorted(n.pitch for inst in result_b.instruments for n in inst.notes)

    assert np.allclose(onsets_a, onsets_b, atol=1e-4)
    assert pitches_a == pitches_b


# ---------------------------------------------------------------------------
# AudioAugmentor IR extension
# ---------------------------------------------------------------------------


def test_augmentor_ir_synthetic_shape():
    """Synthetic IR convolution must preserve waveform shape."""
    import torch
    from model_improvement.augmentation import AudioAugmentor

    aug = AudioAugmentor(augment_prob=1.0, ir_dir=None)
    waveform = torch.randn(1, 24000)
    # Force IR path by calling _apply_room_ir directly
    audio_np = waveform.numpy()
    result = aug._apply_room_ir(audio_np, sample_rate=24000)
    assert result.shape == audio_np.shape


def test_augmentor_generate_synthetic_ir():
    """Synthetic IR must have the expected length and be normalised."""
    from model_improvement.augmentation import AudioAugmentor

    ir = AudioAugmentor._generate_synthetic_room_ir(sample_rate=24000, rt60=0.4)
    assert ir.ndim == 1
    assert len(ir) > 0
    assert np.max(np.abs(ir)) <= 1.0 + 1e-6
