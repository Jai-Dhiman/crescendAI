"""Unit tests for the Transkun-unlocked difficulty features (#137).

Every case is hand-constructed so the expected value is arithmetic, not a regression
snapshot: a feature that silently changes meaning must fail here, not quietly shift a
tau-c by 0.003 three stages downstream.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "difficulty"))

from transkun_features import (  # noqa: E402
    articulation_features,
    duration_features,
    pedal_features,
    release_features,
    transkun_features,
    voicing_features,
)


def note(pitch, onset, offset, velocity=64):
    return {"pitch": pitch, "onset": onset, "offset": offset, "velocity": velocity}


# --------------------------------------------------------------- articulation


def test_staccato_run_is_all_staccato():
    """Notes held 0.1 s with attacks 1.0 s apart -> ratio 0.1, unambiguously staccato."""
    notes = [note(60 + i, float(i), float(i) + 0.1) for i in range(6)]
    f = articulation_features(notes)
    assert f["staccato_fraction"] == 1.0
    assert f["legato_fraction"] == 0.0
    assert f["overlap_fraction"] == 0.0
    assert f["artic_ratio_median"] == pytest.approx(0.1, abs=1e-9)


def test_legato_run_is_all_legato():
    """Each note held exactly until the next attack -> ratio 1.0."""
    notes = [note(60 + i, float(i), float(i) + 1.0) for i in range(6)]
    f = articulation_features(notes)
    assert f["legato_fraction"] == 1.0
    assert f["staccato_fraction"] == 0.0
    assert f["artic_ratio_median"] == pytest.approx(1.0, abs=1e-9)


def test_overlapping_notes_register_as_overlap_not_legato():
    """Held 1.5x past the next attack -> ratio 1.5, which is overlap, not clean legato."""
    notes = [note(60 + i, float(i), float(i) + 1.5) for i in range(6)]
    f = articulation_features(notes)
    assert f["overlap_fraction"] == 1.0
    assert f["legato_fraction"] == 0.0


def test_chord_mates_share_the_ioi_to_the_next_distinct_onset():
    """Two notes struck together must both measure against the NEXT onset, not each
    other -- otherwise a chord would show a zero IOI and an infinite ratio."""
    notes = [note(60, 0.0, 0.5), note(64, 0.0, 0.5), note(67, 1.0, 1.5)]
    f = articulation_features(notes)
    assert f["artic_ratio_median"] == pytest.approx(0.5, abs=1e-9)
    assert math.isfinite(f["artic_ratio_median"])


def test_melodic_legato_counts_only_pitch_adjacent_pairs():
    """Two stepwise pairs overlap; the one >12-semitone leap is a different line and is
    excluded rather than counted as cross-hand 'legato'."""
    notes = [note(60, 0.0, 1.5), note(62, 1.0, 2.5), note(90, 2.0, 2.2),
             note(64, 3.0, 4.5), note(65, 4.0, 5.0)]
    f = articulation_features(notes)
    # pitch-adjacent successive pairs: 60->62 (overlaps) and 64->65 (overlaps) -> 1.0
    assert f["melodic_legato_fraction"] == 1.0


def test_melodic_legato_is_nan_when_every_step_is_a_leap():
    """No pitch-adjacent pair exists -> the quantity is undefined, so nan, not 0.0."""
    notes = [note(60, 0.0, 2.0), note(90, 1.0, 1.2), note(61, 2.0, 3.0)]
    assert math.isnan(articulation_features(notes)["melodic_legato_fraction"])


def test_single_note_is_nan_not_zero():
    """One note cannot define articulation; nan keeps it out of the model rather than
    injecting a fake 0.0 that reads as 'perfectly staccato'."""
    f = articulation_features([note(60, 0.0, 1.0)])
    assert all(math.isnan(v) for v in f.values())


# ------------------------------------------------------------------ duration


def test_uniform_durations_have_zero_entropy_and_zero_cv():
    notes = [note(60 + i, float(i), float(i) + 0.5) for i in range(8)]
    f = duration_features(notes)
    assert f["dur_entropy"] == pytest.approx(0.0, abs=1e-12)
    assert f["dur_cv"] == pytest.approx(0.0, abs=1e-12)
    assert f["dur_range_log"] == pytest.approx(0.0, abs=1e-12)
    assert f["unique_dur_ratio"] == pytest.approx(1 / 8)


def test_two_equally_common_durations_give_one_bit():
    """Half short, half long, in different bins -> H = 1 bit exactly."""
    notes = []
    for i in range(4):
        notes.append(note(60, 2.0 * i, 2.0 * i + 0.06))       # short bin
        notes.append(note(62, 2.0 * i + 1.0, 2.0 * i + 3.5))  # long bin
    f = duration_features(notes)
    assert f["dur_entropy"] == pytest.approx(1.0, abs=1e-12)


def test_duration_separates_pieces_that_ioi_cannot():
    """Same attack times, different note lengths: IOI-based features are identical while
    duration entropy differs. This is the whole reason the family is new."""
    onsets = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    uniform = [note(60, t, t + 0.5) for t in onsets]
    mixed = [note(60, t, t + (0.06 if i % 2 else 3.0)) for i, t in enumerate(onsets)]
    assert duration_features(uniform)["dur_entropy"] < duration_features(mixed)["dur_entropy"]


# ------------------------------------------------------------------- voicing


def test_monophonic_line_has_concurrency_one_and_rests_are_reported_separately():
    """Voice statistics are normalized over SOUNDING time, so the 0.1 s gaps do not drag
    the mean below 1.0; they surface in `frac_time_silent` instead. Blending the two
    would make a sparse melody indistinguishable from a thin continuous texture."""
    notes = [note(60 + i, float(i), float(i) + 0.9) for i in range(5)]
    f = voicing_features(notes)
    assert f["concurrency_mean"] == pytest.approx(1.0, abs=1e-9)
    assert f["concurrency_max"] == 1.0
    assert f["frac_time_polyphonic"] == pytest.approx(0.0, abs=1e-9)
    # 4 gaps of 0.1 s across a 4.9 s span
    assert f["frac_time_silent"] == pytest.approx(0.4 / 4.9, abs=1e-9)


def test_continuous_line_has_no_silence():
    notes = [note(60 + i, float(i), float(i) + 1.0) for i in range(5)]
    assert voicing_features(notes)["frac_time_silent"] == pytest.approx(0.0, abs=1e-12)


def test_held_bass_under_a_melody_is_seen_only_by_true_concurrency():
    """A bass held for 4 s under four melody notes. Onset-cluster polyphony sees four
    isolated single attacks; time-weighted concurrency correctly reports 2 voices."""
    notes = [note(36, 0.0, 4.0)] + [note(72 + i, float(i), float(i) + 1.0) for i in range(4)]
    f = voicing_features(notes)
    assert f["concurrency_mean"] == pytest.approx(2.0, abs=1e-9)
    assert f["frac_time_polyphonic"] == pytest.approx(1.0, abs=1e-9)
    assert f["concurrency_max"] == 2.0


def test_block_chord_reports_its_voice_count():
    notes = [note(p, 0.0, 2.0) for p in (60, 64, 67, 72)]
    f = voicing_features(notes)
    assert f["concurrency_mean"] == pytest.approx(4.0, abs=1e-9)
    assert f["frac_time_ge3_voices"] == pytest.approx(1.0, abs=1e-9)
    assert f["frac_time_ge5_voices"] == pytest.approx(0.0, abs=1e-9)


def test_voice_change_rate_is_per_second_not_a_count():
    """Doubling the piece's length while keeping the same texture must not change the
    rate -- this is the length-confound guard."""
    short = [note(60, 0.0, 1.0), note(64, 1.0, 2.0)]
    long = [note(60, 0.0, 2.0), note(64, 2.0, 4.0)]
    assert voicing_features(short)["voice_change_rate"] == pytest.approx(
        2 * voicing_features(long)["voice_change_rate"])


# ------------------------------------------------------------------- release


def test_block_chord_releases_together():
    notes = [note(p, 0.0, 2.0) for p in (60, 64, 67)]
    f = release_features(notes)
    assert f["release_dispersion_median"] == pytest.approx(0.0, abs=1e-12)
    assert f["frac_chords_released_together"] == 1.0


def test_independent_voice_release_is_dispersed():
    """Same attack, staggered releases -> a 1.0 s offset spread and NOT 'together'."""
    notes = [note(60, 0.0, 1.0), note(64, 0.0, 1.5), note(67, 0.0, 2.0)]
    f = release_features(notes)
    assert f["release_dispersion_median"] == pytest.approx(1.0, abs=1e-9)
    assert f["frac_chords_released_together"] == 0.0


def test_monophonic_piece_has_no_chords_to_measure():
    notes = [note(60 + i, float(i), float(i) + 0.5) for i in range(4)]
    f = release_features(notes)
    assert all(math.isnan(v) for v in f.values())


# --------------------------------------------------------------------- pedal


def test_no_pedal_gives_zero_usage_but_nan_segment_shape():
    """Absence of pedal is a real 0.0 for usage, but segment length is undefined with no
    events -- nan keeps 'never pedalled' distinguishable from 'pedalled briefly'."""
    f = pedal_features([], [], 0.0, 60.0)
    assert f["pedal_change_rate"] == 0.0
    assert f["pedal_on_fraction"] == 0.0
    assert math.isnan(f["pedal_segment_mean_s"])
    assert f["soft_pedal_used"] == 0.0


def test_pedal_on_fraction_is_time_weighted():
    """Down at t=0, up at t=5, over a 10 s piece -> exactly half the piece pedalled."""
    sustain = [{"time": 0.0, "value": 127}, {"time": 5.0, "value": 0}]
    f = pedal_features(sustain, [], 0.0, 10.0)
    assert f["pedal_on_fraction"] == pytest.approx(0.5, abs=1e-9)
    assert f["pedal_change_rate"] == pytest.approx(0.2, abs=1e-9)
    assert f["pedal_segment_mean_s"] == pytest.approx(5.0, abs=1e-9)


def test_on_fraction_stays_a_fraction_when_pedal_precedes_the_first_note():
    """Regression: pedal events are timestamped from the start of the FILE while the
    first note may begin much later. Normalizing by the note span alone let hold-times
    sum past the span and produced an 'on fraction' of 1.24 on real data."""
    sustain = [{"time": 0.0, "value": 127}, {"time": 30.0, "value": 0}]
    f = pedal_features(sustain, [], start_s=20.0, end_s=30.0)
    assert 0.0 <= f["pedal_on_fraction"] <= 1.0
    # Only t=20..30 lies inside the note window, and the pedal is down for all of it.
    assert f["pedal_on_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_pedal_events_after_the_last_note_do_not_inflate_the_fraction():
    sustain = [{"time": 0.0, "value": 127}, {"time": 50.0, "value": 0}]
    f = pedal_features(sustain, [], start_s=0.0, end_s=10.0)
    assert f["pedal_on_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_pedal_rate_is_length_invariant():
    """Same gesture density over a 2x longer piece -> same rate."""
    a = [{"time": float(i), "value": 127} for i in range(10)]
    b = [{"time": 2.0 * i, "value": 127} for i in range(10)]
    assert (pedal_features(a, [], 0.0, 10.0)["pedal_change_rate"]
            == pytest.approx(2 * pedal_features(b, [], 0.0, 20.0)["pedal_change_rate"]))


def test_pedal_depth_features_are_absent_because_transkun_cannot_supply_them():
    """Transkun's pedal head emits only CC64 0 and 127, so depth mean (always 63.5),
    value entropy (always 1.0) and half-pedal fraction (always 0.0) are constants across
    the whole corpus. They must not exist as columns -- a constant feature is dead weight
    that reads as signal in a feature list."""
    f = pedal_features([{"time": 0.0, "value": 127}], [], 0.0, 10.0)
    for dead in ("pedal_depth_mean", "pedal_value_entropy", "pedal_half_fraction"):
        assert dead not in f


# ----------------------------------------------------------------- integration


def test_transkun_features_are_namespaced_and_complete():
    notes = [note(60, 0.0, 1.0), note(64, 0.0, 1.5), note(67, 1.0, 2.0)]
    sustain = [{"time": 0.0, "value": 100}, {"time": 1.0, "value": 0}]
    f = transkun_features(notes, sustain, [])
    assert all(k.startswith("tk_") for k in f)
    # 8 articulation + 6 duration + 8 voicing + 4 release + 6 pedal
    assert len(f) == 32
    assert "tk_concurrency_mean" in f and "tk_pedal_on_fraction" in f


def test_transkun_features_rejects_a_single_note():
    with pytest.raises(ValueError):
        transkun_features([note(60, 0.0, 1.0)], [], [])
