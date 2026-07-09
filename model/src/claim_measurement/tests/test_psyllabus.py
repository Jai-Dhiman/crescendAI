"""Tests for the PianoSyllabus loader (MIREX 2026 difficulty, #104).

Covers the two error-prone pure functions: label->record coercion (grade parsing +
MIDI-presence filtering, no silent drops) and the composer-disjoint split (no composer
may straddle train/test).
"""
from __future__ import annotations

import pytest

from claim_measurement.difficulty.psyllabus import (
    PsRecord,
    build_records,
    composer_disjoint_split,
)


def _labels():
    return {
        "Bach.Invention 1": {"ps": "3", "composer": "Bach J.S.", "duration": 90.0},
        "Bach.Invention 2": {"ps": 5, "composer": "Bach J.S.", "duration": 100.0},
        "Chopin.Etude 10-1": {"ps": "9", "composer": "Chopin F.", "duration": 120.0},
        "Missing.Midi": {"ps": "4", "composer": "Nobody", "duration": 60.0},
        "Bad.Grade": {"ps": None, "composer": "Bach J.S.", "duration": 60.0},
    }


def _available():
    return {
        "mid/Bach.Invention 1.mid", "mid/Bach.Invention 2.mid",
        "mid/Chopin.Etude 10-1.mid", "mid/Bad.Grade.mid",
    }  # note: no MIDI for "Missing.Midi"


def test_build_records_keeps_valid_and_coerces_grade():
    records, report = build_records(_labels(), _available())
    by_key = {r.key: r for r in records}
    assert by_key["Bach.Invention 1"].grade == 3          # "3" -> 3
    assert by_key["Bach.Invention 2"].grade == 5          # already int
    assert isinstance(by_key["Bach.Invention 1"], PsRecord)
    assert report["n_records"] == 3


def test_build_records_skips_missing_midi_loudly():
    _, report = build_records(_labels(), _available())
    assert report["skipped_no_midi"] == 1   # "Missing.Midi"


def test_build_records_skips_bad_grade_loudly():
    records, report = build_records(_labels(), _available())
    assert report["skipped_bad_grade"] == 1                # ps=None
    assert "Bad.Grade" not in {r.key for r in records}


def _many_records():
    # 10 composers x 6 pieces = 60 records
    recs = []
    for ci in range(10):
        for pi in range(6):
            recs.append(PsRecord(key=f"c{ci}-p{pi}", grade=(ci % 11),
                                  composer=f"Composer {ci}", duration=60.0,
                                  midi_name=f"mid/c{ci}-p{pi}.mid"))
    return recs


def test_split_is_composer_disjoint():
    train, test = composer_disjoint_split(_many_records(), test_frac=0.3, seed=1)
    train_composers = {r.composer for r in train}
    test_composers = {r.composer for r in test}
    assert train_composers.isdisjoint(test_composers)
    assert len(train) + len(test) == 60


def test_split_respects_fraction_roughly():
    _, test = composer_disjoint_split(_many_records(), test_frac=0.3, seed=1)
    # whole-composer granularity -> within one composer-block (6) of the 18 target
    assert 12 <= len(test) <= 24


def test_split_is_deterministic():
    a = composer_disjoint_split(_many_records(), test_frac=0.3, seed=7)
    b = composer_disjoint_split(_many_records(), test_frac=0.3, seed=7)
    assert [r.key for r in a[1]] == [r.key for r in b[1]]


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        composer_disjoint_split(_many_records(), test_frac=1.5)
