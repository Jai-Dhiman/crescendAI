"""Articulation offset-gate probe pure functions (#101 FRONT 9)."""
from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.articulation_offset_probe import (
    articulation_ratio,
    match_notes,
)


def _n(pitch, onset, offset):
    return {"pitch": pitch, "onset": onset, "offset": offset, "velocity": 64}


def test_match_within_tol_same_pitch():
    amt = [_n(60, 0.00, 0.5)]
    gt = [_n(60, 0.02, 0.4)]
    pairs = match_notes(amt, gt, tol=0.05)
    assert len(pairs) == 1 and pairs[0][1]["offset"] == 0.4


def test_no_match_across_pitch_or_beyond_tol():
    amt = [_n(60, 0.0, 0.5), _n(72, 0.0, 0.5)]
    gt = [_n(61, 0.0, 0.5), _n(72, 0.2, 0.5)]  # wrong pitch; right pitch but 200ms away
    assert match_notes(amt, gt, tol=0.05) == []


def test_each_gt_note_used_at_most_once():
    amt = [_n(60, 0.0, 0.5), _n(60, 0.01, 0.5)]  # two AMT notes near one GT
    gt = [_n(60, 0.0, 0.5)]
    assert len(match_notes(amt, gt, tol=0.05)) == 1


def test_articulation_ratio_legato_near_one():
    # duration == IOI -> ratio 1.0 (fully legato); need >=6 notes for >=5 IOIs
    notes = [_n(60 + i, i * 1.0, i * 1.0 + 1.0) for i in range(6)]
    assert articulation_ratio(notes) == pytest.approx(1.0)


def test_articulation_ratio_staccato_below_one():
    # duration 0.2, IOI 1.0 -> ratio 0.2 (detached)
    notes = [_n(60 + i, i * 1.0, i * 1.0 + 0.2) for i in range(6)]
    assert articulation_ratio(notes) == pytest.approx(0.2)


def test_articulation_ratio_none_when_too_few_notes():
    assert articulation_ratio([_n(60, 0.0, 0.5), _n(62, 1.0, 1.5)]) is None
