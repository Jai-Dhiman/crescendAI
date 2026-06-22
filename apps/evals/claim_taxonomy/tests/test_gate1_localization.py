"""GATE 1 localization-robustness measurement.

delta(bar) = M_corrupt(bar) - W(M_clean(bar)), where M_* is the SHIPPED
LocationResolver's resolved bar-start audio time and W is the construction-known
corruption warp. A perfectly warp-aware pipeline yields delta == 0 everywhere.
The metric is the fraction of bars within a tolerance band.
"""
from __future__ import annotations

import numpy as np

from claim_taxonomy.gate1.localization import (
    accuracy_at_tolerances,
    bar_localization_deltas,
)


def _measure_table(n_bars: int, bar_dur: float) -> list[dict]:
    return [
        {"bar_number": b, "start_sec": (b - 1) * bar_dur, "start_tick": (b - 1) * 1536}
        for b in range(1, n_bars + 1)
    ]


def _bundle(measure_table, perf_audio_sec, score_audio_sec) -> dict:
    return {
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": list(perf_audio_sec),
            "score_audio_sec": list(score_audio_sec),
        },
        "notes": [],
        "pedal_events": [],
        "substrate_versions": {},
    }


def _warp_map(clean_sec, corrupt_sec) -> dict:
    return {"clean_sec": list(clean_sec), "corrupt_sec": list(corrupt_sec)}


def _apply_warp(warp_map, xs):
    return np.interp(xs, warp_map["clean_sec"], warp_map["corrupt_sec"])


def test_perfect_warp_recovery_gives_zero_delta():
    mt = _measure_table(10, bar_dur=2.0)
    grid = np.linspace(0.0, 20.0, 41)  # dense identity anchors
    clean = _bundle(mt, grid, grid)

    # Speed up [4s,12s] by 2x: that 8s region becomes 4s.
    wm = _warp_map([0.0, 4.0, 12.0, 20.0], [0.0, 4.0, 8.0, 16.0])
    # A perfectly warp-aware corrupt bundle: perf anchors moved to W(score).
    corrupt = _bundle(mt, _apply_warp(wm, grid), grid)

    deltas = bar_localization_deltas(clean, corrupt, wm)
    resolvable = [d for d in deltas if d.resolvable]
    assert len(resolvable) == 10
    assert max(abs(d.delta_sec) for d in resolvable) < 0.05

    acc = accuracy_at_tolerances(deltas, [0.1, 0.5, 1.0])
    assert acc["tolerances"]["0.1"]["within_over_total"] == 1.0
    assert acc["n_resolvable"] == 10


def test_injected_constant_mislocalization_fails_tight_passes_loose():
    mt = _measure_table(10, bar_dur=2.0)
    grid = np.linspace(0.0, 20.0, 41)
    clean = _bundle(mt, grid, grid)

    wm = _warp_map([0.0, 4.0, 12.0, 20.0], [0.0, 4.0, 8.0, 16.0])
    # Same warp, but the corrupt alignment is globally off by +0.5s.
    corrupt = _bundle(mt, _apply_warp(wm, grid) + 0.5, grid)

    deltas = bar_localization_deltas(clean, corrupt, wm)
    resolvable = [d for d in deltas if d.resolvable]
    assert np.median([abs(d.delta_sec) for d in resolvable]) == \
        __import__("pytest").approx(0.5, abs=0.05)

    acc = accuracy_at_tolerances(deltas, [0.1, 1.0])
    assert acc["tolerances"]["0.1"]["within_over_total"] == 0.0
    assert acc["tolerances"]["1.0"]["within_over_total"] == 1.0


def test_unresolvable_bars_count_against_total_not_resolvable():
    mt = _measure_table(10, bar_dur=2.0)
    grid = np.linspace(0.0, 20.0, 41)
    clean = _bundle(mt, grid, grid)
    # Corrupt anchors only span [0,10]s -> bars whose start is past the anchor
    # span still interpolate (np.interp clamps), so force unresolvability with a
    # degenerate single-anchor bundle instead.
    corrupt = _bundle(mt, [0.0], [0.0])

    deltas = bar_localization_deltas(clean, corrupt, _warp_map([0, 20], [0, 20]))
    assert all(not d.resolvable for d in deltas)
    acc = accuracy_at_tolerances(deltas, [1.0])
    assert acc["n_resolvable"] == 0
    assert acc["resolvable_rate"] == 0.0
    assert acc["tolerances"]["1.0"]["within_over_total"] == 0.0
