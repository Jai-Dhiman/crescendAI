"""GATE 1 report aggregation (grouping by corruption kind, pooled accuracy)."""
from __future__ import annotations

import numpy as np

from claim_taxonomy.gate1.analyze import report_from_loaded


def _mt(n_bars: int, bar_dur: float) -> list[dict]:
    return [
        {"bar_number": b, "start_sec": (b - 1) * bar_dur, "start_tick": (b - 1) * 1536}
        for b in range(1, n_bars + 1)
    ]


def _bundle(mt, perf, score) -> dict:
    return {
        "measure_table": mt,
        "anchors": {"perf_audio_sec": list(perf), "score_audio_sec": list(score)},
        "notes": [], "pedal_events": [], "substrate_versions": {},
    }


def _wm(clean, corrupt) -> dict:
    return {"clean_sec": list(clean), "corrupt_sec": list(corrupt)}


def _loaded(kind, clean_b, corrupt_b, wm, piece="p", video="v", spec=None):
    return {
        "piece": piece, "video": video, "spec_id": spec or kind, "kind": kind,
        "clean_bundle": clean_b, "corrupt_bundle": corrupt_b, "warp_map": wm,
    }


def test_report_groups_by_kind_and_pools_bars():
    mt = _mt(10, 2.0)
    grid = np.linspace(0.0, 20.0, 41)
    clean = _bundle(mt, grid, grid)

    # A perfect "clean" re-extraction -> delta 0.
    clean_item = _loaded("clean", clean, _bundle(mt, grid, grid), _wm([0, 20], [0, 20]))
    # A "noise" item globally off by 0.5s -> delta 0.5 everywhere.
    noise_item = _loaded("noise", clean, _bundle(mt, grid + 0.5, grid), _wm([0, 20], [0, 20]))

    report = report_from_loaded([clean_item, noise_item], tolerances_sec=[0.25, 1.0])

    by_kind = report["by_kind"]
    assert by_kind["clean"]["tolerances"]["0.25"]["within_over_total"] == 1.0
    assert by_kind["noise"]["tolerances"]["0.25"]["within_over_total"] == 0.0
    assert by_kind["noise"]["tolerances"]["1.0"]["within_over_total"] == 1.0
    # Per-item rows are retained.
    assert len(report["rows"]) == 2
    assert report["overall"]["n_total"] == 20  # 10 bars x 2 items pooled


def test_report_handles_empty_input():
    report = report_from_loaded([], tolerances_sec=[1.0])
    assert report["overall"]["n_total"] == 0
    assert report["by_kind"] == {}
