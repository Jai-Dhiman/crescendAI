"""Stratified sample selector for the rubric calibration protocol.

Initial implementation: band stratification only. Later tasks layer on era
quotas (T7), holdout reservation (T10), and anchor injection (T13).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_BAND_TARGETS: dict[str, int] = {
    "threshold": 80,
    "high": 40,
    "low": 30,
    "weak_dim": 50,
}


def _row_synth_id(row: dict[str, Any]) -> str:
    return f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"


def _row_composite(row: dict[str, Any]) -> float | None:
    scores = [
        d["score"] for d in row.get("judge_dimensions", [])
        if d.get("score") is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _row_ascf_process(row: dict[str, Any]) -> int | None:
    for d in row.get("judge_dimensions", []):
        if d.get("criterion") == "Audible-Specific Corrective Feedback":
            return d.get("process")
    return None


def _classify_band(row: dict[str, Any]) -> str | None:
    """Return the band membership for the row, or None if it has no judge data.

    Order matters: weak_dim takes priority over composite-band when ASCF
    process <= 1 because the protocol explicitly oversamples weak-ASCF cases
    regardless of where their composite lands.

    Bands partition the composite range with no gap: high >= 2.7,
    threshold in [2.3, 2.7), low < 2.3.
    """
    ascf_p = _row_ascf_process(row)
    if ascf_p is not None and ascf_p <= 1:
        return "weak_dim"
    composite = _row_composite(row)
    if composite is None:
        return None
    if composite >= 2.7:
        return "high"
    if composite >= 2.3:
        return "threshold"
    return "low"


def _load_valid_rows(source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            if not row.get("judge_dimensions"):
                continue
            rows.append(row)
    return rows


def select_sample(
    source_path: Path,
    target_n: int,
    holdout_n: int,
    anchor_n: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = _load_valid_rows(source_path)

    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in _BAND_TARGETS}
    for r in rows:
        band = _classify_band(r)
        if band is None:
            continue
        by_band[band].append(r)

    if sum(_BAND_TARGETS.values()) != target_n:
        raise ValueError(
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}. "
            f"Update _BAND_TARGETS or pass matching target_n."
        )
    band_targets = dict(_BAND_TARGETS)

    main: list[dict[str, Any]] = []
    band_counts: dict[str, int] = {}
    for band, target in band_targets.items():
        pool = by_band[band]
        if len(pool) < target:
            raise ValueError(
                f"Band '{band}' has {len(pool)} rows but needs {target}. "
                f"Source pool too small or distribution too skewed."
            )
        rng.shuffle(pool)
        chosen = pool[:target]
        main.extend(chosen)
        band_counts[band] = len(chosen)

    rng.shuffle(main)

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": None,
                "skill_bucket": r["skill_bucket"],
                "is_anchor_seed": False,
                "anchor_position": None,
            }
            for r in main
        ],
        "anchors": [],
        "holdout": [],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": 0,
            "n_holdout": 0,
            "band_counts": band_counts,
        },
    }
