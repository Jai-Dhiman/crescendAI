"""Stratified sample selector for the rubric calibration protocol.

Band stratification (T3) + era min-quotas (this task). Holdout reservation
arrives in T10, anchor injection in T13.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from teacher_model.calibration.era_lookup import composer_to_era

_BAND_TARGETS: dict[str, int] = {
    "threshold": 80,
    "high": 40,
    "low": 30,
    "weak_dim": 50,
}

_ERA_MIN_QUOTAS: dict[str, int] = {
    "Baroque": 30,
    "Classical": 30,
    "Romantic": 30,
    "Impressionist": 30,
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


def _select_with_quotas(
    by_band: dict[str, list[dict[str, Any]]],
    band_targets: dict[str, int],
    era_min_quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Two-pass selection:
    1. Reserve era min-quotas first by drawing era-pure picks across bands.
    2. Fill remaining band slots from the unreserved pool.
    Raises ValueError if either constraint cannot be satisfied.
    """
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    era_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()

    # Pass 1: era reservation
    for era, min_q in era_min_quotas.items():
        candidates: list[tuple[str, dict[str, Any]]] = []
        for band, pool in by_band.items():
            for r in pool:
                if composer_to_era(r["composer"]) == era and _row_synth_id(r) not in chosen_ids:
                    candidates.append((band, r))
        rng.shuffle(candidates)
        if len(candidates) < min_q:
            raise ValueError(
                f"Era '{era}' has only {len(candidates)} candidates but needs {min_q}."
            )
        for band, r in candidates[:min_q]:
            if band_counter[band] >= band_targets[band]:
                continue
            chosen.append(r)
            chosen_ids.add(_row_synth_id(r))
            era_counter[era] += 1
            band_counter[band] += 1
            if era_counter[era] >= min_q:
                break
        if era_counter[era] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                chosen.append(r)
                chosen_ids.add(_row_synth_id(r))
                era_counter[era] += 1
                band_counter[band] += 1
                if era_counter[era] >= min_q:
                    break

    # Pass 2: fill remaining band slots
    for band, target in band_targets.items():
        remaining = target - band_counter[band]
        if remaining <= 0:
            continue
        pool = [r for r in by_band[band] if _row_synth_id(r) not in chosen_ids]
        rng.shuffle(pool)
        if len(pool) < remaining:
            raise ValueError(
                f"Band '{band}' has only {len(pool)} unreserved rows but needs {remaining} more."
            )
        for r in pool[:remaining]:
            chosen.append(r)
            chosen_ids.add(_row_synth_id(r))
            band_counter[band] += 1
            era_counter[composer_to_era(r["composer"])] += 1

    return chosen


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
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}."
        )
    band_targets = dict(_BAND_TARGETS)

    main = _select_with_quotas(by_band, band_targets, _ERA_MIN_QUOTAS, rng)
    rng.shuffle(main)

    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)

    return {
        "version": 1,
        "seed": seed,
        "source_path": str(source_path),
        "main": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
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
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
        },
    }
