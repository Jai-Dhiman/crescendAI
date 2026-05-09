"""Stratified sample selector for the rubric calibration protocol.

Band stratification (T3) + era min-quotas (this task). Holdout reservation
arrives in T10, anchor injection in T13.
"""
from __future__ import annotations

import hashlib
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

_SKILL_MIN_QUOTAS: dict[str, int] = {
    "beginner": 50,
    "intermediate": 50,
    "advanced": 50,
}


def _skill_group(skill_bucket: int) -> str:
    if skill_bucket <= 2:
        return "beginner"
    if skill_bucket == 3:
        return "intermediate"
    return "advanced"


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


def _make_scrambled_id(synth_id: str, seed: int, position: int) -> str:
    h = hashlib.sha256(f"{seed}:{position}:{synth_id}".encode()).hexdigest()
    return f"anchor-{h[:16]}"


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
    skill_min_quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    era_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    band_counter: Counter[str] = Counter()

    def _take(r: dict[str, Any], band: str) -> None:
        chosen.append(r)
        chosen_ids.add(_row_synth_id(r))
        era_counter[composer_to_era(r["composer"])] += 1
        skill_counter[_skill_group(r["skill_bucket"])] += 1
        band_counter[band] += 1

    # Pass 1: era reservation
    for era, min_q in era_min_quotas.items():
        candidates = [
            (band, r) for band, pool in by_band.items() for r in pool
            if composer_to_era(r["composer"]) == era and _row_synth_id(r) not in chosen_ids
        ]
        rng.shuffle(candidates)
        if len(candidates) < min_q:
            raise ValueError(f"Era '{era}' has {len(candidates)} candidates but needs {min_q}.")
        for band, r in candidates:
            if era_counter[era] >= min_q:
                break
            if band_counter[band] >= band_targets[band]:
                continue
            _take(r, band)
        if era_counter[era] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                _take(r, band)
                if era_counter[era] >= min_q:
                    break

    # Pass 2: skill reservation (covers groups under-met by era pass)
    for group, min_q in skill_min_quotas.items():
        if skill_counter[group] >= min_q:
            continue
        candidates = [
            (band, r) for band, pool in by_band.items() for r in pool
            if _skill_group(r["skill_bucket"]) == group and _row_synth_id(r) not in chosen_ids
        ]
        rng.shuffle(candidates)
        needed = min_q - skill_counter[group]
        if len(candidates) < needed:
            raise ValueError(
                f"Skill group '{group}' has {len(candidates)} unreserved candidates "
                f"but needs {needed} more."
            )
        for band, r in candidates:
            if skill_counter[group] >= min_q:
                break
            if band_counter[band] >= band_targets[band]:
                continue
            _take(r, band)
        if skill_counter[group] < min_q:
            for band, r in candidates:
                if _row_synth_id(r) in chosen_ids:
                    continue
                _take(r, band)
                if skill_counter[group] >= min_q:
                    break

    # Pass 3: fill remaining band slots
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
            _take(r, band)

    # Pass 4: trim overshoot (quota fills can push past target_n)
    while len(chosen) > sum(band_targets.values()):
        for band, target in band_targets.items():
            if band_counter[band] <= target:
                continue
            for i, r in enumerate(chosen):
                if _classify_band(r) != band:
                    continue
                era = composer_to_era(r["composer"])
                grp = _skill_group(r["skill_bucket"])
                if era_counter[era] - 1 < era_min_quotas.get(era, 0):
                    continue
                if skill_counter[grp] - 1 < skill_min_quotas.get(grp, 0):
                    continue
                chosen.pop(i)
                chosen_ids.discard(_row_synth_id(r))
                era_counter[era] -= 1
                skill_counter[grp] -= 1
                band_counter[band] -= 1
                break
            else:
                continue
            break
        else:
            raise ValueError(
                "Cannot satisfy all quotas simultaneously: overshoot is locked by "
                "minima in every band. Loosen a quota or expand source pool."
            )

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

    valid_with_band: list[tuple[dict[str, Any], str]] = []
    for r in rows:
        band = _classify_band(r)
        if band is not None:
            valid_with_band.append((r, band))

    if len(valid_with_band) < holdout_n + target_n:
        raise ValueError(
            f"Source has only {len(valid_with_band)} band-classified rows "
            f"but needs at least {holdout_n + target_n} (holdout + main). "
            f"Band/era feasibility is verified downstream."
        )

    # Use a separate RNG for holdout so the main selection RNG state is unaffected.
    # Addition (not XOR) ensures no two seeds share the same derived seed.
    holdout_rng = random.Random(seed + 0xDEAD)
    holdout_pool = list(valid_with_band)
    holdout_rng.shuffle(holdout_pool)
    holdout_synth_ids = {_row_synth_id(r) for r, _ in holdout_pool[:holdout_n]}
    holdout_rows = [r for r, _ in holdout_pool[:holdout_n]]

    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in _BAND_TARGETS}
    for r, band in valid_with_band:
        if _row_synth_id(r) not in holdout_synth_ids:
            by_band[band].append(r)

    if sum(_BAND_TARGETS.values()) != target_n:
        raise ValueError(
            f"Band targets sum to {sum(_BAND_TARGETS.values())} but target_n={target_n}."
        )
    band_targets = dict(_BAND_TARGETS)

    main = _select_with_quotas(by_band, band_targets, _ERA_MIN_QUOTAS, _SKILL_MIN_QUOTAS, rng)
    rng.shuffle(main)

    band_counts = Counter(_classify_band(r) for r in main)
    era_counts = Counter(composer_to_era(r["composer"]) for r in main)
    skill_group_counts = Counter(_skill_group(r["skill_bucket"]) for r in main)

    if anchor_n > len(main):
        raise ValueError(f"anchor_n={anchor_n} exceeds main size {len(main)}")
    anchor_indices = list(range(len(main)))
    rng.shuffle(anchor_indices)
    anchor_indices = sorted(anchor_indices[:anchor_n])

    anchors: list[dict[str, Any]] = []
    for k, idx in enumerate(anchor_indices):
        original = main[idx]
        synth_id = _row_synth_id(original)
        display_position = len(main) + k
        anchors.append({
            "synth_id": synth_id,
            "synth_id_displayed": _make_scrambled_id(synth_id, seed, display_position),
            "display_position": display_position,
            "original_main_index": idx,
        })

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
                "is_anchor_seed": _row_synth_id(r) in {a["synth_id"] for a in anchors},
                "anchor_position": next(
                    (a["display_position"] for a in anchors
                     if a["synth_id"] == _row_synth_id(r)),
                    None,
                ),
            }
            for r in main
        ],
        "anchors": anchors,
        "holdout": [
            {
                "synth_id": _row_synth_id(r),
                "band": _classify_band(r),
                "era": composer_to_era(r["composer"]),
                "skill_bucket": r["skill_bucket"],
            }
            for r in holdout_rows
        ],
        "stats": {
            "n_main": len(main),
            "n_anchors_silent_dups": len(anchors),
            "n_holdout": len(holdout_rows),
            "band_counts": dict(band_counts),
            "era_counts": dict(era_counts),
            "skill_group_counts": dict(skill_group_counts),
        },
    }
