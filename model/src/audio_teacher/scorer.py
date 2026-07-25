"""Probe scoring: per-(axis, population) accuracy + ex-ante gate verdict.

Populations are NEVER pooled (issue #21 synthetic-gap scar): every number
in the report is keyed "axis/population". The verdict reads ONLY real-
population cells; synthetic cells are informative. Uncertainty (too few
real pairs, too many unparseable responses) FAILS -- the gate defaults
to closed. Thresholds are ex-ante constants; there is deliberately no
override knob.
"""
from __future__ import annotations

import json
from typing import Mapping

from audio_teacher.manifest import ProbeManifest
from audio_teacher.prompts import parse_choice

KILL_THRESHOLD = 0.70
MIN_REAL_PAIRS_PER_AXIS = 20
MAX_UNPARSEABLE_RATE = 0.10


class ProbeIncompleteError(Exception):
    """Responses do not cover the manifest exactly (missing or extra pairs)."""


def score_responses(manifest: ProbeManifest, responses: Mapping[str, str]) -> dict:
    manifest_ids = {p.pair_id for p in manifest.pairs}
    missing = sorted(manifest_ids - set(responses))
    extra = sorted(set(responses) - manifest_ids)
    if missing or extra:
        raise ProbeIncompleteError(
            f"responses do not match manifest: missing={missing} extra={extra}"
        )

    cells: dict[str, dict] = {}
    for pair in manifest.pairs:
        key = f"{pair.axis}/{pair.population}"
        cell = cells.setdefault(key, {"n": 0, "correct": 0, "unparseable": 0})
        cell["n"] += 1
        choice = parse_choice(responses[pair.pair_id])
        if choice is None:
            cell["unparseable"] += 1
        elif choice == pair.degraded:
            cell["correct"] += 1
    for cell in cells.values():
        cell["accuracy"] = cell["correct"] / cell["n"]
        cell["unparseable_rate"] = cell["unparseable"] / cell["n"]

    reasons = _verdict_reasons(manifest, cells)
    return {
        "schema_version": 1,
        "thresholds": {
            "kill_threshold": KILL_THRESHOLD,
            "min_real_pairs_per_axis": MIN_REAL_PAIRS_PER_AXIS,
            "max_unparseable_rate": MAX_UNPARSEABLE_RATE,
        },
        "cells": cells,
        "verdict": "PASS" if not reasons else "FAIL",
        "verdict_reasons": reasons,
    }


def _verdict_reasons(manifest: ProbeManifest, cells: dict[str, dict]) -> list[str]:
    reasons: list[str] = []
    for axis in sorted({p.axis for p in manifest.pairs}):
        real = cells.get(f"{axis}/real")
        if real is None:
            reasons.append(
                f"{axis}: no real-population pairs; synthetic alone never opens "
                f"the gate (issue #21 synthetic-gap)"
            )
            continue
        if real["n"] < MIN_REAL_PAIRS_PER_AXIS:
            reasons.append(
                f"{axis}/real: only {real['n']} pairs, need >= {MIN_REAL_PAIRS_PER_AXIS}"
            )
        if real["unparseable_rate"] > MAX_UNPARSEABLE_RATE:
            reasons.append(
                f"{axis}/real: unparseable rate {real['unparseable_rate']:.2f} "
                f"above {MAX_UNPARSEABLE_RATE:.2f} (ambiguous -> gate stays closed)"
            )
        if real["accuracy"] < KILL_THRESHOLD:
            reasons.append(
                f"{axis}/real: accuracy {real['accuracy']:.2f} below "
                f"{KILL_THRESHOLD:.2f} kill threshold"
            )
    return reasons


def render_report(report: dict) -> str:
    """Deterministic serialization: the same report dict always renders to
    byte-identical text (sorted keys, fixed indent, trailing newline). The
    report carries no timestamps -- volatile run metadata belongs in
    run_meta.json, written by the probe driver."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
