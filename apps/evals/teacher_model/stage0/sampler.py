"""Stratified holdout sampler for Stage 0 synthesis eval.

Produces an n=N sample from the briefing pool stratified by era x skill_bucket.
Era is derived from composer via shared.style_rules.composer_to_era; strata
with fewer than _MIN_STRATUM_POPULATION recordings are merged into a fallback
'Other' era stratum to avoid degenerate single-recording cells.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from shared.style_rules import composer_to_era

_MIN_STRATUM_POPULATION = 5


def _stratum_label(era: str, skill_bucket: int) -> str:
    return f"{era}|sk{skill_bucket}"


def sample_holdout(
    briefings_dir: Path,
    manifests: dict[str, dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Return up to n briefings stratified by era x skill_bucket.

    Args:
        briefings_dir: directory containing one .json per recording.
        manifests: video_id -> {composer, skill_bucket, ...} (from load_manifests).
        n: target sample size.
        seed: deterministic seed.

    Each returned row has keys: recording_id, era, skill_bucket, stratum,
    composer, piece_slug, title, briefing_path.
    """
    cache_files = sorted(briefings_dir.glob("*.json"))

    # Build the candidate pool: only recordings present in manifests.
    candidates: dict[str, list[dict]] = {}
    for path in cache_files:
        if path.name == "_fingerprint.json":
            continue
        rid = path.stem
        meta = manifests.get(rid)
        if meta is None:
            continue
        era = composer_to_era(meta.get("composer", ""))
        skill_bucket = int(meta.get("skill_bucket", 3))
        stratum = _stratum_label(era, skill_bucket)
        candidates.setdefault(stratum, []).append(
            {
                "recording_id": rid,
                "era": era,
                "skill_bucket": skill_bucket,
                "stratum": stratum,
                "composer": meta.get("composer", ""),
                "piece_slug": meta.get("piece_slug", ""),
                "title": meta.get("title", ""),
                "briefing_path": str(path),
            }
        )

    # Drop strata smaller than _MIN_STRATUM_POPULATION; fold into an "Other" stratum.
    keep: dict[str, list[dict]] = {}
    other: list[dict] = []
    for stratum, rows in candidates.items():
        if len(rows) >= _MIN_STRATUM_POPULATION:
            keep[stratum] = rows
        else:
            for r in rows:
                r2 = dict(r)
                r2["stratum"] = "Other"
                other.append(r2)
    if other:
        keep["Other"] = other

    if not keep:
        return []

    # Per-stratum target; floor division then distribute remainder by stratum size.
    rng = random.Random(seed)
    strata_sorted = sorted(keep.keys())
    base = n // len(strata_sorted)
    remainder = n - base * len(strata_sorted)

    # Distribute remainder to the largest strata first (deterministic).
    sizes = [(s, len(keep[s])) for s in strata_sorted]
    sizes.sort(key=lambda x: (-x[1], x[0]))
    extra: dict[str, int] = {s: 0 for s in strata_sorted}
    for s, _ in sizes[:remainder]:
        extra[s] = 1

    out: list[dict] = []
    for stratum in strata_sorted:
        rows = list(keep[stratum])
        rng.shuffle(rows)
        target = min(base + extra[stratum], len(rows))
        out.extend(rows[:target])

    out.sort(key=lambda r: r["recording_id"])  # deterministic output ordering
    return out


def write_holdout(rows: list[dict], out_path: Path) -> None:
    """Persist the holdout to a JSONL file (one row per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
