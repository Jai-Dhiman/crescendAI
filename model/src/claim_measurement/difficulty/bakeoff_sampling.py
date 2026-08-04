"""Manifest+labels join and composer-stratified sampling for the bake-off.

Pure functions: no MIDI parsing here (that's each Backbone adapter's job at
extraction time) -- this module only decides WHICH pieces make the sample.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ManifestEntry:
    seg_id: str
    key: str
    grade: int
    composer: str


def load_bakeoff_manifest(manifest_path: Path, labels_path: Path, transkun_mid_dir: Path) -> list[ManifestEntry]:
    """Join amt_gap_curve/manifest.json (seg_id/key/grade) against
    new_clean_data.json (key -> composer), keeping only entries that have
    BOTH a non-empty composer AND an on-disk Transkun MIDI -- a composer-less
    entry cannot be placed in a composer-disjoint fold, and a piece without a
    Transkun MIDI cannot be embedded."""
    manifest = json.loads(Path(manifest_path).read_text())
    labels = json.loads(Path(labels_path).read_text())
    entries = []
    for m in manifest:
        composer = str(labels.get(m["key"], {}).get("composer", "")).strip()
        if not composer:
            continue
        if not (Path(transkun_mid_dir) / f"{m['seg_id']}.mid").exists():
            continue
        entries.append(ManifestEntry(seg_id=m["seg_id"], key=m["key"],
                                      grade=int(m["grade"]), composer=composer))
    return entries


def composer_stratified_sample(entries: list[ManifestEntry], target_n: int, seed: int) -> list[ManifestEntry]:
    """Draw up to target_n entries so every composer present in `entries`
    contributes at least one piece to the output (when target_n >= the
    number of distinct composers), proportional to each composer's share of
    `entries` beyond that floor, deterministic given `seed`."""
    if target_n <= 0:
        raise ValueError("target_n must be positive")
    if target_n >= len(entries):
        return list(entries)

    by_composer: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        by_composer.setdefault(e.composer, []).append(e)
    composers = sorted(by_composer)
    rng = np.random.default_rng(seed)
    for c in composers:
        rng.shuffle(by_composer[c])  # deterministic per-composer order

    if len(composers) > target_n:
        keep = sorted(rng.permutation(composers)[:target_n].tolist())
        return [by_composer[c][0] for c in keep]

    quotas = {c: 1 for c in composers}
    remaining = target_n - len(composers)
    total = len(entries)
    for c in composers:
        share = len(by_composer[c]) / total
        quotas[c] += int(round(share * remaining))

    # Rounding can push sum(quotas) above target_n. Shrink the surplus
    # (quota - 1) off the composers with the largest surplus first, never
    # below the floor of 1, so every composer keeps at least one piece.
    overflow = sum(quotas.values()) - target_n
    if overflow > 0:
        order = sorted(composers, key=lambda c: quotas[c] - 1, reverse=True)
        i = 0
        while overflow > 0:
            c = order[i % len(order)]
            if quotas[c] > 1:
                quotas[c] -= 1
                overflow -= 1
            i += 1

    sample: list[ManifestEntry] = []
    for c in composers:
        take = min(quotas[c], len(by_composer[c]))
        sample.extend(by_composer[c][:take])

    if len(sample) > target_n:
        # Defensive: quota capping above should already bring sum(sample)
        # to <= target_n, but guard against residual surplus by truncating
        # only entries beyond each composer's first (protected) piece.
        protected_ids = {by_composer[c][0].seg_id for c in composers}
        protected = [e for e in sample if e.seg_id in protected_ids]
        surplus = [e for e in sample if e.seg_id not in protected_ids]
        rng.shuffle(surplus)
        sample = protected + surplus[: target_n - len(protected)]
    elif len(sample) < target_n:
        taken_ids = {e.seg_id for e in sample}
        leftover = [e for e in entries if e.seg_id not in taken_ids]
        rng.shuffle(leftover)
        sample.extend(leftover[: target_n - len(sample)])

    return sample
