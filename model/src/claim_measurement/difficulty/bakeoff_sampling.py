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
