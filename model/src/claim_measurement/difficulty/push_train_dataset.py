"""Hermetic HF Jobs training-bundle staging + upload for #138 Phase 1.

Judgment lives entirely in WHAT gets staged (stage_training_bundle); the
upload itself is three lines behind an injected `uploader` so tests never
touch the network. Staging never fetches anything over the network either --
the MoonBeam fork snapshot and the Transkun MIDIs must already exist locally
(see moonbeam_extract_script.py's SETUP section for the fork clone recipe).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from claim_measurement.difficulty.fold_plan import FoldPlan


@dataclass(frozen=True)
class BundleSources:
    midi_dir: Path
    grades: dict
    repo_snapshot_dir: Path


@dataclass(frozen=True)
class BundleReport:
    n_midis: int
    n_fold_plans: int
    repo_snapshot_files: int
    checksum: str


def _referenced_seg_ids(plans: list) -> list:
    seg_ids: set = set()
    for plan in plans:
        seg_ids.update(plan.train_seg_ids)
        seg_ids.update(plan.val_seg_ids)
        seg_ids.update(plan.test_seg_ids)
    return sorted(seg_ids)


def stage_training_bundle(
    paths: BundleSources, plans: list, staging_dir: Path
) -> BundleReport:
    staging_dir = Path(staging_dir)
    midi_out = staging_dir / "midi"
    midi_out.mkdir(parents=True, exist_ok=True)

    seg_ids = _referenced_seg_ids(plans)
    missing_grades = [s for s in seg_ids if s not in paths.grades]
    if missing_grades:
        raise ValueError(
            f"{len(missing_grades)} piece(s) referenced by a fold plan have no grade: "
            f"{missing_grades[:5]}")

    for seg_id in seg_ids:
        src = Path(paths.midi_dir) / f"{seg_id}.mid"
        if not src.exists():
            raise FileNotFoundError(
                f"fold plan references {seg_id}, but {src} does not exist")
        shutil.copy2(src, midi_out / f"{seg_id}.mid")

    (staging_dir / "grades.json").write_text(json.dumps(
        {seg_id: paths.grades[seg_id] for seg_id in seg_ids}))
    (staging_dir / "fold_plans.json").write_text(
        json.dumps([dataclasses.asdict(p) for p in plans]))

    repo_out = staging_dir / "moonbeam_repo"
    if repo_out.exists():
        shutil.rmtree(repo_out)
    shutil.copytree(paths.repo_snapshot_dir, repo_out)
    repo_files = sum(1 for p in repo_out.rglob("*") if p.is_file())

    hasher = hashlib.sha256()
    for p in sorted(staging_dir.rglob("*")):
        if p.is_file():
            hasher.update(str(p.relative_to(staging_dir)).encode())
            hasher.update(str(p.stat().st_size).encode())

    return BundleReport(n_midis=len(seg_ids), n_fold_plans=len(plans),
                         repo_snapshot_files=repo_files, checksum=hasher.hexdigest())


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 23
