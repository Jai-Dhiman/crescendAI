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


def _referenced_seg_ids(plans: list[FoldPlan]) -> list:
    seg_ids: set = set()
    for plan in plans:
        seg_ids.update(plan.train_seg_ids)
        seg_ids.update(plan.val_seg_ids)
        seg_ids.update(plan.test_seg_ids)
    return sorted(seg_ids)


def stage_training_bundle(
    paths: BundleSources, plans: list[FoldPlan], staging_dir: Path
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


def main(argv=None, uploader=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument(
        "--sample-manifest", type=Path, required=True,
        help="the 900-piece eval sample_manifest.json "
             "(run_bakeoff.py --stage sample's output)")
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--repo-snapshot-dir", type=Path, required=True)
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument(
        "--repo-id", required=True,
        help="private HF dataset repo id, e.g. jaidhiman/phase1-lora-bundle")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val-frac", type=float, default=0.12)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_sampling import load_bakeoff_manifest
    from claim_measurement.difficulty.fold_plan import build_fold_plans

    pool_entries = load_bakeoff_manifest(args.manifest, args.labels, args.midi_dir)
    sample_seg_ids = {e["seg_id"] for e in json.loads(args.sample_manifest.read_text())}
    eval_entries = sorted(
        (e for e in pool_entries if e.seg_id in sample_seg_ids),
        key=lambda e: e.seg_id)

    plans = build_fold_plans(
        eval_entries, pool_entries, args.n_folds, args.seed, args.val_frac)

    grades = {e.seg_id: e.grade for e in pool_entries}
    paths = BundleSources(
        midi_dir=args.midi_dir, grades=grades,
        repo_snapshot_dir=args.repo_snapshot_dir)
    report = stage_training_bundle(paths, plans, args.staging_dir)
    print(f"staged {report.n_midis} MIDIs, {report.n_fold_plans} fold plans, "
          f"{report.repo_snapshot_files} repo files, checksum {report.checksum}")

    if uploader is None:
        from huggingface_hub import HfApi

        def uploader(staged_dir: Path, repo_id: str) -> None:
            api = HfApi()
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(staged_dir), repo_id=repo_id, repo_type="dataset")

    uploader(args.staging_dir, args.repo_id)
    print(f"uploaded to {args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
