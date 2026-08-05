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
    eval_manifest: list


# Junk the fork snapshot carries that must never reach the bundle: .git is the
# fork's full history (the bulk of a 119 MB bundle), and __pycache__/*.pyc are
# stale bytecode from local runs, which would shadow the .py files they were
# compiled from if their timestamps happened to line up in the container.
_REPO_SNAPSHOT_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc")


def build_eval_manifest(features37_dir: Path) -> list[dict]:
    """{seg_id, grade, composer_id} for every eval piece, in the SAME row
    order ft_eval.py's _load_features37 reads -- both come from
    bakeoff_paths.features37_seg_ids. grade/composer_id are read out of the
    features37 .npz files themselves rather than re-joined from the label
    JSON, so the manifest cannot disagree with the arm it is compared against.
    ft_eval.py rejects any emb_fold{F}.npz whose seg_ids do not match this
    order, and train_fold.py emits its rows in exactly this order."""
    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.bakeoff_paths import features37_seg_ids

    features37_dir = Path(features37_dir)
    manifest = []
    for seg_id in features37_seg_ids(features37_dir):
        record = read_embedding_npz(features37_dir / f"{seg_id}.npz")
        manifest.append({"seg_id": seg_id, "grade": record.grade,
                          "composer_id": record.composer_id})
    return manifest


# Modules train_fold.py needs at train time but which `hf jobs uv run` never
# uploads (it uploads only the one file passed on the command line). Staged
# verbatim into the bundle's code/ subdir so train_fold.py can pull them back
# via snapshot_download instead of vendoring a second copy of the loss/tau_c
# implementation that could drift from this one.
_CODE_FILES = ("ranking_loss.py", "bakeoff_cv.py")


@dataclass(frozen=True)
class BundleReport:
    n_midis: int
    n_fold_plans: int
    n_eval_pieces: int
    repo_snapshot_files: int
    code_files: int
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
    (staging_dir / "eval_manifest.json").write_text(
        json.dumps(list(paths.eval_manifest)))

    repo_out = staging_dir / "moonbeam_repo"
    if repo_out.exists():
        shutil.rmtree(repo_out)
    shutil.copytree(paths.repo_snapshot_dir, repo_out,
                     ignore=_REPO_SNAPSHOT_IGNORE)
    repo_files = sum(1 for p in repo_out.rglob("*") if p.is_file())

    code_out = staging_dir / "code"
    code_out.mkdir(parents=True, exist_ok=True)
    module_dir = Path(__file__).resolve().parent
    for name in _CODE_FILES:
        shutil.copy2(module_dir / name, code_out / name)

    hasher = hashlib.sha256()
    for p in sorted(staging_dir.rglob("*")):
        if p.is_file():
            hasher.update(str(p.relative_to(staging_dir)).encode())
            hasher.update(str(p.stat().st_size).encode())

    return BundleReport(n_midis=len(seg_ids), n_fold_plans=len(plans),
                         n_eval_pieces=len(paths.eval_manifest),
                         repo_snapshot_files=repo_files, code_files=len(_CODE_FILES),
                         checksum=hasher.hexdigest())


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
    ap.add_argument(
        "--features37-dir", type=Path, default=None,
        help="the per-piece features37 .npz dir the staged eval_manifest.json "
             "takes its row order, grades and composer ids from; defaults to "
             "<data-root>/results/bakeoff/emb/features37")
    ap.add_argument(
        "--data-root", type=Path, default=None,
        help="only used to locate --features37-dir when that is not given")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val-frac", type=float, default=0.12)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_sampling import load_bakeoff_manifest
    from claim_measurement.difficulty.fold_plan import (
        build_fold_plans,
        check_fold_plans,
    )

    pool_entries = load_bakeoff_manifest(args.manifest, args.labels, args.midi_dir)
    sample_seg_ids = {e["seg_id"] for e in json.loads(args.sample_manifest.read_text())}
    eval_entries = sorted(
        (e for e in pool_entries if e.seg_id in sample_seg_ids),
        key=lambda e: e.seg_id)

    plans = build_fold_plans(
        eval_entries, pool_entries, args.n_folds, args.seed, args.val_frac)

    violations = check_fold_plans(
        plans, eval_entries, pool_entries, args.n_folds, args.seed)
    if violations:
        raise ValueError(
            f"fold plan failed independent re-derivation, refusing to stage or "
            f"upload: {violations}")

    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        resolve_paths,
    )

    f37_dir = (args.features37_dir if args.features37_dir is not None
               else features37_dir(resolve_paths(args.data_root).emb_root))
    eval_manifest = build_eval_manifest(f37_dir)

    grades = {e.seg_id: e.grade for e in pool_entries}
    paths = BundleSources(
        midi_dir=args.midi_dir, grades=grades,
        repo_snapshot_dir=args.repo_snapshot_dir, eval_manifest=eval_manifest)
    report = stage_training_bundle(paths, plans, args.staging_dir)
    print(f"staged {report.n_midis} MIDIs, {report.n_fold_plans} fold plans, "
          f"{report.n_eval_pieces} eval pieces, "
          f"{report.repo_snapshot_files} repo files, {report.code_files} code files, "
          f"checksum {report.checksum}")

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
