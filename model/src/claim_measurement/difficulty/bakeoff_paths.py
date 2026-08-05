"""__file__-anchored default data paths for the backbone bake-off (#138 Phase 0).

Worktrees each have their own (independently populated) model/data/ directory.
The 5798 Transkun MIDIs, manifest.json, and psyllabus labels currently exist
only under the main checkout's model/data/ -- so every path here is overridable
via `resolve_paths(data_root=...)` (and run_bakeoff.py's --data-root), never
hardcoded to an absolute machine path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class BakeoffPaths:
    manifest: Path
    labels: Path
    transkun_mid_dir: Path
    emb_root: Path
    feature37_cache: Path


def resolve_paths(data_root: Path | None = None) -> BakeoffPaths:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return BakeoffPaths(
        manifest=root / "results" / "amt_gap_curve" / "manifest.json",
        labels=root / "raw" / "psyllabus" / "new_clean_data.json",
        transkun_mid_dir=root / "results" / "amt_gap_curve" / "transkun_mid",
        emb_root=root / "results" / "bakeoff",
        feature37_cache=root / "results" / "mirex_137_tk_features.json",
    )


def features37_dir(emb_root: Path) -> Path:
    return Path(emb_root) / "emb" / "features37"


def features37_seg_ids(features37_dir: Path) -> list[str]:
    """The CANONICAL eval-piece row order: the features37 .npz filenames,
    sorted. Every artifact that is compared row-for-row against features37 --
    ft_eval.py's per-fold emb_fold{F}.npz, push_train_dataset.py's staged
    eval_manifest.json -- must be built from this one function, because
    ft_eval.py rejects any emb_fold{F}.npz whose seg_ids do not match it
    exactly (an unpaired comparison is worse than no comparison)."""
    features37_dir = Path(features37_dir)
    paths = sorted(features37_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no features37 .npz files under {features37_dir}")
    return [p.stem for p in paths]
