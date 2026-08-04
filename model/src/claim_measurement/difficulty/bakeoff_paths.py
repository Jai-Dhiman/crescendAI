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
