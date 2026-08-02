"""CLI stage dispatch for the frozen backbone bake-off (#138 Phase 0).

Stages:
    sample           -- draw the composer-stratified Transkun sample
    extract-aria      -- human-lit GPU stage (needs real Aria weights); not
                          wired into this offline CLI, run interactively
                          against AriaBackbone (see docs/specs/2026-08-02-
                          backbone-bakeoff-design.md)
    extract-moonbeam  -- points at moonbeam_extract_script.py, which must run
                          under the isolated MoonBeam venv (see that file's
                          docstring)
    eval              -- composer-disjoint tau-c for whichever backbone(s)
                          have extracted embeddings under
                          --data-root/results/bakeoff/emb/{backbone}/

Usage:
    cd model && uv run python -m claim_measurement.difficulty.run_bakeoff \
        --stage sample [--target-n 900] [--data-root PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import oof_tau_ridge
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.bakeoff_paths import resolve_paths
from claim_measurement.difficulty.bakeoff_sampling import (
    composer_stratified_sample,
    load_bakeoff_manifest,
)

N_FOLDS = 5
SEEDS = list(range(2026, 2031))


def _stage_sample(paths, target_n: int) -> None:
    entries = load_bakeoff_manifest(paths.manifest, paths.labels, paths.transkun_mid_dir)
    sample = composer_stratified_sample(entries, target_n, seed=2026)
    paths.emb_root.mkdir(parents=True, exist_ok=True)
    out = paths.emb_root / "sample_manifest.json"
    out.write_text(json.dumps([e.__dict__ for e in sample], indent=2))
    print(f"sampled {len(sample)}/{len(entries)} eligible pieces -> {out}")


def _stage_eval(paths) -> dict:
    """Per-backbone, per-pooling composer-disjoint tau-c, from whatever
    backbone_dir/*.npz files exist under paths.emb_root/emb/."""
    results = {}
    emb_dir = paths.emb_root / "emb"
    if not emb_dir.exists():
        return results
    for backbone_dir in sorted(p for p in emb_dir.glob("*") if p.is_dir()):
        npz_paths = sorted(backbone_dir.glob("*.npz"))
        if not npz_paths:
            continue
        by_pooling: dict = {}
        grades, composer_ids = [], []
        for npz_path in npz_paths:
            record = read_embedding_npz(npz_path)
            for pooling_name, vec in record.embeddings.items():
                by_pooling.setdefault(pooling_name, []).append(vec)
            grades.append(record.grade)
            composer_ids.append(record.composer_id)
        y = np.array(grades)
        composers = np.array(composer_ids)
        results[backbone_dir.name] = {
            pooling_name: oof_tau_ridge(np.stack(vecs), y, composers, N_FOLDS, SEEDS)
            for pooling_name, vecs in by_pooling.items()
        }
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["sample", "extract-aria", "extract-moonbeam", "eval"])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--target-n", type=int, default=900)
    args = ap.parse_args(argv)

    paths = resolve_paths(args.data_root)

    if args.stage == "sample":
        _stage_sample(paths, args.target_n)
    elif args.stage == "extract-aria":
        print("extract-aria: human-lit GPU stage, use claim_measurement.difficulty.aria_backbone.AriaBackbone "
              "+ claim_measurement.difficulty.extract.extract_embeddings directly (see design spec)")
    elif args.stage == "extract-moonbeam":
        print("Run under the isolated MoonBeam venv: see moonbeam_extract_script.py's module docstring")
    elif args.stage == "eval":
        print(json.dumps(_stage_eval(paths), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
