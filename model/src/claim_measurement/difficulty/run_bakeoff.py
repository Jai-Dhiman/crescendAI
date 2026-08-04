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
    features37        -- write the #137 37-feature vectors into
                          emb/features37/ so `eval` scores the hand-feature
                          baseline through the SAME folds as the encoder arms
                          (offline: reads tk_ablation.py's feature cache, no
                          MIDI parsing, no model)
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
    ManifestEntry,
    composer_stratified_sample,
    load_bakeoff_manifest,
)
from claim_measurement.difficulty.extract import extract_embeddings
from claim_measurement.difficulty.features37_backbone import (
    CachedFeature37Backbone,
    load_feature37_cache,
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


def _stage_features37(paths) -> None:
    """Score-ready feature arm: one .npz per sampled piece holding the 37
    cached hand features, written through the same extract_embeddings path the
    encoder arms used so grades, composer ids, and the .npz contract are
    identical by construction."""
    sample = json.loads((paths.emb_root / "sample_manifest.json").read_text())
    entries = [ManifestEntry(**e) for e in sample]
    backbone = CachedFeature37Backbone(
        by_key=load_feature37_cache(paths.feature37_cache),
        seg_id_to_key={e.seg_id: e.key for e in entries},
    )
    report = extract_embeddings(
        backbone, entries,
        midi_dir=paths.transkun_mid_dir,
        out_dir=paths.emb_root / "emb" / "features37",
        composer_index_path=paths.emb_root / "composer_index.json")
    print(f"features37: ok={report.ok} skipped={report.skipped} "
          f"failed={len(report.failed)}")
    for failure in report.failed[:10]:
        print(f"  fail {failure}")


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
    ap.add_argument("--stage", required=True,
                    choices=["sample", "extract-aria", "extract-moonbeam",
                             "features37", "eval"])
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
    elif args.stage == "features37":
        _stage_features37(paths)
    elif args.stage == "eval":
        print(json.dumps(_stage_eval(paths), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
