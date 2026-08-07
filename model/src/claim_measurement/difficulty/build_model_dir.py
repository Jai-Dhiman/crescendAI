"""#166 (#104 S1): assemble a `score_wav.load_scorer`-shaped model directory.

    <model-dir>/adapter/          the peft LoRA adapter
    <model-dir>/ridge_head.npz    the flattened StandardScaler + Ridge
    <model-dir>/manifest.json     provenance

Two sources, and the difference between them is the whole point:

**`--from-fold F`** builds a directory from one of the five #149 per-fold
artifacts. That fold's adapter trained on a composer-disjoint subset, so the
head is fit with that fold's own TEST pieces excluded -- otherwise the head
would see rows the adapter never trained on and the directory would not be a
deployment of anything we measured. This is a scaffold: it exists so the
end-to-end seam can be executed before the all-data model is trained.

**`--from-all-data`** builds the actual submission directory from an adapter
trained on all 5,798 pieces, with no exclusions, because the MIREX test set is
disjoint from PSyllabus by construction.

**Neither directory may be used to report a tau-c.** Per #104: the recipe is
validated on folds; a model directory is a deployment of that recipe.

    cd model && uv run python -m claim_measurement.difficulty.build_model_dir \\
        --from-fold 0 \\
        --fold-emb-dir .../results/phase1_lora/fold_embeddings \\
        --data-root .../model/data --out-dir .../results/phase1_lora/model_fold0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds
from claim_measurement.difficulty.fold_plan import ALL_DATA_FOLD
from claim_measurement.difficulty.score_wav import (
    ADAPTER_SUBDIR,
    HEAD_FILENAME,
    fit_head_from_fold_embeddings,
    write_manifest,
    write_ridge_head,
)

N_FOLDS, SEED = 5, 2026


def fold_test_seg_ids(seg_ids: list, composer_ids: np.ndarray, fold: int,
                      n_folds: int = N_FOLDS, seed: int = SEED) -> list:
    """The seg_ids fold F holds out. Delegates to the same
    `composer_disjoint_folds` at the same (n_folds, seed) every #149
    measurement used -- "the same folds" only holds if it is literally the same
    function, and a drifted split here would silently fit the head on rows this
    adapter never saw."""
    test_idx = composer_disjoint_folds(composer_ids, n_folds, seed)[fold]
    return [seg_ids[i] for i in test_idx]


def build_model_dir(out_dir: Path, adapter_src: Path, head, **provenance) -> Path:
    """Copy the adapter and write the head + manifest. The adapter is COPIED,
    not symlinked: this directory is what goes into a container image, and a
    symlink into the research tree would resolve to nothing there."""
    out_dir = Path(out_dir)
    adapter_dst = out_dir / ADAPTER_SUBDIR
    if adapter_dst.exists():
        shutil.rmtree(adapter_dst)
    shutil.copytree(Path(adapter_src), adapter_dst)
    write_ridge_head(out_dir / HEAD_FILENAME, head)
    write_manifest(out_dir, **provenance)
    return out_dir


def _build_all_data(artifact_dir: Path, out_dir: Path) -> int:
    """The submission directory. No exclusions: the fold-99 adapter trained on
    every pool piece, so the head is fit on every row of emb_fold99.npz.

    Deliberately does NOT touch features37 or composer_disjoint_folds. There is
    no fold to be consistent with, and reaching for one here would be the first
    step toward reporting a tau-c off this artifact -- which is train-on-test by
    construction, since every labelled piece we hold is in its training set.
    """
    artifact_dir = Path(artifact_dir)
    emb_path = artifact_dir / f"emb_fold{ALL_DATA_FOLD}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} does not exist -- --from-all-data expects the "
            f"fold-{ALL_DATA_FOLD} training run's output dir")

    head = fit_head_from_fold_embeddings(emb_path, exclude_seg_ids=None)
    from claim_measurement.difficulty.train_fold import read_fold_embeddings

    n_rows = len(read_fold_embeddings(emb_path)["seg_ids"])

    build_model_dir(
        out_dir, artifact_dir / ADAPTER_SUBDIR, head,
        kind="SUBMISSION model -- trained on all pool pieces, nothing held out",
        fold=ALL_DATA_FOLD,
        head_train_rows=n_rows,
        held_out_pieces=0,
        embedding_dim=head.n_features,
        fallback_score=head.fallback_score,
        adapter_source=str(artifact_dir / ADAPTER_SUBDIR),
        warning="No tau-c may be reported from this directory: every labelled "
                "piece is in its training set, so any evaluation is "
                "train-on-test. The recipe was validated on folds; see #104.")

    print(f"wrote {out_dir}: SUBMISSION adapter (fold {ALL_DATA_FOLD}), head fit "
          f"on {n_rows} rows with no exclusions, dim={head.n_features}, "
          f"fallback={head.fallback_score}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-fold", type=int, default=None,
                    help="build the scaffold directory from per-fold artifact F")
    ap.add_argument(
        "--from-all-data", type=Path, default=None,
        help="build the SUBMISSION directory from a fold-99 artifact dir "
             "(adapter/ + emb_fold99.npz from the all-data training run)")
    ap.add_argument("--fold-emb-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--features37-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if (args.from_fold is None) == (args.from_all_data is None):
        ap.error("pass exactly one of --from-fold or --from-all-data")

    if args.from_all_data is not None:
        return _build_all_data(args.from_all_data, args.out_dir)

    if args.fold_emb_dir is None:
        ap.error("--fold-emb-dir is required with --from-fold")

    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        features37_seg_ids,
        resolve_paths,
    )

    f37_dir = (args.features37_dir if args.features37_dir is not None
               else features37_dir(resolve_paths(args.data_root).emb_root))
    seg_ids = features37_seg_ids(f37_dir)
    composer_ids = np.array([read_embedding_npz(f37_dir / f"{s}.npz").composer_id
                             for s in seg_ids])
    held_out = fold_test_seg_ids(seg_ids, composer_ids, args.from_fold)

    fold_dir = Path(args.fold_emb_dir) / f"fold{args.from_fold}"
    head = fit_head_from_fold_embeddings(
        Path(args.fold_emb_dir) / f"emb_fold{args.from_fold}.npz",
        exclude_seg_ids=held_out)

    build_model_dir(
        args.out_dir, fold_dir / ADAPTER_SUBDIR, head,
        kind="per-fold scaffold -- NOT the submission model",
        fold=args.from_fold, n_folds=N_FOLDS, seed=SEED,
        head_train_rows=len(seg_ids) - len(held_out),
        held_out_pieces=len(held_out),
        embedding_dim=head.n_features,
        fallback_score=head.fallback_score,
        adapter_source=str(fold_dir / ADAPTER_SUBDIR),
        warning="No tau-c may be reported from this directory; see #104.")

    print(f"wrote {args.out_dir}: adapter from fold {args.from_fold}, head fit "
          f"on {len(seg_ids) - len(held_out)} rows "
          f"({len(held_out)} held-out pieces excluded), "
          f"dim={head.n_features}, fallback={head.fallback_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
