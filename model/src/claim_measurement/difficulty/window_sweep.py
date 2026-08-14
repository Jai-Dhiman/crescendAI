# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "scipy>=1.10.0", "torch>=2.0.0", "peft==0.11.1",
#     "scikit-learn",  # lazy in bakeoff_cv; unused on this path, declared anyway
#     "pretty_midi",  # lazily imported by write_notes_midi -- fatal at CALL time
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#168 S3 lever 1: sweep the EVAL-time window (chunk) length.

train_fold.py trains on one random max_len=1024-token crop per piece
(`_random_window`) and extracts full-piece embeddings chunked at the same
1024 (`_extract_full_piece`). The chunk length used at extraction has never
been swept; Track B's post-mortem says window granularity, not head
architecture, was the lever there. This harness re-extracts the 900 eval
pieces through the EXISTING five fold adapters at other chunk lengths -- no
retraining.

Measured before building this (note counts over the 900 bundle eval MIDIs,
one compound token per note): median 456, p75 987 -- 76% of pieces fit in a
single 1024-token window, so W>1024 arms differ only on the 24% long tail
while short-W arms bind on most of the set.

`extract`: one arm = <out-root>/w{W}/emb_fold{0..4}.npz in ft_eval.py's
fold-embeddings contract, all 900 pieces per fold, resumable per fold.
CPU-only. Run under the ISOLATED uv env this file's `# /// script` header
defines, NOT the shared model/.venv -- same reason as audio_emb_extract.py
(the fork's vendored transformers needs tokenizers 0.19, peft 0.11.1):

    cd model/src/claim_measurement/difficulty
    uv run --no-project --script window_sweep.py extract \\
        --window-len 512 --out-root .../phase1_lora/window_sweep \\
        --bundle-dir .../phase1_lora/bundle \\
        --adapter-root .../phase1_lora/fold_embeddings \\
        --checkpoint .../moonbeam_839M.pt --repo-root .../moonbeam/repo \\
        --model-config .../repo/src/llama_recipes/configs/model_config.json

`score`: ft_eval's exact protocol (seed 2026 folds, per-fold RidgeCV, pooled
OOF tau-c) on every w* arm under --sweep-root, paired-bootstrapped against
the CPU W=1024 arm. The CPU re-extraction, not the stored emb_fold{F}.npz,
is the paired baseline: those were extracted on A100 and #166 measured
CPU/GPU score differences up to +0.0435, so a sweep arm extracted on CPU
must be compared device-matched. The stored GPU embeddings are reported
alongside as the anchor (0.8395).

    cd model && uv run python -m claim_measurement.difficulty.window_sweep \\
        score --data-root data \\
        --sweep-root data/results/phase1_lora/window_sweep \\
        --anchor-dir data/results/phase1_lora/fold_embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Under `uv run --script` this file runs as a loose script, so the package it
# imports from is not on sys.path. __file__-anchored, never CWD-relative.
_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from claim_measurement.difficulty.audio_emb_extract import (  # noqa: E402
    build_fold_embedder,
)
from claim_measurement.difficulty.bakeoff_cv import paired_boot, tau_c  # noqa: E402
from claim_measurement.difficulty.ft_eval import (  # noqa: E402
    N_FOLDS,
    SEED,
    _load_features37,
    oof_tau_per_fold,
)
from claim_measurement.difficulty.train_fold import (  # noqa: E402
    read_fold_embeddings,
    write_fold_embeddings,
)


def load_eval_manifest(path: Path):
    """seg_ids in the bundle's canonical row order (= features37's order, the
    order every emb_fold{F}.npz row is paired on), plus grades/composer_ids."""
    pieces = json.loads(Path(path).read_text())
    seg_ids = [p["seg_id"] for p in pieces]
    grades = np.array([p["grade"] for p in pieces], dtype=np.int32)
    composer_ids = np.array([p["composer_id"] for p in pieces], dtype=np.int32)
    return seg_ids, grades, composer_ids


def extract_arm(bundle_dir: Path, out_dir: Path, embedder_for_fold,
                n_folds: int = N_FOLDS, log_every: int = 50) -> None:
    """One sweep arm: emb_fold{F}.npz for every fold, each holding ALL eval
    pieces extracted through fold F's adapter (ft_eval scores fold F's train
    AND test ridge rows from that one matrix). Resumable at fold granularity;
    an existing fold file is never re-extracted and its adapter never loaded.

    A failed or non-finite extraction raises: this is the research harness,
    where a silent skip would unpair the arm against every other arm."""
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    seg_ids, grades, composer_ids = load_eval_manifest(
        bundle_dir / "eval_manifest.json")
    midi_dir = bundle_dir / "midi"

    for fold in range(n_folds):
        out_path = out_dir / f"emb_fold{fold}.npz"
        if out_path.exists():
            print(f"fold {fold}: {out_path} exists, skipping", flush=True)
            continue
        embed = embedder_for_fold(fold)
        vectors = []
        started = time.monotonic()
        for i, seg_id in enumerate(seg_ids):
            vector = embed(midi_dir / f"{seg_id}.mid")
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"non-finite embedding for {seg_id} (fold {fold})")
            vectors.append(vector)
            if (i + 1) % log_every == 0:
                rate = (i + 1) / (time.monotonic() - started)
                eta_min = (len(seg_ids) - i - 1) / rate / 60
                print(f"fold {fold}: {i + 1}/{len(seg_ids)} "
                      f"{rate:.2f} pieces/s eta {eta_min:.0f}m", flush=True)
        write_fold_embeddings(out_path, seg_ids, np.stack(vectors),
                               grades, composer_ids)
        print(f"fold {fold}: wrote {out_path} "
              f"({time.monotonic() - started:.0f}s)", flush=True)


def arm_oof(arm_dir: Path, y: np.ndarray, composers: np.ndarray,
            seg_ids: list, n_folds: int = N_FOLDS, seed: int = SEED
            ) -> np.ndarray:
    """Pooled OOF predictions for one arm dir, on ft_eval's exact protocol.
    The row-order check is the pairing guarantee: a mismatched arm would not
    crash downstream, it would silently score permuted embeddings."""
    emb_by_fold = {}
    for f in range(n_folds):
        fold_data = read_fold_embeddings(Path(arm_dir) / f"emb_fold{f}.npz")
        if fold_data["seg_ids"] != seg_ids:
            raise SystemExit(
                f"{arm_dir}/emb_fold{f}.npz row order does not match "
                f"features37's seg_id order; the comparison would be unpaired")
        emb_by_fold[f] = fold_data["embeddings"]
    return oof_tau_per_fold(emb_by_fold, y, composers, n_folds, seed)


def score(sweep_root: Path, baseline: str, data_root: Path | None,
          anchor_dir: Path | None, n_folds: int = N_FOLDS,
          seed: int = SEED) -> int:
    from claim_measurement.difficulty.bakeoff_paths import resolve_paths

    emb_root = resolve_paths(data_root).emb_root
    _, y, composers, seg_ids = _load_features37(emb_root)

    sweep_root = Path(sweep_root)
    arm_dirs = sorted((d for d in sweep_root.iterdir()
                       if d.is_dir() and d.name.startswith("w")),
                      key=lambda d: int(d.name[1:]))
    if baseline not in {d.name for d in arm_dirs}:
        raise SystemExit(f"baseline arm {baseline!r} not found under {sweep_root}")

    oofs = {d.name: arm_oof(d, y, composers, seg_ids, n_folds, seed)
            for d in arm_dirs}
    base_oof = oofs[baseline]

    print(f"n={len(y)} pieces, {len(set(composers))} composers, "
          f"seed {seed}, {n_folds} folds")
    if anchor_dir is not None:
        anchor = arm_oof(anchor_dir, y, composers, seg_ids, n_folds, seed)
        d, lo, hi, p = paired_boot(anchor, base_oof, y, seed=seed)
        print(f"anchor (stored GPU emb)  tau-c {tau_c(anchor, y):.4f}")
        print(f"{baseline} - anchor: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
              f"P(diff<=0)={p:.3f}  (device effect, CPU vs A100 extraction)")
    for name, oof in oofs.items():
        line = f"{name:<8} tau-c {tau_c(oof, y):.4f}"
        if name != baseline:
            d, lo, hi, p = paired_boot(base_oof, oof, y, seed=seed)
            verdict = "SIG" if (lo > 0 or hi < 0) else "noise"
            line += (f"  vs {baseline}: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
                     f"P(diff<=0)={p:.3f} {verdict}")
        print(line)
    return 0


def main(argv=None, embedder_for_fold=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    ex = sub.add_parser("extract", help="extract one arm's fold embeddings")
    ex.add_argument("--window-len", type=int, required=True)
    ex.add_argument("--out-root", type=Path, required=True,
                    help="arm is written to <out-root>/w{window-len}/")
    ex.add_argument("--bundle-dir", type=Path, required=True,
                    help="phase1_lora/bundle (eval_manifest.json + midi/)")
    ex.add_argument("--adapter-root", type=Path, required=True,
                    help="fold F's adapter at <adapter-root>/fold{F}/adapter")
    ex.add_argument("--checkpoint", type=Path, default=None)
    ex.add_argument("--repo-root", type=Path, default=None)
    ex.add_argument("--model-config", type=Path, default=None)
    ex.add_argument("--n-folds", type=int, default=N_FOLDS)
    ex.add_argument("--log-every", type=int, default=50)

    sc = sub.add_parser("score", help="tau-c + paired bootstrap per arm")
    sc.add_argument("--sweep-root", type=Path, required=True)
    sc.add_argument("--baseline", default="w1024",
                    help="arm dir name every other arm is paired against")
    sc.add_argument("--data-root", type=Path, default=None)
    sc.add_argument("--anchor-dir", type=Path, default=None,
                    help="stored (GPU-extracted) fold_embeddings dir, reported "
                         "as the 0.8395 anchor and the device-effect delta")
    sc.add_argument("--n-folds", type=int, default=N_FOLDS)
    sc.add_argument("--seed", type=int, default=SEED)

    args = ap.parse_args(argv)

    if args.mode == "score":
        return score(args.sweep_root, args.baseline, args.data_root,
                     args.anchor_dir, args.n_folds, args.seed)

    if embedder_for_fold is None:
        for flag, value in (("--checkpoint", args.checkpoint),
                            ("--repo-root", args.repo_root),
                            ("--model-config", args.model_config)):
            if value is None:
                ap.error(f"{flag} is required when the real backbone is used")

        def embedder_for_fold(fold: int):
            adapter_dir = Path(args.adapter_root) / f"fold{fold}" / "adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"fold {fold}'s adapter is missing: {adapter_dir}")
            return build_fold_embedder(adapter_dir, args.checkpoint,
                                        args.repo_root, args.model_config,
                                        window_len=args.window_len)

    extract_arm(args.bundle_dir, args.out_root / f"w{args.window_len}",
                embedder_for_fold, args.n_folds, args.log_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
