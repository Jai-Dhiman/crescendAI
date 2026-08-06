"""Assemble the Docker build context for the MIREX Track A container (#166).

There is no directory in this repo that can serve as a build context directly:
the code lives under `model/src/claim_measurement/difficulty/`, the transcriber
shim under `apps/inference/amt/`, and the model artifacts under the gitignored
`model/data/` -- including a 1.6GB checkpoint that is not in git at all. This
copies exactly what the Dockerfile expects into one tree and refuses to proceed
if anything is missing.

    uv run python apps/inference/mirex-difficulty/build_context.py \\
        --model-dir  <data>/results/phase1_lora/model_alldata \\
        --moonbeam-dir <data>/weights/moonbeam \\
        --out-dir /tmp/mirex-ctx
    docker build -t crescendai/mirex-difficulty /tmp/mirex-ctx

The context is ~5GB (checkpoint 1.6GB + fork + adapter), so it is written
outside the repo by default and never committed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# score_wav.py plus every local module reachable from it at import time. Listed
# explicitly rather than copying the whole package: the difficulty/ directory
# also holds the research harness (tk_ablation, phase3c_explore, ~5MB of test
# files), none of which belongs in a submission artifact.
CODE_FILES = (
    "__init__.py",
    "score_wav.py",
    "audio_emb_extract.py",
    "train_fold.py",
    "realaudio_check.py",
    "bakeoff_cv.py",
    "bakeoff_npz.py",
    "bakeoff_paths.py",
    "bakeoff_sampling.py",
    "psyllabus.py",
    "ranking_loss.py",
)

# The Dockerfile's own files, copied in so `docker build <ctx>` needs no -f.
DOCKER_FILES = ("Dockerfile", "predict.sh")


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} does not exist: {path}")
    return path


def build_context(out_dir: Path, difficulty_dir: Path, amt_dir: Path,
                  model_dir: Path, moonbeam_dir: Path, here: Path) -> dict:
    """Copy the context and return a small report. Every input is checked
    before anything is written, so a missing artifact fails in seconds rather
    than after copying 1.6GB."""
    out_dir = Path(out_dir)
    _require(model_dir / "ridge_head.npz", "the ridge head")
    _require(model_dir / "adapter" / "adapter_config.json", "the LoRA adapter")
    _require(moonbeam_dir / "moonbeam_839M.pt", "the MoonBeam checkpoint")
    _require(moonbeam_dir / "repo" / "src" / "llama_recipes" / "configs"
             / "model_config.json", "the MoonBeam fork's 839M model config")
    _require(amt_dir / "transkun_cli.py", "the Transkun shim")
    for name in CODE_FILES:
        _require(difficulty_dir / name, f"code file {name}")
    for name in DOCKER_FILES:
        _require(here / name, f"docker file {name}")

    if out_dir.exists():
        # Merging into a previous context would leave a stale adapter or an
        # older score_wav.py beside the new ones, and the image would be built
        # from a mixture nobody chose.
        shutil.rmtree(out_dir)

    pkg = out_dir / "claim_measurement" / "difficulty"
    pkg.mkdir(parents=True)
    (out_dir / "claim_measurement" / "__init__.py").write_text("")
    for name in CODE_FILES:
        shutil.copy2(difficulty_dir / name, pkg / name)

    amt_out = out_dir / "apps" / "inference" / "amt"
    amt_out.mkdir(parents=True)
    shutil.copy2(amt_dir / "transkun_cli.py", amt_out / "transkun_cli.py")

    shutil.copytree(model_dir, out_dir / "model")
    moonbeam_out = out_dir / "moonbeam"
    moonbeam_out.mkdir()
    shutil.copy2(moonbeam_dir / "moonbeam_839M.pt",
                 moonbeam_out / "moonbeam_839M.pt")
    # .git is the fork's full history and is the bulk of a naive copy; the same
    # exclusion push_train_dataset.py applies when staging the training bundle.
    shutil.copytree(moonbeam_dir / "repo", moonbeam_out / "repo",
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))

    for name in DOCKER_FILES:
        shutil.copy2(here / name, out_dir / name)

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    return {
        "out_dir": str(out_dir),
        "code_files": len(CODE_FILES),
        "total_gb": round(total_bytes / 1e9, 2),
    }


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="a build_model_dir.py output: adapter/ + ridge_head.npz")
    ap.add_argument("--moonbeam-dir", type=Path, required=True,
                    help="<data>/weights/moonbeam: moonbeam_839M.pt + repo/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--difficulty-dir", type=Path,
                    default=(repo_root / "model" / "src" / "claim_measurement"
                             / "difficulty"))
    ap.add_argument("--amt-dir", type=Path,
                    default=repo_root / "apps" / "inference" / "amt")
    args = ap.parse_args(argv)

    report = build_context(args.out_dir, args.difficulty_dir, args.amt_dir,
                           args.model_dir, args.moonbeam_dir, here)
    print(f"context at {report['out_dir']}: {report['code_files']} code files, "
          f"{report['total_gb']} GB")
    print(f"next: docker build -t crescendai/mirex-difficulty {report['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
