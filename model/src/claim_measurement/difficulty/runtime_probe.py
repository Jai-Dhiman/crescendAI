# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "scipy>=1.10.0", "torch>=2.0.0", "peft==0.11.1",
#     "scikit-learn", "pretty_midi", "soundfile",
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#166 (#104 S1): measure per-item scoring runtime on a real GPU.

S1's exit criterion has one unmet clause -- "inside the runtime budget" -- and
it cannot be closed on a machine with no CUDA. The CPU fit (~17.7s fixed +
~0.55x realtime, Docker-on-macOS) says the whole test set would take between
470 and 1,700 items per 24h. Whether that is a crisis or a non-issue depends
entirely on a number nobody has measured.

    hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN \\
        model/src/claim_measurement/difficulty/runtime_probe.py \\
        --probe-repo Jai-D/mirex-runtime-probe --device cuda

`hf jobs uv run` uploads ONLY this file, so every dependency arrives over the
Hub: the probe bundle (WAVs + the ridge head + an importable copy of this
package), the MoonBeam fork snapshot from the training bundle, the LoRA
adapter, and the public 839M checkpoint.

## What it reports, and what it does not

It fits `seconds = fixed + slope * duration` across the probe's duration spread
and extrapolates to plausible test-set sizes. It does NOT report a tau-c: every
labelled piece we hold is in the submission model's training set, so any score
this job prints about quality would be train-on-test. Scores are written out
only so a run can be checked for determinism against the CPU value.

## Two failure modes are results, not bugs

Transcription is what `--device cuda` newly reaches (see transkun_cli.py). If
Transkun's isolated env resolves a CPU-only torch wheel, the slope will simply
not improve, and that is the finding. If the longest item is OOM-killed, the
per-item stream on disk keeps every shorter measurement -- which is why items
run shortest first.
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
from pathlib import Path

BUNDLE_REPO = "Jai-D/phase1-lora-bundle"
ADAPTER_REPO = "Jai-D/phase1-lora-alldata"
CHECKPOINT_REPO = "guozixunnicolas/moonbeam-midi-foundation-model"
CHECKPOINT_FILE = "moonbeam_839M.pt"


def fit_runtime(rows: list[dict]) -> dict:
    """Least-squares `seconds = fixed + slope * duration` over measured items.

    Returns the fit plus the residual spread, because a tight fit and a loose
    one justify very different extrapolations and the extrapolation is the
    whole point. Fewer than two successful items cannot define a line and say
    so rather than returning a fixed term that is really just one measurement.
    """
    import numpy as np

    ok = [r for r in rows if r["ok"]]
    if len(ok) < 2:
        raise ValueError(
            f"need at least 2 successful items to fit a line, got {len(ok)}")
    x = np.array([r["audio_s"] for r in ok], dtype=float)
    y = np.array([r["score_s"] for r in ok], dtype=float)
    slope, fixed = np.polyfit(x, y, 1)
    resid = y - (fixed + slope * x)
    return {
        "n": len(ok),
        "fixed_s": float(fixed),
        "slope_x_realtime": float(slope),
        "max_abs_residual_s": float(np.max(np.abs(resid))),
    }


def project_budget(fit: dict, mean_piece_s: float, budget_h: float = 24.0
                   ) -> dict:
    """How many items of a given mean length fit the 24h/1-GPU clause."""
    per_item = fit["fixed_s"] + fit["slope_x_realtime"] * mean_piece_s
    return {
        "mean_piece_s": mean_piece_s,
        "per_item_s": round(per_item, 2),
        "items_in_budget": int(budget_h * 3600 / per_item) if per_item > 0 else 0,
    }


def build_transcriber(probe_dir: Path, device: str):
    """Import transkun_cli out of the probe bundle, with the device bound.

    `load_scorer` would otherwise build its own transcriber via
    `realaudio_check._import_transcribe_wav`, which locates transkun_cli by
    walking `Path(__file__).resolve()`'s parents for `apps/inference/amt`. That
    lookup is correct in a real checkout and in the container, and it CANNOT
    work against a Hub snapshot: snapshot_download materialises every file as a
    symlink into `<repo>/blobs/<sha256>`, so `.resolve()` leaves the snapshot
    tree entirely and the walk searches the blob store. It cost one cancelled
    GPU job to find, because a locally staged bundle holds real files and the
    rehearsal therefore could not reproduce it.

    Binding the device here is load-bearing: load_scorer only binds its own
    `--device` onto a transcriber it built itself, so a caller that supplies
    one owns that responsibility. An unbound transcriber would leave Transkun
    on CPU and the probe would measure the thing it exists to rule out.
    """
    amt_dir = probe_dir / "code" / "apps" / "inference" / "amt"
    if not (amt_dir / "transkun_cli.py").exists():
        raise FileNotFoundError(
            f"probe bundle has no transcriber at {amt_dir}. Re-stage it with "
            "push_runtime_probe.py -- without one this job measures nothing.")
    sys.path.insert(0, str(amt_dir))
    import transkun_cli  # noqa: PLC0415 -- path must be set first

    return functools.partial(transkun_cli.transcribe_wav, device=device)


def assemble_model_dir(adapter_dir: Path, head_dir: Path, dest: Path) -> Path:
    """load_scorer wants ONE directory holding `adapter/` + `ridge_head.npz`.

    The two halves live in different Hub repos -- the adapter is a 46MB model
    repo, the head is 46KB that exists only in the checkout -- so they are
    joined here rather than by re-uploading one into the other's repo, which
    would make two copies of the adapter that can drift apart.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(adapter_dir / "adapter", dest / "adapter")
    for name in ("ridge_head.npz", "manifest.json"):
        shutil.copy2(head_dir / name, dest / name)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe-repo", default="Jai-D/mirex-runtime-probe")
    ap.add_argument("--probe-dir", type=Path, default=None,
                    help="use an already-downloaded probe bundle instead of "
                         "--probe-repo. This is how the job is rehearsed "
                         "locally before any GPU time is spent")
    ap.add_argument("--device", default="cuda",
                    help="passed to BOTH Transkun and the MoonBeam forward "
                         "pass. cpu reproduces the pessimistic baseline")
    ap.add_argument("--out-dir", type=Path, default=Path("/data/runtime_probe"))
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N (shortest) items -- a cheap "
                         "smoke run before committing to the full probe")
    # The offline trio. Three of the first HF Jobs launches in #149 died on
    # environment plumbing rather than on anything they were measuring, so the
    # whole path is rehearsable on CPU against local files first. Any of these
    # given skips the corresponding download.
    ap.add_argument("--adapter-dir", type=Path, default=None,
                    help="local dir holding adapter/, instead of the Hub repo")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="local MoonBeam fork checkout, instead of the bundle")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="local moonbeam_839M.pt, instead of the Hub copy")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download, snapshot_download

    if args.probe_dir is not None:
        probe_dir = args.probe_dir
    else:
        probe_dir = Path(snapshot_download(args.probe_repo, repo_type="dataset"))
    print(f"probe bundle: {probe_dir}", flush=True)

    sys.path.insert(0, str(probe_dir / "code"))

    if args.repo_root is not None:
        repo_root = args.repo_root
    else:
        repo_root = Path(snapshot_download(
            BUNDLE_REPO, repo_type="dataset",
            allow_patterns=["moonbeam_repo/**"])) / "moonbeam_repo"
    adapter_dir = (args.adapter_dir if args.adapter_dir is not None
                   else Path(snapshot_download(ADAPTER_REPO,
                                               allow_patterns=["adapter/**"])))
    checkpoint = (args.checkpoint if args.checkpoint is not None
                  else Path(hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = assemble_model_dir(adapter_dir, probe_dir / "head",
                                   args.out_dir / "model")

    from claim_measurement.difficulty.score_wav import load_scorer

    load_t0 = time.perf_counter()
    scorer = load_scorer(
        model_dir,
        checkpoint=checkpoint,
        repo_root=repo_root,
        model_config=(repo_root / "src" / "llama_recipes" / "configs"
                      / "model_config.json"),
        on_failure="fallback",
        device=args.device,
        transcribe=build_transcriber(probe_dir, args.device),
    )
    load_s = time.perf_counter() - load_t0
    print(f"scorer loaded in {load_s:.1f}s on device={args.device}", flush=True)

    manifest = json.loads((probe_dir / "manifest.json").read_text())
    items = manifest["items"]
    if args.limit is not None:
        items = items[:args.limit]

    # Streamed per item, not buffered: the longest item is where an OOM kill is
    # plausible, and nothing in Python catches SIGKILL. A buffered write would
    # discard every measurement that already succeeded.
    rows_path = args.out_dir / "runtime_rows.jsonl"
    rows: list[dict] = []
    with rows_path.open("w") as fh:
        for item in items:
            wav = probe_dir / "wav" / item["wav"]
            t0 = time.perf_counter()
            score, ok = scorer(wav)
            elapsed = time.perf_counter() - t0
            row = {"wav": item["wav"], "audio_s": item["seconds"],
                   "score_s": round(elapsed, 2), "ok": bool(ok),
                   "score": float(score)}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{item['seconds']:8.1f}s audio -> {elapsed:8.1f}s  "
                  f"ok={ok}  score={score:.4f}", flush=True)

    # A one-item smoke run (--limit 1) is a legitimate way to spend the least
    # possible GPU time proving the plumbing works. Exiting non-zero on it
    # would flag the job ERROR and hide the fact that scoring succeeded, so
    # too-few-points is reported rather than raised. fit_runtime still refuses
    # to draw the line -- one point does not define one.
    fit = None
    if sum(1 for r in rows if r["ok"]) >= 2:
        fit = fit_runtime(rows)
        print(f"\nfit: {fit['fixed_s']:.1f}s fixed + "
              f"{fit['slope_x_realtime']:.3f}x realtime  "
              f"(n={fit['n']}, max residual {fit['max_abs_residual_s']:.1f}s)")
        print("\nprojection against the 24h/1-GPU clause:")
        for mean_s in (60.0, 103.0, 200.0, 400.0):
            p = project_budget(fit, mean_s)
            print(f"  mean piece {p['mean_piece_s']:6.0f}s -> "
                  f"{p['per_item_s']:7.1f}s/item -> "
                  f"{p['items_in_budget']:6d} items in 24h")
    else:
        print("\nfewer than 2 successful items: no fit, no projection. "
              "Per-item times above are the whole result of this run.")

    failures = [r for r in rows if not r["ok"]]
    print(f"\nfailure rate: {len(failures)}/{len(rows)} "
          f"({100.0 * len(failures) / max(1, len(rows)):.1f}%) "
          f"-- the contract excludes a submission above 5%")

    summary = {"device": args.device, "load_s": round(load_s, 1),
               "fit": fit, "rows": rows}
    (args.out_dir / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {rows_path} and {args.out_dir / 'runtime_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
