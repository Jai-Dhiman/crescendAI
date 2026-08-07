"""Stage + upload the GPU runtime-probe bundle for #166 (#104 S1).

The one unmet part of S1's exit criterion is "inside the runtime budget", and
it is unmeasurable on this machine: there is no local CUDA, and the CPU numbers
(~17.7s fixed + ~0.55x realtime, Docker-on-macOS) are pessimistic by an unknown
factor. This bundle is what makes the measurement runnable on HF Jobs.

`hf jobs uv run <file>` uploads exactly ONE file, so everything else the probe
needs has to arrive over the Hub. What is already there is not re-staged:

  - the LoRA adapter          -> Jai-D/phase1-lora-alldata (model repo)
  - the MoonBeam fork + config -> Jai-D/phase1-lora-bundle/moonbeam_repo
  - moonbeam_839M.pt          -> guozixunnicolas/... , public, fetched in-job

What is local-only, and therefore staged here:

  - `wav/`   the probe recordings themselves
  - `head/`  ridge_head.npz + manifest.json, which live only in this checkout
  - `code/`  an importable tree, NOT a flat file list

`code/` mirrors the real layout -- `code/claim_measurement/difficulty/` and
`code/apps/inference/amt/transkun_cli.py` -- because the closure is a package,
not a set of loose modules. `score_wav.py` imports
`claim_measurement.difficulty.audio_emb_extract`, which imports three more
siblings; flattening those into one directory (what the training bundle's
`code/` does for two dependency-free helpers) would break every one of those
imports. The `apps/` path is equally deliberate:
`realaudio_check._import_transcribe_wav` locates `transkun_cli.py` by walking
parents for `apps/inference/amt`, so putting it there means the probe needs no
special-casing to find its transcriber -- the production lookup just works.

## Why the probe set is chosen by duration

Per-item cost is a fixed term plus a term linear in audio duration, and
transcription owns the linear one. A probe set clustered near the median would
fit the fixed term well and the slope badly, so pieces are picked nearest to
targets spanning the REAL corpus spread (12.8s min / 103s median / 734s p95 /
4793s max over 1,233 recordings). Ascending order is load-bearing: a 30-minute
item is where an OOM is plausible, and `score_wav --runtime-out` streams per
item, so a kill on the last row still leaves every shorter measurement on disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Targets in seconds, spanning the measured corpus spread rather than a round
# ladder: the fit is only trustworthy across the range it was anchored on.
DEFAULT_TARGETS_S = (15.0, 30.0, 60.0, 100.0, 200.0, 400.0, 750.0, 1800.0)

TRANSKUN_REL = Path("apps/inference/amt/transkun_cli.py")


def wav_durations(wav_dir: Path, probe) -> list[tuple[str, float]]:
    """[(filename, seconds)] for every readable WAV, sorted by name.

    `probe` is injected (`soundfile.info` in production) so the selection logic
    is testable without audio on disk. A WAV that will not open is dropped and
    named on stderr -- silently including it would put an unmeasurable item in
    a bundle whose entire purpose is measurement.
    """
    out = []
    for path in sorted(wav_dir.glob("*.wav")):
        try:
            info = probe(str(path))
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            print(f"unreadable, skipped: {path.name}: {exc!r}", file=sys.stderr)
            continue
        out.append((path.name, float(info.frames) / float(info.samplerate)))
    return out


def select_probe_set(durations: list[tuple[str, float]],
                     targets=DEFAULT_TARGETS_S) -> list[dict]:
    """One recording per target, nearest by duration, no repeats, ascending.

    A target with no unused candidate left is dropped rather than filled with a
    duplicate: eight rows at seven distinct durations would report a slope
    fitted on less spread than it appears to have.
    """
    if not durations:
        raise ValueError("no readable WAVs to select from")
    taken: set[str] = set()
    chosen: list[dict] = []
    for target in targets:
        pool = [(name, d) for name, d in durations if name not in taken]
        if not pool:
            break
        name, seconds = min(pool, key=lambda nd: abs(nd[1] - target))
        taken.add(name)
        chosen.append({"wav": name, "seconds": round(seconds, 2),
                       "target_s": target})
    chosen.sort(key=lambda row: row["seconds"])
    return chosen


def stage_probe_bundle(wav_dir: Path, head_dir: Path, module_dir: Path,
                       staging_dir: Path, selection: list[dict]) -> dict:
    """Materialise wav/ + head/ + code/ + manifest.json. Returns the report."""
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "wav").mkdir(parents=True)
    (staging_dir / "head").mkdir()
    (staging_dir / "code").mkdir()

    for row in selection:
        shutil.copy2(wav_dir / row["wav"], staging_dir / "wav" / row["wav"])

    for name in ("ridge_head.npz", "manifest.json"):
        src = head_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"the submission head is incomplete: {src} is missing. "
                "Build it with build_model_dir.py before staging a probe.")
        shutil.copy2(src, staging_dir / "head" / name)

    # The whole package, .py only: 46 modules and ~1.2MB of source, against
    # which hand-listing an import closure is the more expensive option and
    # goes stale the first time a sibling import is added.
    src_root = module_dir.parents[1]
    pkg_dest = staging_dir / "code" / "claim_measurement"
    shutil.copytree(src_root / "claim_measurement", pkg_dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    repo_root = module_dir.parents[3]
    transkun = repo_root / TRANSKUN_REL
    if not transkun.exists():
        raise FileNotFoundError(
            f"transkun_cli.py not found at {transkun}. Without it the probe "
            "container has no transcriber and measures nothing.")
    transkun_dest = staging_dir / "code" / TRANSKUN_REL
    transkun_dest.parent.mkdir(parents=True)
    shutil.copy2(transkun, transkun_dest)

    manifest = {
        "purpose": "GPU runtime probe for #166 (#104 S1)",
        "items": selection,
        "total_audio_s": round(sum(r["seconds"] for r in selection), 2),
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None, uploader=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wav-dir", type=Path, required=True,
                    help="directory of real recordings to pick the probe from")
    ap.add_argument("--head-dir", type=Path, required=True,
                    help="the submission model dir holding ridge_head.npz")
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument("--repo-id", default="Jai-D/mirex-runtime-probe")
    ap.add_argument("--no-upload", action="store_true",
                    help="stage and report only. The staged tree is what gets "
                         "uploaded, so this is how you inspect it first")
    args = ap.parse_args(argv)

    import soundfile as sf

    module_dir = Path(__file__).resolve().parent
    durations = wav_durations(args.wav_dir, sf.info)
    selection = select_probe_set(durations)
    manifest = stage_probe_bundle(args.wav_dir, args.head_dir, module_dir,
                                  args.staging_dir, selection)

    print(f"staged {len(selection)} probe WAVs, "
          f"{manifest['total_audio_s']:.0f}s of audio total")
    for row in selection:
        print(f"  {row['seconds']:8.1f}s  (target {row['target_s']:.0f}s)  "
              f"{row['wav']}")
    if args.no_upload:
        print(f"not uploading (--no-upload). Staged at {args.staging_dir}")
        return 0

    if uploader is None:
        uploader = _default_uploader
    uploader(args.staging_dir, args.repo_id)
    print(f"uploaded to {args.repo_id}")
    return 0


def _default_uploader(staged_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(staged_dir), repo_id=repo_id,
                      repo_type="dataset")


if __name__ == "__main__":
    raise SystemExit(main())
