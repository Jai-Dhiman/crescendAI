#!/usr/bin/env python3
"""Thin HF Jobs launcher for CrescendAI model-v2 training (issue #74).

One command to launch a tracked cloud GPU run. Wraps `hf jobs uv run`, injecting a
consistent contract for every job: the HF token as a secret, a Trackio Space for a
persistent dashboard, and the flavor echoed into the job env. Presets pick the
right hardware so callers don't have to remember flavors.

Presets:
  smoke  cpu-basic ($0.01/hr)  jobs/smoke_train.py   -- cheap end-to-end validation
  aria   a100-large ($2.50/hr) jobs/train_aria.py    -- the real Aria 650M fine-tune

Examples:
  # cheap validation that the whole cloud path works (~$0.001):
  uv run model/jobs/hf_launch.py smoke

  # the real Aria run once #72 (AMT MIDI) + #73 (score MIDI) inputs exist:
  uv run model/jobs/hf_launch.py aria --timeout 6h --detach

  # any uv script on any flavor:
  uv run model/jobs/hf_launch.py custom --script path/to.py --flavor l4x1

Runbook / local-MPS alternative (no HF Jobs): the same training entrypoint runs
locally on Apple MPS via `python -m model_improvement.training` with CRESCEND_DEVICE
set; HF Jobs is only needed for the A100-class memory Aria 650M requires.

Requires: `hf` CLI authenticated (`hf auth whoami`), and HF Jobs access on the
account/namespace.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent

PRESETS: dict[str, dict] = {
    "smoke": {"script": JOBS_DIR / "smoke_train.py", "flavor": "cpu-basic", "timeout": "20m"},
    "aria": {"script": JOBS_DIR / "train_aria.py", "flavor": "a100-large", "timeout": "6h"},
}


def build_command(args: argparse.Namespace) -> list[str]:
    if args.preset == "custom":
        if not args.script:
            sys.exit("custom preset requires --script")
        script = Path(args.script)
        flavor = args.flavor or "l4x1"
        timeout = args.timeout or "1h"
    else:
        p = PRESETS[args.preset]
        script = Path(args.script) if args.script else p["script"]
        flavor = args.flavor or p["flavor"]
        timeout = args.timeout or p["timeout"]

    if not script.exists():
        sys.exit(f"job script not found: {script}")

    cmd = [
        "hf", "jobs", "uv", "run",
        "--flavor", flavor,
        "--timeout", timeout,
        "--secrets", "HF_TOKEN",      # injects the caller's HF token into the job
        "-e", f"HF_FLAVOR={flavor}",
    ]
    if args.trackio_space:
        cmd += ["-e", f"TRACKIO_SPACE_ID={args.trackio_space}"]
    if args.ckpt_repo:
        cmd += ["-e", f"CKPT_REPO={args.ckpt_repo}"]
    for kv in args.env or []:
        cmd += ["-e", kv]
    if args.detach:
        cmd.append("--detach")
    cmd.append(str(script))
    cmd += args.script_args
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=[*PRESETS, "custom"])
    ap.add_argument("--script", default=None, help="Override the preset's uv script.")
    ap.add_argument("--flavor", default=None, help="Override the preset's HF hardware flavor.")
    ap.add_argument("--timeout", default=None, help="Max duration, e.g. 20m, 6h.")
    ap.add_argument("--trackio-space", default=None, help="HF Space id for the Trackio dashboard.")
    ap.add_argument("--ckpt-repo", default=None, help="HF model repo to upload the checkpoint to.")
    ap.add_argument("-e", "--env", action="append", help="Extra env var KEY=VALUE (repeatable).")
    ap.add_argument("--detach", action="store_true", help="Run in background; print the job id.")
    ap.add_argument("--dry-run", action="store_true", help="Print the hf command without running it.")
    ap.add_argument("script_args", nargs="*", help="Args passed through to the job script.")
    args = ap.parse_args()

    cmd = build_command(args)
    print("+ " + " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
