"""Download prod eval samples from R2 eval-queue/ prefix and merge into a JSONL.

Requires R2 access via wrangler or aws CLI configured for the Cloudflare R2 endpoint.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.drain_eval_queue --out results/prod_samples.jsonl
    uv run python -m teaching_knowledge.scripts.drain_eval_queue --list   # list available keys
    uv run python -m teaching_knowledge.scripts.drain_eval_queue --delete-after-download
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = EVALS_ROOT / "results"
R2_BUCKET = "crescendai-bucket"
R2_PREFIX = "eval-queue/"


def _r2_list() -> list[str]:
    """Return list of R2 object keys under eval-queue/ using wrangler."""
    result = subprocess.run(
        ["bun", "x", "wrangler", "r2", "object", "list", R2_BUCKET, "--prefix", R2_PREFIX],
        capture_output=True,
        text=True,
        check=True,
    )
    keys: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(R2_PREFIX) and line.endswith(".json"):
            keys.append(line)
    return keys


def _r2_get(key: str) -> bytes:
    result = subprocess.run(
        ["bun", "x", "wrangler", "r2", "object", "get", R2_BUCKET, key, "--pipe"],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _r2_delete(key: str) -> None:
    subprocess.run(
        ["bun", "x", "wrangler", "r2", "object", "delete", R2_BUCKET, key],
        capture_output=True,
        check=True,
    )


def drain(out_path: Path, list_only: bool = False, delete_after: bool = False) -> None:
    keys = _r2_list()
    print(f"Found {len(keys)} eval samples in R2")

    if list_only:
        for k in keys:
            print(f"  {k}")
        return

    if not keys:
        print("Nothing to drain.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0

    with out_path.open("a") as fout:
        for key in keys:
            session_id = Path(key).stem
            try:
                raw = _r2_get(key)
                sample = json.loads(raw)
                fout.write(json.dumps(sample) + "\n")
                fout.flush()
                written += 1
                print(f"  [{written}] {session_id}")
                if delete_after:
                    _r2_delete(key)
            except Exception as exc:
                errors += 1
                print(f"  ERROR {session_id}: {exc}")

    print(f"\nDone. written={written} errors={errors}")
    print(f"Output: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain R2 eval queue into JSONL")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "prod_samples.jsonl")
    parser.add_argument("--list", action="store_true", help="List keys only, no download")
    parser.add_argument("--delete-after-download", action="store_true",
                        help="Delete R2 objects after successful download")
    args = parser.parse_args()
    drain(args.out, list_only=args.list, delete_after=args.delete_after_download)


if __name__ == "__main__":
    main()
