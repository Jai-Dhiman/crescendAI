#!/usr/bin/env python3
"""
Deduplicate audio across one or more manifests using Chromaprint (fpcalc).

Outputs a JSON file with groups of duplicates by fingerprint. Falls back to sha1-only grouping if fpcalc is not available and sha1 exists.

Usage:
- uv run python -m src.tools.dedup_audio --manifests data/manifests/percepiano.jsonl data/manifests/maestro.jsonl --out data/manifests/dedup_pairs.json

Requirements:
- chromaprint (fpcalc) installed (macOS: `brew install chromaprint`).

Explicit exceptions: if fpcalc is missing and no sha1 fields exist, raise with instructions.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Args:
    manifests: List[Path]
    out: Path
    fpcalc_path: str
    length: int
    method: str


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Deduplicate audio across manifests using Chromaprint (fpcalc)")
    ap.add_argument("--manifests", nargs="+", type=Path, required=True, help="One or more JSONL manifests")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON file with duplicate groups")
    ap.add_argument("--fpcalc-path", default="fpcalc", type=str, help="Path to fpcalc binary (default in PATH)")
    ap.add_argument("--length", default=120, type=int, help="Audio length (seconds) to analyze for fingerprint")
    ap.add_argument("--method", default="chromaprint", choices=["chromaprint", "sha1"], help="Dedup method")
    ns = ap.parse_args()
    return Args(
        manifests=[m.resolve() for m in ns.manifests],
        out=ns.out.resolve(),
        fpcalc_path=ns.fpcalc_path,
        length=int(ns.length),
        method=ns.method,
    )


def read_manifest_lines(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception as e:
                raise ValueError(f"Invalid JSON line in {path}: {e}")
    return items


def fpcalc_available(fpcalc_path: str) -> bool:
    return shutil.which(fpcalc_path) is not None


def fingerprint_file(fpcalc_path: str, file_path: Path, length: int) -> str:
    # Call: fpcalc -length 120 "file"
    # Parse line starting with "FINGERPRINT="
    try:
        proc = subprocess.run(
            [fpcalc_path, "-length", str(length), str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"fpcalc failed: {file_path}")
        for line in (proc.stdout or "").splitlines():
            if line.startswith("FINGERPRINT="):
                return line.split("=", 1)[1].strip()
        # Some versions may output lowercase or different formatting; fallback parse
        for line in (proc.stdout or "").splitlines():
            if "FINGERPRINT" in line:
                return line.split("=", 1)[1].strip()
        raise RuntimeError(f"Could not parse fingerprint from fpcalc output for: {file_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError("fpcalc not found. Install Chromaprint: brew install chromaprint") from e


def main() -> None:
    args = parse_args()

    # Load manifests
    items: List[Tuple[str, dict]] = []  # (manifest_path, entry)
    for m in args.manifests:
        if not m.exists():
            raise FileNotFoundError(f"Manifest not found: {m}")
        for obj in read_manifest_lines(m):
            items.append((str(m), obj))

    if not items:
        raise RuntimeError("No items loaded from manifests")

    # Prepare output
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Choose method
    if args.method == "chromaprint":
        if not fpcalc_available(args.fpcalc_path):
            # Fall back to sha1-based grouping if available, else error
            if any("sha1" in obj for _, obj in items):
                print("Warning: fpcalc not found; falling back to sha1 grouping", flush=True)
                args.method = "sha1"
            else:
                raise FileNotFoundError(
                    "fpcalc not found in PATH. Install Chromaprint (brew install chromaprint) or use --method sha1 with manifests that include sha1."
                )

    groups: Dict[str, List[dict]] = {}

    if args.method == "chromaprint":
        for mpath, obj in items:
            p = Path(obj.get("path", ""))
            if not p.exists():
                # skip missing files; explicit but non-fatal
                continue
            try:
                fp = fingerprint_file(args.fpcalc_path, p, args.length)
                groups.setdefault(fp, []).append({"manifest": mpath, **obj})
            except Exception as e:
                # Explicit exception: stop process (dedup quality depends on fingerprints)
                raise RuntimeError(f"Fingerprint failed for {p}: {e}") from e
    else:  # sha1
        for mpath, obj in items:
            sha1 = obj.get("sha1")
            if not sha1:
                continue
            groups.setdefault(sha1, []).append({"manifest": mpath, **obj})

    # Keep only groups with more than one item
    dup_groups = [v for v in groups.values() if len(v) > 1]

    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"duplicate_groups": dup_groups, "method": args.method}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
