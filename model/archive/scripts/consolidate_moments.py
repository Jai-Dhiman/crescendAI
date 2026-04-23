#!/usr/bin/env python3
"""Consolidate per-video teaching moment JSONL files into a single file.

Usage:
    python scripts/consolidate_moments.py [--data-dir DATA_DIR] [--output OUTPUT]

The Rust pipeline's `export` command does the same thing, but this script
avoids a Rust compile and works directly with the gitignored data directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def consolidate(moments_dir: Path, output_path: Path) -> int:
    """Concatenate all per-video JSONL files into a single file.

    Returns the total number of moments written.
    """
    if not moments_dir.is_dir():
        raise FileNotFoundError(f"Teaching moments directory not found: {moments_dir}")

    jsonl_files = sorted(moments_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {moments_dir}")

    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for jf in jsonl_files:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Validate JSON
                    json.loads(line)
                    out.write(line + "\n")
                    count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate teaching moments")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/masterclass_pipeline"),
        help="Masterclass pipeline data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: DATA_DIR/all_moments.jsonl)",
    )
    args = parser.parse_args()

    moments_dir = args.data_dir / "teaching_moments"
    output_path = args.output or (args.data_dir / "all_moments.jsonl")

    count = consolidate(moments_dir, output_path)
    print(f"Consolidated {count} teaching moments into {output_path}")


if __name__ == "__main__":
    main()
