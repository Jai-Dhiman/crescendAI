"""SHALLOW curation script: contrast-pair CSV -> YAML probe manifest.

Deliberately not a rendering framework (Gate 0 scope decision). CSV
columns: id,axis,population,clip_a,clip_b,degraded,description with clip
paths relative to --repo-root (default: model/). The generated YAML is
round-tripped through load_manifest before this script reports success,
so a written manifest is loadable by construction (clips exist locally,
headers valid, schema well-formed).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from audio_teacher.manifest import AXES, MODEL_ROOT, POPULATIONS, load_manifest

_COLUMNS = ["id", "axis", "population", "clip_a", "clip_b", "degraded", "description"]


class CurationError(Exception):
    """A CSV row is invalid; nothing is written."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args(argv)

    with args.pairs_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _COLUMNS:
            raise CurationError(
                f"{args.pairs_csv}: header must be {','.join(_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise CurationError(f"{args.pairs_csv}: no pairs")

    entries = []
    for i, row in enumerate(rows):
        if row["axis"] not in AXES:
            raise CurationError(
                f"{args.pairs_csv} row {i}: axis {row['axis']!r} not in {AXES}"
            )
        if row["population"] not in POPULATIONS:
            raise CurationError(
                f"{args.pairs_csv} row {i}: population {row['population']!r} "
                f"not in {POPULATIONS}"
            )
        if row["degraded"] not in ("a", "b"):
            raise CurationError(
                f"{args.pairs_csv} row {i}: degraded must be 'a' or 'b', "
                f"got {row['degraded']!r}"
            )
        entries.append({k: row[k] for k in _COLUMNS})

    args.out.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "sample_rate": args.sample_rate, "pairs": entries},
            sort_keys=False,
        )
    )
    manifest = load_manifest(args.out, repo_root=args.repo_root)
    print(f"wrote {args.out} ({len(manifest.pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
