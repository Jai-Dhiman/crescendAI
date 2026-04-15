"""Enrich inference-cache entries with composer_era, skill, duration tags.

Reads the auto-t5_http inference cache plus skill_eval manifests, produces
dataset_index.jsonl -- one row per cached recording with the tags that
split.py and aggregate.py stratify over.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.tag_dataset
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shared.style_rules import composer_to_era


@dataclass(frozen=True)
class RecordingTags:
    recording_id: str
    composer_era: str
    skill_bucket: int
    duration_bucket: str


def _duration_bucket(seconds: float) -> str:
    if seconds < 30:
        return "<30s"
    if seconds < 60:
        return "30-60s"
    return "60s+"


def tag_recording(
    recording_id: str,
    manifest_entry: dict,
    cache_entry: dict,
) -> RecordingTags:
    composer = manifest_entry.get("composer", "Unknown")
    return RecordingTags(
        recording_id=recording_id,
        composer_era=composer_to_era(composer),
        skill_bucket=int(manifest_entry.get("skill_bucket", 3)),
        duration_bucket=_duration_bucket(float(cache_entry.get("total_duration_seconds", 0.0))),
    )


def build_dataset_index(
    manifest_lookup: dict[str, dict],
    cache_dir: Path,
    out_path: Path,
) -> None:
    """Walk the inference cache and emit a tagged JSONL index."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for cache_file in sorted(cache_dir.glob("*.json")):
            if cache_file.name == "_fingerprint.json":
                continue
            cache_entry = json.loads(cache_file.read_text())
            recording_id = cache_entry.get("recording_id", cache_file.stem)
            manifest_entry = manifest_lookup.get(recording_id)
            if manifest_entry is None:
                continue
            tags = tag_recording(recording_id, manifest_entry, cache_entry)
            fout.write(json.dumps(asdict(tags)) + "\n")


def main() -> None:
    from teaching_knowledge.run_eval import CACHE_DIR, EVALS_ROOT, load_manifests

    parser = argparse.ArgumentParser(description="Build dataset_index.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "dataset_index.jsonl",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    manifest_lookup = load_manifests()
    build_dataset_index(manifest_lookup, args.cache_dir, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
