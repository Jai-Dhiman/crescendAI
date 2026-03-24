"""Generate practice eval scenario files from T5 skill_eval manifests.

Reads manifest.yaml files from model/data/evals/skill_eval/<piece>/
and writes scenario YAML files to apps/evals/pipeline/practice_eval/scenarios/
in the format expected by eval_practice.py's load_scenarios().

Key behavior:
- Maps skill_bucket -> skill_level (same 1-5 scale, renamed)
- Only includes recordings where downloaded: true
- Deliberately omits piece_query to force automatic piece identification
- Sets include: true on all candidates

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.generate_t5_scenarios
    uv run python -m pipeline.practice_eval.generate_t5_scenarios --piece fur_elise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from paths import MODEL_DATA

SKILL_EVAL_DIR = MODEL_DATA / "evals" / "skill_eval"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"

TARGET_PIECES = [
    "bach_prelude_c_wtc1",
    "bach_invention_1",
    "fur_elise",
    "nocturne_op9no2",
]


def load_manifest(manifest_path: Path) -> dict:
    """Load a manifest YAML file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def manifest_to_scenario(manifest: dict) -> dict:
    """Convert a manifest dict to the scenario format expected by load_scenarios().

    Only includes recordings where downloaded is True.
    Maps skill_bucket to skill_level.
    Deliberately omits piece_query to force automatic piece identification.
    """
    candidates = []
    for rec in manifest.get("recordings", []):
        if not rec.get("downloaded"):
            continue
        bucket = rec.get("skill_bucket")
        candidates.append({
            "video_id": rec["video_id"],
            "include": True,
            "skill_level": bucket,
            "title": rec["title"],
            "general_notes": f"T5 skill corpus, bucket {bucket}",
        })
    return {"candidates": candidates}


def generate_scenario_file(manifest_path: Path, output_path: Path) -> Path:
    """Read a manifest and write the corresponding scenario file.

    Returns the output path.
    """
    manifest = load_manifest(manifest_path)
    scenario = manifest_to_scenario(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate practice eval scenarios from T5 skill_eval manifests."
    )
    parser.add_argument(
        "--piece",
        help="Generate for a single piece (default: all target pieces).",
    )
    args = parser.parse_args()

    pieces = [args.piece] if args.piece else TARGET_PIECES

    for piece_id in pieces:
        manifest_path = SKILL_EVAL_DIR / piece_id / "manifest.yaml"
        if not manifest_path.exists():
            print(f"SKIP {piece_id}: no manifest at {manifest_path}")
            continue

        output_path = SCENARIOS_DIR / f"t5_{piece_id}.yaml"
        generate_scenario_file(manifest_path, output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)
        n = len(data.get("candidates", []))
        print(f"OK   {piece_id}: {n} candidates -> {output_path}")


if __name__ == "__main__":
    main()
