"""Stratified train/holdout split for eval dataset.

Keyed on (composer_era, skill_bucket). Holdout is never touched during
prompt iteration -- document this in every doc that mentions the harness.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.scripts.split --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.scripts.tag_dataset import RecordingTags


@dataclass(frozen=True)
class Split:
    train: list[str]
    holdout: list[str]


def stratified_split(
    tags: list[RecordingTags],
    seed: int,
    holdout_ratio: float = 0.2,
) -> Split:
    """Stratify on (composer_era, skill_bucket), seeded for determinism."""
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, int], list[str]] = defaultdict(list)
    for tag in tags:
        key = (tag.composer_era, tag.skill_bucket)
        by_stratum[key].append(tag.recording_id)

    train: list[str] = []
    holdout: list[str] = []

    for key in sorted(by_stratum.keys()):
        ids = sorted(by_stratum[key])  # deterministic base order
        rng.shuffle(ids)
        n_holdout = max(0, round(len(ids) * holdout_ratio))
        # Never steal the only recording in a stratum for holdout
        if n_holdout == len(ids) and len(ids) > 1:
            n_holdout = len(ids) - 1
        holdout.extend(ids[:n_holdout])
        train.extend(ids[n_holdout:])

    return Split(train=sorted(train), holdout=sorted(holdout))


def write_split(split: Split, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"train": split.train, "holdout": split.holdout},
            indent=2,
        )
    )


def load_split(split_path: Path, which: str) -> set[str]:
    """Load a split file. which in {"train","holdout","all"}."""
    if which not in {"train", "holdout", "all"}:
        raise ValueError(f"invalid split selector: {which!r}")
    blob = json.loads(split_path.read_text())
    if which == "train":
        return set(blob["train"])
    if which == "holdout":
        return set(blob["holdout"])
    return set(blob["train"]) | set(blob["holdout"])


def main() -> None:
    from teaching_knowledge.run_eval import EVALS_ROOT

    parser = argparse.ArgumentParser(description="Split dataset into train/holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "dataset_index.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EVALS_ROOT / "teaching_knowledge" / "data" / "splits.json",
    )
    args = parser.parse_args()

    tags: list[RecordingTags] = []
    for line in args.dataset_index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tags.append(
            RecordingTags(
                recording_id=row["recording_id"],
                composer_era=row["composer_era"],
                skill_bucket=int(row["skill_bucket"]),
                duration_bucket=row["duration_bucket"],
            )
        )

    split = stratified_split(tags, seed=args.seed, holdout_ratio=args.holdout_ratio)
    write_split(split, args.out)
    print(f"wrote {args.out}")
    print(f"  train:   {len(split.train)}")
    print(f"  holdout: {len(split.holdout)}")


if __name__ == "__main__":
    main()
