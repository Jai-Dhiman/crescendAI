"""T5 data integrity checks. Run before split generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_all_t5_manifests(skill_eval_dir: Path) -> list[dict[str, Any]]:
    """Load all T5 recordings from manifest.yaml files."""
    all_recordings = []
    for manifest_path in sorted(skill_eval_dir.glob("*/manifest.yaml")):
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        piece = manifest["piece"]
        for rec in manifest.get("recordings", []):
            rec["piece"] = piece
            all_recordings.append(rec)
    return all_recordings


def check_integrity(
    recordings: list[dict[str, Any]],
    audio_dir: Path | None = None,
    embedding_dir: Path | None = None,
) -> list[str]:
    """Run all integrity checks on T5 recordings.

    Returns:
        List of error strings. Empty list means all checks pass.
    """
    errors: list[str] = []

    # Check 1: duplicate video_ids
    seen_ids: dict[str, str] = {}
    for rec in recordings:
        vid = rec["video_id"]
        piece = rec["piece"]
        if vid in seen_ids:
            errors.append(
                f"Duplicate video_id {vid}: in {seen_ids[vid]} and {piece}"
            )
        seen_ids[vid] = piece

    # Check 2: bucket balance (min 3 per piece+bucket)
    groups: dict[tuple, int] = defaultdict(int)
    for rec in recordings:
        groups[(rec["piece"], rec["skill_bucket"])] += 1
    for (piece, bucket), count in sorted(groups.items()):
        if count < 3:
            errors.append(
                f"piece={piece}, bucket={bucket}: only {count} recordings (need >=3)"
            )

    # Check 3: audio files exist (if audio_dir provided)
    if audio_dir is not None:
        for rec in recordings:
            piece = rec["piece"]
            vid = rec["video_id"]
            audio_path = audio_dir / piece / f"{vid}.wav"
            if not audio_path.exists():
                alt_paths = [
                    audio_dir / piece / f"{vid}.opus",
                    audio_dir / piece / f"{vid}.webm",
                ]
                if not any(p.exists() for p in alt_paths):
                    errors.append(f"Missing audio: {piece}/{vid}")

    # Check 4: embeddings exist and are non-empty (if embedding_dir provided)
    if embedding_dir is not None:
        for rec in recordings:
            vid = rec["video_id"]
            emb_path = embedding_dir / f"{vid}.pt"
            if not emb_path.exists():
                errors.append(f"Missing embedding: {vid}.pt")
            elif emb_path.stat().st_size == 0:
                errors.append(f"Empty embedding: {vid}.pt")

    return errors
