from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class StaleCacheError(Exception):
    """Raised when the inference cache fingerprint does not match the expected value."""


def find_cache_dir(cache_root: str | Path) -> Path:
    """Find the most recent cache directory under cache_root.

    Returns the subdirectory with the most recent modification time.
    Cache directories are named by model fingerprint (e.g., "a1-max_muq-l9-12").
    """
    cache_root = Path(cache_root)
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    subdirs = sorted(
        [d for d in cache_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No cache directories found in: {cache_root}")

    return subdirs[0]


def load_recording(cache_dir: str | Path, recording_id: str) -> dict[str, Any]:
    """Load a single recording's JSON from the cache directory."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{recording_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")
    return json.loads(path.read_text())


def load_all_recordings(cache_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load all recording JSON files from the cache directory.

    Returns a dict mapping recording_id to its data.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    recordings: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "_fingerprint.json":
            continue
        recording_id = path.stem
        recordings[recording_id] = json.loads(path.read_text())

    return recordings


def validate_cache(
    cache_dir: str | Path,
    expected_fingerprint: str | None = None,
) -> str:
    """Validate cache integrity and return its fingerprint.

    The fingerprint is a SHA-256 hash of sorted filenames and their sizes.
    If expected_fingerprint is provided and does not match, raises StaleCacheError.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    entries = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "_fingerprint.json":
            continue
        entries.append(f"{path.name}:{path.stat().st_size}")

    fingerprint = hashlib.sha256("|".join(entries).encode()).hexdigest()[:16]

    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise StaleCacheError(
            f"Cache fingerprint mismatch: expected {expected_fingerprint}, "
            f"got {fingerprint}. Re-run inference to regenerate cache."
        )

    return fingerprint
