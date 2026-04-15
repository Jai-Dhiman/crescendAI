from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    git_sha: str
    git_dirty: bool


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def make_run_provenance(suffix: str | None = None) -> RunProvenance:
    """Stamp a run with a filesystem-safe ID, git SHA, and dirty-tree flag.

    Falls back to git_sha="unknown" + git_dirty=True when the git binary is
    unavailable (e.g., sandboxed CI). Always produces a run_id.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        print("warn: git unavailable, using unknown SHA", file=sys.stderr)
        git_sha = "unknown"
        git_dirty = True
    else:
        git_sha = sha
        porcelain = _git("status", "--porcelain")
        git_dirty = bool(porcelain) if porcelain is not None else True

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_sha = git_sha[:7] if git_sha != "unknown" else "nosha"
    run_id = f"{timestamp}_{short_sha}"
    if suffix:
        run_id = f"{run_id}_{suffix}"

    return RunProvenance(run_id=run_id, git_sha=git_sha, git_dirty=git_dirty)
