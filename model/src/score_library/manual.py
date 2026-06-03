"""Ranked-source manual ingestion driver.

ingest_manifest fetches each piece's MIDI from a ranked list of public-domain
URLs, parses it via parse_score_midi, validates via validate_score, and pins the
winning (url + sha256) to a build-written lockfile. The first source that passes
the gate wins. Winning JSONs are staged in a temp dir and only moved into
scores_dir after EVERY piece resolves, so a HALT is all-or-nothing at the
filesystem boundary. If every source for a piece fails, it raises
SourceResolutionError with a per-candidate failure table -- no silent skip, no
partial JSONs left in scores_dir, no lockfile written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from score_library.parse import parse_score_midi
from score_library.schema import ScoreData
from score_library.validate import ExpectedMeta, validate_score


class SourceResolutionError(Exception):
    """Raised when no candidate source for a piece passes the validation gate."""


@dataclass(frozen=True)
class IngestReport:
    """Summary of a successful ingest run: piece_id -> {resolved_url, sha256}."""

    resolved: dict[str, dict[str, str]]


def _http_fetch(url: str) -> bytes:
    """Fetch raw bytes from a URL using stdlib urllib (no third-party deps)."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (PD URLs from a pinned manifest)
        return resp.read()


def _expected_from_entry(entry: dict) -> ExpectedMeta:
    return ExpectedMeta(
        piece_id=entry["piece_id"],
        expected_key=entry["expected_key"],
        expected_bars=entry["expected_bars"],
    )


def ingest_manifest(
    manifest_path: Path,
    scores_dir: Path,
    lock_path: Path,
    fetch_fn=None,
) -> IngestReport:
    """Resolve every piece in the manifest, writing score JSONs + a lockfile."""
    # Resolve the default at call time (not def time) so monkeypatching the
    # module-level _http_fetch is observed by callers that pass no fetch_fn.
    if fetch_fn is None:
        fetch_fn = _http_fetch
    manifest_path = Path(manifest_path)
    scores_dir = Path(scores_dir)
    lock_path = Path(lock_path)
    scores_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lock: dict[str, dict[str, str]] = {}
    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text())

    resolved: dict[str, dict[str, str]] = {}
    entries = json.loads(manifest_path.read_text())

    # Stage every winning JSON in a temp dir; only move all of them into
    # scores_dir (and write the lockfile) after EVERY piece resolves. A mid-run
    # HALT leaves scores_dir and the lockfile untouched (CONCERN 2: all-or-nothing
    # at the filesystem boundary). The staging dir is auto-removed on exception.
    manifest_dir = manifest_path.parent

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)

        for entry in entries:
            piece_id = entry["piece_id"]
            expected = _expected_from_entry(entry)
            pinned_sha = existing_lock.get(piece_id, {}).get("sha256")
            candidate_failures: list[str] = []
            won = False

            for url in entry["sources"]:
                if url.startswith("http://") or url.startswith("https://"):
                    raw = fetch_fn(url)
                else:
                    local_path = manifest_dir / url
                    raw = local_path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()

                if pinned_sha is not None and sha != pinned_sha:
                    candidate_failures.append(f"{url}: hash_mismatch (got {sha[:12]}, pinned {pinned_sha[:12]})")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
                    tmp.write(raw)
                    tmp.flush()
                    score: ScoreData = parse_score_midi(
                        tmp.name, piece_id, entry["composer"], entry["title"]
                    )

                violations = validate_score(score, expected)
                if violations:
                    detail = "; ".join(f"{v.check}:{v.detail}" for v in violations)
                    candidate_failures.append(f"{url}: {detail}")
                    continue

                staged_path = staging_dir / f"{piece_id}.json"
                with open(staged_path, "w") as f:
                    json.dump(score.model_dump(), f, indent=2)
                resolved[piece_id] = {"resolved_url": url, "sha256": sha}
                won = True
                break

            if not won:
                # HALT: nothing has been moved into scores_dir, no lockfile written.
                table = "\n".join(f"  - {line}" for line in candidate_failures)
                raise SourceResolutionError(
                    f"No source resolved for {piece_id}. Candidate failures:\n{table}"
                )

        # All pieces resolved: commit staged JSONs to scores_dir, then the lockfile.
        for piece_id in resolved:
            shutil.move(str(staging_dir / f"{piece_id}.json"), str(scores_dir / f"{piece_id}.json"))

    with open(lock_path, "w") as f:
        json.dump(resolved, f, indent=2)

    return IngestReport(resolved=resolved)
