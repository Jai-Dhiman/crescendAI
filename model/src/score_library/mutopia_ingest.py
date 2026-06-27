"""Ingest the net-new Mutopia keyboard pieces into the score-library catalog.

Reuses the self-consistency gate (mirrors kernscores_bulk.py): expected_key =
Krumhansl argmax of the piece's own pitch histogram, expected_bars = parsed
total_bars, then validate_score applies key-agreement + 16th-quant + pitch-range
+ bar-count well-formedness. Gate failures are reported and EXCLUDED. piece_ids
are namespaced `mutopia.<sanitized stem>` (unique); a string collision HALTs
(should not occur after content dedup).

Run:  cd model && uv run python -m score_library.mutopia_ingest
Reads <staging>/mutopia_netnew_candidates.txt; writes data/scores/mutopia.*.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from score_library.key_estimate import estimate_key
from score_library.parse import parse_score_midi
from score_library.validate import ExpectedMeta, validate_score

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES = _MODEL_ROOT / "data" / "scores"
_STAGE = Path(os.environ.get("MUTOPIA_STAGING_DIR", str(Path.home() / "crescendai_corpus_staging")))
_MIDI = _STAGE / "mutopia_midi"
_NETNEW = _STAGE / "mutopia_netnew_candidates.txt"


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _piece_id(stem: str) -> str:
    return f"mutopia.{_sanitize(stem)}"


def _composer(code: str) -> str:
    """'BachJS'->'Bach J.S.', 'BeethovenLv'->'Beethoven L.v.', 'Traditional'->'Traditional'."""
    m = re.match(r"([A-Z][a-z]+(?:-[A-Z][a-z]+)?)([A-Z].*)?$", code)
    if not m:
        return code
    surname, initials = m.group(1), (m.group(2) or "")
    return f"{surname} {'.'.join(initials)}." if initials else surname


def _title(parts: list[str]) -> str:
    return (parts[-1] if parts else "?").replace("_", " ").replace("-", " ").strip()


def main() -> None:
    files = [l.strip() for l in _NETNEW.read_text().splitlines() if l.strip()]
    print(f"net-new candidates: {len(files)}", flush=True)

    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    collisions: list[str] = []
    for fname in files:
        midi = _MIDI / fname
        if not midi.exists():
            failed.append((fname, "MIDI not found"))
            continue
        stem = fname[:-4] if fname.endswith(".mid") else fname
        parts = stem.split("__")
        pid = _piece_id(stem)
        dest = _SCORES / f"{pid}.json"
        if dest.exists():
            collisions.append(pid)
            continue
        try:
            score = parse_score_midi(midi, pid, _composer(parts[0]) if parts else "Unknown", _title(parts))
            expected = ExpectedMeta(piece_id=pid, expected_key=estimate_key(score), expected_bars=score.total_bars)
            violations = validate_score(score, expected)
        except Exception as e:  # noqa: BLE001 -- surface parse/estimate failures
            failed.append((fname, f"{type(e).__name__}: {e}"))
            continue
        if violations:
            failed.append((fname, "; ".join(f"{v.check}: {v.detail}" for v in violations)))
            continue
        with open(dest, "w") as f:
            json.dump(score.model_dump(), f, indent=2)
        passed.append(pid)

    print(f"\nPASSED: {len(passed)}   FAILED: {len(failed)}   COLLISIONS: {len(collisions)}")
    if collisions:
        for c in collisions[:20]:
            print(f"  collision: {c}")
        raise SystemExit("ABORT: piece_id collision -- dedup namespace breach")
    reasons = Counter(r.split(":")[0].split(";")[0].strip() for _, r in failed)
    print("-- failure reasons --")
    for k, n in reasons.most_common():
        print(f"  {n:4d}  {k}")
    print(f"catalog now: {len(list(_SCORES.glob('*.json')))} JSONs")
    if not passed:
        sys.exit("ABORT: zero pieces ingested")


if __name__ == "__main__":
    main()
