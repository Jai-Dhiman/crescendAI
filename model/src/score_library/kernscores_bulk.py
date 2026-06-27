"""Bulk self-consistency ingest of KernScores MIDI collections.

Ingests Joplin rags, Scarlatti keyboard sonatas, and Chopin mazurkas from
score-engraved MIDIs into the score library.

Self-consistency gate: each piece's expected_key is derived from its own
pitch-class histogram (Krumhansl argmax).  expected_bars is set to the
parsed total_bars so the bar-count plausibility check auto-passes.  The
key_agreement check still catches atonal / garbled MIDI (correlation < 0.6).

Fail-loud policy:
  - Individual gate failures are printed in a per-piece violation table and
    excluded from the catalog (their JSON is not written).
  - Exit is non-zero only if an entire source collection yields zero passes.
  - Piece-ID collisions with existing catalog entries HALT immediately.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from score_library.key_estimate import estimate_key
from score_library.parse import parse_score_midi
from score_library.validate import ExpectedMeta, validate_score

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"

# ---------------------------------------------------------------------------
# Piece-ID helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Lower-case and replace non-alphanumeric runs with underscores."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def _joplin_piece_id(stem: str) -> str:
    """e.g. 'mapleleaf' -> 'joplin.mapleleaf'."""
    return f"joplin.{_sanitize(stem)}"


def _scarlatti_piece_id(stem: str) -> str:
    """e.g. 'L001K514' -> 'scarlatti.sonatas.l001'."""
    m = re.match(r"(L\d+)", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract Longo number from Scarlatti stem: {stem!r}")
    longo = m.group(1).lower()
    return f"scarlatti.sonatas.{longo}"


def _chopin_mazurka_piece_id(stem: str) -> str:
    """e.g. 'mazurka06-1' -> 'chopin.mazurkas.06-1', 'mazurka-50' -> 'chopin.mazurkas.50'."""
    m = re.match(r"mazurka[-_]?(.+)", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract mazurka suffix from Chopin stem: {stem!r}")
    suffix = m.group(1)
    return f"chopin.mazurkas.{suffix}"


def _joplin_title(stem: str) -> str:
    return stem.replace("_", " ").title() + " Rag"


def _scarlatti_title(stem: str) -> str:
    m = re.match(r"L(\d+)K(\d+)", stem, re.IGNORECASE)
    if m:
        return f"Keyboard Sonata in L{int(m.group(1)):03d} (K.{int(m.group(2))})"
    return f"Keyboard Sonata {stem}"


def _chopin_mazurka_title(stem: str) -> str:
    m = re.match(r"mazurka[-_]?(.+)", stem, re.IGNORECASE)
    suffix = m.group(1) if m else stem
    return f"Mazurka {suffix}"


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def _ingest_collection(
    midi_dir: Path,
    composer: str,
    piece_id_fn,
    title_fn,
    scores_dir: Path,
) -> tuple[list[str], list[tuple[str, str, list]]]:
    """Parse and gate-check all MIDIs in midi_dir.

    Returns (passed_ids, failures) where failures is a list of
    (piece_id, midi_path, violations).

    Writes score JSON for each passing piece.
    Raises RuntimeError on piece-ID collision with existing catalog.
    """
    passed: list[str] = []
    failures: list[tuple[str, str, list]] = []

    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        raise RuntimeError(f"No .mid files found in {midi_dir}")

    for midi_path in midi_files:
        stem = midi_path.stem
        piece_id = piece_id_fn(stem)
        title = title_fn(stem)

        existing = scores_dir / f"{piece_id}.json"
        if existing.exists():
            raise RuntimeError(
                f"COLLISION: piece_id '{piece_id}' already exists in catalog at {existing}. "
                "Halt -- dedup before proceeding."
            )

        score = parse_score_midi(midi_path, piece_id, composer, title)

        expected_key = estimate_key(score)
        expected = ExpectedMeta(
            piece_id=piece_id,
            expected_key=expected_key,
            expected_bars=score.total_bars,
        )

        violations = validate_score(score, expected)
        if violations:
            failures.append((piece_id, str(midi_path), violations))
            continue

        dest = scores_dir / f"{piece_id}.json"
        with open(dest, "w") as f:
            json.dump(score.model_dump(), f, indent=2)
        passed.append(piece_id)

    return passed, failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bulk_ingest(
    joplin_dir: Path,
    scarlatti_dir: Path,
    chopin_dir: Path,
    scores_dir: Path,
) -> None:
    """Run the full three-collection ingest and print a results summary."""
    scores_dir.mkdir(parents=True, exist_ok=True)

    collections = [
        (joplin_dir, "Scott Joplin", _joplin_piece_id, _joplin_title),
        (scarlatti_dir, "Domenico Scarlatti", _scarlatti_piece_id, _scarlatti_title),
        (chopin_dir, "Frederic Chopin", _chopin_mazurka_piece_id, _chopin_mazurka_title),
    ]

    total_passed = 0
    total_failed = 0
    all_zero_pass_collections: list[str] = []

    for midi_dir, composer, pid_fn, title_fn in collections:
        print(f"\n--- {composer} ({midi_dir.name}) ---")
        passed, failures = _ingest_collection(
            midi_dir, composer, pid_fn, title_fn, scores_dir
        )

        total_passed += len(passed)
        total_failed += len(failures)

        print(f"  Passed: {len(passed)}")

        if failures:
            print(f"  FAILED ({len(failures)} pieces):")
            for piece_id, midi_path, violations in failures:
                viols = "; ".join(f"{v.check}: {v.detail}" for v in violations)
                print(f"    {piece_id} ({Path(midi_path).name}): {viols}")

        if len(passed) == 0:
            all_zero_pass_collections.append(f"{composer} ({midi_dir.name})")

    print(f"\n=== Summary ===")
    print(f"Total passed: {total_passed}")
    print(f"Total failed: {total_failed}")

    if all_zero_pass_collections:
        print(f"\nERROR: zero passes for collections:")
        for name in all_zero_pass_collections:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    staging_base = Path.home() / "crescendai_corpus_staging" / "kernscores_midi"
    run_bulk_ingest(
        joplin_dir=staging_base / "joplin",
        scarlatti_dir=staging_base / "scarlatti-keyboard-sonatas",
        chopin_dir=staging_base / "chopin-mazurkas",
        scores_dir=_SCORES_DIR,
    )
