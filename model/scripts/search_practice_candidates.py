"""Search YouTube for piano practice/rehearsal recordings for the practice_eval corpus.

Finds recordings where someone is practicing (slow work, repeats, restarts, partial
play, error correction) rather than performing top-to-bottom.  Output goes to
model/data/evals/practice_eval/<piece>/candidates.yaml.

Uses yt-dlp flat-playlist search (no API key required -- same pattern as
model/src/data_collection/youtube_search.py).

Usage:
    # Search all 17 pieces
    uv run python scripts/search_practice_candidates.py --all

    # Search a single piece by slug
    uv run python scripts/search_practice_candidates.py --piece fur_elise

    # Dry-run: print queries without hitting YouTube
    uv run python scripts/search_practice_candidates.py --all --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Reuse search primitives from the existing skill_eval module
# ---------------------------------------------------------------------------

# Add model/src to path so we can import without package install friction
_MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MODEL_ROOT / "src"))

from data_collection.youtube_search import (  # type: ignore[import-not-found]  # noqa: E402
    Candidate,
    entry_to_candidate,
    search_youtube,
)

PRACTICE_EVAL_DIR = _MODEL_ROOT / "data" / "evals" / "practice_eval"
SKILL_EVAL_DIR    = _MODEL_ROOT / "data" / "evals" / "skill_eval"

# ---------------------------------------------------------------------------
# Piece definitions (17 pieces that have score JSONs + audio in skill_eval)
# ---------------------------------------------------------------------------

# Each entry: (piece_slug, display_title, composer, query_title)
# query_title is the short name used to build search queries.

PIECES: list[tuple[str, str, str, str]] = [
    ("bach_invention_1",         "Bach Invention No. 1",                    "Bach",          "Bach Invention No 1"),
    ("bach_prelude_c_wtc1",      "Bach Prelude in C Major WTC Book 1",      "Bach",          "Bach Prelude C major"),
    ("chopin_ballade_1",         "Chopin Ballade No. 1",                    "Chopin",        "Chopin Ballade No 1"),
    ("chopin_etude_op10no4",     "Chopin Etude Op. 10 No. 4",               "Chopin",        "Chopin Etude Op 10 No 4"),
    ("chopin_waltz_csm",         "Chopin Waltz in C-sharp Minor",           "Chopin",        "Chopin Waltz C sharp minor"),
    ("clair_de_lune",            "Debussy Clair de Lune",                   "Debussy",       "Clair de Lune"),
    ("debussy_arabesque_1",      "Debussy Arabesque No. 1",                 "Debussy",       "Debussy Arabesque No 1"),
    ("fantaisie_impromptu",      "Chopin Fantaisie-Impromptu",              "Chopin",        "Chopin Fantaisie Impromptu"),
    ("fur_elise",                "Fur Elise",                               "Beethoven",     "Fur Elise"),
    ("liszt_liebestraum_3",      "Liszt Liebestraum No. 3",                 "Liszt",         "Liszt Liebestraum No 3"),
    ("moonlight_sonata_mvt1",    "Beethoven Moonlight Sonata 1st Movement", "Beethoven",     "Moonlight Sonata first movement"),
    ("mozart_k545_mvt1",         "Mozart Sonata K.545 1st Movement",        "Mozart",        "Mozart Sonata K545 first movement"),
    ("nocturne_op9no2",          "Chopin Nocturne Op. 9 No. 2",             "Chopin",        "Chopin Nocturne Op 9 No 2"),
    ("pathetique_mvt2",          "Beethoven Pathetique Sonata 2nd Movement","Beethoven",     "Pathetique second movement"),
    ("rachmaninoff_prelude_csm", "Rachmaninoff Prelude in C-sharp Minor",   "Rachmaninoff",  "Rachmaninoff Prelude C sharp minor"),
    ("schumann_traumerei",       "Schumann Traumerei",                      "Schumann",      "Schumann Traumerei"),
]

# Practice-targeted query templates.  {t} is substituted with query_title.
QUERY_TEMPLATES = [
    "{t} slow practice",
    "{t} practice session",
    "{t} working on",
    "{t} learning",
    "{t} rehearsal",
    "{t} sight reading",
]

# Per-query result count
PER_QUERY = 25

# Rate-limit between queries (seconds)
RATE_LIMIT_QUERY = 1.5
RATE_LIMIT_PIECE = 2.0

# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _load_skill_eval_ids(piece_slug: str) -> set[str]:
    """Return all video_ids already present in skill_eval for this piece."""
    ids: set[str] = set()
    for filename in ("candidates.yaml", "manifest.yaml"):
        path = SKILL_EVAL_DIR / piece_slug / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        for rec in data.get("recordings", []):
            vid = rec.get("video_id")
            if vid:
                ids.add(vid)
    return ids


def _load_existing_practice_ids(piece_slug: str) -> set[str]:
    """Return video_ids already in practice_eval/candidates.yaml for this piece."""
    path = PRACTICE_EVAL_DIR / piece_slug / "candidates.yaml"
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        return set()
    return {
        rec["video_id"]
        for rec in data.get("recordings", [])
        if rec.get("video_id")
    }


# ---------------------------------------------------------------------------
# Candidate -> YAML dict
# ---------------------------------------------------------------------------

def _candidate_to_dict(c: Candidate, query_source: str) -> dict:
    return {
        "video_id":        c.video_id,
        "title":           c.title,
        "channel":         c.channel,
        "duration_seconds": c.duration_seconds,
        "view_count":      c.view_count,
        "url":             f"https://www.youtube.com/watch?v={c.video_id}",
        "query_source":    query_source,
        "approved":        None,
        "review_notes":    "",
    }


# ---------------------------------------------------------------------------
# Write candidates.yaml, merging with any existing content
# ---------------------------------------------------------------------------

def write_candidates_yaml(
    piece_slug: str,
    title: str,
    composer: str,
    new_candidates: list[tuple[Candidate, str]],  # (candidate, query_source)
) -> tuple[Path, int]:
    """Merge new candidates into practice_eval/<piece>/candidates.yaml.

    Existing records (with any review state) are preserved.
    New records are appended.
    """
    piece_dir = PRACTICE_EVAL_DIR / piece_slug
    piece_dir.mkdir(parents=True, exist_ok=True)
    out_path = piece_dir / "candidates.yaml"

    # Load existing records to preserve review state
    existing_records: list[dict] = []
    existing_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            existing_data = yaml.safe_load(f)
        if existing_data and existing_data.get("recordings"):
            for rec in existing_data["recordings"]:
                existing_records.append(rec)
                existing_ids.add(rec["video_id"])

    appended = 0
    for candidate, query_source in new_candidates:
        if candidate.video_id in existing_ids:
            continue
        existing_records.append(_candidate_to_dict(candidate, query_source))
        existing_ids.add(candidate.video_id)
        appended += 1

    data = {
        "piece":      piece_slug,
        "title":      title,
        "composer":   composer,
        "recordings": existing_records,
    }

    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return out_path, appended


# ---------------------------------------------------------------------------
# Search one piece
# ---------------------------------------------------------------------------

def search_piece(
    piece_slug: str,
    title: str,
    composer: str,
    query_title: str,
    dry_run: bool = False,
) -> int:
    """Run all practice query templates for a piece and write candidates.yaml.

    Returns total new candidates appended.
    """
    skill_blacklist  = _load_skill_eval_ids(piece_slug)
    practice_existing = _load_existing_practice_ids(piece_slug)
    seen_ids = skill_blacklist | practice_existing

    queries = [t.format(t=query_title) for t in QUERY_TEMPLATES]

    print(f"  Deduping against {len(skill_blacklist)} skill_eval IDs, "
          f"{len(practice_existing)} existing practice IDs")

    collected: list[tuple[Candidate, str]] = []
    seen_this_run: set[str] = set()

    for i, (query, template) in enumerate(zip(queries, QUERY_TEMPLATES)):
        print(f"  [{i + 1}/{len(queries)}] {query}")

        if dry_run:
            print("    (dry-run, skipping)")
            if i < len(queries) - 1:
                time.sleep(0.1)
            continue

        try:
            entries = search_youtube(query, max_results=PER_QUERY)
        except RuntimeError as e:
            print(f"    [error] {e}", file=sys.stderr)
            if i < len(queries) - 1:
                time.sleep(RATE_LIMIT_QUERY)
            continue

        new_this_query = 0
        for entry in entries:
            candidate = entry_to_candidate(entry)
            if candidate is None:
                continue
            if candidate.video_id in seen_ids:
                continue
            if candidate.video_id in seen_this_run:
                continue
            seen_this_run.add(candidate.video_id)
            collected.append((candidate, template))
            new_this_query += 1

        print(f"    -> {new_this_query} new ({len(collected)} total this piece)")

        if i < len(queries) - 1:
            time.sleep(RATE_LIMIT_QUERY)

    if dry_run:
        print(f"  (dry-run) would have written {len(queries)} queries")
        return 0

    out_path, appended = write_candidates_yaml(piece_slug, title, composer, collected)
    print(f"  Wrote {appended} new candidates -> {out_path}")
    return appended


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search YouTube for practice-mode piano recordings (practice_eval corpus)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Search all 17 pieces",
    )
    parser.add_argument(
        "--piece",
        type=str,
        metavar="SLUG",
        help="Single piece slug to search (e.g. fur_elise)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries without hitting YouTube",
    )
    args = parser.parse_args()

    if not args.all and not args.piece:
        parser.print_help()
        sys.exit(1)

    if args.piece:
        matches = [p for p in PIECES if p[0] == args.piece]
        if not matches:
            slugs = ", ".join(p[0] for p in PIECES)
            raise ValueError(f"Unknown piece slug '{args.piece}'. Valid: {slugs}")
        pieces_to_search = matches
    else:
        pieces_to_search = PIECES

    print(f"Output directory: {PRACTICE_EVAL_DIR}")
    print(f"Pieces to search: {len(pieces_to_search)}")
    if args.dry_run:
        print("Mode: DRY RUN (no YouTube requests)")
    print()

    total_new = 0
    for i, (slug, title, composer, query_title) in enumerate(pieces_to_search):
        print(f"=== [{i + 1}/{len(pieces_to_search)}] {title} ({slug}) ===")
        new = search_piece(slug, title, composer, query_title, dry_run=args.dry_run)
        total_new += new
        print()
        if i < len(pieces_to_search) - 1 and not args.dry_run:
            time.sleep(RATE_LIMIT_PIECE)

    print(f"Done. New candidates added: {total_new} across {len(pieces_to_search)} pieces")


if __name__ == "__main__":
    main()
