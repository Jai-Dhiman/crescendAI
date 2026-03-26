"""YouTube search + candidate generation for T5 Skill Corpus.

Uses yt-dlp flat-playlist search (no API key) to find piano recordings
across skill levels. Outputs candidates.yaml matching the manifest format.

Usage:
    # Search a single piece
    python -m src.data_collection.youtube_search \
        --piece moonlight_sonata_mvt1 \
        --title "Beethoven Moonlight Sonata" \
        --composer Beethoven \
        --queries "Beethoven Moonlight Sonata first movement piano" \
                  "Moonlight Sonata piano beginner" \
                  "Moonlight Sonata piano student recital" \
        --target 60

    # Run all 14 pieces
    python -m src.data_collection.youtube_search --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Piece definitions
# ---------------------------------------------------------------------------

@dataclass
class PieceSpec:
    piece: str
    title: str
    composer: str
    queries: list[str]
    target: int  # target candidate count


# Query strategy: beginner/progress queries FIRST (they're harder to find),
# then professional/general queries to fill remaining slots.
# "progress" videos are the best source for buckets 1-3.


def _make_queries(piece_name: str, alt_names: list[str] | None = None) -> list[str]:
    """Generate search queries for a piece, beginner-first ordering.

    Front-loads queries that find low-skill recordings (progress videos,
    student recitals, amateur performances), then adds general queries
    that tend to surface advanced/professional results.
    """
    names = [piece_name] + (alt_names or [])
    queries = []

    # --- Phase 1: Beginner/student content (run first, most important) ---
    for name in names:
        queries.extend([
            f"{name} piano progress",
            f"{name} adult beginner piano",
            f"{name} piano student recital",
            f"{name} piano amateur",
        ])

    # Additional beginner-signal queries with the primary name
    queries.extend([
        f"{piece_name} months piano progress",
        f"{piece_name} year piano progress",
        f"{piece_name} self taught piano",
        f"{piece_name} piano recital child",
        f"{piece_name} piano recital kid",
        f"{piece_name} grade piano",
    ])

    # --- Phase 2: General queries (tend to surface advanced/professional) ---
    for name in names:
        queries.append(f"{name} piano")

    queries.extend([
        f"{piece_name} piano performance",
        f"{piece_name} piano concert",
    ])

    return queries


PIECES: list[PieceSpec] = [
    # --- Core pieces (60-75 candidates) ---
    PieceSpec(
        piece="moonlight_sonata_mvt1",
        title="Beethoven Moonlight Sonata 1st Movement",
        composer="Beethoven",
        target=75,
        queries=_make_queries(
            "Moonlight Sonata first movement",
            ["Beethoven Moonlight Sonata", "Beethoven Sonata 14 first movement"],
        ),
    ),
    PieceSpec(
        piece="clair_de_lune",
        title="Debussy Clair de Lune",
        composer="Debussy",
        target=75,
        queries=_make_queries(
            "Clair de Lune",
            ["Debussy Clair de Lune"],
        ),
    ),
    PieceSpec(
        piece="bach_prelude_c_wtc1",
        title="Bach Prelude in C Major WTC Book 1",
        composer="Bach",
        target=75,
        queries=_make_queries(
            "Bach Prelude C major",
            ["Bach WTC 1 Prelude 1", "Bach Prelude BWV 846"],
        ),
    ),
    PieceSpec(
        piece="mozart_k545_mvt1",
        title="Mozart Sonata K.545 1st Movement",
        composer="Mozart",
        target=75,
        queries=_make_queries(
            "Mozart Sonata K545 first movement",
            ["Mozart Sonata Facile", "Mozart Sonata 16 C major"],
        ),
    ),
    PieceSpec(
        piece="chopin_waltz_csm",
        title="Chopin Waltz in C-sharp Minor",
        composer="Chopin",
        target=75,
        queries=_make_queries(
            "Chopin Waltz C sharp minor",
            ["Chopin Waltz Op 64 No 2"],
        ),
    ),
    PieceSpec(
        piece="liszt_liebestraum_3",
        title="Liszt Liebestraum No. 3",
        composer="Liszt",
        target=75,
        queries=_make_queries(
            "Liszt Liebestraum No 3",
            ["Liszt Liebestraum"],
        ),
    ),
    # --- Breadth pieces (40-50 candidates) ---
    PieceSpec(
        piece="chopin_etude_op10no4",
        title="Chopin Etude Op. 10 No. 4",
        composer="Chopin",
        target=55,
        queries=_make_queries(
            "Chopin Etude Op 10 No 4",
        ),
    ),
    PieceSpec(
        piece="pathetique_mvt2",
        title="Beethoven Pathetique Sonata 2nd Movement",
        composer="Beethoven",
        target=55,
        queries=_make_queries(
            "Pathetique second movement",
            ["Beethoven Pathetique Adagio", "Beethoven Sonata 8 second movement"],
        ),
    ),
    PieceSpec(
        piece="debussy_arabesque_1",
        title="Debussy Arabesque No. 1",
        composer="Debussy",
        target=55,
        queries=_make_queries(
            "Debussy Arabesque No 1",
            ["Debussy Arabesque 1"],
        ),
    ),
    PieceSpec(
        piece="chopin_ballade_1",
        title="Chopin Ballade No. 1",
        composer="Chopin",
        target=55,
        queries=_make_queries(
            "Chopin Ballade No 1",
            ["Chopin Ballade 1 G minor"],
        ),
    ),
    PieceSpec(
        piece="rachmaninoff_prelude_csm",
        title="Rachmaninoff Prelude in C-sharp Minor",
        composer="Rachmaninoff",
        target=55,
        queries=_make_queries(
            "Rachmaninoff Prelude C sharp minor",
            ["Rachmaninoff Prelude Op 3 No 2"],
        ),
    ),
    PieceSpec(
        piece="schumann_traumerei",
        title="Schumann Traumerei",
        composer="Schumann",
        target=55,
        queries=_make_queries(
            "Schumann Traumerei",
            ["Schumann Kinderszenen Traumerei"],
        ),
    ),
    PieceSpec(
        piece="bach_invention_1",
        title="Bach Invention No. 1",
        composer="Bach",
        target=55,
        queries=_make_queries(
            "Bach Invention No 1",
            ["Bach Invention 1 C major", "Bach Two Part Invention 1"],
        ),
    ),
    PieceSpec(
        piece="fantaisie_impromptu",
        title="Chopin Fantaisie-Impromptu",
        composer="Chopin",
        target=55,
        queries=_make_queries(
            "Chopin Fantaisie Impromptu",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Duration filtering constants
# ---------------------------------------------------------------------------

MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 900  # 15 minutes

# Keywords to exclude (compilations, tutorials, non-performance content)
EXCLUDE_KEYWORDS = [
    # Compilations / playlists
    "compilation",
    "mix",
    "playlist",
    "1 hour",
    "2 hour",
    "3 hour",
    "full album",
    "complete works",
    "all movements",
    # Ambient / non-performance
    "sleep music",
    "study music",
    "relaxing",
    "rain sounds",
    "asmr",
    # Tutorials and lessons (teacher talking, not performing)
    "tutorial",
    "lesson",
    "how to play",
    "learn to play",
    "piano lesson",
    "teaching",
    "step by step",
    "easy piano",
    "piano tutorial",
    "slow practice",
    "practice tips",
    "masterclass",  # usually teacher talking, not student performing
    "analysis",
    "explained",
    "sheet music",
    "midi visualization",
    "synthesia",
    # Other non-performance
    "reaction",
    "tier list",
    "ranking",
    "comparison",
    "vs",
    "top 10",
]


# ---------------------------------------------------------------------------
# Candidate data structure
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    video_id: str
    title: str
    channel: str
    duration_seconds: int
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    skill_bucket: Optional[int] = None
    label_rationale: Optional[str] = None
    downloaded: bool = False
    download_error: Optional[str] = None


# ---------------------------------------------------------------------------
# YouTube search via yt-dlp
# ---------------------------------------------------------------------------

def search_youtube(query: str, max_results: int = 30) -> list[dict]:
    """Run a yt-dlp search and return raw JSON entries."""
    search_url = f"ytsearch{max_results}:{query}"
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
        search_url,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp search failed for query '{query}': {result.stderr.strip()}"
        )

    entries = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [warn] Skipping malformed JSON line: {e}", file=sys.stderr)
    return entries


def _is_excluded(title: str) -> bool:
    """Check if a title matches any exclusion keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in EXCLUDE_KEYWORDS)


def entry_to_candidate(entry: dict) -> Optional[Candidate]:
    """Convert a yt-dlp JSON entry to a Candidate, or None if filtered out."""
    video_id = entry.get("id")
    title = entry.get("title", "")
    duration = entry.get("duration")

    if not video_id or not title:
        return None

    # Duration filter
    if duration is None:
        return None
    if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
        return None

    # Exclusion filter
    if _is_excluded(title):
        return None

    channel = entry.get("channel") or entry.get("uploader") or "Unknown"
    view_count = entry.get("view_count")
    upload_date = entry.get("upload_date")

    return Candidate(
        video_id=video_id,
        title=title,
        channel=channel,
        duration_seconds=int(duration),
        view_count=int(view_count) if view_count else None,
        upload_date=upload_date,
    )


def search_piece(spec: PieceSpec) -> list[Candidate]:
    """Search all query variants for a piece and deduplicate."""
    seen_ids: set[str] = set()
    candidates: list[Candidate] = []

    for i, query in enumerate(spec.queries):
        # Always run all queries -- beginner queries (run first) are most
        # valuable even if target is already met from later general queries.
        per_query = 25
        print(f"  [{i + 1}/{len(spec.queries)}] Searching: {query} (requesting {per_query})")

        try:
            entries = search_youtube(query, max_results=per_query)
        except RuntimeError as e:
            print(f"  [error] {e}", file=sys.stderr)
            continue

        new_this_query = 0
        for entry in entries:
            candidate = entry_to_candidate(entry)
            if candidate is None:
                continue
            if candidate.video_id in seen_ids:
                continue
            seen_ids.add(candidate.video_id)
            candidates.append(candidate)
            new_this_query += 1

        print(f"    -> {new_this_query} new candidates ({len(candidates)} total)")

        # Rate limit between queries
        if i < len(spec.queries) - 1:
            time.sleep(1.5)

    return candidates


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def _candidate_to_dict(c: Candidate) -> dict:
    """Convert Candidate to dict for YAML serialization."""
    d = {
        "video_id": c.video_id,
        "title": c.title,
        "channel": c.channel,
        "duration_seconds": c.duration_seconds,
    }
    if c.view_count is not None:
        d["view_count"] = c.view_count
    if c.upload_date is not None:
        d["upload_date"] = c.upload_date
    d["skill_bucket"] = c.skill_bucket
    d["label_rationale"] = c.label_rationale
    d["downloaded"] = c.downloaded
    d["download_error"] = c.download_error
    return d


def write_candidates_yaml(
    spec: PieceSpec,
    candidates: list[Candidate],
    output_dir: Path,
) -> Path:
    """Write candidates.yaml for a piece, merging with existing labels if present."""
    piece_dir = output_dir / spec.piece
    piece_dir.mkdir(parents=True, exist_ok=True)
    out_path = piece_dir / "candidates.yaml"

    # Preserve existing labels if candidates.yaml already has curated data
    existing_labels: dict[str, dict] = {}
    existing_status = "uncurated"
    if out_path.exists():
        with open(out_path) as f:
            existing = yaml.safe_load(f)
        if existing and existing.get("recordings"):
            for rec in existing["recordings"]:
                if rec.get("skill_bucket") is not None or rec.get("label_rationale") == "skip":
                    existing_labels[rec["video_id"]] = {
                        "skill_bucket": rec.get("skill_bucket"),
                        "label_rationale": rec.get("label_rationale"),
                        "downloaded": rec.get("downloaded", False),
                        "download_error": rec.get("download_error"),
                    }
            existing_status = existing.get("status", "uncurated")

    new_recordings = []
    for c in candidates:
        d = _candidate_to_dict(c)
        if d["video_id"] in existing_labels:
            d.update(existing_labels[d["video_id"]])
        new_recordings.append(d)

    # Preserve recordings that were in the old file but not in the new search
    new_vids = {c.video_id for c in candidates}
    if existing_labels:
        with open(out_path) as f:
            existing = yaml.safe_load(f)
        for rec in existing["recordings"]:
            if rec["video_id"] not in new_vids and rec["video_id"] in existing_labels:
                new_recordings.append(rec)

    # Recompute status
    n_labeled = sum(
        1 for r in new_recordings
        if r.get("skill_bucket") is not None or r.get("label_rationale") == "skip"
    )
    if n_labeled == len(new_recordings) and n_labeled > 0:
        status = "complete"
    elif n_labeled > 0:
        status = "partial"
    else:
        status = "uncurated"

    if existing_labels:
        print(f"  Preserved {len(existing_labels)} existing labels (was {existing_status})")

    data = {
        "piece": spec.piece,
        "title": spec.title,
        "composer": spec.composer,
        "status": status,
        "target_per_bucket": 10 if spec.target >= 60 else 6,
        "recordings": new_recordings,
    }

    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search YouTube for T5 Skill Corpus candidates"
    )
    parser.add_argument("--all", action="store_true", help="Search all 14 pieces")
    parser.add_argument("--piece", type=str, help="Single piece slug to search")
    parser.add_argument("--title", type=str, help="Piece title (for single piece mode)")
    parser.add_argument("--composer", type=str, help="Composer name (for single piece mode)")
    parser.add_argument("--queries", nargs="+", help="Search queries (for single piece mode)")
    parser.add_argument("--target", type=int, default=50, help="Target candidate count")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "evals" / "skill_eval",
        help="Output directory for candidates.yaml files",
    )

    args = parser.parse_args()

    if args.all:
        specs = PIECES
    elif args.piece:
        if not args.queries:
            raise ValueError("--queries required when using --piece")
        specs = [
            PieceSpec(
                piece=args.piece,
                title=args.title or args.piece,
                composer=args.composer or "Unknown",
                queries=args.queries,
                target=args.target,
            )
        ]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Output directory: {args.output_dir}")
    print(f"Pieces to search: {len(specs)}")
    print()

    total_candidates = 0
    for i, spec in enumerate(specs):
        print(f"=== [{i + 1}/{len(specs)}] {spec.title} ({spec.piece}) ===")
        print(f"  Target: {spec.target} candidates")

        candidates = search_piece(spec)
        out_path = write_candidates_yaml(spec, candidates, args.output_dir)

        print(f"  -> Wrote {len(candidates)} candidates to {out_path}")
        total_candidates += len(candidates)
        print()

        # Rate limit between pieces
        if i < len(specs) - 1:
            time.sleep(2)

    print(f"Done. Total candidates across {len(specs)} pieces: {total_candidates}")


if __name__ == "__main__":
    main()
