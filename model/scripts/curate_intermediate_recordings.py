"""Search YouTube for intermediate piano recordings and curate candidates.

Two modes:
  1. Search (default): Discover candidates from YouTube, filter, write candidates.jsonl
  2. Download (--download): Download approved candidates and extract MuQ embeddings

Run from model/ directory:
    python scripts/curate_intermediate_recordings.py
    python scripts/curate_intermediate_recordings.py --download
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

import jsonlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Search queries targeting intermediate-level piano performances
SEARCH_QUERIES = [
    "piano student recital performance",
    "intermediate piano recital",
    "piano diploma exam performance",
    "amateur piano competition performance",
    "piano grade 8 recital",
    "piano student concert performance",
    "high school piano recital",
    "university piano student recital",
    "ABRSM piano performance",
    "RCM piano exam performance",
]

# Title substrings that indicate non-performance content
EXCLUDE_KEYWORDS = [
    "lesson", "tutorial", "how to", "learn to", "beginner",
    "practice tips", "technique", "exercises", "scales",
    "review", "unboxing", "reaction", "analysis", "masterclass",
    "sheet music", "synthesia", "midi", "karaoke",
]

MIN_DURATION = 120   # 2 minutes
MAX_DURATION = 900   # 15 minutes
MAX_RESULTS_PER_QUERY = 20


def search_youtube(query: str, max_results: int) -> list[dict]:
    """Search YouTube with yt-dlp and return video metadata."""
    search_url = f"ytsearch{max_results}:{query}"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--no-download",
                search_url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("yt-dlp search failed for '%s': %s", query, e)
        return []

    if result.returncode != 0:
        logger.warning("yt-dlp error for '%s': %s", query, result.stderr[:200])
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = data.get("id", "")
        if not video_id:
            continue

        videos.append({
            "id": video_id,
            "title": data.get("title", ""),
            "channel": data.get("channel", data.get("uploader", "")),
            "duration": data.get("duration") or 0,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "query": query,
        })

    return videos


def filter_candidates(videos: list[dict]) -> list[dict]:
    """Filter videos by duration and title heuristics."""
    filtered = []
    seen_ids = set()

    for v in videos:
        vid = v["id"]
        if vid in seen_ids:
            continue
        seen_ids.add(vid)

        duration = v.get("duration", 0)
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        title_lower = v.get("title", "").lower()
        if any(kw in title_lower for kw in EXCLUDE_KEYWORDS):
            continue

        filtered.append(v)

    return filtered


def search_mode(cache_dir: Path) -> None:
    """Search YouTube and write candidates.jsonl."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = cache_dir / "candidates.jsonl"

    all_videos = []
    for query in SEARCH_QUERIES:
        logger.info("Searching: '%s'", query)
        videos = search_youtube(query, MAX_RESULTS_PER_QUERY)
        logger.info("  Found %d raw results", len(videos))
        all_videos.extend(videos)

    logger.info("Total raw results: %d", len(all_videos))
    candidates = filter_candidates(all_videos)
    logger.info("After filtering: %d candidates", len(candidates))

    # Load existing candidates to avoid duplicates
    existing_ids = set()
    if candidates_path.exists():
        with jsonlines.open(candidates_path) as reader:
            for r in reader:
                existing_ids.add(r["id"])

    new_count = 0
    with jsonlines.open(candidates_path, mode="a") as writer:
        for c in candidates:
            if c["id"] not in existing_ids:
                writer.write(c)
                existing_ids.add(c["id"])
                new_count += 1

    logger.info("Wrote %d new candidates to %s", new_count, candidates_path)
    logger.info("Total candidates: %d", len(existing_ids))

    # Print summary for manual review
    print("\n=== Candidates for Manual Review ===\n")
    all_candidates = []
    with jsonlines.open(candidates_path) as reader:
        all_candidates = list(reader)

    for i, c in enumerate(all_candidates):
        dur = int(c["duration"])
        mins = dur // 60
        secs = dur % 60
        approved = " [APPROVED]" if c.get("approved") else ""
        print(f"  {i+1:3d}. [{mins}:{secs:02d}] {c['title'][:70]}")
        print(f"       {c['url']}{approved}")
        print(f"       Channel: {c['channel'][:50]}  |  Query: {c['query']}")
        print()

    print(f"Total: {len(all_candidates)} candidates")
    print(f"\nTo approve candidates, edit {candidates_path}")
    print('and add "approved": true to each line you want to download.')
    print(f"Then run: python {Path(__file__).name} --download")


def download_mode(cache_dir: Path) -> None:
    """Download approved candidates and extract MuQ embeddings."""
    from model_improvement.youtube_piano import (
        download_piano_audio,
        segment_and_embed_piano,
    )

    candidates_path = cache_dir / "candidates.jsonl"
    if not candidates_path.exists():
        logger.error("No candidates file at %s. Run search mode first.", candidates_path)
        sys.exit(1)

    with jsonlines.open(candidates_path) as reader:
        candidates = list(reader)

    approved = [c for c in candidates if c.get("approved")]
    if not approved:
        logger.error(
            "No approved candidates. Edit %s and add '\"approved\": true' "
            "to candidates you want to download.",
            candidates_path,
        )
        sys.exit(1)

    logger.info("Found %d approved candidates", len(approved))

    # Convert to format expected by download_piano_audio
    videos = [
        {
            "id": c["id"],
            "title": c.get("title", ""),
            "channel": c.get("channel", ""),
            "url": c["url"],
        }
        for c in approved
    ]

    logger.info("Downloading audio...")
    records = download_piano_audio(videos, cache_dir)
    logger.info("Downloaded %d new recordings", len(records))

    logger.info("Segmenting and extracting MuQ embeddings...")
    n = segment_and_embed_piano(cache_dir)
    logger.info("Processed %d new segments", n)

    emb_dir = cache_dir / "muq_embeddings"
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0
    logger.info("Total embeddings: %d", n_emb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate intermediate piano recordings from YouTube"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "intermediate_cache",
        help="Output directory",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download approved candidates (run search first)",
    )
    args = parser.parse_args()

    if args.download:
        download_mode(args.cache_dir)
    else:
        search_mode(args.cache_dir)


if __name__ == "__main__":
    main()
