"""Download YouTube transcripts with quality gating.

Usage:
  uv run python -m apps.evals.teaching_knowledge.download_transcripts \
    --source t2 --limit 50
  uv run python -m apps.evals.teaching_knowledge.download_transcripts \
    --source search --query "piano masterclass feedback" --limit 100
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "transcripts"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
T2_MANIFEST_DIR = PROJECT_ROOT / "model" / "data" / "evals" / "skill_eval"

NOISE_TOKENS = re.compile(r"\[(?:inaudible|Music|Applause|Laughter)\]", re.IGNORECASE)


def download_transcript(video_id: str, output_dir: Path) -> Path | None:
    """Download auto-generated subtitles for a YouTube video.
    Returns path to transcript file, or None if download fails or quality gate fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / video_id

    # Skip if already downloaded
    srt_file = out_path.with_suffix(".en.vtt")
    if srt_file.exists():
        return srt_file

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--output", str(out_path),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  SKIP {video_id}: yt-dlp error: {result.stderr[:200]}")
            return None
    except FileNotFoundError:
        raise FileNotFoundError("yt-dlp not found. Install with: brew install yt-dlp")
    except subprocess.TimeoutExpired:
        print(f"  SKIP {video_id}: download timeout")
        return None

    # Find the downloaded subtitle file (yt-dlp may use .vtt or .srt)
    for ext in [".en.vtt", ".en.srt", ".vtt", ".srt"]:
        candidate = out_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    print(f"  SKIP {video_id}: no subtitle file found")
    return None


def parse_vtt(path: Path) -> str:
    """Extract plain text from VTT/SRT subtitle file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text_lines = []
    for line in lines:
        # Skip timestamps, headers, empty lines
        if "-->" in line or line.strip().isdigit() or line.startswith("WEBVTT") or not line.strip():
            continue
        # Remove VTT formatting tags
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in text_lines[-1:]:  # deduplicate consecutive
            text_lines.append(clean)
    return " ".join(text_lines)


def quality_gate(text: str, min_words: int = 500, max_noise_ratio: float = 0.3) -> bool:
    """Check if transcript passes quality thresholds."""
    words = text.split()
    if len(words) < min_words:
        return False
    noise_count = len(NOISE_TOKENS.findall(text))
    if noise_count / max(len(words), 1) > max_noise_ratio:
        return False
    return True


def get_t2_video_ids(limit: int = 50) -> list[str]:
    """Extract video IDs from T2 masterclass manifest files."""
    video_ids = []
    for manifest_dir in sorted(T2_MANIFEST_DIR.iterdir()):
        manifest = manifest_dir / "manifest.yaml"
        if not manifest.exists():
            continue
        # Simple YAML parsing for video_id fields
        text = manifest.read_text()
        for match in re.finditer(r"video_id:\s*([A-Za-z0-9_-]+)", text):
            video_ids.append(match.group(1))
    # Deduplicate (same video may appear in multiple pieces)
    seen = set()
    unique = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique[:limit]


def search_youtube(query: str, limit: int = 100) -> list[str]:
    """Search YouTube for video IDs matching a query."""
    try:
        result = subprocess.run(
            ["yt-dlp", f"ytsearch{limit}:{query}", "--get-id", "--flat-playlist"],
            capture_output=True, text=True, timeout=120,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError("yt-dlp not found. Install with: brew install yt-dlp")
    except subprocess.TimeoutExpired:
        print(f"  WARNING: yt-dlp search timed out for '{query}'")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download YouTube transcripts")
    parser.add_argument("--source", choices=["t2", "search"], required=True)
    parser.add_argument("--query", type=str, help="Search query (for --source search)")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.source == "t2":
        video_ids = get_t2_video_ids(args.limit)
        print(f"Found {len(video_ids)} T2 video IDs (limit={args.limit})")
    else:
        if not args.query:
            print("ERROR: --query required for --source search")
            sys.exit(1)
        video_ids = search_youtube(args.query, args.limit)
        print(f"Found {len(video_ids)} videos for '{args.query}'")

    results = {"downloaded": 0, "quality_pass": 0, "quality_fail": 0, "error": 0}
    manifest = []

    for i, vid in enumerate(video_ids):
        print(f"[{i+1}/{len(video_ids)}] {vid}...", end=" ")
        path = download_transcript(vid, args.output_dir)
        if path is None:
            results["error"] += 1
            continue

        results["downloaded"] += 1
        text = parse_vtt(path)

        if quality_gate(text):
            results["quality_pass"] += 1
            manifest.append({
                "video_id": vid,
                "path": str(path),
                "word_count": len(text.split()),
                "source": args.source,
            })
            print(f"PASS ({len(text.split())} words)")
        else:
            results["quality_fail"] += 1
            print(f"FAIL (quality gate)")

    # Save manifest
    manifest_path = args.output_dir / f"manifest_{args.source}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nResults: {results}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
