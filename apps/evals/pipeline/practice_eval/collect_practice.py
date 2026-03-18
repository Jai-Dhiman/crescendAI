"""Collect YouTube practice recordings for pipeline eval.

Searches for students practicing Fur Elise and Nocturne Op. 9 No. 2,
filters for actual practice sessions, outputs YAML scenario cards
for human review.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.collect_practice --piece fur_elise --search-only
    uv run python -m pipeline.practice_eval.collect_practice --piece nocturne_op9no2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
AUDIO_DIR = Path(__file__).parent / "audio"

PIECES = {
    "fur_elise": {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "piece_query": "beethoven fur elise",
        "duration_range": (60, 360),
        "searches": [
            "fur elise practice session piano",
            "fur elise learning piano slow",
            "fur elise beginner practicing",
            "fur elise piano progress practice",
            "fur elise hands separate practice",
        ],
    },
    "nocturne_op9no2": {
        "title": "Chopin Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "piece_query": "chopin nocturne op 9 no 2",
        "duration_range": (60, 420),
        "searches": [
            "chopin nocturne op 9 no 2 practice piano",
            "chopin nocturne practicing slow",
            "chopin nocturne piano progress learning",
            "chopin nocturne beginner practice session",
        ],
    },
}

SKIP_KEYWORDS = [
    "tutorial", "synthesia", "sheet music", "how to play",
    "easy piano", "piano lesson", "slow tutorial", "learn to play",
    "midi", "piano tiles", "roblox",
]


def search_youtube(query: str, max_results: int = 15) -> list[dict]:
    """Search YouTube via yt-dlp."""
    cmd = [
        "yt-dlp", f"ytsearch{max_results}:{query}",
        "--dump-json", "--flat-playlist", "--no-download",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return []
    entries = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def collect_candidates(piece_id: str) -> list[dict]:
    """Search YouTube and collect practice video candidates."""
    piece = PIECES[piece_id]
    min_dur, max_dur = piece["duration_range"]
    seen: set[str] = set()
    candidates = []

    for query in piece["searches"]:
        print(f"  Searching: {query}")
        for entry in search_youtube(query):
            vid = entry.get("id", "")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            title = entry.get("title", "")
            dur = entry.get("duration", 0)
            if dur and (dur < min_dur or dur > max_dur):
                continue
            if any(kw in title.lower() for kw in SKIP_KEYWORDS):
                continue
            candidates.append({
                "video_id": vid,
                "title": title,
                "channel": entry.get("channel", entry.get("uploader", "")),
                "duration_seconds": dur or 0,
                "url": f"https://youtube.com/watch?v={vid}",
                "include": False,
                "skill_level": 0,
                "general_notes": "",
                "audio_quality": "",
                "expected_stop": True,
            })
        time.sleep(2)

    print(f"  Found {len(candidates)} candidates")
    return candidates


def save_candidates(piece_id: str, candidates: list[dict]) -> Path:
    """Save candidates to YAML for human review."""
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    data = {
        "piece": piece_id,
        "title": PIECES[piece_id]["title"],
        "composer": PIECES[piece_id]["composer"],
        "piece_query": PIECES[piece_id]["piece_query"],
        "candidates": candidates,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Saved to {path}")
    print("  NEXT: Review YAML, set include=true on practice videos, fill skill_level/notes")
    return path


def download_included(piece_id: str):
    """Download audio for included recordings."""
    path = SCENARIOS_DIR / f"{piece_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    included = [c for c in data["candidates"] if c.get("include")]
    print(f"  Downloading {len(included)} included recordings...")
    for i, rec in enumerate(included):
        vid = rec["video_id"]
        out = AUDIO_DIR / f"{vid}.wav"
        if out.exists():
            print(f"  [{i+1}/{len(included)}] {vid} -- exists")
            continue
        print(f"  [{i+1}/{len(included)}] {vid} -- downloading...")
        cmd = [
            "yt-dlp", f"https://youtube.com/watch?v={vid}",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "-o", str(out), "--no-playlist", "--quiet",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"    Timeout downloading {vid}")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--piece", required=True, choices=list(PIECES.keys()))
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    print(f"=== Practice: {PIECES[args.piece]['title']} ===")
    scenario_path = SCENARIOS_DIR / f"{args.piece}.yaml"
    if scenario_path.exists() and not args.search_only:
        download_included(args.piece)
    else:
        save_candidates(args.piece, collect_candidates(args.piece))
        if not args.search_only:
            print("\n  Review YAML first, then re-run without --search-only.")


if __name__ == "__main__":
    main()
