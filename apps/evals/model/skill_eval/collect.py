"""YouTube collection for skill eval.

Searches YouTube for piano performances at various skill levels,
downloads audio, and builds a YAML manifest with metadata-derived
skill labels.

Usage:
    cd apps/evals/

    # Original: search + download for a defined piece
    uv run python -m model.skill_eval.collect --piece fur_elise
    uv run python -m model.skill_eval.collect --piece fur_elise --search-only

    # Download from existing manifest + upload to R2 + delete local
    uv run --extra model python -m model.skill_eval.collect --manifest chopin_ballade_1 --r2
    uv run --extra model python -m model.skill_eval.collect --manifest all --r2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

from paths import MODEL_DATA

DATA_DIR = MODEL_DATA / "evals" / "skill_eval"

PIECES = {
    "fur_elise": {
        "title": "Fur Elise",
        "composer": "Beethoven",
        "duration_range": (90, 360),  # 1.5 - 6 min
        "searches": [
            ("fur elise beginner piano", [1, 2]),
            ("fur elise piano progress", [1, 2]),
            ("fur elise piano 1 year", [1, 2]),
            ("fur elise grade 3 piano", [2, 3]),
            ("fur elise piano", [3]),
            ("fur elise piano recital", [3, 4]),
            ("fur elise piano performance", [4, 5]),
            ("fur elise piano concert professional", [5]),
        ],
    },
    "nocturne_op9no2": {
        "title": "Chopin Nocturne Op. 9 No. 2",
        "composer": "Chopin",
        "duration_range": (180, 420),  # 3 - 7 min
        "searches": [
            ("chopin nocturne op 9 no 2 beginner piano", [1, 2]),
            ("chopin nocturne op 9 no 2 piano progress", [1, 2]),
            ("chopin nocturne op 9 no 2 piano student", [2, 3]),
            ("chopin nocturne op 9 no 2 piano", [3]),
            ("chopin nocturne op 9 no 2 piano recital", [3, 4]),
            ("chopin nocturne op 9 no 2 piano performance", [4, 5]),
            ("chopin nocturne op 9 no 2 piano concert", [5]),
            ("chopin nocturne op 9 no 2 lang lang horowitz argerich", [5]),
        ],
    },
}

# Keyword patterns for skill bucket assignment (case-insensitive)
BUCKET_KEYWORDS = {
    1: [
        "beginner", "first year", "1 year progress", "6 month",
        "learning", "grade 1", "grade 2", "abrsm 1", "abrsm 2",
        "first time", "self taught beginner", "starting piano",
    ],
    2: [
        "grade 3", "grade 4", "abrsm 3", "abrsm 4", "rcm 3", "rcm 4",
        "2 year", "3 year", "progress", "improving",
    ],
    4: [
        "grade 7", "grade 8", "abrsm 7", "abrsm 8", "rcm 7", "rcm 8",
        "diploma", "conservatory", "conservatoire", "music school",
        "competition", "masterclass", "recital", "exam",
        "advanced", "university", "college", "audition",
    ],
    5: [
        "concert pianist", "professional", "world class",
    ],
}

# Known professional pianists (partial match on channel or title)
KNOWN_PROFESSIONALS = [
    "lang lang", "horowitz", "argerich", "zimerman", "pollini",
    "rubinstein", "kissin", "trifonov", "yuja wang", "barenboim",
    "ashkenazy", "brendel", "pogorelich", "lugansky", "sokolov",
    "deutsche grammophon", "decca classics", "warner classics",
    "sony classical", "harmonia mundi", "tonebase piano",
]

# Titles containing these are tutorials/non-performances -- skip
TUTORIAL_KEYWORDS = [
    "tutorial", "synthesia", "sheet music", "how to play",
    "easy piano", "piano lesson", "slow tutorial", "learn to play",
    "piano tutorial", "right hand only", "left hand only",
    "midi", "piano tiles", "roblox", "fortnite",
]


def classify_skill_bucket(title: str, channel: str, description: str) -> tuple[int, str]:
    """Assign a skill bucket (1-5) based on metadata keywords.

    Returns (bucket, rationale).
    """
    text = f"{title} {channel} {description}".lower()

    # Check professionals first
    for pro in KNOWN_PROFESSIONALS:
        if pro in text:
            return 5, f"known professional: '{pro}'"

    # Check keyword patterns (highest specificity first)
    for bucket in [1, 4, 2]:  # Check extremes before middle
        for kw in BUCKET_KEYWORDS[bucket]:
            if kw in text:
                return bucket, f"keyword match: '{kw}'"

    # Default: intermediate (no distinguishing signals)
    return 3, "no distinguishing keywords, defaulting to intermediate"


def search_youtube(query: str, max_results: int = 20) -> list[dict]:
    """Search YouTube via yt-dlp and return metadata for each result."""
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  yt-dlp search failed: {result.stderr[:200]}")
        return []

    entries = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return entries


def build_manifest(piece_id: str) -> list[dict]:
    """Search YouTube and build a manifest of recordings with skill labels."""
    piece = PIECES[piece_id]
    min_dur, max_dur = piece["duration_range"]

    seen_ids: set[str] = set()
    recordings: list[dict] = []
    bucket_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for query, target_buckets in piece["searches"]:
        print(f"  Searching: {query}")
        entries = search_youtube(query, max_results=15)
        time.sleep(2)  # Rate limiting

        for entry in entries:
            video_id = entry.get("id", "")
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            title = entry.get("title", "")
            channel = entry.get("channel", entry.get("uploader", ""))
            duration = entry.get("duration", 0)
            description = entry.get("description", "")

            # Duration filter
            if duration and (duration < min_dur or duration > max_dur):
                continue

            # Tutorial/non-performance filter
            title_lower = title.lower()
            if any(kw in title_lower for kw in TUTORIAL_KEYWORDS):
                continue

            # Classify skill level
            bucket, rationale = classify_skill_bucket(title, channel, description)

            # Skip if we already have enough for this bucket
            if bucket_counts.get(bucket, 0) >= 15:
                continue

            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            recordings.append({
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "duration_seconds": duration or 0,
                "skill_bucket": bucket,
                "label_rationale": rationale,
                "downloaded": False,
                "download_error": None,
            })

    # Sort by bucket for readability
    recordings.sort(key=lambda r: (r["skill_bucket"], r["title"]))

    print(f"  Found {len(recordings)} recordings: {dict(sorted(bucket_counts.items()))}")
    return recordings


def download_audio(piece_id: str, recordings: list[dict]) -> list[dict]:
    """Download audio for all recordings in the manifest."""
    audio_dir = DATA_DIR / piece_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(recordings):
        video_id = rec["video_id"]
        output_path = audio_dir / f"{video_id}.wav"

        if output_path.exists():
            rec["downloaded"] = True
            print(f"  [{i+1}/{len(recordings)}] {video_id} -- already downloaded")
            continue

        print(f"  [{i+1}/{len(recordings)}] {video_id} -- downloading...")

        cmd = [
            "yt-dlp",
            f"https://youtube.com/watch?v={video_id}",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "-o", str(output_path),
            "--no-playlist",
            "--quiet",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if output_path.exists():
                rec["downloaded"] = True
            else:
                # yt-dlp may add extension
                wav_files = list(audio_dir.glob(f"{video_id}*"))
                if wav_files:
                    wav_files[0].rename(output_path)
                    rec["downloaded"] = True
                else:
                    rec["download_error"] = "output file not found after download"
        except subprocess.TimeoutExpired:
            rec["download_error"] = "download timed out"
        except Exception as e:
            rec["download_error"] = str(e)

        time.sleep(2)  # Rate limiting

    downloaded = sum(1 for r in recordings if r["downloaded"])
    print(f"  Downloaded {downloaded}/{len(recordings)}")
    return recordings


def save_manifest(piece_id: str, recordings: list[dict]) -> Path:
    """Save manifest to YAML."""
    manifest_dir = DATA_DIR / piece_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.yaml"

    piece = PIECES[piece_id]
    data = {
        "piece": piece_id,
        "title": piece["title"],
        "composer": piece["composer"],
        "recordings": recordings,
    }

    with open(manifest_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Manifest saved: {manifest_path}")
    return manifest_path


def download_and_upload_r2(piece_id: str, batch_size: int = 20):
    """Download audio from manifest, upload to R2, delete local.

    Processes in batches to limit disk usage. Each batch:
    1. Download up to batch_size recordings
    2. Upload all to R2
    3. Delete local WAV files
    4. Save manifest progress
    """
    from model.skill_eval.r2_sync import BUCKET, R2_PREFIX, get_s3_client

    manifest_path = DATA_DIR / piece_id / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest for {piece_id}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    recordings = data.get("recordings", [])
    to_download = [r for r in recordings if not r.get("downloaded")]

    if not to_download:
        print(f"  {piece_id}: all {len(recordings)} recordings already downloaded")
        return

    print(f"  {piece_id}: {len(to_download)} to download ({len(recordings)} total)")

    audio_dir = DATA_DIR / piece_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()

    total_uploaded = 0
    total_errors = 0

    for batch_start in range(0, len(to_download), batch_size):
        batch = to_download[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(to_download) + batch_size - 1) // batch_size
        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} recordings)")

        # Download batch
        local_files = []
        for i, rec in enumerate(batch):
            video_id = rec["video_id"]
            output_path = audio_dir / f"{video_id}.wav"
            idx = batch_start + i + 1
            total = len(to_download)

            print(f"    [{idx}/{total}] {video_id} -- downloading...")

            cmd = [
                "yt-dlp",
                f"https://youtube.com/watch?v={video_id}",
                "-x", "--audio-format", "wav",
                "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
                "-o", str(output_path),
                "--no-playlist",
                "--quiet",
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if output_path.exists():
                    local_files.append((rec, output_path))
                else:
                    wav_files = list(audio_dir.glob(f"{video_id}*"))
                    if wav_files:
                        wav_files[0].rename(output_path)
                        local_files.append((rec, output_path))
                    else:
                        rec["download_error"] = "output file not found"
                        total_errors += 1
            except subprocess.TimeoutExpired:
                rec["download_error"] = "download timed out"
                total_errors += 1
            except Exception as e:
                rec["download_error"] = str(e)
                total_errors += 1

            time.sleep(2)

        # Upload batch to R2
        if local_files:
            uploaded_paths = []
            print(f"    Uploading up to {len(local_files)} files to R2...")
            for rec, wav_path in local_files:
                if not wav_path.exists():
                    rec["download_error"] = "file disappeared before upload"
                    total_errors += 1
                    continue
                try:
                    key = f"{R2_PREFIX}/{piece_id}/{wav_path.name}"
                    s3.upload_file(
                        str(wav_path), BUCKET, key,
                        ExtraArgs={"ContentType": "audio/wav"},
                    )
                    rec["downloaded"] = True
                    uploaded_paths.append(wav_path)
                    total_uploaded += 1
                except Exception as e:
                    rec["download_error"] = f"R2 upload failed: {e}"
                    total_errors += 1

            # Delete successfully uploaded local files
            for wav_path in uploaded_paths:
                if wav_path.exists():
                    wav_path.unlink()

            print(f"    Uploaded {len(uploaded_paths)}/{len(local_files)} files")

        # Save manifest progress after each batch
        with open(manifest_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n  Done: {total_uploaded} uploaded to R2, {total_errors} errors")


def get_manifest_pieces() -> list[str]:
    """List piece IDs that have manifest.yaml files with recordings to download."""
    pieces = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "manifest.yaml"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            data = yaml.safe_load(f)
        recordings = data.get("recordings", [])
        remaining = sum(1 for r in recordings if not r.get("downloaded"))
        if remaining > 0:
            pieces.append(d.name)
    return pieces


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube recordings for skill eval")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--piece", choices=list(PIECES.keys()), help="Search + download a defined piece")
    group.add_argument("--manifest", type=str, help="Download from existing manifest (piece_id or 'all')")
    parser.add_argument("--search-only", action="store_true", help="Build manifest without downloading")
    parser.add_argument("--r2", action="store_true", help="Upload to R2 and delete local after download")
    parser.add_argument("--batch-size", type=int, default=20, help="Files per R2 upload batch (default: 20)")
    args = parser.parse_args()

    if args.manifest:
        # Manifest mode: download existing manifests + optionally upload to R2
        if args.manifest == "all":
            pieces = get_manifest_pieces()
            if not pieces:
                print("No pieces with remaining downloads.")
                return
            print(f"=== Processing {len(pieces)} pieces ===")
        else:
            pieces = [args.manifest]

        for piece_id in pieces:
            print(f"\n=== {piece_id} ===")
            if args.r2:
                download_and_upload_r2(piece_id, batch_size=args.batch_size)
            else:
                manifest_path = DATA_DIR / piece_id / "manifest.yaml"
                with open(manifest_path) as f:
                    data = yaml.safe_load(f)
                recordings = download_audio(piece_id, data.get("recordings", []))
                data["recordings"] = recordings
                with open(manifest_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return

    # Original search + download mode
    print(f"=== Collecting: {PIECES[args.piece]['title']} ===")

    manifest_path = DATA_DIR / args.piece / "manifest.yaml"
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = yaml.safe_load(f)
        recordings = existing.get("recordings", [])
        print(f"  Loaded existing manifest with {len(recordings)} recordings")
    else:
        recordings = build_manifest(args.piece)

    save_manifest(args.piece, recordings)

    if not args.search_only:
        recordings = download_audio(args.piece, recordings)
        save_manifest(args.piece, recordings)

    # Summary
    downloaded = sum(1 for r in recordings if r["downloaded"])
    bucket_counts = {}
    for r in recordings:
        b = r["skill_bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    print(f"\n=== Summary: {PIECES[args.piece]['title']} ===")
    print(f"Total: {len(recordings)}, Downloaded: {downloaded}")
    print(f"Buckets: {dict(sorted(bucket_counts.items()))}")


if __name__ == "__main__":
    main()
