"""Sync T5 skill eval audio to/from Cloudflare R2.

Upload local audio to R2 for persistent storage, download from R2
when needed for inference. Frees local disk by treating R2 as the
source of truth for audio files.

Usage:
    # Upload all local audio to R2
    cd apps/evals/
    uv run python -m model.skill_eval.r2_sync upload

    # Upload a single piece
    uv run python -m model.skill_eval.r2_sync upload --piece fur_elise

    # Upload and delete local copies
    uv run python -m model.skill_eval.r2_sync upload --delete-local

    # Download a piece from R2 for local inference
    uv run python -m model.skill_eval.r2_sync download --piece fur_elise

    # Download all pieces
    uv run python -m model.skill_eval.r2_sync download

    # List what's in R2
    uv run python -m model.skill_eval.r2_sync list

    # Show storage summary (local vs R2)
    uv run python -m model.skill_eval.r2_sync status

Requires environment variables (or .env file in apps/evals/):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3

from paths import MODEL_DATA

# Load .env from apps/evals/ if present
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

DATA_DIR = MODEL_DATA / "evals" / "skill_eval"
BUCKET = "crescendai-bucket"
R2_PREFIX = "t5-audio"


def get_s3_client():
    """Create S3 client for R2."""
    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def get_local_pieces() -> list[str]:
    """List piece directories that have audio locally."""
    pieces = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "audio").is_dir():
            wav_count = len(list((d / "audio").glob("*.wav")))
            if wav_count > 0:
                pieces.append(d.name)
    return pieces


def get_all_pieces() -> list[str]:
    """List all piece directories (with or without audio)."""
    return [
        d.name for d in sorted(DATA_DIR.iterdir())
        if d.is_dir() and (d / "manifest.yaml").exists()
    ]


def cmd_upload(piece: str | None, delete_local: bool):
    """Upload local audio to R2."""
    s3 = get_s3_client()

    if piece:
        pieces = [piece]
    else:
        pieces = get_local_pieces()

    if not pieces:
        print("No local audio found to upload.")
        return

    total_uploaded = 0
    total_bytes = 0

    for p in pieces:
        audio_dir = DATA_DIR / p / "audio"
        if not audio_dir.exists():
            print(f"  {p}: no audio directory, skipping")
            continue

        wav_files = sorted(audio_dir.glob("*.wav"))
        if not wav_files:
            print(f"  {p}: no WAV files, skipping")
            continue

        print(f"  {p}: uploading {len(wav_files)} files...")
        for wf in wav_files:
            key = f"{R2_PREFIX}/{p}/{wf.name}"
            s3.upload_file(
                str(wf), BUCKET, key,
                ExtraArgs={"ContentType": "audio/wav"},
            )
            total_bytes += wf.stat().st_size
            total_uploaded += 1

        if delete_local:
            for wf in wav_files:
                wf.unlink()
            print(f"    deleted {len(wav_files)} local files")

    print(f"\nUploaded {total_uploaded} files ({total_bytes / 1e6:.0f} MB) to R2")


def cmd_download(piece: str | None):
    """Download audio from R2 to local disk."""
    s3 = get_s3_client()

    if piece:
        pieces = [piece]
    else:
        pieces = get_all_pieces()

    total_downloaded = 0

    for p in pieces:
        audio_dir = DATA_DIR / p / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # List objects in R2 for this piece
        prefix = f"{R2_PREFIX}/{p}/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        objects = response.get("Contents", [])

        if not objects:
            print(f"  {p}: nothing in R2")
            continue

        # Skip files that already exist locally
        to_download = []
        for obj in objects:
            filename = obj["Key"].split("/")[-1]
            local_path = audio_dir / filename
            if not local_path.exists():
                to_download.append((obj["Key"], local_path))

        if not to_download:
            existing = len(list(audio_dir.glob("*.wav")))
            print(f"  {p}: all {existing} files already local")
            continue

        print(f"  {p}: downloading {len(to_download)} files...")
        for key, local_path in to_download:
            s3.download_file(BUCKET, key, str(local_path))
            total_downloaded += 1

    print(f"\nDownloaded {total_downloaded} files from R2")


def cmd_list():
    """List what's stored in R2."""
    s3 = get_s3_client()

    # List all objects under t5-audio/
    paginator = s3.get_paginator("list_objects_v2")
    by_piece: dict[str, list[dict]] = {}
    total_size = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{R2_PREFIX}/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 3:
                piece = parts[1]
                by_piece.setdefault(piece, []).append(obj)
                total_size += obj["Size"]

    if not by_piece:
        print("No T5 audio in R2 yet.")
        return

    print(f"R2 t5-audio/ contents ({total_size / 1e9:.2f} GB total):\n")
    for piece in sorted(by_piece):
        objects = by_piece[piece]
        piece_size = sum(o["Size"] for o in objects)
        print(f"  {piece:30s}  {len(objects):4d} files  {piece_size / 1e6:7.0f} MB")

    print(f"\n  {'TOTAL':30s}  {sum(len(v) for v in by_piece.values()):4d} files  {total_size / 1e6:7.0f} MB")


def cmd_status():
    """Show local vs R2 storage comparison."""
    # Local status
    print("LOCAL:")
    local_total_files = 0
    local_total_bytes = 0
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_dir():
            continue
        audio_dir = p / "audio"
        if not audio_dir.exists():
            continue
        wav_files = list(audio_dir.glob("*.wav"))
        if wav_files:
            size = sum(f.stat().st_size for f in wav_files)
            local_total_files += len(wav_files)
            local_total_bytes += size
            print(f"  {p.name:30s}  {len(wav_files):4d} files  {size / 1e6:7.0f} MB")

    print(f"  {'TOTAL':30s}  {local_total_files:4d} files  {local_total_bytes / 1e6:7.0f} MB")

    # R2 status
    try:
        s3 = get_s3_client()
        print("\nR2:")
        paginator = s3.get_paginator("list_objects_v2")
        by_piece: dict[str, list[dict]] = {}
        r2_total_size = 0

        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{R2_PREFIX}/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) >= 3:
                    piece = parts[1]
                    by_piece.setdefault(piece, []).append(obj)
                    r2_total_size += obj["Size"]

        if not by_piece:
            print("  (empty)")
        else:
            for piece in sorted(by_piece):
                objects = by_piece[piece]
                piece_size = sum(o["Size"] for o in objects)
                print(f"  {piece:30s}  {len(objects):4d} files  {piece_size / 1e6:7.0f} MB")
            print(f"  {'TOTAL':30s}  {sum(len(v) for v in by_piece.values()):4d} files  {r2_total_size / 1e6:7.0f} MB")

        # Savings
        if local_total_bytes > 0 and r2_total_size > 0:
            print(f"\n  Potential local savings: {local_total_bytes / 1e6:.0f} MB (delete local after R2 upload)")
    except Exception as e:
        print(f"\nR2: could not connect ({e})")


def main():
    parser = argparse.ArgumentParser(description="Sync T5 audio to/from R2")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload local audio to R2")
    up.add_argument("--piece", type=str, help="Upload only this piece")
    up.add_argument("--delete-local", action="store_true", help="Delete local files after upload")

    dl = sub.add_parser("download", help="Download audio from R2")
    dl.add_argument("--piece", type=str, help="Download only this piece")

    sub.add_parser("list", help="List R2 contents")
    sub.add_parser("status", help="Compare local vs R2 storage")

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args.piece, args.delete_local)
    elif args.command == "download":
        cmd_download(args.piece)
    elif args.command == "list":
        cmd_list()
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
