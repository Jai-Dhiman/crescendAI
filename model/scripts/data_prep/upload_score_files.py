#!/usr/bin/env python3
"""
Upload score XML files to Google Drive.

This script uploads the MusicXML score files that are required for
score-aligned training. Without these files, the model cannot compute
score alignment features and training will produce R^2 = 0.

Usage:
    python scripts/upload_score_files.py

    # Dry run (check what would be uploaded):
    python scripts/upload_score_files.py --dry-run

    # Custom paths:
    python scripts/upload_score_files.py --local-dir /path/to/scores --remote gdrive:custom/path
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Default paths
DEFAULT_LOCAL_SCORE_DIR = Path("data/raw/PercePiano/virtuoso/data/score_xml")
DEFAULT_GDRIVE_DEST = "gdrive:percepiano_data/PercePiano/virtuoso/data/score_xml"
EXPECTED_FILE_COUNT = 262  # Known count of MusicXML files


class UploadError(Exception):
    """Raised when upload fails."""

    pass


def check_rclone_available() -> None:
    """
    Verify rclone is installed and gdrive remote is configured.

    Raises:
        UploadError: If rclone is not available or gdrive not configured
    """
    try:
        result = subprocess.run(
            ["rclone", "listremotes"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise UploadError(
            "rclone is not installed.\n"
            "\n"
            "Install rclone:\n"
            "  macOS: brew install rclone\n"
            "  Linux: curl https://rclone.org/install.sh | sudo bash\n"
            "\n"
            "Then configure Google Drive:\n"
            "  rclone config"
        )
    except subprocess.TimeoutExpired:
        raise UploadError("rclone command timed out - check your network connection")

    if result.returncode != 0:
        raise UploadError(f"rclone listremotes failed: {result.stderr}")

    if "gdrive:" not in result.stdout:
        raise UploadError(
            "rclone 'gdrive' remote not configured.\n"
            "\n"
            "Configure Google Drive remote:\n"
            "  rclone config\n"
            "\n"
            "Select 'n' for new remote, name it 'gdrive', choose 'drive' for Google Drive."
        )

    print("[OK] rclone is available and gdrive remote is configured")


def verify_local_files(local_dir: Path) -> int:
    """
    Verify local score files exist.

    Args:
        local_dir: Path to local score directory

    Returns:
        Number of MusicXML files found

    Raises:
        UploadError: If directory doesn't exist or no files found
    """
    if not local_dir.exists():
        raise UploadError(
            f"Score directory not found: {local_dir}\n"
            "\n"
            "Expected location: data/raw/PercePiano/virtuoso/data/score_xml/\n"
            "\n"
            "If you haven't cloned the PercePiano repository:\n"
            "  cd data/raw\n"
            "  git clone https://github.com/your-repo/PercePiano.git"
        )

    if not local_dir.is_dir():
        raise UploadError(f"Path exists but is not a directory: {local_dir}")

    score_files = list(local_dir.glob("*.musicxml"))

    if len(score_files) == 0:
        raise UploadError(
            f"No MusicXML files found in {local_dir}\n"
            "Expected *.musicxml files for score alignment."
        )

    print(f"[OK] Found {len(score_files)} MusicXML files in {local_dir}")

    if len(score_files) < EXPECTED_FILE_COUNT:
        print(f"[WARN] Expected {EXPECTED_FILE_COUNT} files, found {len(score_files)}")

    return len(score_files)


def check_existing_remote_files(remote_path: str) -> int:
    """
    Check if files already exist on remote.

    Args:
        remote_path: rclone remote path

    Returns:
        Number of files already on remote
    """
    try:
        result = subprocess.run(
            ["rclone", "ls", remote_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            # Directory might not exist yet
            return 0

        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        return len(lines)

    except subprocess.TimeoutExpired:
        print("[WARN] Timeout checking remote files - proceeding with upload")
        return 0


def upload_files(local_dir: Path, remote_path: str, dry_run: bool = False) -> None:
    """
    Upload files to Google Drive using rclone.

    Args:
        local_dir: Path to local score directory
        remote_path: rclone remote path
        dry_run: If True, only show what would be uploaded

    Raises:
        UploadError: If upload fails
    """
    cmd = [
        "rclone",
        "copy",
        str(local_dir),
        remote_path,
        "--progress",
        "--stats",
        "1s",
    ]

    if dry_run:
        cmd.append("--dry-run")
        print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}\n")
    else:
        print(f"\nUploading to {remote_path}...")

    try:
        result = subprocess.run(
            cmd,
            check=True,  # FAIL if rclone fails - no silent fallback
            timeout=600,  # 10 minute timeout for upload
        )
    except subprocess.CalledProcessError as e:
        raise UploadError(
            f"Upload failed with exit code {e.returncode}.\n"
            "Check your network connection and rclone configuration."
        )
    except subprocess.TimeoutExpired:
        raise UploadError(
            "Upload timed out after 10 minutes.\n"
            "Try again with a better network connection."
        )

    if not dry_run:
        print("\n[OK] Upload completed")


def verify_upload(remote_path: str, expected_count: int) -> None:
    """
    Verify upload was successful by counting remote files.

    Args:
        remote_path: rclone remote path
        expected_count: Expected number of files

    Raises:
        UploadError: If verification fails
    """
    print("\nVerifying upload...")

    try:
        result = subprocess.run(
            ["rclone", "ls", remote_path],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise UploadError(f"Failed to verify upload: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise UploadError("Verification timed out")

    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    uploaded_count = len(lines)

    if uploaded_count < expected_count:
        raise UploadError(
            f"Upload verification failed!\n"
            f"Expected {expected_count} files, found {uploaded_count} on remote.\n"
            "Some files may have failed to upload. Try running the script again."
        )

    print(f"[OK] Verified {uploaded_count} files on remote")


def main():
    parser = argparse.ArgumentParser(
        description="Upload score XML files to Google Drive for score-aligned training"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_SCORE_DIR,
        help=f"Local directory containing MusicXML files (default: {DEFAULT_LOCAL_SCORE_DIR})",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=DEFAULT_GDRIVE_DEST,
        help=f"rclone remote path (default: {DEFAULT_GDRIVE_DEST})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip upload verification step"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCORE FILE UPLOAD")
    print("=" * 60)
    print(f"Local:  {args.local_dir}")
    print(f"Remote: {args.remote}")
    print("=" * 60)

    try:
        # Step 1: Check rclone
        check_rclone_available()

        # Step 2: Verify local files
        local_count = verify_local_files(args.local_dir)

        # Step 3: Check existing remote files
        remote_count = check_existing_remote_files(args.remote)
        if remote_count > 0:
            print(f"[INFO] Found {remote_count} existing files on remote")
            if remote_count >= local_count:
                print("[INFO] Remote already has all files - skipping upload")
                return

        # Step 4: Upload
        upload_files(args.local_dir, args.remote, args.dry_run)

        # Step 5: Verify
        if not args.dry_run and not args.skip_verify:
            verify_upload(args.remote, local_count)

        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(f"Uploaded {local_count} score files to {args.remote}")
        print("\nNext steps:")
        print("1. Update training config to use remote score path")
        print("2. Run pre-flight validation: python -m src.utils.preflight_validation")
        print("3. Start training: python scripts/train_score_aligned.py")

    except UploadError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
