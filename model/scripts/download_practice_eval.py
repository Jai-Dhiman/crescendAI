"""Download approved practice_eval audio clips via yt-dlp.

Walks `model/data/evals/practice_eval/<piece>/candidates.yaml`, filters
`approved: true` (optionally constrained by --piece and --video-ids), and
downloads each video as a 24kHz mono WAV under
`model/data/evals/practice_eval/<piece>/audio/<video_id>.wav`.

Idempotent: skips existing files. Errors per video are logged and do not
abort the batch; final exit code is non-zero if any download failed.

Usage:
    uv run python scripts/download_practice_eval.py                    # all approved
    uv run python scripts/download_practice_eval.py --piece fur_elise  # one piece
    uv run python scripts/download_practice_eval.py --video-ids a,b,c  # explicit ids
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

MODEL_ROOT = Path(__file__).resolve().parents[1]
PRACTICE_EVAL_DIR = MODEL_ROOT / "data" / "evals" / "practice_eval"


def _download_one(url: str, out_wav: Path) -> None:
    """Download a single video as 24kHz mono WAV. Raises on failure."""
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "--output", str(out_wav.with_suffix(".%(ext)s")),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {result.stderr[:500]}"
        )
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"yt-dlp produced empty file: {out_wav}")


def _select_targets(piece_filter: str | None, video_ids: set[str] | None) -> list[tuple[str, dict]]:
    """Walk practice_eval pieces and return [(piece, recording_dict), ...] for approved+filter matches."""
    targets: list[tuple[str, dict]] = []
    for piece_dir in sorted(PRACTICE_EVAL_DIR.iterdir()):
        if not piece_dir.is_dir():
            continue
        if piece_filter is not None and piece_dir.name != piece_filter:
            continue
        cand = piece_dir / "candidates.yaml"
        if not cand.exists():
            continue
        data = yaml.safe_load(cand.read_text())
        for r in data.get("recordings", []):
            if r.get("approved") is not True:
                continue
            if video_ids is not None and r.get("video_id") not in video_ids:
                continue
            targets.append((piece_dir.name, r))
    return targets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--piece", help="restrict to one piece slug (e.g., fur_elise)")
    p.add_argument("--video-ids", help="comma-separated video_ids to fetch (subset)")
    args = p.parse_args()

    video_ids: set[str] | None = None
    if args.video_ids:
        video_ids = {v.strip() for v in args.video_ids.split(",") if v.strip()}

    targets = _select_targets(args.piece, video_ids)
    if not targets:
        print("no matching approved videos found", file=sys.stderr)
        return 1

    print(f"found {len(targets)} target(s)")
    failures: list[tuple[str, str, str]] = []
    skipped = 0
    downloaded = 0

    for piece, rec in targets:
        video_id = rec["video_id"]
        url = rec.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        audio_dir = PRACTICE_EVAL_DIR / piece / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        out_wav = audio_dir / f"{video_id}.wav"

        if out_wav.exists() and out_wav.stat().st_size > 0:
            print(f"  SKIP {piece}/{video_id} (exists)")
            skipped += 1
            continue

        print(f"  GET  {piece}/{video_id} ({rec.get('duration_seconds', '?')}s) ...", flush=True)
        try:
            _download_one(url, out_wav)
            downloaded += 1
        except Exception as exc:
            failures.append((piece, video_id, str(exc)))
            print(f"       FAIL: {exc}", file=sys.stderr)

    print()
    print(f"downloaded={downloaded} skipped={skipped} failed={len(failures)}")
    if failures:
        for piece, vid, msg in failures:
            print(f"  FAIL {piece}/{vid}: {msg[:200]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
