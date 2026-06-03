"""yt-dlp audio acquisition (cache-miss only).

Downloads a single YouTube video as a mono 16 kHz WAV. Called only when
the expected WAV file is missing from the audio cache directory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class AcquireError(RuntimeError):
    pass


def acquire_audio(
    video_id: str,
    out_dir: Path,
    cookies_file: Path | None = None,
) -> Path:
    """Download audio for video_id to out_dir/{video_id}.wav using yt-dlp.

    Returns the path to the downloaded WAV file.

    Raises:
        AcquireError: if yt-dlp exits non-zero or the output file is missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.wav"
    if out_path.exists():
        return out_path

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
    ]
    if cookies_file is not None:
        if not cookies_file.exists():
            raise AcquireError(f"cookies file not found: {cookies_file}")
        cmd += ["--cookies", str(cookies_file)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AcquireError(
            f"yt-dlp failed for {video_id} (exit {res.returncode}): {res.stderr[:500]}"
        )
    if not out_path.exists():
        raise AcquireError(
            f"yt-dlp succeeded but output file not found: {out_path}"
        )
    return out_path
