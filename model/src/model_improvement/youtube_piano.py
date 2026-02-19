"""T4: Unlabeled piano audio at scale from YouTube.

Downloads audio from curated piano YouTube channels, segments into 30s clips,
extracts clean MuQ embeddings, and optionally generates augmented pairs for
invariance training.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonlines
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PianoRecording:
    video_id: str
    title: str
    channel: str
    duration_seconds: float
    audio_path: str
    source_url: str


def load_channel_list(channels_path: Path) -> list[dict]:
    """Load curated YouTube piano channel list from YAML.

    Args:
        channels_path: Path to channels.yaml.

    Returns:
        List of dicts with keys: url, name, category.

    Raises:
        FileNotFoundError: If channels_path does not exist.
    """
    channels_path = Path(channels_path)
    if not channels_path.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_path}")

    with open(channels_path) as f:
        data = yaml.safe_load(f)

    return data.get("channels", [])


def discover_channel_videos(
    channel_url: str,
    max_videos: int = 100,
) -> list[dict]:
    """Discover videos from a YouTube channel using yt-dlp.

    Args:
        channel_url: YouTube channel URL.
        max_videos: Maximum number of videos to return.

    Returns:
        List of dicts with keys: id, title, duration, url.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--playlist-end", str(max_videos),
                channel_url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("yt-dlp failed for %s: %s", channel_url, e)
        return []

    if result.returncode != 0:
        logger.warning("yt-dlp error for %s: %s", channel_url, result.stderr[:200])
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
            "duration": data.get("duration", 0),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": data.get("channel", data.get("uploader", "")),
        })

    return videos


def _download_audio_yt_dlp(url: str, output_path: Path) -> None:
    """Download audio from YouTube as 24kHz mono WAV."""
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "--output", str(output_path.with_suffix(".%(ext)s")),
            "--no-playlist",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {result.stderr[:500]}"
        )


def download_piano_audio(
    videos: list[dict],
    cache_dir: Path,
) -> list[PianoRecording]:
    """Download audio from YouTube videos.

    Args:
        videos: List of video dicts from discover_channel_videos().
        cache_dir: Output directory.

    Returns:
        List of PianoRecording for successfully downloaded videos.
    """
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "recordings.jsonl"

    # Load existing downloads for idempotency
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for r in reader:
                existing_ids.add(r["video_id"])

    records: list[PianoRecording] = []

    for video in videos:
        video_id = video["id"]
        wav_path = audio_dir / f"{video_id}.wav"

        if video_id in existing_ids and wav_path.exists():
            logger.debug("Skipping %s (already downloaded)", video_id)
            continue

        url = video.get("url", f"https://www.youtube.com/watch?v={video_id}")

        logger.info("Downloading %s: %s", video_id, video.get("title", ""))

        try:
            _download_audio_yt_dlp(url, wav_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", video_id, e)
            continue

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            logger.error("Download produced empty file: %s", video_id)
            continue

        import soundfile as sf
        try:
            info = sf.info(str(wav_path))
            duration = info.duration
        except Exception:
            duration = 0.0

        record = PianoRecording(
            video_id=video_id,
            title=video.get("title", ""),
            channel=video.get("channel", ""),
            duration_seconds=duration,
            audio_path=f"audio/{video_id}.wav",
            source_url=url,
        )
        records.append(record)

        with jsonlines.open(metadata_path, mode="a") as writer:
            writer.write(asdict(record))

    logger.info("Downloaded %d new recordings", len(records))
    return records


def segment_and_embed_piano(
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment YouTube piano audio and extract clean MuQ embeddings.

    Reads recordings from cache_dir/audio/*.wav and cache_dir/recordings.jsonl.
    Writes:
    - cache_dir/metadata.jsonl with per-segment metadata
    - cache_dir/muq_embeddings/{segment_id}.pt per segment

    Returns count of newly processed segments.
    """
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio, segment_audio

    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"
    metadata_path = cache_dir / "metadata.jsonl"
    recordings_path = cache_dir / "recordings.jsonl"

    if not recordings_path.exists():
        logger.warning("No recordings metadata at %s", recordings_path)
        return 0

    with jsonlines.open(recordings_path) as reader:
        recordings = list(reader)

    existing_segments: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    emb_dir.mkdir(parents=True, exist_ok=True)
    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for i, recording in enumerate(recordings):
        video_id = recording["video_id"]
        wav_path = audio_dir / f"{video_id}.wav"

        if not wav_path.exists():
            continue

        base_id = f"yt_{video_id}"
        if any(sid.startswith(base_id) for sid in existing_segments):
            continue

        logger.info("[%d/%d] Segmenting %s", i + 1, len(recordings), video_id)

        audio, sr = load_audio(wav_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for j, seg in enumerate(segments):
            segment_id = f"{base_id}_seg{j:03d}"

            if segment_id in existing_segments:
                continue

            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            seg_record = {
                "segment_id": segment_id,
                "video_id": video_id,
                "title": recording.get("title", ""),
                "channel": recording.get("channel", ""),
                "segment_start": seg["start_sec"],
                "segment_end": seg["end_sec"],
            }

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(seg_record)

            existing_segments.add(segment_id)
            new_count += 1

    del extractor
    logger.info("Processed %d new YouTube piano segments", new_count)
    return new_count
