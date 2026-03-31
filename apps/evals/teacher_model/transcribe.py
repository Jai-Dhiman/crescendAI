"""
YouTube audio transcription pipeline for the CPT teacher model corpus.

Downloads YouTube audio, transcribes with Cohere Transcribe, scores relevance,
and saves to corpus/ with provenance tracking.

Usage:
    uv run python -m teacher_model.transcribe --url "https://youtube.com/watch?v=XXX"
    uv run python -m teacher_model.transcribe --playlist "https://youtube.com/playlist?list=XXX"
    uv run python -m teacher_model.transcribe --channel "@tonebasePiano" --limit 50
    uv run python -m teacher_model.transcribe --no-filter
    uv run python -m teacher_model.transcribe --tier tier1_youtube
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cohere
import tiktoken

from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
RELEVANCE_THRESHOLD = 0.4  # minimum score to save a transcript


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def download_audio(url: str, output_dir: Path) -> tuple[Path, dict]:
    """
    Download YouTube audio as WAV 16kHz mono via yt-dlp.

    Returns (audio_path, metadata_dict).
    Raises subprocess.CalledProcessError on yt-dlp failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metadata first (--print-json + --skip-download)
    info_cmd = [
        "yt-dlp",
        "--skip-download",
        "--print-json",
        "--no-playlist",
        url,
    ]
    info_result = subprocess.run(
        info_cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(info_result.stdout)
    video_id = metadata["id"]

    # Download audio, convert to WAV 16kHz mono
    audio_path = output_dir / f"{video_id}.wav"
    dl_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", str(audio_path.with_suffix("")),  # yt-dlp appends .wav
        url,
    ]
    subprocess.run(dl_cmd, capture_output=True, text=True, check=True)

    # yt-dlp may produce {id}.wav directly or {id}.wav after conversion
    if not audio_path.exists():
        # Search for any wav produced in output_dir with the video_id prefix
        candidates = list(output_dir.glob(f"{video_id}*.wav"))
        if not candidates:
            raise FileNotFoundError(
                f"yt-dlp did not produce a WAV file for video {video_id} in {output_dir}"
            )
        audio_path = candidates[0]

    return audio_path, metadata


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe an audio file using Cohere Transcribe.

    Returns the transcribed text string.
    Raises cohere.CohereError on API failure.
    """
    co = cohere.ClientV2()
    with open(audio_path, "rb") as f:
        response = co.audio.transcriptions.create(
            model="cohere-transcribe-03-2026",
            language="en",
            file=f,
        )
    return response.text


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def process_video(
    url: str,
    manifest: ProvenanceManifest,
    classifier: Optional[PedagogyRelevanceClassifier] = None,
    source_tier: str = "tier1_youtube",
) -> dict:
    """
    Full pipeline: download -> transcribe -> score relevance -> save -> record provenance.

    Args:
        url: YouTube video URL.
        manifest: ProvenanceManifest instance to record provenance.
        classifier: Optional relevance classifier. If None, all transcripts are saved.
        source_tier: Provenance tier label.

    Returns:
        dict with keys: video_id, title, word_count, token_count, relevance_score, saved.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        logger.info("Downloading audio: %s", url)
        audio_path, metadata = download_audio(url, tmp_path)

        video_id = metadata["id"]
        title = metadata.get("title", "")
        channel = metadata.get("uploader", metadata.get("channel", "unknown"))
        license_claimed = metadata.get("license", "unknown") or "unknown"
        duration = metadata.get("duration", 0)

        logger.info("Transcribing %s (%ss): %s", video_id, duration, title)
        transcript = transcribe_audio(audio_path)

    word_count = len(transcript.split())
    token_count = count_tokens(transcript)

    relevance_score: Optional[float] = None
    if classifier is not None:
        relevance_score = classifier.score(transcript[:4000])  # score first 4K chars

    saved = False
    if classifier is None or (relevance_score is not None and relevance_score >= RELEVANCE_THRESHOLD):
        out_path = CORPUS_DIR / f"{video_id}.txt"
        out_path.write_text(transcript, encoding="utf-8")
        saved = True
        logger.info(
            "Saved %s (%d words, %d tokens, relevance=%.3f)",
            video_id,
            word_count,
            token_count,
            relevance_score if relevance_score is not None else -1.0,
        )

        record = ProvenanceRecord(
            url=url,
            title=title,
            channel_or_publisher=channel,
            download_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            license_claimed=license_claimed,
            word_count=word_count,
            inclusion_threshold_score=relevance_score,
            source_tier=source_tier,
        )
        manifest.add(record)
    else:
        logger.info(
            "Skipped %s (relevance=%.3f < %.3f threshold)",
            video_id,
            relevance_score,
            RELEVANCE_THRESHOLD,
        )

    return {
        "video_id": video_id,
        "title": title,
        "word_count": word_count,
        "token_count": token_count,
        "relevance_score": relevance_score,
        "saved": saved,
    }


def get_playlist_urls(playlist_url: str, limit: Optional[int] = None) -> list[str]:
    """
    Extract video URLs from a YouTube playlist or channel using yt-dlp --flat-playlist.

    Args:
        playlist_url: YouTube playlist or channel URL.
        limit: Maximum number of URLs to return. None = all.

    Returns:
        List of full YouTube video URLs.

    Raises:
        subprocess.CalledProcessError on yt-dlp failure.
        ValueError if no URLs are found.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s",
        "--no-warnings",
    ]
    if limit is not None:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(playlist_url)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    video_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not video_ids:
        raise ValueError(f"No videos found at {playlist_url}")

    urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    logger.info("Found %d videos in playlist/channel", len(urls))
    return urls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and transcribe YouTube videos into the teacher model corpus."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Single YouTube video URL")
    source.add_argument("--playlist", help="YouTube playlist URL")
    source.add_argument("--channel", help="YouTube channel handle or URL (e.g. @tonebasePiano)")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process (playlist/channel only)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip relevance filtering -- save all transcripts",
    )
    parser.add_argument(
        "--tier",
        default="tier1_youtube",
        choices=["tier1_youtube", "tier2_literature", "tier3_musicology", "tier4_own"],
        help="Source tier label for provenance (default: tier1_youtube)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to provenance JSONL manifest (default: teacher_model/data/provenance.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = ProvenanceManifest(path=args.manifest)

    classifier: Optional[PedagogyRelevanceClassifier] = None
    if not args.no_filter:
        logger.info("Loading relevance classifier...")
        classifier = PedagogyRelevanceClassifier()
        logger.info("Classifier loaded (threshold=%.3f)", classifier._threshold)  # type: ignore[union-attr]

    # Collect URLs
    if args.url:
        urls = [args.url]
    elif args.playlist:
        urls = get_playlist_urls(args.playlist, limit=args.limit)
    else:
        # Channel
        channel_url = args.channel
        if not channel_url.startswith("http"):
            channel_url = f"https://www.youtube.com/{args.channel}/videos"
        urls = get_playlist_urls(channel_url, limit=args.limit)

    logger.info("Processing %d video(s)...", len(urls))

    saved_count = 0
    skipped_count = 0
    failed: list[tuple[str, str]] = []

    for i, url in enumerate(urls, 1):
        logger.info("[%d/%d] %s", i, len(urls), url)
        try:
            result = process_video(
                url=url,
                manifest=manifest,
                classifier=classifier,
                source_tier=args.tier,
            )
            if result["saved"]:
                saved_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            logger.error("Failed to process %s: %s", url, exc)
            failed.append((url, str(exc)))

    print(f"\nDone. saved={saved_count} skipped={skipped_count} failed={len(failed)}")
    if failed:
        print("Failed URLs:")
        for url, err in failed:
            print(f"  {url}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
