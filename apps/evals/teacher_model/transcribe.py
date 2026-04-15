"""
YouTube audio transcription pipeline for the CPT teacher model corpus.

Downloads YouTube audio, transcribes with a configurable provider, scores
relevance, and saves to corpus/ with provenance tracking.

Providers:
  assemblyai   - AssemblyAI Universal-3 Pro with speaker diarization (pilot)
  workers-ai   - Cloudflare Workers AI Whisper-L-V3, cheapest bulk option

Usage:
    uv run python -m teacher_model.transcribe --url "https://youtube.com/watch?v=XXX"
    uv run python -m teacher_model.transcribe --playlist "https://youtube.com/playlist?list=XXX"
    uv run python -m teacher_model.transcribe --channel "@tonebasePiano" --limit 50
    uv run python -m teacher_model.transcribe --provider workers-ai --no-filter
    uv run python -m teacher_model.transcribe --tier tier1_youtube
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tiktoken

# Load apps/evals/.env into os.environ if present (same pattern as r2_sync.py)
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
RELEVANCE_THRESHOLD = 0.4  # minimum score to save a transcript


# ---------------------------------------------------------------------------
# Transcription providers
# ---------------------------------------------------------------------------

# Each provider returns (full_text, utterances).
# utterances is a list of {"speaker": str, "text": str, "start": int, "end": int}.
# For providers without diarization, utterances is [].


def _transcribe_assemblyai(audio_path: Path) -> tuple[str, list[dict]]:
    """
    Transcribe with AssemblyAI Universal-3 Pro, speaker diarization enabled.

    Requires ASSEMBLYAI_API_KEY in environment.
    Raises RuntimeError on transcription failure.
    """
    import assemblyai as aai

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in environment")

    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    utterances = [
        {
            "speaker": u.speaker,
            "text": u.text,
            "start": u.start,
            "end": u.end,
        }
        for u in (transcript.utterances or [])
    ]
    return transcript.text or "", utterances


def _transcribe_workers_ai(audio_path: Path) -> tuple[str, list[dict]]:
    """
    Transcribe with Cloudflare Workers AI Whisper-Large-V3.

    Requires CF_ACCOUNT_ID and CF_API_TOKEN in environment.
    Raises httpx.HTTPStatusError on API failure.
    """
    import httpx

    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id or not api_token:
        raise RuntimeError("CF_ACCOUNT_ID and CF_API_TOKEN must be set for workers-ai provider")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        "/ai/run/@cf/openai/whisper-large-v3"
    )
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        content=audio_bytes,
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    text = result.get("result", {}).get("text", "")
    return text, []


def transcribe_audio(audio_path: Path, provider: str = "assemblyai") -> tuple[str, list[dict]]:
    """
    Transcribe an audio file using the specified provider.

    Args:
        audio_path: Path to the WAV file (16kHz mono).
        provider: "assemblyai" or "workers-ai".

    Returns:
        (full_text, utterances) where utterances are speaker-labeled segments.
        utterances is empty for providers without diarization.
    """
    if provider == "assemblyai":
        return _transcribe_assemblyai(audio_path)
    elif provider == "workers-ai":
        return _transcribe_workers_ai(audio_path)
    else:
        raise ValueError(f"Unknown transcription provider: {provider!r}. Choose assemblyai or workers-ai.")


# ---------------------------------------------------------------------------
# Diarization utilities
# ---------------------------------------------------------------------------


def extract_dominant_speaker_text(utterances: list[dict]) -> tuple[str, str]:
    """
    Identify the dominant speaker by word count and return their concatenated text.

    The dominant speaker is assumed to be the teacher (most words in a masterclass).

    Args:
        utterances: List of {"speaker": str, "text": str, "start": int, "end": int}.

    Returns:
        (dominant_speaker_label, teacher_text) where teacher_text is the
        concatenation of all utterances from that speaker.
    """
    if not utterances:
        return "", ""

    word_counts: Counter = Counter()
    for u in utterances:
        word_counts[u["speaker"]] += len(u["text"].split())

    dominant_speaker = word_counts.most_common(1)[0][0]
    teacher_text = " ".join(
        u["text"] for u in utterances if u["speaker"] == dominant_speaker
    )

    total_words = sum(word_counts.values())
    teacher_words = word_counts[dominant_speaker]
    logger.info(
        "Speaker distribution: %s. Dominant=%s (%.0f%% of words)",
        dict(word_counts),
        dominant_speaker,
        100 * teacher_words / total_words if total_words > 0 else 0,
    )

    return dominant_speaker, teacher_text


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------


def download_audio(
    url: str,
    output_dir: Path,
    cookies_from_browser: Optional[str] = None,
) -> tuple[Path, dict]:
    """
    Download YouTube audio as WAV 16kHz mono via yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to write the WAV file.
        cookies_from_browser: Browser name to pull cookies from (e.g. "chrome",
            "safari", "firefox"). Pass when YouTube requires authentication.

    Returns (audio_path, metadata_dict).
    Raises subprocess.CalledProcessError on yt-dlp failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cookie_args = ["--cookies-from-browser", cookies_from_browser] if cookies_from_browser else []

    info_cmd = [
        "yt-dlp",
        "--skip-download",
        "--print-json",
        "--no-playlist",
        *cookie_args,
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

    audio_path = output_dir / f"{video_id}.wav"
    dl_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--sleep-interval", "5",
        "--max-sleep-interval", "15",
        "--output", str(audio_path.with_suffix("")),
        *cookie_args,
        url,
    ]
    subprocess.run(dl_cmd, capture_output=True, text=True, check=True)

    if not audio_path.exists():
        candidates = list(output_dir.glob(f"{video_id}*.wav"))
        if not candidates:
            raise FileNotFoundError(
                f"yt-dlp did not produce a WAV file for video {video_id} in {output_dir}"
            )
        audio_path = candidates[0]

    return audio_path, metadata


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def process_video(
    url: str,
    manifest: ProvenanceManifest,
    classifier: Optional[PedagogyRelevanceClassifier] = None,
    source_tier: str = "tier1_youtube",
    provider: str = "assemblyai",
    cookies_from_browser: Optional[str] = None,
) -> dict:
    """
    Full pipeline: download -> transcribe -> (diarize) -> score relevance -> save -> provenance.

    When using assemblyai, the dominant speaker's text is extracted before relevance
    scoring. The saved corpus file contains only teacher speech, not the full transcript.

    Args:
        url: YouTube video URL.
        manifest: ProvenanceManifest to record provenance.
        classifier: Relevance classifier. If None, all transcripts are saved.
        source_tier: Provenance tier label.
        provider: Transcription provider ("assemblyai" or "workers-ai").

    Returns:
        dict with keys: video_id, title, word_count, token_count, relevance_score,
        saved, dominant_speaker, teacher_word_share.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        logger.info("Downloading audio: %s", url)
        audio_path, metadata = download_audio(url, tmp_path, cookies_from_browser=cookies_from_browser)

        video_id = metadata["id"]
        title = metadata.get("title", "")
        channel = metadata.get("uploader", metadata.get("channel", "unknown"))
        license_claimed = metadata.get("license", "unknown") or "unknown"
        duration = metadata.get("duration", 0)

        logger.info("Transcribing %s (%ss) via %s: %s", video_id, duration, provider, title)
        full_text, utterances = transcribe_audio(audio_path, provider=provider)

    # If diarization returned utterances, extract dominant (teacher) speaker text.
    # Otherwise fall back to the full transcript.
    dominant_speaker = ""
    teacher_word_share = 1.0

    if utterances:
        dominant_speaker, corpus_text = extract_dominant_speaker_text(utterances)
        if corpus_text:
            total_words = sum(len(u["text"].split()) for u in utterances)
            teacher_words = len(corpus_text.split())
            teacher_word_share = teacher_words / total_words if total_words > 0 else 1.0
        else:
            corpus_text = full_text
    else:
        corpus_text = full_text

    word_count = len(corpus_text.split())
    token_count = count_tokens(corpus_text)

    relevance_score: Optional[float] = None
    if classifier is not None:
        relevance_score = classifier.score(corpus_text[:4000])

    saved = False
    if classifier is None or (relevance_score is not None and relevance_score >= RELEVANCE_THRESHOLD):
        out_path = CORPUS_DIR / f"{video_id}.txt"
        out_path.write_text(corpus_text, encoding="utf-8")
        saved = True
        rel = round(relevance_score, 3) if relevance_score is not None else -1.0
        share_pct = round(teacher_word_share * 100)
        logger.info(
            "Saved %s: words=%d tok_count=%d relevance=%s dominant=%s share=%s%%",
            video_id,
            word_count,
            token_count,
            rel,
            dominant_speaker or "n/a",
            share_pct,
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
        "dominant_speaker": dominant_speaker,
        "teacher_word_share": teacher_word_share,
    }


def get_playlist_urls(
    playlist_url: str,
    limit: Optional[int] = None,
    cookies_from_browser: Optional[str] = None,
) -> list[str]:
    """
    Extract video URLs from a YouTube playlist or channel using yt-dlp --flat-playlist.

    Args:
        playlist_url: YouTube playlist or channel URL.
        limit: Maximum number of URLs to return. None = all.
        cookies_from_browser: Browser name for cookie auth (e.g. "chrome", "safari").

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
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
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
        "--provider",
        default="assemblyai",
        choices=["assemblyai", "workers-ai"],
        help="Transcription provider (default: assemblyai)",
    )
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
        "--cookies-from-browser",
        dest="cookies_from_browser",
        default=None,
        metavar="BROWSER",
        help="Pass cookies from BROWSER to yt-dlp (e.g. chrome, safari, firefox). "
             "Required when YouTube returns bot-detection errors.",
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

    if args.url:
        urls = [args.url]
    elif args.playlist:
        urls = get_playlist_urls(
            args.playlist,
            limit=args.limit,
            cookies_from_browser=args.cookies_from_browser,
        )
    else:
        channel_url = args.channel
        if not channel_url.startswith("http"):
            channel_url = f"https://www.youtube.com/{args.channel}/videos"
        urls = get_playlist_urls(
            channel_url,
            limit=args.limit,
            cookies_from_browser=args.cookies_from_browser,
        )

    logger.info("Processing %d video(s) via %s...", len(urls), args.provider)

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
                provider=args.provider,
                cookies_from_browser=args.cookies_from_browser,
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
