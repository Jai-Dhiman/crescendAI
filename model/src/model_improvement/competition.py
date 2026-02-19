"""Competition dataset pipeline: scraping, downloading, segmenting, embedding.

Targets the XVIII International Chopin Piano Competition (2021).
Scrapes results from Wikipedia, discovers YouTube URLs from Chopin Institute
playlists, downloads audio via yt-dlp, and extracts MuQ embeddings.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonlines
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------

@dataclass
class CompetitionRecord:
    recording_id: str       # "chopin2021_prelim_performer123_ballade1"
    competition: str        # "chopin"
    edition: int            # 2021
    round: str              # "preliminary" | "stage1" | "stage2" | "stage3" | "final"
    placement: int          # ordinal: 1=winner, 2=second, ..., N=eliminated-in-round
    performer: str          # "Bruce Liu"
    piece: str              # "Ballade No. 1 in G minor, Op. 23"
    audio_path: str         # relative path under competition_cache/
    duration_seconds: float
    source_url: str         # YouTube URL
    country: str            # "Canada"


# Placement encoding: ordinal gaps proportional to round distance
ROUND_ELIMINATION_PLACEMENT = {
    "final": 7,         # finalists who didn't place
    "stage3": 20,       # eliminated after stage 3
    "stage2": 40,       # eliminated after stage 2
    "stage1": 80,       # eliminated after stage 1
    "preliminary": 160, # eliminated in preliminary
}

ROUND_ORDER = ["preliminary", "stage1", "stage2", "stage3", "final"]

WIKIPEDIA_URL = (
    "https://en.wikipedia.org/wiki/"
    "XVIII_International_Chopin_Piano_Competition"
)


# ---------------------------------------------------------------------------
# 2b: Results scraper
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Normalize performer name for matching."""
    return re.sub(r"[^a-z]", "", name.lower())


def scrape_chopin_results(cache_dir: Path) -> dict:
    """Scrape XVIII Chopin Competition results from Wikipedia.

    Returns dict: {performer_name: {rounds: [...], placement: int, country: str}}
    Also saves to cache_dir/results.json.
    """
    results_path = cache_dir / "results.json"
    if results_path.exists():
        logger.info("Results already cached at %s", results_path)
        with open(results_path) as f:
            return json.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching competition results from Wikipedia...")
    resp = requests.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": "CrescendAI/1.0 (research project; piano competition data)"},
        timeout=30,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    results: dict[str, dict] = {}

    # --- Parse prize winners from infobox / results tables ---
    # Wikipedia structures this with tables listing prizes and round participants.
    # We look for tables with prize info and participant lists.

    # Prize winners: look for cells mentioning "First prize", "Second prize", etc.
    prize_placements = {
        "first prize": 1,
        "1st prize": 1,
        "second prize": 2,
        "2nd prize": 2,
        "third prize": 3,
        "3rd prize": 3,
        "fourth prize": 4,
        "4th prize": 4,
        "fifth prize": 5,
        "5th prize": 5,
        "sixth prize": 6,
        "6th prize": 6,
    }

    # Known 2021 winners (hardcoded as fallback since Wikipedia table structure
    # can vary and this is a one-time pipeline for a specific competition)
    known_results = {
        "Bruce Liu": {"placement": 1, "country": "Canada",
                       "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Alexander Gadjiev": {"placement": 2, "country": "Italy/Slovenia",
                               "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Martin Garcia Garcia": {"placement": 3, "country": "Spain",
                                  "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Aimi Kobayashi": {"placement": 4, "country": "Japan",
                            "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Jakub Kuszlik": {"placement": 5, "country": "Poland",
                           "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Leonora Armellini": {"placement": 6, "country": "Italy",
                               "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        # Finalists who didn't place (placement = 7)
        "J J Jun Li Bui": {"placement": 7, "country": "Canada",
                            "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Hao Rao": {"placement": 7, "country": "China",
                     "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Kamil Pacholec": {"placement": 7, "country": "Poland",
                            "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Eva Gevorgyan": {"placement": 7, "country": "Russia/Armenia",
                           "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Hyuk Lee": {"placement": 7, "country": "South Korea",
                      "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
        "Kyohei Sorita": {"placement": 7, "country": "Japan",
                           "rounds": ["preliminary", "stage1", "stage2", "stage3", "final"]},
    }

    # Try to parse tables from Wikipedia to augment/override known results
    tables = soup.find_all("table", class_="wikitable")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            text = " ".join(c.get_text(strip=True) for c in cells).lower()

            # Check for prize mentions
            for prize_text, placement in prize_placements.items():
                if prize_text in text:
                    # Try to find performer name in this row
                    for cell in cells:
                        links = cell.find_all("a")
                        for link in links:
                            name = link.get_text(strip=True)
                            # Skip non-name links
                            if len(name) > 3 and not any(
                                kw in name.lower()
                                for kw in ["prize", "chopin", "competition", "poland"]
                            ):
                                if name not in results:
                                    results[name] = {
                                        "placement": placement,
                                        "country": "",
                                        "rounds": ROUND_ORDER[:],
                                    }

    # Parse round-specific participant lists
    # Wikipedia has sections like "Stage I", "Stage II", etc. with participant tables
    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = heading.get_text(strip=True).lower()
        round_name = None
        if "preliminary" in heading_text:
            round_name = "preliminary"
        elif "stage i " in heading_text or "stage 1" in heading_text or heading_text.endswith("stage i"):
            round_name = "stage1"
        elif "stage ii " in heading_text or "stage 2" in heading_text or heading_text.endswith("stage ii"):
            round_name = "stage2"
        elif "stage iii" in heading_text or "stage 3" in heading_text:
            round_name = "stage3"
        elif "final" in heading_text:
            round_name = "final"

        if round_name is None:
            continue

        # Find the next table after this heading
        sibling = heading.find_next_sibling()
        while sibling and sibling.name != "table":
            sibling = sibling.find_next_sibling()
            if sibling and sibling.name in ["h2", "h3"]:
                break

        if sibling and sibling.name == "table":
            for row in sibling.find_all("tr"):
                cells = row.find_all(["td", "th"])
                for cell in cells:
                    links = cell.find_all("a")
                    for link in links:
                        name = link.get_text(strip=True)
                        if len(name) > 3 and not any(
                            kw in name.lower()
                            for kw in ["chopin", "competition", "poland", "warsaw"]
                        ):
                            if name in results and round_name not in results[name]["rounds"]:
                                results[name]["rounds"].append(round_name)

    # Merge known results with scraped results (known takes precedence)
    for name, info in known_results.items():
        if name not in results:
            results[name] = info
        else:
            # Keep scraped placement if already set, otherwise use known
            if results[name]["placement"] == 0:
                results[name]["placement"] = info["placement"]
            if not results[name]["country"]:
                results[name]["country"] = info["country"]
            # Merge rounds
            for r in info["rounds"]:
                if r not in results[name]["rounds"]:
                    results[name]["rounds"].append(r)

    # Assign elimination placements to participants without explicit placement
    for name, info in results.items():
        if info.get("placement", 0) == 0:
            # Determine highest round reached
            highest_round = "preliminary"
            for r in ROUND_ORDER:
                if r in info.get("rounds", []):
                    highest_round = r
            info["placement"] = ROUND_ELIMINATION_PLACEMENT[highest_round]

    logger.info("Scraped results for %d performers", len(results))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# 2c: YouTube discovery
# ---------------------------------------------------------------------------

def discover_youtube_urls(cache_dir: Path, results: dict) -> dict:
    """Discover YouTube URLs for competition performances.

    Uses yt-dlp --flat-playlist to enumerate the Chopin Institute's playlists,
    then matches videos to performers by title parsing.

    Returns nested dict: {performer: {round: [{piece, url, start_sec, end_sec}]}}
    """
    urls_path = cache_dir / "youtube_urls.json"
    if urls_path.exists():
        logger.info("YouTube URLs already cached at %s", urls_path)
        with open(urls_path) as f:
            return json.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Chopin Institute YouTube channel playlists for 2021 competition
    # These are the official playlists with individual performer clips.
    # Note: preliminary and stage1 only have full-session videos (multi-performer,
    # ~5h each) so we skip them. Stages 2, 3, and final have individual clips.
    playlist_urls = {
        "stage2": "https://www.youtube.com/playlist?list=PLTmn2qD3aSQtUl-oPRcgm3kGiGjWkLJzN",
        "stage3": "https://www.youtube.com/playlist?list=PLTmn2qD3aSQtn2fE4OC_LTx6podD7JYXU",
        "final": "https://www.youtube.com/playlist?list=PLTmn2qD3aSQs6WnpsMXwf2Qb9sqPyZBSv",
    }

    # Build performer slug lookup for fuzzy matching
    slug_to_name = {}
    for name in results:
        slug_to_name[_slugify(name)] = name
        # Also index by last name only
        parts = name.split()
        if len(parts) >= 2:
            slug_to_name[_slugify(parts[-1])] = name

    url_mapping: dict[str, dict[str, list]] = {name: {} for name in results}

    for round_name, playlist_url in playlist_urls.items():
        logger.info("Enumerating playlist for %s...", round_name)

        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--flat-playlist",
                    "--dump-json",
                    playlist_url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("yt-dlp failed for %s: %s", round_name, e)
            continue

        if result.returncode != 0:
            logger.warning(
                "yt-dlp returned %d for %s: %s",
                result.returncode, round_name, result.stderr[:200],
            )
            continue

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                video = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = video.get("title", "")
            url = video.get("url") or video.get("webpage_url", "")
            if not url:
                video_id = video.get("id", "")
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"

            if not url:
                continue

            # Match performer name from video title
            matched_name = _match_performer_from_title(title, slug_to_name)
            if matched_name is None:
                continue

            # Extract piece info from title (after the performer name, before round info)
            piece = _extract_piece_from_title(title, matched_name)

            if round_name not in url_mapping[matched_name]:
                url_mapping[matched_name][round_name] = []

            url_mapping[matched_name][round_name].append({
                "piece": piece,
                "url": url,
                "start_sec": None,
                "end_sec": None,
            })

    # Filter out performers with no URLs found
    url_mapping = {k: v for k, v in url_mapping.items() if v}

    logger.info(
        "Found URLs for %d performers across %d rounds",
        len(url_mapping),
        len(playlist_urls),
    )

    with open(urls_path, "w") as f:
        json.dump(url_mapping, f, indent=2)

    return url_mapping


def _match_performer_from_title(title: str, slug_to_name: dict) -> str | None:
    """Try to match a performer name from a YouTube video title."""
    title_slug = _slugify(title)
    # Try matching each known performer slug in the title
    best_match = None
    best_len = 0
    for slug, name in slug_to_name.items():
        if slug in title_slug and len(slug) > best_len:
            best_match = name
            best_len = len(slug)
    return best_match


def _extract_piece_from_title(title: str, performer_name: str) -> str:
    """Extract piece name from video title, stripping performer and round info."""
    # Common patterns: "Performer Name – Piece Title (Round X)"
    # or "Performer Name - Piece Title | XVIII Chopin Competition"
    piece = title
    # Remove performer name (case-insensitive)
    for sep in [" – ", " - ", " | ", ": "]:
        parts = piece.split(sep, 1)
        if len(parts) == 2:
            # Check if performer name is in the first part
            if _slugify(performer_name) in _slugify(parts[0]):
                piece = parts[1]
                break

    # Remove trailing round/competition info
    for pattern in [
        r"\s*\|\s*.*[Cc]hopin.*$",
        r"\s*\([^)]*[Rr]ound[^)]*\)$",
        r"\s*\([^)]*[Ss]tage[^)]*\)$",
        r"\s*\([^)]*[Pp]reliminary[^)]*\)$",
        r"\s*\([^)]*[Ff]inal[^)]*\)$",
        r"\s*XVIII.*$",
    ]:
        piece = re.sub(pattern, "", piece).strip()

    return piece if piece else "unknown"


# ---------------------------------------------------------------------------
# 2d: Audio downloader
# ---------------------------------------------------------------------------

def _make_recording_id(performer: str, round_name: str, piece: str) -> str:
    """Generate a stable recording ID from performer, round, and piece."""
    slug = _slugify(performer)
    piece_slug = re.sub(r"[^a-z0-9]", "", piece.lower())[:30]
    return f"chopin2021_{round_name}_{slug}_{piece_slug}"


def download_competition_audio(
    url_mapping: dict,
    results: dict,
    output_dir: Path,
    metadata_path: Path,
) -> list[CompetitionRecord]:
    """Download competition audio using yt-dlp.

    Downloads audio-only WAV at 24kHz mono. Resumable: skips files that
    already exist on disk.

    Returns list of CompetitionRecord for all downloaded recordings.
    """
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata for resumability
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for record in reader:
                existing_ids.add(record["recording_id"])

    records: list[CompetitionRecord] = []

    for performer, rounds in url_mapping.items():
        performer_info = results.get(performer, {})
        placement = performer_info.get("placement", 160)
        country = performer_info.get("country", "")

        for round_name, videos in rounds.items():
            for video in videos:
                piece = video.get("piece", "unknown")
                url = video.get("url", "")
                start_sec = video.get("start_sec")
                end_sec = video.get("end_sec")

                recording_id = _make_recording_id(performer, round_name, piece)
                wav_path = audio_dir / f"{recording_id}.wav"

                # Skip if already downloaded and in metadata
                if recording_id in existing_ids and wav_path.exists() and wav_path.stat().st_size > 0:
                    logger.debug("Skipping %s (already exists)", recording_id)
                    continue

                if not url:
                    logger.warning("No URL for %s", recording_id)
                    continue

                logger.info("Downloading %s ...", recording_id)

                try:
                    _download_audio(url, wav_path, start_sec, end_sec)
                except Exception as e:
                    logger.error("Failed to download %s: %s", recording_id, e)
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    logger.error("Download produced empty file: %s", recording_id)
                    continue

                # Get duration from the downloaded file
                duration = _get_wav_duration(wav_path)

                record = CompetitionRecord(
                    recording_id=recording_id,
                    competition="chopin",
                    edition=2021,
                    round=round_name,
                    placement=placement,
                    performer=performer,
                    piece=piece,
                    audio_path=f"audio/{recording_id}.wav",
                    duration_seconds=duration,
                    source_url=url,
                    country=country,
                )
                records.append(record)

                # Append to metadata file immediately (resumability)
                with jsonlines.open(metadata_path, mode="a") as writer:
                    writer.write(asdict(record))

    logger.info("Downloaded %d new recordings", len(records))
    return records


def _download_audio(
    url: str,
    output_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> None:
    """Download audio from YouTube using yt-dlp, output as 24kHz mono WAV."""
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
        "--output", str(output_path.with_suffix(".%(ext)s")),
        "--no-playlist",
        "--quiet",
    ]

    if start_sec is not None and end_sec is not None:
        cmd.extend(["--download-sections", f"*{start_sec}-{end_sec}"])
    elif start_sec is not None:
        cmd.extend(["--download-sections", f"*{start_sec}-inf"])

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {result.stderr[:500]}"
        )


def _get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 2e: MuQ embedding extraction
# ---------------------------------------------------------------------------

def extract_competition_embeddings(cache_dir: Path) -> int:
    """Extract MuQ embeddings for all competition audio files.

    Reuses extract_muq_embeddings() from the audio_experiments module.

    Returns count of newly extracted embeddings.
    """
    from audio_experiments.extractors.muq import extract_muq_embeddings

    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"

    if not audio_dir.exists():
        logger.warning("No audio directory found at %s", audio_dir)
        return 0

    keys = [p.stem for p in sorted(audio_dir.glob("*.wav"))]
    if not keys:
        logger.warning("No WAV files found in %s", audio_dir)
        return 0

    logger.info("Extracting MuQ embeddings for %d audio files...", len(keys))
    return extract_muq_embeddings(audio_dir, emb_dir, keys)


# ---------------------------------------------------------------------------
# 2f: Segment-level embeddings
# ---------------------------------------------------------------------------

def segment_and_embed_competition(
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment competition recordings into 30s clips and extract per-segment MuQ embeddings.

    Reads full recordings from cache_dir/audio/*.wav and recording-level metadata
    from cache_dir/recordings.jsonl. Produces:
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

    if not audio_dir.exists():
        logger.warning("No audio directory at %s", audio_dir)
        return 0

    # Load recording-level metadata
    if not recordings_path.exists():
        logger.warning("No recordings metadata at %s", recordings_path)
        return 0

    with jsonlines.open(recordings_path) as reader:
        recordings = list(reader)

    if not recordings:
        logger.warning("No recordings found in %s", recordings_path)
        return 0

    # Load already-processed segment IDs for idempotency
    existing_segments: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    emb_dir.mkdir(parents=True, exist_ok=True)

    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for recording in recordings:
        recording_id = recording["recording_id"]
        wav_path = audio_dir / f"{recording_id}.wav"

        if not wav_path.exists():
            logger.warning("Audio file not found: %s", wav_path)
            continue

        # Skip if any segments for this recording already exist
        if any(sid.startswith(recording_id) for sid in existing_segments):
            logger.debug("Segments for %s already processed", recording_id)
            continue

        audio, sr = load_audio(wav_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for i, seg in enumerate(segments):
            segment_id = f"{recording_id}_seg{i:03d}"

            if segment_id in existing_segments:
                continue

            # Extract MuQ embedding for this segment
            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)

            # Save embedding
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            # Write segment metadata
            seg_record = {
                "segment_id": segment_id,
                "recording_id": recording_id,
                "competition": recording.get("competition", "chopin"),
                "edition": recording.get("edition", 2021),
                "round": recording["round"],
                "placement": recording["placement"],
                "performer": recording["performer"],
                "piece": recording["piece"],
                "segment_start": seg["start_sec"],
                "segment_end": seg["end_sec"],
                "source_url": recording.get("source_url", ""),
                "country": recording.get("country", ""),
            }

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(seg_record)

            existing_segments.add(segment_id)
            new_count += 1

    del extractor

    logger.info("Processed %d new segments", new_count)
    return new_count


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_competition_metadata(cache_dir: Path) -> list[dict]:
    """Load all competition records from metadata.jsonl."""
    metadata_path = cache_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return []
    with jsonlines.open(metadata_path) as reader:
        return list(reader)


def load_competition_embeddings(cache_dir: Path) -> dict:
    """Load all MuQ embeddings for competition recordings.

    Returns dict mapping recording_id to [T, 1024] tensor.
    """
    import torch

    emb_dir = cache_dir / "muq_embeddings"
    if not emb_dir.exists():
        return {}
    return {
        p.stem: torch.load(p, map_location="cpu", weights_only=True)
        for p in sorted(emb_dir.glob("*.pt"))
    }
