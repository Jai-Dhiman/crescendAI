#!/usr/bin/env python3
"""
Ingest YouTube piano masterclass transcripts into CrescendAI RAG system.

This script:
1. Fetches transcripts from YouTube videos using youtube-transcript-api
2. Chunks by time segments (~30-60 seconds)
3. Extracts metadata (speaker, composers, techniques)
4. Generates text_with_context with header injection
5. Outputs JSON for ingestion via wrangler CLI

Usage:
    python ingest_youtube.py --output youtube_chunks.json
    python ingest_youtube.py --video-id dQw4w9WgXcQ --output chunks.json
    python ingest_youtube.py --config masterclasses.json --output chunks.json

Requirements:
    uv add youtube-transcript-api
"""

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    print("Error: youtube-transcript-api not installed")
    print("Install with: uv add youtube-transcript-api")
    exit(1)


SOURCE_TYPE = "masterclass"

# Sample piano masterclass videos (educational, public content)
DEFAULT_MASTERCLASSES = [
    {
        "video_id": "example-video-id-1",
        "title": "Chopin Ballade No. 1 Masterclass",
        "speaker": "Example Pianist",
        "composers": ["Chopin"],
    },
]

# Known composers
KNOWN_COMPOSERS = [
    "Bach", "Beethoven", "Brahms", "Chopin", "Czerny", "Debussy", "Grieg",
    "Handel", "Haydn", "Liszt", "Mendelssohn", "Mozart", "Rachmaninoff",
    "Scarlatti", "Schubert", "Schumann", "Scriabin", "Tchaikovsky", "Wagner",
    "Paderewski", "Prokofiev", "Ravel", "Satie", "Bartok", "Gershwin"
]

# Common piano techniques
KNOWN_TECHNIQUES = [
    "legato", "staccato", "touch", "tone", "pedal", "pedaling", "dynamics",
    "phrasing", "articulation", "fingering", "scales", "arpeggios", "octaves",
    "trills", "voicing", "tempo", "rubato", "expression", "interpretation",
    "technique", "practice", "memorization", "sight-reading", "ear training",
    "hand position", "wrist", "arm weight", "relaxation", "tension",
    "balance", "color", "singing tone", "cantabile"
]


@dataclass
class PedagogyChunk:
    """A chunk of pedagogical content with full citation metadata."""
    chunk_id: str
    text: str
    text_with_context: str
    source_type: str
    source_title: str
    source_author: str
    source_url: Optional[str]
    page_number: Optional[int]
    section_title: Optional[str]
    paragraph_index: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    timestamp_start: Optional[float]
    timestamp_end: Optional[float]
    speaker: Optional[str]
    composers: list[str]
    pieces: list[str]
    techniques: list[str]
    source_hash: str


@dataclass
class VectorMetadata:
    """Metadata for Vectorize index."""
    difficulty: Optional[str]
    topic: Optional[str]
    content_type: Optional[str]
    composer: Optional[str]
    source_type: Optional[str]


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return as-is if already an ID."""
    # Already a video ID (11 characters)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    # Full YouTube URL patterns
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript from YouTube video.

    Returns list of dicts with 'text', 'start', 'duration' keys.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created transcripts over auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])

        return transcript.fetch()

    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video {video_id}")
    except NoTranscriptFound:
        raise ValueError(f"No English transcript found for video {video_id}")
    except VideoUnavailable:
        raise ValueError(f"Video {video_id} is unavailable")


def extract_composers(text: str, hint_composers: list[str] = None) -> list[str]:
    """Extract mentioned composers from text."""
    found = set()
    text_lower = text.lower()

    # Add hint composers if they appear
    if hint_composers:
        for composer in hint_composers:
            if composer.lower() in text_lower:
                found.add(composer)

    # Check known composers
    for composer in KNOWN_COMPOSERS:
        if composer.lower() in text_lower:
            found.add(composer)

    return list(found)


def extract_techniques(text: str) -> list[str]:
    """Extract mentioned techniques from text."""
    found = []
    text_lower = text.lower()
    for technique in KNOWN_TECHNIQUES:
        if technique.lower() in text_lower:
            found.append(technique)
    return list(set(found))


def extract_pieces(text: str) -> list[str]:
    """Extract mentioned pieces from text."""
    pieces = []

    # Opus patterns
    opus_pattern = re.compile(r'(?:Op\.|Opus)\s*\d+(?:\s*No\.\s*\d+)?', re.IGNORECASE)
    pieces.extend(opus_pattern.findall(text))

    # Common piece types
    piece_pattern = re.compile(
        r'(?:Nocturne|Ballade|Scherzo|Polonaise|Mazurka|Waltz|Etude|Prelude|Sonata|Concerto|Rhapsody|Impromptu)'
        r'(?:\s+(?:in\s+)?[A-G](?:\s+(?:major|minor|flat|sharp))?)?'
        r'(?:\s*(?:No\.\s*)?\d+)?',
        re.IGNORECASE
    )
    pieces.extend(piece_pattern.findall(text))

    return list(set(pieces))


def chunk_transcript(
    transcript: list[dict],
    target_duration: float = 45.0,  # seconds
    min_duration: float = 20.0,
    max_duration: float = 90.0,
) -> list[tuple[str, float, float]]:
    """
    Chunk transcript by time segments.

    Returns list of (text, start_time, end_time) tuples.
    """
    if not transcript:
        return []

    chunks = []
    current_texts = []
    current_start = transcript[0]['start']
    current_duration = 0.0

    for segment in transcript:
        segment_text = segment['text'].strip()
        segment_start = segment['start']
        segment_duration = segment.get('duration', 2.0)

        # Check if adding this segment would exceed max duration
        potential_duration = (segment_start + segment_duration) - current_start

        # End current chunk if we've reached target duration and hit a pause
        should_split = (
            current_duration >= target_duration and
            (segment_text.endswith('.') or segment_text.endswith('?') or
             potential_duration > max_duration)
        )

        if should_split and current_texts:
            chunk_text = ' '.join(current_texts)
            chunk_end = current_start + current_duration
            chunks.append((chunk_text, current_start, chunk_end))

            # Start new chunk
            current_texts = []
            current_start = segment_start
            current_duration = 0.0

        current_texts.append(segment_text)
        current_duration = (segment_start + segment_duration) - current_start

    # Don't forget last chunk
    if current_texts:
        chunk_text = ' '.join(current_texts)
        chunk_end = current_start + current_duration
        if current_duration >= min_duration:
            chunks.append((chunk_text, current_start, chunk_end))
        elif chunks:
            # Merge with previous chunk if too short
            prev_text, prev_start, _ = chunks[-1]
            chunks[-1] = (prev_text + ' ' + chunk_text, prev_start, chunk_end)

    return chunks


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def create_text_with_context(
    text: str,
    title: str,
    speaker: str,
    timestamp_start: float,
    composers: list[str],
    techniques: list[str],
) -> str:
    """Create text with contextual header injection."""
    header_lines = [
        f"[Source: {title}]",
        f"[Speaker: {speaker}]",
        f"[Timestamp: {format_timestamp(timestamp_start)}]",
    ]

    context_parts = []
    if composers:
        context_parts.append(", ".join(composers[:3]))
    if techniques:
        context_parts.append(", ".join(techniques[:3]))

    if context_parts:
        header_lines.append(f"[Context: {' - '.join(context_parts)}]")

    header = "\n".join(header_lines)
    return f"{header}\n\n{text}"


def compute_source_hash(video_id: str, timestamp_start: float, text: str) -> str:
    """Compute unique hash for deduplication."""
    content = f"{video_id}:{timestamp_start:.1f}:{text[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def infer_topic(techniques: list[str]) -> Optional[str]:
    """Infer topic category from techniques."""
    technique_topics = {
        "technique/articulation": ["touch", "legato", "staccato", "articulation", "fingering"],
        "technique/pedaling": ["pedal", "pedaling"],
        "interpretation/phrasing": ["phrasing", "expression", "rubato", "interpretation", "cantabile"],
        "interpretation/dynamics": ["dynamics", "tone", "voicing", "balance", "color"],
        "practice/method": ["practice", "memorization", "scales", "arpeggios"],
    }

    for topic, keywords in technique_topics.items():
        if any(t.lower() in keywords for t in techniques):
            return topic

    return None


def infer_difficulty(title: str, text: str) -> str:
    """Infer difficulty level from content."""
    text_lower = (title + " " + text).lower()

    advanced_keywords = ["virtuoso", "advanced", "concert", "professional", "difficult"]
    beginner_keywords = ["beginner", "basic", "introduction", "first", "simple"]

    if any(kw in text_lower for kw in advanced_keywords):
        return "advanced"
    if any(kw in text_lower for kw in beginner_keywords):
        return "beginner"
    return "intermediate"


def process_video(
    video_id: str,
    title: str,
    speaker: str,
    hint_composers: list[str] = None,
) -> tuple[list[PedagogyChunk], list[VectorMetadata]]:
    """Process a single YouTube video into chunks."""
    print(f"  Fetching transcript for {video_id}...")

    try:
        transcript = fetch_transcript(video_id)
    except ValueError as e:
        print(f"  Error: {e}")
        return [], []

    print(f"  Got {len(transcript)} transcript segments")

    # Chunk the transcript
    chunked = chunk_transcript(transcript)
    print(f"  Created {len(chunked)} chunks")

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    chunks = []
    metadata_list = []

    for chunk_idx, (text, start_time, end_time) in enumerate(chunked):
        # Extract metadata
        composers = extract_composers(text, hint_composers)
        techniques = extract_techniques(text)
        pieces = extract_pieces(text)

        # Create context-enriched text
        text_with_context = create_text_with_context(
            text, title, speaker, start_time, composers, techniques
        )

        # Generate IDs
        chunk_id = str(uuid.uuid4())
        source_hash = compute_source_hash(video_id, start_time, text)

        chunk = PedagogyChunk(
            chunk_id=chunk_id,
            text=text,
            text_with_context=text_with_context,
            source_type=SOURCE_TYPE,
            source_title=title,
            source_author=speaker,
            source_url=youtube_url,
            page_number=None,
            section_title=f"Part {chunk_idx + 1}",
            paragraph_index=chunk_idx,
            char_start=None,
            char_end=None,
            timestamp_start=start_time,
            timestamp_end=end_time,
            speaker=speaker,
            composers=composers or (hint_composers or []),
            pieces=pieces,
            techniques=techniques,
            source_hash=source_hash
        )
        chunks.append(chunk)

        # Create vector metadata
        topic = infer_topic(techniques)
        difficulty = infer_difficulty(title, text)
        primary_composer = composers[0] if composers else (hint_composers[0] if hint_composers else None)

        metadata = VectorMetadata(
            difficulty=difficulty,
            topic=topic,
            content_type="explanation",
            composer=primary_composer,
            source_type=SOURCE_TYPE
        )
        metadata_list.append(metadata)

    return chunks, metadata_list


def generate_sql_inserts(chunks: list[PedagogyChunk]) -> str:
    """Generate SQL INSERT statements for wrangler d1 execute."""
    statements = []

    for chunk in chunks:
        composers_json = json.dumps(chunk.composers)
        pieces_json = json.dumps(chunk.pieces)
        techniques_json = json.dumps(chunk.techniques)

        # Escape single quotes
        text_escaped = chunk.text.replace("'", "''")
        text_with_context_escaped = chunk.text_with_context.replace("'", "''")
        section_title_escaped = (chunk.section_title or "").replace("'", "''")
        speaker_escaped = (chunk.speaker or "").replace("'", "''")
        title_escaped = chunk.source_title.replace("'", "''")
        author_escaped = chunk.source_author.replace("'", "''")

        sql = f"""INSERT OR REPLACE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, paragraph_index, char_start, char_end,
    timestamp_start, timestamp_end, speaker, composers, pieces, techniques, source_hash
) VALUES (
    '{chunk.chunk_id}',
    '{text_escaped}',
    '{text_with_context_escaped}',
    '{chunk.source_type}',
    '{title_escaped}',
    '{author_escaped}',
    '{chunk.source_url or ""}',
    NULL,
    '{section_title_escaped}',
    {chunk.paragraph_index or 'NULL'},
    NULL,
    NULL,
    {chunk.timestamp_start if chunk.timestamp_start is not None else 'NULL'},
    {chunk.timestamp_end if chunk.timestamp_end is not None else 'NULL'},
    '{speaker_escaped}',
    '{composers_json}',
    '{pieces_json}',
    '{techniques_json}',
    '{chunk.source_hash}'
);"""
        statements.append(sql)

    return "\n\n".join(statements)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest YouTube piano masterclass transcripts into CrescendAI RAG"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("youtube_chunks.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--sql-output",
        type=Path,
        help="Output SQL file for wrangler d1 execute"
    )
    parser.add_argument(
        "--video-id",
        help="Single YouTube video ID to process"
    )
    parser.add_argument(
        "--title",
        help="Video title (required with --video-id)"
    )
    parser.add_argument(
        "--speaker",
        help="Speaker/teacher name (required with --video-id)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file with list of videos to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't write output"
    )

    args = parser.parse_args()

    all_chunks = []
    all_metadata = []

    if args.video_id:
        # Process single video
        if not args.title or not args.speaker:
            parser.error("--title and --speaker are required with --video-id")

        print(f"Processing video: {args.video_id}")
        chunks, metadata = process_video(
            video_id=args.video_id,
            title=args.title,
            speaker=args.speaker,
        )
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)

    elif args.config:
        # Process videos from config file
        config = json.loads(args.config.read_text())
        videos = config.get("videos", config) if isinstance(config, dict) else config

        for video_info in videos:
            video_id = extract_video_id(video_info.get("video_id") or video_info.get("url"))
            title = video_info["title"]
            speaker = video_info["speaker"]
            composers = video_info.get("composers", [])

            print(f"\nProcessing: {title}")
            chunks, metadata = process_video(
                video_id=video_id,
                title=title,
                speaker=speaker,
                hint_composers=composers,
            )
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    else:
        # Use default masterclass list
        print("No videos specified, using default list...")
        print("Provide --video-id or --config for custom videos")

        for video_info in DEFAULT_MASTERCLASSES:
            if video_info["video_id"].startswith("example"):
                print(f"Skipping example video: {video_info['title']}")
                continue

            print(f"\nProcessing: {video_info['title']}")
            chunks, metadata = process_video(
                video_id=video_info["video_id"],
                title=video_info["title"],
                speaker=video_info["speaker"],
                hint_composers=video_info.get("composers", []),
            )
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    print(f"\n{'=' * 50}")
    print(f"Total chunks generated: {len(all_chunks)}")

    if all_chunks:
        print("\nSample chunk:")
        sample = all_chunks[0]
        print(f"  ID: {sample.chunk_id}")
        print(f"  Title: {sample.source_title}")
        print(f"  Speaker: {sample.speaker}")
        print(f"  Timestamp: {format_timestamp(sample.timestamp_start or 0)}")
        print(f"  Composers: {sample.composers}")
        print(f"  Techniques: {sample.techniques}")
        print(f"  Text preview: {sample.text[:200]}...")

    if args.dry_run:
        print("\nDry run - no files written")
        return

    if not all_chunks:
        print("\nNo chunks to write")
        return

    # Write JSON output
    output_data = {
        "chunks": [asdict(c) for c in all_chunks],
        "metadata": [asdict(m) for m in all_metadata]
    }

    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nWrote {len(all_chunks)} chunks to {args.output}")

    # Write SQL output if requested
    if args.sql_output:
        sql = generate_sql_inserts(all_chunks)
        args.sql_output.write_text(sql)
        print(f"Wrote SQL to {args.sql_output}")

    print("\nTo ingest into D1 (local):")
    print(f"  wrangler d1 execute crescendai-db --local --file {args.sql_output or 'youtube.sql'}")


if __name__ == "__main__":
    main()
