#!/usr/bin/env python3
"""
Ingest Piano Mastery by Harriette Brower from Project Gutenberg into CrescendAI RAG system.

This script:
1. Downloads the text from Gutenberg
2. Parses into chapters/interviews
3. Chunks at ~400-512 tokens with 15% overlap
4. Extracts metadata (composers, pieces, techniques)
5. Generates text_with_context with header injection
6. Outputs JSON for ingestion via wrangler CLI

Usage:
    python ingest_gutenberg.py --output chunks.json
    python ingest_gutenberg.py --output chunks.json --dry-run
"""

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.request import urlopen


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/15604/pg15604.txt"
SOURCE_TITLE = "Piano Mastery"
SOURCE_AUTHOR = "Harriette Brower"
SOURCE_TYPE = "book"

# Known composers mentioned in Piano Mastery
KNOWN_COMPOSERS = [
    "Bach", "Beethoven", "Brahms", "Chopin", "Czerny", "Debussy", "Grieg",
    "Handel", "Haydn", "Liszt", "Mendelssohn", "Mozart", "Rachmaninoff",
    "Scarlatti", "Schubert", "Schumann", "Scriabin", "Tchaikovsky", "Wagner",
    "Paderewski", "Godowsky", "Leschetizky", "Busoni", "Hofmann", "Rosenthal",
    "Gabrilowitch", "Bauer", "Grainger", "Hutcheson"
]

# Common piano techniques
KNOWN_TECHNIQUES = [
    "legato", "staccato", "touch", "tone", "pedal", "pedaling", "dynamics",
    "phrasing", "articulation", "fingering", "scales", "arpeggios", "octaves",
    "trills", "voicing", "tempo", "rubato", "expression", "interpretation",
    "technique", "practice", "memorization", "sight-reading", "ear training",
    "hand position", "wrist", "arm weight", "relaxation", "tension"
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


def download_gutenberg_text(url: str) -> str:
    """Download text from Project Gutenberg."""
    print(f"Downloading from {url}...")
    with urlopen(url) as response:
        content = response.read().decode("utf-8-sig")
    # Normalize line endings (Gutenberg uses CRLF)
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    print(f"Downloaded {len(content):,} characters")
    return content


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Gutenberg header and footer boilerplate."""
    # Find start of actual content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find next double newline after marker
            newline_idx = text.find("\n\n", idx)
            if newline_idx != -1:
                start_idx = newline_idx + 2
                break

    # Find end of actual content
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = min(end_idx, idx)

    return text[start_idx:end_idx].strip()


def parse_chapters(text: str) -> list[tuple[str, str, Optional[str]]]:
    """
    Parse text into chapters/interviews.

    Returns list of (title, content, speaker) tuples.

    Piano Mastery format: Roman numeral followed by pianist name in all caps.
    Example:
        I

        IGNACE JAN PADEREWSKI

        One of the most consummate masters...
    """
    chapters = []

    # Pattern: Roman numeral on its own line, followed by blank line, then ALL CAPS title
    # The Roman numeral can be I, II, III, IV, V, VI, VII, VIII, IX, X, XI, etc.
    chapter_pattern = re.compile(
        r'\n\n([IVXLC]+)\n\n+([A-Z][A-Z\s\.\-\']+[A-Z])\n\n',
        re.MULTILINE
    )

    matches = list(chapter_pattern.finditer(text))

    if not matches:
        # Fallback: try simpler pattern - just all caps names
        alt_pattern = re.compile(
            r'\n\n([A-Z][A-Z\s\.\-\']{5,}[A-Z])\n\n',
            re.MULTILINE
        )
        matches = list(alt_pattern.finditer(text))

    if not matches:
        # Last fallback: treat entire text as one chapter
        chapters.append(("Piano Mastery", text, None))
        return chapters

    # Get content before first chapter as introduction (if substantial)
    first_match = matches[0]
    intro_text = text[:first_match.start()].strip()
    # Strip Gutenberg header from intro
    if "*** START OF" in intro_text:
        intro_text = intro_text.split("*** START OF")[-1]
        intro_text = intro_text.split("\n\n", 1)[-1] if "\n\n" in intro_text else ""
    if len(intro_text) > 500:
        chapters.append(("Introduction", intro_text, SOURCE_AUTHOR))

    for i, match in enumerate(matches):
        groups = match.groups()
        if len(groups) == 2:
            # Roman numeral + title format
            roman, title = groups
            title = title.strip()
        else:
            # Just title format
            title = groups[0].strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Extract speaker (pianist name) from title
        speaker = title.title()
        # Clean up common suffixes/titles
        speaker = re.sub(r'\s+SECOND SERIES.*$', '', speaker, flags=re.IGNORECASE)

        if content and len(content) > 100:  # Skip tiny chapters
            chapters.append((title.title(), content, speaker))

    return chapters


def extract_composers(text: str) -> list[str]:
    """Extract mentioned composers from text."""
    found = []
    text_lower = text.lower()
    for composer in KNOWN_COMPOSERS:
        if composer.lower() in text_lower:
            found.append(composer)
    return list(set(found))


def extract_techniques(text: str) -> list[str]:
    """Extract mentioned techniques from text."""
    found = []
    text_lower = text.lower()
    for technique in KNOWN_TECHNIQUES:
        if technique.lower() in text_lower:
            found.append(technique)
    return list(set(found))


def extract_pieces(text: str) -> list[str]:
    """
    Extract mentioned pieces from text.

    Looks for patterns like:
    - "Opus 23"
    - "Nocturne in E flat"
    - "Sonata in C minor"
    """
    pieces = []

    # Opus patterns
    opus_pattern = re.compile(r'(?:Op\.|Opus)\s*\d+(?:\s*No\.\s*\d+)?', re.IGNORECASE)
    pieces.extend(opus_pattern.findall(text))

    # Common piece types
    piece_types = [
        r'(?:Nocturne|Ballade|Scherzo|Polonaise|Mazurka|Waltz|Etude|Prelude|Sonata|Concerto|Rhapsody)\s+(?:in\s+)?[A-G](?:\s+(?:major|minor|flat|sharp))?(?:\s*(?:No\.\s*)?\d+)?'
    ]
    for pattern in piece_types:
        matches = re.findall(pattern, text, re.IGNORECASE)
        pieces.extend(matches)

    return list(set(pieces))


def estimate_page_number(char_position: int, chars_per_page: int = 2500) -> int:
    """Estimate page number based on character position."""
    return (char_position // chars_per_page) + 1


def split_into_chunks(
    text: str,
    target_tokens: int = 450,
    overlap_ratio: float = 0.15,
    chars_per_token: float = 4.0
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Returns list of (chunk_text, char_start, char_end) tuples.
    """
    target_chars = int(target_tokens * chars_per_token)
    overlap_chars = int(target_chars * overlap_ratio)

    # Split by paragraphs first
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_start = 0
    char_position = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_position += 2  # Account for split newlines
            continue

        para_length = len(para)

        # If adding this paragraph exceeds target, finalize current chunk
        if current_length + para_length > target_chars and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, chunk_start, char_position))

            # Start new chunk with overlap
            # Find paragraphs that fit in overlap
            overlap_paras = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p) + 2
                else:
                    break

            current_chunk = overlap_paras
            current_length = sum(len(p) + 2 for p in current_chunk)
            chunk_start = char_position - current_length

        current_chunk.append(para)
        current_length += para_length + 2
        char_position += para_length + 2

    # Don't forget last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append((chunk_text, chunk_start, char_position))

    return chunks


def create_text_with_context(
    text: str,
    section_title: str,
    page_number: int,
    composers: list[str],
    techniques: list[str],
    speaker: Optional[str] = None
) -> str:
    """Create text with contextual header injection."""
    header_lines = [
        f"[Source: {SOURCE_TITLE} by {SOURCE_AUTHOR}, p.{page_number}]"
    ]

    if section_title:
        header_lines.append(f"[Section: {section_title}]")

    if speaker:
        header_lines.append(f"[Speaker: {speaker}]")

    context_parts = []
    if composers:
        context_parts.append(", ".join(composers[:3]))  # Limit to 3
    if techniques:
        context_parts.append(", ".join(techniques[:3]))

    if context_parts:
        header_lines.append(f"[Context: {' - '.join(context_parts)}]")

    header = "\n".join(header_lines)
    return f"{header}\n\n{text}"


def compute_source_hash(text: str, section: str, char_start: int) -> str:
    """Compute unique hash for deduplication."""
    content = f"{SOURCE_TITLE}:{section}:{char_start}:{text[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def infer_topic(techniques: list[str]) -> Optional[str]:
    """Infer topic category from techniques."""
    technique_topics = {
        "technique/articulation": ["touch", "legato", "staccato", "articulation", "fingering"],
        "technique/pedaling": ["pedal", "pedaling", "sustain"],
        "interpretation/phrasing": ["phrasing", "expression", "rubato", "interpretation"],
        "interpretation/dynamics": ["dynamics", "tone", "voicing"],
        "practice/method": ["practice", "memorization", "scales", "arpeggios"],
    }

    for topic, keywords in technique_topics.items():
        if any(t.lower() in keywords for t in techniques):
            return topic

    return None


def process_gutenberg(url: str) -> tuple[list[PedagogyChunk], list[VectorMetadata]]:
    """Process Gutenberg text into chunks with metadata."""
    # Download and clean text
    raw_text = download_gutenberg_text(url)
    clean_text = strip_gutenberg_header_footer(raw_text)

    print(f"Clean text: {len(clean_text):,} characters")

    # Parse into chapters
    chapters = parse_chapters(clean_text)
    print(f"Found {len(chapters)} chapters/sections")

    chunks = []
    metadata_list = []
    global_char_offset = 0

    for chapter_idx, (chapter_title, chapter_content, speaker) in enumerate(chapters):
        print(f"Processing: {chapter_title}")

        # Chunk the chapter
        chapter_chunks = split_into_chunks(chapter_content)

        for chunk_idx, (chunk_text, local_start, local_end) in enumerate(chapter_chunks):
            char_start = global_char_offset + local_start
            char_end = global_char_offset + local_end

            # Extract metadata from chunk
            composers = extract_composers(chunk_text)
            techniques = extract_techniques(chunk_text)
            pieces = extract_pieces(chunk_text)

            # Estimate page number
            page_number = estimate_page_number(char_start)

            # Create context-enriched text
            text_with_context = create_text_with_context(
                chunk_text,
                chapter_title,
                page_number,
                composers,
                techniques,
                speaker
            )

            # Generate unique ID and hash
            chunk_id = str(uuid.uuid4())
            source_hash = compute_source_hash(chunk_text, chapter_title, local_start)

            chunk = PedagogyChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                text_with_context=text_with_context,
                source_type=SOURCE_TYPE,
                source_title=SOURCE_TITLE,
                source_author=SOURCE_AUTHOR,
                source_url=f"https://www.gutenberg.org/ebooks/26663",
                page_number=page_number,
                section_title=chapter_title,
                paragraph_index=chunk_idx,
                char_start=char_start,
                char_end=char_end,
                timestamp_start=None,
                timestamp_end=None,
                speaker=speaker,
                composers=composers,
                pieces=pieces,
                techniques=techniques,
                source_hash=source_hash
            )
            chunks.append(chunk)

            # Create vector metadata
            topic = infer_topic(techniques)
            primary_composer = composers[0] if composers else None

            metadata = VectorMetadata(
                difficulty="intermediate",  # Piano Mastery targets intermediate+ pianists
                topic=topic,
                content_type="explanation",
                composer=primary_composer,
                source_type=SOURCE_TYPE
            )
            metadata_list.append(metadata)

        global_char_offset += len(chapter_content) + 2

    return chunks, metadata_list


def generate_sql_inserts(chunks: list[PedagogyChunk]) -> str:
    """Generate SQL INSERT statements for wrangler d1 execute."""
    statements = []

    for chunk in chunks:
        composers_json = json.dumps(chunk.composers)
        pieces_json = json.dumps(chunk.pieces)
        techniques_json = json.dumps(chunk.techniques)

        # Escape single quotes in text
        text_escaped = chunk.text.replace("'", "''")
        text_with_context_escaped = chunk.text_with_context.replace("'", "''")
        section_title_escaped = (chunk.section_title or "").replace("'", "''")
        speaker_escaped = (chunk.speaker or "").replace("'", "''")

        sql = f"""INSERT OR REPLACE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, paragraph_index, char_start, char_end,
    timestamp_start, timestamp_end, speaker, composers, pieces, techniques, source_hash
) VALUES (
    '{chunk.chunk_id}',
    '{text_escaped}',
    '{text_with_context_escaped}',
    '{chunk.source_type}',
    '{chunk.source_title}',
    '{chunk.source_author}',
    '{chunk.source_url or ""}',
    {chunk.page_number or 'NULL'},
    '{section_title_escaped}',
    {chunk.paragraph_index or 'NULL'},
    {chunk.char_start or 'NULL'},
    {chunk.char_end or 'NULL'},
    NULL,
    NULL,
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
        description="Ingest Piano Mastery from Project Gutenberg into CrescendAI RAG"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("gutenberg_chunks.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--sql-output",
        type=Path,
        help="Output SQL file for wrangler d1 execute"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't write output"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks (for testing)"
    )

    args = parser.parse_args()

    # Process the book
    chunks, metadata_list = process_gutenberg(GUTENBERG_URL)

    if args.limit:
        chunks = chunks[:args.limit]
        metadata_list = metadata_list[:args.limit]

    print(f"\nGenerated {len(chunks)} chunks")

    # Show sample
    if chunks:
        print("\nSample chunk:")
        sample = chunks[0]
        print(f"  ID: {sample.chunk_id}")
        print(f"  Section: {sample.section_title}")
        print(f"  Page: {sample.page_number}")
        print(f"  Composers: {sample.composers}")
        print(f"  Techniques: {sample.techniques}")
        print(f"  Text preview: {sample.text[:200]}...")

    if args.dry_run:
        print("\nDry run - no files written")
        return

    # Write JSON output
    output_data = {
        "chunks": [asdict(c) for c in chunks],
        "metadata": [asdict(m) for m in metadata_list]
    }

    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nWrote {len(chunks)} chunks to {args.output}")

    # Write SQL output if requested
    if args.sql_output:
        sql = generate_sql_inserts(chunks)
        args.sql_output.write_text(sql)
        print(f"Wrote SQL to {args.sql_output}")

    print("\nTo ingest into D1 (local):")
    print(f"  wrangler d1 execute crescendai-db --local --file {args.sql_output or 'chunks.sql'}")
    print("\nTo ingest into D1 (production):")
    print(f"  wrangler d1 execute crescendai-db --file {args.sql_output or 'chunks.sql'}")


if __name__ == "__main__":
    main()
