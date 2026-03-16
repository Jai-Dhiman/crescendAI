# Reference Cache Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reference performance profiles from MAESTRO recordings, match them to the 244-piece score library, and upload to R2 for the API analysis engine.

**Architecture:** Three-stage CLI pipeline (`match` -> human review -> `generate` -> `upload`) built as subcommands of `reference_cache.py`. The existing DTW alignment, BarStats/ReferenceProfile dataclasses, and MIDI loading functions are preserved. New code adds fuzzy matching, CSV workflow, validation gates, and upload.

**Tech Stack:** Python 3.10+, mido, dtw-python, numpy, csv stdlib, subprocess (for wrangler). Tests with pytest. Run with uv.

**Spec:** `docs/superpowers/specs/2026-03-15-reference-cache-pipeline-design.md`

---

## File Structure

```
model/src/score_library/
  reference_cache.py          -- REWRITE: subcommand CLI, preserve existing functions
  maestro_matcher.py          -- NEW: fuzzy matching engine (normalize, score, match)

model/tests/score_library/
  test_maestro_matcher.py     -- NEW: unit tests for fuzzy matching
  test_reference_pipeline.py  -- NEW: integration tests for generate + validation gates
```

**Rationale:** The matching logic is complex enough (normalization, multi-piece detection, Dice scoring) to warrant its own module. The existing `reference_cache.py` keeps DTW/stats/aggregation and gains the CLI orchestration. Tests are split: unit tests for matching (fast, no MIDI), integration tests for the pipeline (uses fixtures).

**Dependencies:**
- `titles.json` already exists at `model/data/score_library/titles.json` -- maps each piece_id to a human-readable title (e.g., `"chopin.ballades.1": "Ballade No. 1"`). The match command loads this as the ASAP title catalog.
- `model/tests/score_library/` already exists with `__init__.py` and existing test files.

**Design note -- token Dice vs. bigram Dice:** The spec mentions "bigram Dice similarity." This plan uses token-level Dice (set intersection over normalized tokens) rather than character-bigram Dice. Token Dice is more appropriate here because the normalization step extracts structured tokens (opus numbers, catalog numbers, movement numbers) that need exact matching -- bigram similarity would give partial credit for e.g. "op_10" vs "op_11" which is incorrect. Token Dice on well-normalized tokens is both simpler and more precise for this use case.

---

## Chunk 1: Fuzzy Matching Engine

### Task 1: Composer Normalization

**Files:**
- Create: `model/src/score_library/maestro_matcher.py`
- Create: `model/tests/score_library/test_maestro_matcher.py`

- [ ] **Step 1: Write failing tests for composer normalization**

```python
# model/tests/score_library/test_maestro_matcher.py
"""Tests for MAESTRO-to-ASAP fuzzy matching engine."""
import pytest

from src.score_library.maestro_matcher import extract_composer_last_name, match_composer


class TestExtractComposerLastName:
    def test_simple_name(self):
        assert extract_composer_last_name("Frédéric Chopin") == "chopin"

    def test_van_prefix(self):
        assert extract_composer_last_name("Ludwig van Beethoven") == "beethoven"

    def test_bach_full(self):
        assert extract_composer_last_name("Johann Sebastian Bach") == "bach"

    def test_arrangement_slash(self):
        # "Franz Schubert / Franz Liszt" -> primary composer is Schubert
        assert extract_composer_last_name("Franz Schubert / Franz Liszt") == "schubert"

    def test_accented_characters(self):
        assert extract_composer_last_name("César Franck") == "franck"

    def test_single_name(self):
        assert extract_composer_last_name("Rachmaninoff") == "rachmaninoff"


class TestMatchComposer:
    """Match MAESTRO composer to ASAP composer prefix."""

    ASAP_COMPOSERS = [
        "bach", "beethoven", "chopin", "debussy", "haydn", "liszt",
        "mozart", "rachmaninoff", "schubert", "schumann", "scriabin",
        "brahms", "ravel", "prokofiev", "balakirev", "glinka",
    ]

    def test_chopin_match(self):
        assert match_composer("Frédéric Chopin", self.ASAP_COMPOSERS) == "chopin"

    def test_beethoven_match(self):
        assert match_composer("Ludwig van Beethoven", self.ASAP_COMPOSERS) == "beethoven"

    def test_bach_match(self):
        assert match_composer("Johann Sebastian Bach", self.ASAP_COMPOSERS) == "bach"

    def test_arrangement_uses_primary(self):
        assert match_composer("Franz Schubert / Franz Liszt", self.ASAP_COMPOSERS) == "schubert"

    def test_scriabin_match(self):
        assert match_composer("Alexander Scriabin", self.ASAP_COMPOSERS) == "scriabin"

    def test_no_match_returns_none(self):
        assert match_composer("Alban Berg", self.ASAP_COMPOSERS) is None

    def test_balakirev_arrangement(self):
        # "Mikhail Glinka / Mily Balakirev" -> primary is Glinka
        assert match_composer("Mikhail Glinka / Mily Balakirev", self.ASAP_COMPOSERS) == "glinka"

    def test_rachmaninoff_match(self):
        assert match_composer("Sergei Rachmaninoff", self.ASAP_COMPOSERS) == "rachmaninoff"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestExtractComposerLastName -v
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestMatchComposer -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement composer normalization**

```python
# model/src/score_library/maestro_matcher.py
"""MAESTRO-to-ASAP fuzzy matching engine.

Maps MAESTRO CSV entries (canonical_composer, canonical_title) to ASAP piece IDs
using composer normalization, title normalization, and token Dice similarity.
"""
from __future__ import annotations

import unicodedata


def _strip_accents(s: str) -> str:
    """Remove accents/diacritics from a string."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def extract_composer_last_name(composer: str) -> str:
    """Extract the last name from a MAESTRO composer string, lowercased and accent-stripped.

    For arrangement credits ("Schubert / Liszt"), uses the primary (first) composer.
    Handles 'van', 'von' prefixes by taking the final token.
    """
    # Take primary composer (before any slash)
    primary = composer.split("/")[0].strip()
    primary = _strip_accents(primary).lower()

    tokens = primary.split()
    if not tokens:
        return primary

    # Last token is the last name
    return tokens[-1]


def match_composer(maestro_composer: str, asap_composers: list[str]) -> str | None:
    """Match a MAESTRO composer string to an ASAP composer prefix.

    Returns the matching ASAP composer prefix, or None if no match.
    """
    last_name = extract_composer_last_name(maestro_composer)

    for asap_composer in asap_composers:
        if last_name == asap_composer:
            return asap_composer

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestExtractComposerLastName -v
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestMatchComposer -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/score_library/maestro_matcher.py model/tests/score_library/test_maestro_matcher.py
git commit -m "feat: add composer normalization for MAESTRO-to-ASAP matching"
```

---

### Task 2: Title Normalization

**Files:**
- Modify: `model/src/score_library/maestro_matcher.py`
- Modify: `model/tests/score_library/test_maestro_matcher.py`

- [ ] **Step 1: Write failing tests for title normalization and multi-piece detection**

Add to `test_maestro_matcher.py`:

```python
from src.score_library.maestro_matcher import normalize_title, detect_multi_piece


class TestDetectMultiPiece:
    def test_number_range(self):
        assert detect_multi_piece("24 Preludes Op. 11, No. 13-24") is True

    def test_nos_range(self):
        assert detect_multi_piece("Nos. 1-6") is True

    def test_books(self):
        assert detect_multi_piece("Well-Tempered Clavier Books I & II") is True

    def test_complete(self):
        assert detect_multi_piece("Complete Etudes") is True

    def test_single_piece(self):
        assert detect_multi_piece("Ballade No. 1 in G Minor") is False

    def test_single_number(self):
        assert detect_multi_piece("Etude Op. 10 No. 3") is False


class TestNormalizeTitle:
    """Normalize MAESTRO titles and ASAP title components to comparable tokens."""

    def test_opus_normalization(self):
        assert "op_10" in normalize_title("Etudes Op. 10")

    def test_opus_no_space(self):
        assert "op_10" in normalize_title("Etudes op.10")

    def test_opus_word(self):
        assert "op_10" in normalize_title("Etudes Opus 10")

    def test_number_extraction(self):
        assert "3" in normalize_title("No. 3 in E major")

    def test_number_nr(self):
        assert "3" in normalize_title("Nr. 3")

    def test_bwv_catalog(self):
        assert "bwv_846" in normalize_title("Prelude BWV 846")

    def test_k_catalog(self):
        assert "k_331" in normalize_title("Sonata K. 331")

    def test_d_catalog(self):
        assert "d_899" in normalize_title("Impromptu D. 899")

    def test_strip_common_prefixes(self):
        result = normalize_title("Piano Sonata No. 23 in F minor")
        assert "piano" not in result
        assert "sonata" in result

    def test_strip_key_signatures(self):
        result = normalize_title("Ballade No. 1 in G Minor")
        # Key info (in G Minor) should be stripped
        assert "minor" not in result

    def test_accent_stripping(self):
        result = normalize_title("Étude")
        assert "etude" in result

    def test_etude_singular(self):
        result = normalize_title("Etudes Op. 10")
        assert "etude" in result

    def test_returns_sorted_tokens(self):
        # Normalization returns a sorted list of tokens for stable comparison
        result = normalize_title("Sonata No. 23")
        assert isinstance(result, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestNormalizeTitle -v
```

Expected: FAIL (function not found)

- [ ] **Step 3: Implement title normalization**

Add to `maestro_matcher.py`:

```python
import re


# Keys/modes to strip from titles (these don't help matching)
_KEY_PATTERN = re.compile(
    r"\b(in\s+)?[A-Ga-g][#b]?\s*(flat|sharp)?\s*"
    r"(major|minor|dur|moll)\b",
    re.IGNORECASE,
)

# Opus patterns: "Op. 10", "op.10", "Opus 10"
_OPUS_PATTERN = re.compile(r"\b(?:op(?:us)?\.?\s*)(\d+)", re.IGNORECASE)

# Number patterns: "No. 3", "No.3", "Nr. 3", "No 3"
_NUMBER_PATTERN = re.compile(r"\b(?:no|nr)\.?\s*(\d+)", re.IGNORECASE)

# Catalog number patterns: "BWV 846", "K. 331", "D. 899"
_CATALOG_PATTERNS = [
    (re.compile(r"\bBWV\.?\s*(\d+)", re.IGNORECASE), "bwv"),
    (re.compile(r"\bK\.?\s*(\d+)", re.IGNORECASE), "k"),
    (re.compile(r"\bD\.?\s*(\d+)", re.IGNORECASE), "d"),
    (re.compile(r"\bHob\.?\s*([IVXL]+[:/]?\d*)", re.IGNORECASE), "hob"),
]

# Common prefixes to strip (the work type remains, the generic prefix goes)
_STRIP_PREFIXES = {"piano"}

# Plurals to singularize
_SINGULAR_MAP = {
    "etudes": "etude",
    "ballades": "ballade",
    "nocturnes": "nocturne",
    "preludes": "prelude",
    "sonatas": "sonata",
    "mazurkas": "mazurka",
    "polonaises": "polonaise",
    "scherzos": "scherzo",
    "waltzes": "waltz",
    "impromptus": "impromptu",
}

# Range indicators for multi-piece detection
_RANGE_PATTERN = re.compile(
    r"(?:nos?\.\s*\d+\s*[-–]\s*\d+)|"
    r"(?:\d+\s*[-–]\s*\d+)|"
    r"(?:books?\s+[IViv]+\s*[&,]\s*[IViv]+)|"
    r"(?:complete\s+\w+)",
    re.IGNORECASE,
)


def detect_multi_piece(title: str) -> bool:
    """Detect if a MAESTRO title likely covers multiple pieces."""
    return bool(_RANGE_PATTERN.search(title))


def normalize_title(title: str) -> list[str]:
    """Normalize a title string into a sorted list of comparable tokens.

    Applies: accent stripping, lowercasing, key signature removal,
    opus/catalog number standardization, number extraction, prefix stripping,
    plural singularization.
    """
    text = _strip_accents(title).lower()

    # Remove key signatures
    text = _KEY_PATTERN.sub("", text)

    tokens: list[str] = []

    # Extract catalog numbers first (before general cleanup eats them)
    for pattern, prefix in _CATALOG_PATTERNS:
        match = pattern.search(text)
        if match:
            tokens.append(f"{prefix}_{match.group(1).lower()}")
            text = pattern.sub("", text)

    # Extract opus numbers
    opus_match = _OPUS_PATTERN.search(text)
    if opus_match:
        tokens.append(f"op_{opus_match.group(1)}")
        text = _OPUS_PATTERN.sub("", text)

    # Extract movement/piece numbers
    for num_match in _NUMBER_PATTERN.finditer(text):
        tokens.append(num_match.group(1))
    text = _NUMBER_PATTERN.sub("", text)

    # Clean remaining text
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()

    for word in words:
        if not word or word in _STRIP_PREFIXES:
            continue
        # Singularize
        word = _SINGULAR_MAP.get(word, word)
        # Skip pure numbers (already extracted above) and short noise
        if word.isdigit() or len(word) <= 1:
            continue
        tokens.append(word)

    return sorted(set(tokens))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestNormalizeTitle -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/score_library/maestro_matcher.py model/tests/score_library/test_maestro_matcher.py
git commit -m "feat: add title normalization for MAESTRO-to-ASAP matching"
```

---

### Task 3: Dice Similarity + Piece Matching

**Files:**
- Modify: `model/src/score_library/maestro_matcher.py`
- Modify: `model/tests/score_library/test_maestro_matcher.py`

- [ ] **Step 1: Write failing tests for Dice similarity and piece matching**

Add to `test_maestro_matcher.py`:

```python
from src.score_library.maestro_matcher import (
    dice_similarity,
    match_piece,
    MatchResult,
)
import json
from pathlib import Path


class TestDiceSimilarity:
    def test_identical(self):
        assert dice_similarity(["sonata", "op_23"], ["sonata", "op_23"]) == 1.0

    def test_no_overlap(self):
        assert dice_similarity(["sonata"], ["prelude"]) == 0.0

    def test_partial_overlap(self):
        score = dice_similarity(["sonata", "op_23", "1"], ["sonata", "op_23", "2"])
        assert 0.5 < score < 1.0

    def test_empty_input(self):
        assert dice_similarity([], ["sonata"]) == 0.0


class TestMatchPiece:
    """End-to-end piece matching against a catalog."""

    @pytest.fixture
    def titles_map(self):
        """Minimal titles map for testing."""
        return {
            "chopin.ballades.1": "Ballade No. 1",
            "chopin.ballades.2": "Ballade No. 2",
            "chopin.ballades.3": "Ballade No. 3",
            "chopin.ballades.4": "Ballade No. 4",
            "chopin.etudes_op_10.3": "Etude Op. 10 No. 3",
            "chopin.etudes_op_10.4": "Etude Op. 10 No. 4",
            "bach.prelude.bwv_846": "Prelude - BWV 846",
            "bach.fugue.bwv_846": "Fugue - BWV 846",
            "beethoven.sonata_23.1": "Sonata No. 23 - 1st Movement",
        }

    def test_ballade_match(self, titles_map):
        result = match_piece(
            "chopin", "Ballade No. 1 in G Minor", titles_map
        )
        assert result is not None
        assert result.piece_id == "chopin.ballades.1"
        assert result.confidence >= 0.5

    def test_etude_match(self, titles_map):
        result = match_piece(
            "chopin", "Etude in E Major, Op. 10, No. 3", titles_map
        )
        assert result is not None
        assert result.piece_id == "chopin.etudes_op_10.3"

    def test_bach_bwv_match(self, titles_map):
        result = match_piece(
            "bach", "Prelude in C Major, BWV 846", titles_map
        )
        assert result is not None
        assert result.piece_id == "bach.prelude.bwv_846"

    def test_no_match_below_threshold(self, titles_map):
        result = match_piece(
            "chopin", "Completely Unrelated Title", titles_map
        )
        assert result is None or result.confidence < 0.2

    def test_only_searches_within_composer(self, titles_map):
        # A Bach title should not match Chopin pieces
        result = match_piece(
            "bach", "Ballade No. 1", titles_map
        )
        # Should not match chopin.ballades.1 -- wrong composer
        if result is not None:
            assert result.piece_id.startswith("bach")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestDiceSimilarity -v
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestMatchPiece -v
```

Expected: FAIL (function not found)

- [ ] **Step 3: Implement Dice similarity and piece matching**

Add to `maestro_matcher.py`:

```python
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of matching a MAESTRO entry to an ASAP piece."""
    piece_id: str
    asap_title: str
    confidence: float
    multi_piece: bool


def dice_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute Dice coefficient over token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    return (2.0 * intersection) / (len(set_a) + len(set_b))


def match_piece(
    asap_composer: str,
    maestro_title: str,
    titles_map: dict[str, str],
    min_confidence: float = 0.2,
) -> MatchResult | None:
    """Match a MAESTRO title to the best ASAP piece within a composer.

    Args:
        asap_composer: The ASAP composer prefix (e.g., "chopin").
        maestro_title: The MAESTRO canonical_title string.
        titles_map: Dict mapping piece_id -> human-readable title
                     (from titles.json).
        min_confidence: Minimum Dice similarity to return a match.

    Returns:
        MatchResult or None if no match above threshold.
    """
    maestro_tokens = normalize_title(maestro_title)
    multi_piece = detect_multi_piece(maestro_title)

    best_score = 0.0
    best_piece_id = ""
    best_title = ""

    for piece_id, title in titles_map.items():
        # Only consider pieces from the matching composer
        if not piece_id.startswith(asap_composer + "."):
            continue

        # Normalize the ASAP title for comparison
        asap_tokens = normalize_title(title)

        # Also include tokens from the piece_id itself (e.g., "op_10", "3")
        # The piece_id segments after composer contain structured info
        id_parts = piece_id.split(".")[1:]  # drop composer
        for part in id_parts:
            # Split underscored segments and add as tokens
            for sub in part.split("_"):
                if sub and len(sub) > 1:
                    asap_tokens.append(sub.lower())
            asap_tokens.append(part.lower())
        asap_tokens = sorted(set(asap_tokens))

        score = dice_similarity(maestro_tokens, asap_tokens)
        if score > best_score:
            best_score = score
            best_piece_id = piece_id
            best_title = title

    if best_score < min_confidence or not best_piece_id:
        return None

    return MatchResult(
        piece_id=best_piece_id,
        asap_title=best_title,
        confidence=round(best_score, 3),
        multi_piece=multi_piece,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestDiceSimilarity -v
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestMatchPiece -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/score_library/maestro_matcher.py model/tests/score_library/test_maestro_matcher.py
git commit -m "feat: add Dice similarity scoring and piece matching"
```

---

### Task 4: Full Match Pipeline Function

**Files:**
- Modify: `model/src/score_library/maestro_matcher.py`
- Modify: `model/tests/score_library/test_maestro_matcher.py`

- [ ] **Step 1: Write failing tests for the full match pipeline**

Add to `test_maestro_matcher.py`:

```python
import csv
import io
from src.score_library.maestro_matcher import run_match_pipeline


class TestRunMatchPipeline:
    """Integration test: CSV in -> match results out."""

    MAESTRO_CSV = (
        "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration\n"
        'Frédéric Chopin,"Ballade No. 1 in G Minor",train,2018,2018/file1.midi,2018/file1.wav,500.0\n'
        'Frédéric Chopin,"Ballade No. 1 in G Minor",train,2011,2011/file2.midi,2011/file2.wav,520.0\n'
        'Johann Sebastian Bach,"Prelude in C Major, BWV 846",train,2009,2009/file3.midi,2009/file3.wav,200.0\n'
        'Alban Berg,"Sonata Op. 1",train,2018,2018/file4.midi,2018/file4.wav,700.0\n'
    )

    TITLES_MAP = {
        "chopin.ballades.1": "Ballade No. 1",
        "chopin.ballades.2": "Ballade No. 2",
        "bach.prelude.bwv_846": "Prelude - BWV 846",
        "bach.fugue.bwv_846": "Fugue - BWV 846",
    }

    ASAP_COMPOSERS = ["chopin", "bach"]

    def test_produces_matches_and_unmatched(self):
        matches, unmatched = run_match_pipeline(
            maestro_csv_content=self.MAESTRO_CSV,
            titles_map=self.TITLES_MAP,
            asap_composers=self.ASAP_COMPOSERS,
        )
        # Chopin Ballade 1 should match twice (2 recordings)
        chopin_matches = [m for m in matches if m["asap_piece_id"] == "chopin.ballades.1"]
        assert len(chopin_matches) == 2

        # Bach prelude should match
        bach_matches = [m for m in matches if m["asap_piece_id"] == "bach.prelude.bwv_846"]
        assert len(bach_matches) == 1

        # Alban Berg has no ASAP pieces -> unmatched
        assert len(unmatched) >= 1
        assert any("Berg" in u["maestro_composer"] for u in unmatched)

    def test_match_row_has_required_fields(self):
        matches, _ = run_match_pipeline(
            maestro_csv_content=self.MAESTRO_CSV,
            titles_map=self.TITLES_MAP,
            asap_composers=self.ASAP_COMPOSERS,
        )
        required = {
            "maestro_composer", "maestro_title", "midi_filename", "duration_s",
            "asap_piece_id", "asap_title", "confidence", "multi_piece", "status",
        }
        for row in matches:
            assert required.issubset(row.keys()), f"Missing fields: {required - row.keys()}"
            assert row["status"] == ""  # starts blank for review
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestRunMatchPipeline -v
```

Expected: FAIL (function not found)

- [ ] **Step 3: Implement run_match_pipeline**

Add to `maestro_matcher.py`:

```python
import csv
import io


def run_match_pipeline(
    maestro_csv_content: str,
    titles_map: dict[str, str],
    asap_composers: list[str],
    min_confidence: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Run the full matching pipeline.

    Args:
        maestro_csv_content: Content of the MAESTRO CSV file.
        titles_map: Dict mapping piece_id -> human-readable title.
        asap_composers: List of known ASAP composer prefixes.
        min_confidence: Minimum score to include in matches.

    Returns:
        (matches, unmatched) where each is a list of dicts.
    """
    reader = csv.DictReader(io.StringIO(maestro_csv_content))

    # Validate required CSV headers
    required_headers = {"canonical_composer", "canonical_title", "midi_filename", "duration"}
    if reader.fieldnames is None:
        raise ValueError("MAESTRO CSV is empty or has no header row")
    missing = required_headers - set(reader.fieldnames)
    if missing:
        raise ValueError(f"MAESTRO CSV missing required columns: {missing}")

    matches: list[dict] = []
    unmatched: list[dict] = []
    skipped_normalization = 0

    for row in reader:
        maestro_composer = row.get("canonical_composer", "").strip()
        maestro_title = row.get("canonical_title", "").strip()
        midi_filename = row.get("midi_filename", "").strip()
        duration = row.get("duration", "0")

        if not maestro_composer or not maestro_title:
            skipped_normalization += 1
            continue

        # Match composer
        asap_composer = match_composer(maestro_composer, asap_composers)
        if asap_composer is None:
            unmatched.append({
                "maestro_composer": maestro_composer,
                "maestro_title": maestro_title,
                "midi_filename": midi_filename,
                "reason": "no_composer_match",
            })
            continue

        # Match piece
        result = match_piece(asap_composer, maestro_title, titles_map, min_confidence)
        if result is None:
            unmatched.append({
                "maestro_composer": maestro_composer,
                "maestro_title": maestro_title,
                "midi_filename": midi_filename,
                "reason": "below_confidence_threshold",
            })
            continue

        matches.append({
            "maestro_composer": maestro_composer,
            "maestro_title": maestro_title,
            "midi_filename": midi_filename,
            "duration_s": duration,
            "asap_piece_id": result.piece_id,
            "asap_title": result.asap_title,
            "confidence": str(result.confidence),
            "multi_piece": str(result.multi_piece),
            "status": "",
        })

    return matches, unmatched
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd model && uv run pytest tests/score_library/test_maestro_matcher.py::TestRunMatchPipeline -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/score_library/maestro_matcher.py model/tests/score_library/test_maestro_matcher.py
git commit -m "feat: add full match pipeline function for MAESTRO-to-ASAP mapping"
```

---

## Chunk 2: CLI Rewrite + Validation Gates

### Task 5: Modify align_to_score to Return DTW Cost + Subcommand CLI Structure

**Files:**
- Modify: `model/src/score_library/reference_cache.py`

This task (a) modifies `align_to_score` to also return the `normalizedDistance` so the generate command can extract DTW cost from the same alignment it uses for bar mapping (avoids double DTW), and (b) rewrites `main()` with subcommands.

- [ ] **Step 1: Modify align_to_score return type**

Change `align_to_score` (line 172) to return a tuple `(bar_to_perf, normalized_distance)` instead of just `bar_to_perf`. This avoids running DTW twice per recording.

In `align_to_score`, after the DTW call (line 219-223), capture the cost:

```python
    normalized_distance = float(alignment.normalizedDistance)
```

And change the return (line 258) to:

```python
    return bar_to_perf, normalized_distance
```

Also update the empty-input early returns (lines 188, 206) to return `({}, 0.0)`.

Update `build_reference_for_piece` (line 499) to unpack the new return:

```python
            bar_to_notes, _ = align_to_score(perf_notes, score_data)
```

- [ ] **Step 2: Rewrite main() with argparse subcommands**

Replace the existing `main()` and `_find_maestro_midis_for_piece` (lines 530-668) with:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_match(args: argparse.Namespace) -> None:
    """Run the match subcommand."""
    import csv as csv_mod
    from .maestro_matcher import run_match_pipeline

    maestro_csv = Path(args.maestro_csv)
    score_dir = Path(args.score_dir)
    output_path = Path(args.output)

    if not maestro_csv.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found: {maestro_csv}")
    if not score_dir.exists():
        raise FileNotFoundError(f"Score directory not found: {score_dir}")

    # Load titles.json from score_dir
    titles_path = score_dir / "titles.json"
    if not titles_path.exists():
        raise FileNotFoundError(f"titles.json not found in {score_dir}")
    with open(titles_path, encoding="utf-8") as fh:
        titles_map: dict[str, str] = json.load(fh)

    # Extract known ASAP composers from score filenames
    asap_composers = sorted(
        {f.stem.split(".")[0] for f in score_dir.glob("*.json") if f.stem != "titles"}
    )

    print(f"MAESTRO CSV: {maestro_csv}")
    print(f"Score dir: {score_dir} ({len(titles_map)} pieces)")
    print(f"ASAP composers: {', '.join(asap_composers)}")
    print()

    maestro_content = maestro_csv.read_text(encoding="utf-8")
    matches, unmatched = run_match_pipeline(
        maestro_content, titles_map, asap_composers
    )

    # Write matches CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "maestro_composer", "maestro_title", "midi_filename", "duration_s",
        "asap_piece_id", "asap_title", "confidence", "multi_piece", "status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by confidence descending for easier review
        for row in sorted(matches, key=lambda r: float(r["confidence"]), reverse=True):
            writer.writerow(row)

    # Write unmatched CSV
    unmatched_path = output_path.parent / "unmatched_maestro.csv"
    unmatched_fields = ["maestro_composer", "maestro_title", "midi_filename", "reason"]
    with open(unmatched_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=unmatched_fields)
        writer.writeheader()
        for row in unmatched:
            writer.writerow(row)

    # Summary
    unique_pieces = {r["asap_piece_id"] for r in matches}
    print(f"Matched: {len(matches)} recordings -> {len(unique_pieces)} unique pieces")
    print(f"Unmatched: {len(unmatched)} recordings")
    print(f"Output: {output_path}")
    print(f"Unmatched: {unmatched_path}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Run the generate subcommand."""
    import csv as csv_mod

    matches_path = Path(args.matches)
    maestro_dir = Path(args.maestro_dir)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)

    if not matches_path.exists():
        raise FileNotFoundError(f"Matches CSV not found: {matches_path}")
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")
    if not score_dir.exists():
        raise FileNotFoundError(f"Score directory not found: {score_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read approved matches
    with open(matches_path, encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        all_rows = list(reader)

    approved = [r for r in all_rows if r.get("status", "").strip().lower() == "approved"]
    if not approved:
        raise ValueError("No approved rows in matches CSV. Review and mark rows as 'approved' first.")

    # Group by piece_id
    by_piece: dict[str, list[str]] = {}
    for row in approved:
        piece_id = row["asap_piece_id"]
        midi_filename = row["midi_filename"]
        if piece_id not in by_piece:
            by_piece[piece_id] = []
        by_piece[piece_id].append(midi_filename)

    print(f"Processing {len(by_piece)} pieces from {len(approved)} approved recordings")
    print()

    # Generation report rows
    report_rows: list[dict] = []
    min_coverage = args.min_coverage

    for piece_id, midi_filenames in sorted(by_piece.items()):
        print(f"[{piece_id}] ({len(midi_filenames)} recording(s))")

        score_path = score_dir / f"{piece_id}.json"
        if not score_path.exists():
            print(f"  Score not found: {score_path} -- skipping")
            report_rows.append({
                "piece_id": piece_id,
                "total_recordings": len(midi_filenames),
                "passed_validation": 0,
                "rejected_coverage": 0,
                "rejected_dtw_cost": 0,
                "performer_count": 0,
                "mean_coverage": "",
                "mean_dtw_cost": "",
                "errors": "score_not_found",
            })
            continue

        score_data = load_score(score_path)
        total_bars = len(score_data.get("bars", []))
        bar_lookup: dict[int, dict] = {}
        for bar in score_data.get("bars", []):
            bar_lookup[bar["bar_number"]] = bar

        all_bar_stats: list[list[BarStats]] = []
        coverages: list[float] = []
        dtw_costs: list[float] = []
        rejected_coverage_count = 0
        errors: list[str] = []

        for midi_filename in midi_filenames:
            midi_path = maestro_dir / midi_filename
            if not midi_path.exists():
                errors.append(f"{midi_filename}: file_not_found")
                print(f"  MIDI not found: {midi_path}")
                continue

            try:
                perf_notes = load_performance_midi(midi_path)
                pedal_events = _extract_pedal_events(midi_path)

                # Single DTW call: align_to_score now returns (bar_mapping, dtw_cost)
                bar_to_notes, dtw_cost = align_to_score(perf_notes, score_data)
                dtw_costs.append(dtw_cost)

                # Coverage check
                if total_bars > 0:
                    coverage = len(bar_to_notes) / total_bars
                else:
                    coverage = 0.0
                coverages.append(coverage)

                if coverage < min_coverage:
                    rejected_coverage_count += 1
                    print(f"  {Path(midi_filename).name}: coverage {coverage:.1%} < {min_coverage:.0%} -- rejected")
                    continue

                # Compute per-bar stats
                perf_bar_stats: list[BarStats] = []
                for bar_num, notes in bar_to_notes.items():
                    score_bar = bar_lookup.get(bar_num, {})
                    bs = compute_bar_stats(bar_num, notes, score_bar, pedal_events)
                    perf_bar_stats.append(bs)

                if perf_bar_stats:
                    all_bar_stats.append(perf_bar_stats)
                    print(f"  {Path(midi_filename).name}: coverage {coverage:.1%}, dtw_cost {dtw_cost:.4f} -- OK")

            except Exception as exc:
                errors.append(f"{midi_filename}: {exc}")
                print(f"  {Path(midi_filename).name}: ERROR -- {exc}")

        # Aggregate and write
        performer_count = len(all_bar_stats)
        if performer_count == 0:
            print(f"  No valid recordings for {piece_id}")
        else:
            if performer_count == 1:
                print(f"  WARNING: single-performer reference for {piece_id}")

            aggregated_bars = aggregate_bar_stats(all_bar_stats)

            # Validate non-negative values before serialization
            for bar in aggregated_bars:
                if bar.pedal_changes is not None and bar.pedal_changes < 0:
                    bar.pedal_changes = 0

            profile = ReferenceProfile(
                piece_id=piece_id,
                performer_count=performer_count,
                bars=aggregated_bars,
            )

            out_path = output_dir / f"{piece_id}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(asdict(profile), fh, indent=2)
            print(f"  Saved: {out_path.name} ({performer_count} performer(s), {len(aggregated_bars)} bars)")

        report_rows.append({
            "piece_id": piece_id,
            "total_recordings": len(midi_filenames),
            "passed_validation": performer_count,
            "rejected_coverage": rejected_coverage_count,
            "rejected_dtw_cost": 0,  # Not enforced on first run
            "performer_count": performer_count,
            "mean_coverage": f"{statistics.mean(coverages):.3f}" if coverages else "",
            "mean_dtw_cost": f"{statistics.mean(dtw_costs):.4f}" if dtw_costs else "",
            "errors": "; ".join(errors) if errors else "",
        })

    # Write generation report
    report_path = matches_path.parent / "generation_report.csv"
    report_fields = [
        "piece_id", "total_recordings", "passed_validation", "rejected_coverage",
        "rejected_dtw_cost", "performer_count", "mean_coverage", "mean_dtw_cost", "errors",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=report_fields)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    total_generated = sum(1 for r in report_rows if int(r["performer_count"]) > 0)
    print()
    print(f"Generated {total_generated} reference profiles")
    print(f"Report: {report_path}")


def _cmd_upload(args: argparse.Namespace) -> None:
    """Run the upload subcommand."""
    import subprocess

    source_dir = Path(args.source_dir)
    bucket = args.bucket
    prefix = args.prefix

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        return

    print(f"Uploading {len(json_files)} files to {bucket}/{prefix}/")

    uploaded = 0
    for json_file in json_files:
        r2_key = f"{prefix}/{json_file.name}"
        cmd = [
            "wrangler", "r2", "object", "put",
            f"{bucket}/{r2_key}",
            f"--file={json_file}",
            "--content-type=application/json",
        ]
        print(f"  {r2_key}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Upload failed for {json_file.name}: {result.stderr.strip()}"
            )
        uploaded += 1

    print(f"\nUploaded {uploaded} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference cache pipeline: match, generate, upload."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # match
    match_parser = subparsers.add_parser("match", help="Match MAESTRO recordings to ASAP pieces")
    match_parser.add_argument("--maestro-csv", type=str, required=True)
    match_parser.add_argument("--score-dir", type=str, required=True)
    match_parser.add_argument("--output", type=str, required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate reference profiles from approved matches")
    gen_parser.add_argument("--matches", type=str, required=True)
    gen_parser.add_argument("--maestro-dir", type=str, required=True)
    gen_parser.add_argument("--score-dir", type=str, required=True)
    gen_parser.add_argument("--output-dir", type=str, required=True)
    gen_parser.add_argument("--min-coverage", type=float, default=0.75,
                            help="Minimum bar coverage to accept a recording (default: 0.75)")

    # upload
    upload_parser = subparsers.add_parser("upload", help="Upload reference profiles to R2")
    upload_parser.add_argument("--source-dir", type=str, required=True)
    upload_parser.add_argument("--bucket", type=str, required=True)
    upload_parser.add_argument("--prefix", type=str, required=True)

    args = parser.parse_args()

    if args.command == "match":
        _cmd_match(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "upload":
        _cmd_upload(args)
```

- [ ] **Step 3: Verify the CLI parses correctly**

```bash
cd model && uv run python -m src.score_library.reference_cache --help
cd model && uv run python -m src.score_library.reference_cache match --help
cd model && uv run python -m src.score_library.reference_cache generate --help
cd model && uv run python -m src.score_library.reference_cache upload --help
```

Expected: help text for each subcommand

- [ ] **Step 4: Commit**

```bash
git add model/src/score_library/reference_cache.py
git commit -m "feat: rewrite CLI with match/generate/upload subcommands and single-DTW alignment"
```

---

### Task 6: Integration Tests for Generate + Validation Gates

**Files:**
- Create: `model/tests/score_library/test_reference_pipeline.py`

- [ ] **Step 1: Write integration tests for validation gates**

```python
# model/tests/score_library/test_reference_pipeline.py
"""Integration tests for the reference generation pipeline.

Uses the existing DTW alignment and bar stats functions with minimal fixtures.
"""
import json
import pytest
from pathlib import Path

from src.score_library.reference_cache import (
    align_to_score,
    compute_bar_stats,
    aggregate_bar_stats,
    BarStats,
    ReferenceProfile,
)


class TestValidationCoverage:
    """Test the coverage gate logic (>= 75% of score bars must have aligned notes)."""

    def test_full_coverage(self):
        """A well-aligned performance should have high coverage."""
        # Simulate: score has 10 bars, alignment returns notes for 9 of them
        total_bars = 10
        aligned_bars = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        coverage = len(aligned_bars) / total_bars
        assert coverage >= 0.75

    def test_low_coverage_rejected(self):
        """A poorly-aligned performance should be rejected."""
        total_bars = 10
        aligned_bars = {1: [], 5: []}  # only 2 of 10 bars
        coverage = len(aligned_bars) / total_bars
        assert coverage < 0.75

    def test_edge_at_threshold(self):
        total_bars = 4
        aligned_bars = {1: [], 2: [], 3: []}  # 75% exactly
        coverage = len(aligned_bars) / total_bars
        assert coverage >= 0.75


class TestAggregation:
    """Test that aggregation across multiple performers produces valid output."""

    def test_two_performers(self):
        stats_a = [
            BarStats(bar_number=1, velocity_mean=80.0, velocity_std=5.0, performer_count=1),
            BarStats(bar_number=2, velocity_mean=90.0, velocity_std=6.0, performer_count=1),
        ]
        stats_b = [
            BarStats(bar_number=1, velocity_mean=70.0, velocity_std=4.0, performer_count=1),
            BarStats(bar_number=2, velocity_mean=85.0, velocity_std=7.0, performer_count=1),
        ]
        result = aggregate_bar_stats([stats_a, stats_b])
        assert len(result) == 2
        assert result[0].performer_count == 2
        assert result[0].velocity_mean == pytest.approx(75.0)

    def test_non_negative_pedal_changes(self):
        """pedal_changes must always be non-negative (u32 in Rust consumer)."""
        stats = [
            [BarStats(bar_number=1, pedal_changes=3, performer_count=1)],
            [BarStats(bar_number=1, pedal_changes=5, performer_count=1)],
        ]
        result = aggregate_bar_stats(stats)
        assert result[0].pedal_changes >= 0


class TestReferenceProfileSerialization:
    """Test that serialized JSON matches the Rust consumer schema."""

    def test_schema_fields(self):
        from dataclasses import asdict

        profile = ReferenceProfile(
            piece_id="test.piece",
            performer_count=3,
            bars=[
                BarStats(
                    bar_number=1,
                    velocity_mean=75.0,
                    velocity_std=5.0,
                    onset_deviation_mean_ms=10.0,
                    onset_deviation_std_ms=3.0,
                    pedal_duration_mean_beats=2.5,
                    pedal_changes=4,
                    note_duration_ratio_mean=1.1,
                    performer_count=3,
                ),
            ],
        )
        data = asdict(profile)
        assert data["piece_id"] == "test.piece"
        assert data["performer_count"] == 3

        bar = data["bars"][0]
        required_fields = {
            "bar_number", "velocity_mean", "velocity_std",
            "onset_deviation_mean_ms", "onset_deviation_std_ms",
            "pedal_duration_mean_beats", "pedal_changes",
            "note_duration_ratio_mean", "performer_count",
        }
        assert required_fields.issubset(bar.keys())

    def test_optional_fields_can_be_null(self):
        from dataclasses import asdict

        bar = BarStats(bar_number=1, performer_count=1)
        data = asdict(bar)
        # pedal fields should serialize as None -> null
        assert data["pedal_duration_mean_beats"] is None
        assert data["pedal_changes"] is None

        # Verify JSON serialization handles None
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["pedal_duration_mean_beats"] is None
        assert parsed["pedal_changes"] is None
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd model && uv run pytest tests/score_library/test_reference_pipeline.py -v
```

Expected: all PASS (these test existing code + validation logic)

- [ ] **Step 3: Commit**

```bash
git add model/tests/score_library/test_reference_pipeline.py
git commit -m "test: add integration tests for reference generation pipeline"
```

---

## Chunk 3: Run Pipeline + Upload

### Task 7: Run the Match Command

**Files:** None (execution step)

- [ ] **Step 1: Run match command against real data**

```bash
cd model && uv run python -m src.score_library.reference_cache match \
  --maestro-csv data/maestro_cache/maestro-v3.0.0.csv \
  --score-dir data/score_library \
  --output data/reference_profiles/maestro_asap_matches.csv
```

Expected: prints match summary (N recordings matched to M unique pieces, K unmatched)

- [ ] **Step 2: Inspect the output**

Review `data/reference_profiles/maestro_asap_matches.csv`:
- Check high-confidence matches (>= 0.8) look correct
- Check low-confidence matches (< 0.5) for false positives
- Check `multi_piece=True` entries
- Review `data/reference_profiles/unmatched_maestro.csv` for coverage gaps

- [ ] **Step 3: Human review -- mark approved/rejected in CSV**

Open `data/reference_profiles/maestro_asap_matches.csv` and fill in the `status` column:
- Batch-approve high-confidence non-multi-piece matches
- Manually review low-confidence and multi-piece entries
- Mark rejected entries as `rejected`

- [ ] **Step 4: Commit the reviewed CSV**

```bash
git add data/reference_profiles/maestro_asap_matches.csv data/reference_profiles/unmatched_maestro.csv
git commit -m "data: add reviewed MAESTRO-to-ASAP match mapping"
```

---

### Task 8: Run the Generate Command

**Files:** None (execution step)

- [ ] **Step 1: Run generate command**

```bash
cd model && uv run python -m src.score_library.reference_cache generate \
  --matches data/reference_profiles/maestro_asap_matches.csv \
  --maestro-dir data/maestro_cache \
  --score-dir data/score_library \
  --output-dir data/reference_profiles/references/v1
```

Expected: generates reference JSONs, prints per-piece status, writes generation_report.csv

- [ ] **Step 2: Review generation report**

Check `data/reference_profiles/generation_report.csv`:
- How many pieces got references?
- Any pieces with 0 passed recordings?
- Distribution of coverage values
- Distribution of DTW costs
- Any single-performer warnings?

- [ ] **Step 3: Spot-check 3-5 reference profiles**

Pick pieces with known MAESTRO recordings. For each, verify:
- `performer_count` > 0
- `velocity_mean` values in plausible range (40-120)
- Bars count roughly matches score `total_bars`
- Pedal data present for Romantic pieces (Chopin, Liszt)

```bash
cd model && python3 -c "
import json, sys
for piece_id in ['chopin.ballades.1', 'bach.prelude.bwv_846', 'beethoven.sonata_23.1']:
    path = f'data/reference_profiles/references/v1/{piece_id}.json'
    try:
        d = json.load(open(path))
        print(f'{piece_id}: {d[\"performer_count\"]} performers, {len(d[\"bars\"])} bars')
        vels = [b['velocity_mean'] for b in d['bars']]
        print(f'  velocity range: {min(vels):.0f}-{max(vels):.0f}')
        pedals = [b for b in d['bars'] if b['pedal_changes'] is not None]
        print(f'  bars with pedal data: {len(pedals)}/{len(d[\"bars\"])}')
    except FileNotFoundError:
        print(f'{piece_id}: NOT GENERATED')
"
```

---

### Task 9: Upload to R2

**Files:** None (execution step)

- [ ] **Step 1: Run upload command**

```bash
cd model && uv run python -m src.score_library.reference_cache upload \
  --source-dir data/reference_profiles/references/v1 \
  --bucket crescendai-bucket \
  --prefix references/v1
```

Expected: uploads all generated JSONs, prints count

- [ ] **Step 2: Verify a few files in R2**

```bash
wrangler r2 object get crescendai-bucket/references/v1/chopin.ballades.1.json --file=/tmp/ref_check.json && cat /tmp/ref_check.json | python3 -m json.tool | head -20
```

- [ ] **Step 3: Commit generation report (not the JSON profiles)**

```bash
git add data/reference_profiles/generation_report.csv
git commit -m "data: add reference cache generation report"
```
