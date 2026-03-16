"""MAESTRO-to-ASAP fuzzy matching engine.

Maps MAESTRO CSV entries (canonical_composer, canonical_title) to ASAP piece IDs
using composer normalization, title normalization, and token Dice similarity.
"""
from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------

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
    r"(?:nos?\.\s*\d+\s*[-\u2013]\s*\d+)|"
    r"(?:\d+\s*[-\u2013]\s*\d+)|"
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
