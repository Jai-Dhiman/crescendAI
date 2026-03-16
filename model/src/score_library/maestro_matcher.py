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
