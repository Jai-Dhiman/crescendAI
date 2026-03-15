"""Human-readable title generation from ASAP directory names."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def load_titles(titles_path: Path | None) -> dict[str, str]:
    """Load piece_id -> title mapping from a JSON file.

    Returns an empty dict if the path is None or the file does not exist.
    """
    if titles_path is None:
        return {}
    if not titles_path.exists():
        logger.warning("Titles file not found: %s", titles_path)
        return {}
    with titles_path.open() as f:
        data = json.load(f)
    logger.info("Loaded %d titles from %s", len(data), titles_path)
    return data


_ABBREVIATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bop[_\s](\d+)", re.IGNORECASE), r"Op. \1"),
    (re.compile(r"\bbwv[_\s](\d+)", re.IGNORECASE), r"BWV \1"),
    (re.compile(r"\bno[_\s](\d+)", re.IGNORECASE), r"No. \1"),
]


_PRESERVE_CASE = {"BWV", "Op.", "No.", "WTC", "S145"}


def _clean_segment(segment: str) -> str:
    """Clean a single path segment: underscores to spaces, apply abbreviations, capitalize."""
    text = segment.replace("_", " ")
    for pattern, replacement in _ABBREVIATION_PATTERNS:
        text = pattern.sub(replacement, text)
    words = text.split()
    capitalized = []
    for w in words:
        if not w:
            continue
        if w in _PRESERVE_CASE or w.rstrip(".") in {"BWV", "Op", "No", "WTC", "S145"}:
            capitalized.append(w)
        elif w[0].isdigit():
            capitalized.append(w)
        else:
            capitalized.append(w.capitalize())
    return " ".join(capitalized)


def clean_title_from_path(piece_dir: Path, base_dir: Path) -> str:
    """Generate a human-readable title from a piece directory path.

    The base_dir is the ASAP root; piece_dir is the leaf directory containing
    score MIDIs. The first component (composer) is stripped.

    Examples:
        Chopin/Etudes_op_10/3 -> "Etudes Op. 10 No. 3"
        Balakirev/Islamey     -> "Islamey"
        Bach/WTC_I/Prelude/1  -> "WTC I - Prelude - 1"
    """
    rel = piece_dir.relative_to(base_dir)
    parts = rel.parts[1:]  # skip composer

    if not parts:
        return _clean_segment(piece_dir.name)

    cleaned = [_clean_segment(p) for p in parts]

    if len(cleaned) == 1:
        return cleaned[0]
    elif len(cleaned) == 2:
        # "Etudes Op. 10" + "3" -> "Etudes Op. 10 No. 3"
        # Only add "No." if the second part is purely a number
        base_title = cleaned[0]
        suffix = cleaned[1]
        if suffix.isdigit():
            return f"{base_title} No. {suffix}"
        return f"{base_title} - {suffix}"
    else:
        return " - ".join(cleaned)
