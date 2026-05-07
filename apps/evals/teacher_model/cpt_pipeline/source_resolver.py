"""Source resolution for cpt_pipeline.

Classifies corpus files into a `<coarse>:<fine>` source string.
Coarse source comes from filename pattern; fine source from URL-hash / video-id
lookup against provenance JSONLs.
"""
from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename.

    Coarse classification:
      - 11-char alphanumeric stem -> "youtube"
      - else -> "unknown"
    Fine source comes from `provenance_index[stem]`; "unknown" if absent.
    """
    stem = Path(filename).stem
    if YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
