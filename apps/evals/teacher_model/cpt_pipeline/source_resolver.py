"""Source resolution for cpt_pipeline."""
from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PDF_PATTERN = re.compile(r"^pdf_[a-f0-9]{12}$")
WEB_PATTERN = re.compile(r"^web_[a-f0-9]{12}$")


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename."""
    stem = Path(filename).stem
    if PDF_PATTERN.match(stem):
        coarse = "academic_pdf"
    elif WEB_PATTERN.match(stem):
        coarse = "web_scrape"
    elif YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
