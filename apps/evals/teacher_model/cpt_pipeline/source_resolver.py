"""Source resolution for cpt_pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PDF_PATTERN = re.compile(r"^pdf_[a-f0-9]{12}$")
WEB_PATTERN = re.compile(r"^web_[a-f0-9]{12}$")


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _extract_youtube_id(url: str) -> str | None:
    if "youtu.be/" in url:
        tail = url.split("youtu.be/", 1)[1]
        return tail.split("?", 1)[0].split("/", 1)[0]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    candidates = qs.get("v", [])
    return candidates[0] if candidates else None


def build_provenance_index(provenance_dir: Path) -> dict[str, str]:
    """Walk `provenance_*.jsonl` files in `provenance_dir` and return a
    mapping of expected filename stems to harvester names.

    For YouTube rows: extracts `v=` (or `youtu.be/`) id from the url.
    For non-YouTube rows: produces both `pdf_{sha256(url)[:12]}` and
    `web_{sha256(url)[:12]}` index entries (the harvester used one or the other,
    we cannot tell from the JSONL alone).
    """
    index: dict[str, str] = {}
    for path in sorted(Path(provenance_dir).glob("provenance_*.jsonl")):
        harvester = path.stem.replace("provenance_", "")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path}: {line!r}") from exc
                url = row.get("url", "")
                vid = _extract_youtube_id(url)
                if vid and YOUTUBE_ID_PATTERN.match(vid):
                    index[vid] = harvester
                else:
                    h = _hash_url(url)
                    index.setdefault(f"pdf_{h}", harvester)
                    index.setdefault(f"web_{h}", harvester)
    return index


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
