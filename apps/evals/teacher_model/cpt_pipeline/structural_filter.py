"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

MIN_CHARS = 100
MAX_NON_ASCII_RATIO = 0.5

REFS_PATTERN = re.compile(r"^(References|Bibliography|Works Cited|REFERENCES)\s*$")


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _strip_references(text: str) -> str:
    """Truncate at the last line matching REFS_PATTERN, drop that line and everything after."""
    lines = text.splitlines(keepends=True)
    last_idx = -1
    for i, line in enumerate(lines):
        if REFS_PATTERN.match(line.strip()):
            last_idx = i
    if last_idx == -1:
        return text
    return "".join(lines[:last_idx]).rstrip() + "\n"


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            if _non_ascii_ratio(text) > MAX_NON_ASCII_RATIO:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_ascii_ratio"}) + "\n")
                continue
            if not _is_english(text):
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_english"}) + "\n")
                continue
            source = row.get("source", "")
            if source.startswith("academic_pdf:"):
                text = _strip_references(text)
            row["text"] = text
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
