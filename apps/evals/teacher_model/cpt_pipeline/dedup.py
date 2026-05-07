"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from teacher_model.dedup import find_duplicates

WITHIN_DOC_REPEAT_THRESHOLD = 3
MIN_LINE_LEN_FOR_STRIP = 30
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


def _normalize_line(line: str) -> str:
    """Lowercase + collapse internal whitespace + strip leading/trailing whitespace."""
    return " ".join(line.lower().split())


def _is_strippable(normalized: str) -> bool:
    """Lines short enough to legitimately repeat (e.g., 'C major') are exempt."""
    return len(normalized) >= MIN_LINE_LEN_FOR_STRIP and not PAGE_NUMBER_RE.match(normalized)


def _strip_within_doc(text: str) -> str:
    """Drop strippable lines that repeat >= WITHIN_DOC_REPEAT_THRESHOLD times within `text`,
    keeping only the first occurrence."""
    lines = text.splitlines(keepends=True)
    counts: Counter[str] = Counter()
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm):
            counts[norm] += 1
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm) and counts[norm] >= WITHIN_DOC_REPEAT_THRESHOLD:
            if norm in seen:
                continue
            seen.add(norm)
        out_lines.append(line)
    return "".join(out_lines)


def _write_corpus_for_3a(rows: list[dict], scratch: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        doc_id = row["doc_id"]
        (scratch / f"{doc_id}.txt").write_text(row["text"], encoding="utf-8")
        index[doc_id] = row
    return index


def _stage_3a_remove_doc_dups(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, float]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir)
        index = _write_corpus_for_3a(rows, scratch)
        pairs = find_duplicates(scratch, threshold=0.8)
        to_remove: set[str] = set()
        for file1, file2, _sim in pairs:
            id1 = Path(file1).stem
            id2 = Path(file2).stem
            keeper, dup = sorted([id1, id2])
            if keeper not in to_remove:
                to_remove.add(dup)
        surviving = [index[doc_id] for doc_id in index if doc_id not in to_remove]
        return surviving, pairs


def run_dedup(manifest_in: Path, out_dir: Path) -> Path:
    """Three-pass dedup."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    rows, _pairs = _stage_3a_remove_doc_dups(rows)
    for row in rows:
        row["text"] = _strip_within_doc(row["text"])

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
