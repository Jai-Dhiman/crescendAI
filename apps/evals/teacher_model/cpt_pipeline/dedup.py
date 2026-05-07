"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from teacher_model.dedup import find_duplicates

WITHIN_DOC_REPEAT_THRESHOLD = 3
CORPUS_WIDE_LINE_DOC_THRESHOLD = 20
MIN_LINE_LEN_FOR_STRIP = 30
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


def _normalize_line(line: str) -> str:
    return " ".join(line.lower().split())


def _is_strippable(normalized: str) -> bool:
    return len(normalized) >= MIN_LINE_LEN_FOR_STRIP and not PAGE_NUMBER_RE.match(normalized)


def _strip_within_doc(text: str) -> str:
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


def _build_global_line_doc_counts(rows: list[dict]) -> Counter[str]:
    """Count how many distinct docs contain each strippable normalized line."""
    counts: Counter[str] = Counter()
    for row in rows:
        seen: set[str] = set()
        for line in row["text"].splitlines():
            norm = _normalize_line(line)
            if _is_strippable(norm):
                seen.add(norm)
        for norm in seen:
            counts[norm] += 1
    return counts


def _strip_corpus_wide(text: str, banned: set[str]) -> str:
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        norm = _normalize_line(line)
        if norm in banned:
            continue
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
        pairs = find_duplicates(scratch, threshold=0.93, num_perm=512)
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
    """Three-pass dedup.

    Pass order: 3c (corpus-wide boilerplate strip) -> 3a (doc-level near-dup removal)
    -> 3b (within-doc repeated line strip).

    3c runs first so that corpus-wide boilerplate does not inflate MinHash similarity
    and cause legitimate documents to be incorrectly collapsed by 3a.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    # 3c: compute banned lines from the full corpus, strip them globally
    line_doc_counts = _build_global_line_doc_counts(rows)
    banned = {norm for norm, c in line_doc_counts.items() if c > CORPUS_WIDE_LINE_DOC_THRESHOLD}
    for row in rows:
        row["text"] = _strip_corpus_wide(row["text"], banned)

    # 3a: remove whole-doc near-duplicates (now that boilerplate is stripped)
    rows, _pairs = _stage_3a_remove_doc_dups(rows)

    # 3b: strip within-doc repeated lines
    for row in rows:
        row["text"] = _strip_within_doc(row["text"])

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
