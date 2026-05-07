"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from teacher_model.dedup import find_duplicates


def _write_corpus_for_3a(rows: list[dict], scratch: Path) -> dict[str, dict]:
    """Materialize each row's text to a .txt file under scratch named by doc_id.
    Returns {doc_id: row} index for downstream lookup."""
    index: dict[str, dict] = {}
    for row in rows:
        doc_id = row["doc_id"]
        (scratch / f"{doc_id}.txt").write_text(row["text"], encoding="utf-8")
        index[doc_id] = row
    return index


def _stage_3a_remove_doc_dups(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """Run existing teacher_model.dedup.find_duplicates on a temp corpus dir.
    Returns (surviving_rows, dup_pairs)."""
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
    """Three-pass dedup. Returns path to output manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    rows, _pairs = _stage_3a_remove_doc_dups(rows)

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
