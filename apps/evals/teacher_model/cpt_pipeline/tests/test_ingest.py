"""Ingest behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.ingest import run_ingest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_doc_id_is_filename_stem(tiny_corpus, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = _read_jsonl(manifest_path)
    doc_ids = {r["doc_id"] for r in rows}
    assert "abcdefghijk" in doc_ids
    assert "lmnopqrstuv" in doc_ids
    # No row has a doc_id ending in .txt
    assert all(not r["doc_id"].endswith(".txt") for r in rows)
