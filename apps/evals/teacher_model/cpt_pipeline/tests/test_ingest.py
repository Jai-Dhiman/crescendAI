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


def test_source_field_resolves_coarse_and_fine(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"]: r for r in _read_jsonl(manifest_path)}
    assert rows["abcdefghijk"]["source"] == "youtube:tonebase"
    assert rows[f"pdf_{fixture_ids['pdf_h_1']}"]["source"] == "academic_pdf:openalex"


def test_corrupt_file_logged_to_drops(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"] for r in _read_jsonl(manifest_path)}
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert fixture_ids["yt_corrupt"] not in rows, "corrupt doc should not be in manifest"
    assert any(d["doc_id"] == fixture_ids["yt_corrupt"] and d["drop_reason"] == "decode_error" for d in drops)
    # Other docs survived (pipeline continued)
    assert "abcdefghijk" in rows


def test_word_count_matches_whitespace_split(tiny_corpus, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"]: r for r in _read_jsonl(manifest_path)}
    row = rows["abcdefghijk"]
    expected = len(row["text"].split())
    assert row["word_count"] == expected, f"expected {expected}, got {row['word_count']}"
