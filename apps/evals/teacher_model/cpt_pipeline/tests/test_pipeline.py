"""End-to-end pipeline test."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.pipeline import run_pipeline


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_end_to_end_produces_clean_dataset(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus

    exit_code = run_pipeline([
        "run",
        "--corpus-dir", str(corpus_dir),
        "--provenance-dir", str(provenance_dir),
        "--out-dir", str(tmp_path / "pipeline_out"),
        "--repo-id", "Jai-D/test-repo",
        "--push-disabled",
    ])
    assert exit_code == 0, f"pipeline exited with {exit_code}"

    train = _read_jsonl(tmp_path / "pipeline_out" / "4_split" / "train.jsonl")
    val = _read_jsonl(tmp_path / "pipeline_out" / "4_split" / "validation.jsonl")
    all_rows = train + val
    surviving_ids = {r["doc_id"] for r in all_rows}

    # Dropped expected:
    assert fixture_ids["yt_short"] not in surviving_ids, "short doc not dropped"
    assert fixture_ids["yt_french"] not in surviving_ids, "french doc not dropped"
    assert fixture_ids["yt_nonascii"] not in surviving_ids, "non-ASCII doc not dropped"
    assert fixture_ids["yt_corrupt"] not in surviving_ids, "corrupt doc not dropped"

    # Dedup of two near-dup docs: only one should survive
    dup_ids = {fixture_ids["yt_dup_a"], fixture_ids["yt_dup_b"]}
    surviving_dups = surviving_ids & dup_ids
    assert len(surviving_dups) == 1, f"expected exactly one dup to survive, got {surviving_dups}"

    # 3c boilerplate (legal disclaimer, present in 25 web docs) should be stripped
    boilerplate = "The author and publisher disclaim all such representations and warranties for a particular purpose."
    assert all(boilerplate not in r["text"] for r in all_rows), "boilerplate not stripped corpus-wide"

    # Refs stripped on PDF docs
    pdf_doc = next((r for r in all_rows if r["doc_id"] == f"pdf_{fixture_ids['pdf_h_refs']}"), None)
    if pdf_doc is not None:
        assert "Debussy, C. (1905)" not in pdf_doc["text"], "refs not stripped on academic_pdf doc"
