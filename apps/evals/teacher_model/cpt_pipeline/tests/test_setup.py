"""Smoke test: package imports and tiny_corpus fixture builds correctly."""
from pathlib import Path


def test_package_imports():
    import teacher_model.cpt_pipeline  # noqa: F401


def test_tiny_corpus_fixture_has_expected_files(tiny_corpus):
    corpus_dir, provenance_dir = tiny_corpus
    txt_files = sorted(Path(corpus_dir).glob("*.txt"))
    jsonl_files = sorted(Path(provenance_dir).glob("provenance_*.jsonl"))
    assert len(txt_files) >= 14, f"expected >=14 fixture .txt files, got {len(txt_files)}"
    assert len(jsonl_files) >= 3, f"expected >=3 fixture provenance JSONLs, got {len(jsonl_files)}"
