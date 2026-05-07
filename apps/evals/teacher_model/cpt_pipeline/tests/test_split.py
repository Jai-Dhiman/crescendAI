"""Split behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.split import run_split


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_stratifies_one_per_source_per_100(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = []
    for i in range(200):
        rows.append({"doc_id": f"A{i:010d}", "source": "youtube:tonebase", "text": "x", "word_count": 1})
    for i in range(150):
        rows.append({"doc_id": f"B{i:010d}", "source": "academic_pdf:openalex", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    train_path, val_path = run_split(manifest_in, out_dir, seed=42)

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    val_yt = [r for r in val_rows if r["source"] == "youtube:tonebase"]
    val_pdf = [r for r in val_rows if r["source"] == "academic_pdf:openalex"]
    assert len(val_yt) == 2, f"200 yt docs -> 2 in val, got {len(val_yt)}"
    assert len(val_pdf) == 1, f"150 pdf docs -> 1 in val, got {len(val_pdf)}"
    assert len(train_rows) + len(val_rows) == 350


def test_small_source_skips_validation(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = []
    for i in range(99):
        rows.append({"doc_id": f"S{i:010d}", "source": "web_scrape:henle", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    _train_path, val_path = run_split(manifest_in, out_dir, seed=42)

    val_rows = _read_jsonl(val_path)
    assert len(val_rows) == 0, f"99-doc source should produce 0 val rows, got {len(val_rows)}"


def test_same_seed_produces_identical_output(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    rows = []
    for i in range(200):
        rows.append({"doc_id": f"D{i:010d}", "source": "youtube:tonebase", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    train_a, val_a = run_split(manifest_in, out_dir_a, seed=42)
    train_b, val_b = run_split(manifest_in, out_dir_b, seed=42)

    assert train_a.read_bytes() == train_b.read_bytes(), "train output not deterministic with same seed"
    assert val_a.read_bytes() == val_b.read_bytes(), "validation output not deterministic with same seed"
