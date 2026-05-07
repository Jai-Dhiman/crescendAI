"""HF publish behavior tests."""
import json
from pathlib import Path

import pytest

from teacher_model.cpt_pipeline.hf_publish import run_publish


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_missing_hf_token_raises_runtime_error(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [{"doc_id": "a", "source": "youtube:tonebase", "text": "Some content here that is long enough.", "word_count": 7}])
    _write_manifest(val, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run_publish(train, val, repo_id="Jai-D/test-repo", private=True)
