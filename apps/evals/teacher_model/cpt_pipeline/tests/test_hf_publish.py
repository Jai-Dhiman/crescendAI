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


def test_push_uses_correct_repo_args(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Some content here that is long enough.", "word_count": 7},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "Validation content here that is long enough.", "word_count": 7},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured = {}

    class FakeHfApi:
        def create_repo(self, repo_id, private, repo_type, exist_ok, token):
            captured["create"] = {
                "repo_id": repo_id, "private": private, "repo_type": repo_type,
                "exist_ok": exist_ok, "token": token,
            }
        def repo_info(self, repo_id, repo_type, token):
            captured["info"] = {"repo_id": repo_id, "repo_type": repo_type}
            class _R:
                pass
            return _R()

    captured_push = {}
    def fake_push_to_hub(self, repo_id, private, token):
        captured_push["repo_id"] = repo_id
        captured_push["private"] = private
        captured_push["token"] = token

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub)

    url = run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    assert captured["create"]["repo_id"] == "Jai-D/test-repo"
    assert captured["create"]["private"] is True
    assert captured["create"]["repo_type"] == "dataset"
    assert captured["create"]["exist_ok"] is True
    assert captured_push["repo_id"] == "Jai-D/test-repo"
    assert captured_push["private"] is True
    assert "Jai-D/test-repo" in url
