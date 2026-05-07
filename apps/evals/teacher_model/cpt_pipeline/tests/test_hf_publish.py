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
        def upload_file(self, **kwargs): pass

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


def test_published_schema_is_exactly_three_columns(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Long enough text content here for ingestion.", "word_count": 8, "extra_internal_field": "should_not_appear"},
    ])
    _write_manifest(val, [])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured_dataset = {}

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
        def upload_file(self, **kwargs): pass
    def fake_push_to_hub(self, repo_id, private, token):
        captured_dataset["features"] = {split: ds.features for split, ds in self.items()}

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    train_features = captured_dataset["features"]["train"]
    assert sorted(train_features.keys()) == ["doc_id", "source", "text"], \
        f"unexpected schema columns: {sorted(train_features.keys())}"


def test_dataset_dict_has_train_and_validation_only(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Long enough text content here for ingestion.", "word_count": 8},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "Validation content here for ingestion.", "word_count": 6},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured = {}

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
        def upload_file(self, **kwargs): pass
    def fake_push(self, repo_id, private, token):
        captured["splits"] = sorted(self.keys())

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    assert captured["splits"] == ["train", "validation"], f"unexpected splits: {captured['splits']}"


def test_dataset_card_includes_counts_and_per_source(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "ten words here ten words here ten words here ten words here", "word_count": 12},
        {"doc_id": "c", "source": "academic_pdf:openalex", "text": "twenty words here twenty words here twenty words here twenty words here twenty words here", "word_count": 20},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "five words here ok content for validation set", "word_count": 8},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
        def upload_file(self, **kwargs): pass
    def fake_push(self, repo_id, private, token): pass

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True, card_out_dir=tmp_path)

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "train" in card.lower()
    assert "validation" in card.lower()
    assert "youtube:tonebase" in card
    assert "academic_pdf:openalex" in card
    # train rows = 2, val rows = 1, total = 3
    assert "3" in card or "2" in card


def test_dataset_card_pushed_to_hub(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Long enough text content here for ingestion.", "word_count": 8},
    ])
    _write_manifest(val, [])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    upload_calls = []

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, token):
            upload_calls.append({"path_in_repo": path_in_repo, "repo_id": repo_id, "repo_type": repo_type})

    def fake_push(self, repo_id, private, token): pass

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    assert any(
        c["path_in_repo"] == "README.md" and c["repo_id"] == "Jai-D/test-repo" and c["repo_type"] == "dataset"
        for c in upload_calls
    ), f"expected upload_file call for README.md, got {upload_calls}"
