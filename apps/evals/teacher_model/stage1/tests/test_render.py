import hashlib
import json
from pathlib import Path

import pytest

from teacher_model.stage1.render import (
    TokenizerPinMismatchError,
    verify_tokenizer_pin,
)


def _make_tokenizer_dir(tmp_path: Path, contents: dict[str, str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for filename, body in contents.items():
        (tmp_path / filename).write_text(body)
    return tmp_path


def _make_pin(tmp_path: Path, files: dict[str, str], pin_path: Path) -> Path:
    sha = hashlib.sha256()
    for filename in sorted(files):
        sha.update(filename.encode())
        sha.update(b"\0")
        sha.update(files[filename].encode())
        sha.update(b"\0")
    pin_path.write_text(
        json.dumps(
            {
                "model_id": "qwen/qwen3.6-35b-a3b",
                "files": sorted(files),
                "sha256": sha.hexdigest(),
            }
        )
    )
    return pin_path


def test_verify_tokenizer_pin_succeeds_on_match(tmp_path: Path):
    files = {"tokenizer.json": "{}", "chat_template.jinja": "{{ messages }}"}
    tok_dir = _make_tokenizer_dir(tmp_path / "tok", files)
    pin = _make_pin(tmp_path, files, tmp_path / "pin.json")
    # Should not raise
    verify_tokenizer_pin(tok_dir, pin)


def test_verify_tokenizer_pin_raises_on_mutation(tmp_path: Path):
    files = {"tokenizer.json": "{}", "chat_template.jinja": "{{ messages }}"}
    tok_dir = _make_tokenizer_dir(tmp_path / "tok", files)
    pin = _make_pin(tmp_path, files, tmp_path / "pin.json")
    # Mutate the chat template
    (tok_dir / "chat_template.jinja").write_text("{{ messages | reverse }}")
    with pytest.raises(TokenizerPinMismatchError) as exc_info:
        verify_tokenizer_pin(tok_dir, pin)
    assert "chat_template.jinja" in str(exc_info.value) or "sha256" in str(exc_info.value)
