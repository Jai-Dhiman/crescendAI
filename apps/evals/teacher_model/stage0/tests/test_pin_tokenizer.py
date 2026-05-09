"""Hash stability + mutation sensitivity + chat-template-required behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.stage0.pin_tokenizer import (
    MissingChatTemplateError,
    pin_tokenizer,
)


def test_pin_is_stable_across_two_invocations(tmp_path: Path) -> None:
    """Pinning the same model id twice produces the same hash."""
    out1 = tmp_path / "pin1.json"
    out2 = tmp_path / "pin2.json"
    # gpt2 has a chat_template baked in via tokenizer_config in modern transformers;
    # if not present, the test below will surface MissingChatTemplateError instead.
    # We use a model id that ships a chat template: Qwen/Qwen2.5-0.5B-Instruct.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    pin_a = pin_tokenizer(model_id=model_id, out_path=out1)
    pin_b = pin_tokenizer(model_id=model_id, out_path=out2)
    assert pin_a.sha256 == pin_b.sha256
    assert json.loads(out1.read_text())["sha256"] == pin_a.sha256


def test_pin_changes_when_a_source_file_is_mutated(tmp_path: Path) -> None:
    """Mutating one of the pinned files changes the hash."""
    out = tmp_path / "pin.json"
    pin_a = pin_tokenizer(model_id="Qwen/Qwen2.5-0.5B-Instruct", out_path=out)
    # Mutate the cached tokenizer_config.json file in the cache dir
    cache_dir = Path(pin_a.cache_dir)
    targets = list(cache_dir.rglob("tokenizer_config.json"))
    assert targets, "expected tokenizer_config.json in cache"
    target = targets[0]
    original = target.read_text()
    target.write_text(original + "\n")  # one-byte mutation
    try:
        pin_b = pin_tokenizer(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            out_path=tmp_path / "pin2.json",
            cache_dir=cache_dir,
            force_local=True,
        )
        assert pin_a.sha256 != pin_b.sha256
    finally:
        target.write_text(original)


def test_missing_chat_template_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model with no chat_template must raise MissingChatTemplateError."""
    # Use gpt2 which has no chat template by default in older snapshots.
    # If gpt2 ever ships a chat template, swap to another template-less model id.
    with pytest.raises(MissingChatTemplateError):
        pin_tokenizer(model_id="gpt2", out_path=tmp_path / "pin.json")
