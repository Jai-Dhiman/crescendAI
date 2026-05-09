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


import json

from teacher_model.stage1.render import render
from teacher_model.stage1.schema import (
    Stage1AssistantTurn,
    Stage1Example,
    Stage1TextBlock,
    Stage1ToolUseBlock,
)


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        self.calls.append(
            {"messages": messages, "tools": tools, "add_generation_prompt": add_generation_prompt}
        )
        return f"<rendered:{len(messages)}msgs:{len(tools or [])}tools>"


def test_render_passes_messages_and_tools_to_apply_chat_template():
    example = Stage1Example(
        shape="chat",
        system_blocks=["UNIFIED_TEACHER_SYSTEM"],
        messages=[
            {"role": "user", "content": "show me bars 5-8 of chopin.ballades.1"},
        ],
        assistant=Stage1AssistantTurn(
            content=[
                Stage1TextBlock(text="Here you go."),
                Stage1ToolUseBlock(
                    id="t1",
                    name="score_highlight",
                    input={
                        "piece_id": "chopin.ballades.1",
                        "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
                    },
                ),
            ]
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "score_highlight", "description": "stub", "parameters": {}},
        }
    ]
    tok = _RecordingTokenizer()

    out = render(example, tok, tools)

    assert "rendered" in out
    assert len(tok.calls) == 1
    call = tok.calls[0]
    assert call["tools"] == tools
    assert call["messages"][0]["role"] == "system"
    assert "UNIFIED_TEACHER_SYSTEM" in call["messages"][0]["content"]
    assert call["messages"][1]["role"] == "user"
    assert call["messages"][-1]["role"] == "assistant"
    assert call["add_generation_prompt"] is False

    assistant_msg = call["messages"][-1]
    assert "tool_calls" in assistant_msg
    args_str = assistant_msg["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args_str, str), (
        f"arguments must be JSON-encoded string, got {type(args_str).__name__}"
    )
    assert json.loads(args_str) == {
        "piece_id": "chopin.ballades.1",
        "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
    }
