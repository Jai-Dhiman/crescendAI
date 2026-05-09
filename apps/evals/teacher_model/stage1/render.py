import hashlib
import json
from pathlib import Path
from typing import Any

from teacher_model.stage1.schema import (
    Stage1Example,
    Stage1TextBlock,
    Stage1ToolUseBlock,
)


class TokenizerPinMismatchError(Exception):
    pass


def _hash_files(directory: Path, filenames: list[str]) -> str:
    sha = hashlib.sha256()
    for filename in sorted(filenames):
        path = directory / filename
        if not path.exists():
            raise TokenizerPinMismatchError(
                f"Pinned file missing from tokenizer dir: {filename}"
            )
        sha.update(filename.encode())
        sha.update(b"\0")
        sha.update(path.read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()


def verify_tokenizer_pin(tokenizer_dir: Path, pin_path: Path) -> None:
    pin = json.loads(pin_path.read_text())
    expected = pin["sha256"]
    actual = _hash_files(tokenizer_dir, pin["files"])
    if actual != expected:
        raise TokenizerPinMismatchError(
            f"sha256 mismatch: expected {expected}, got {actual}"
        )


def render(example: Stage1Example, tokenizer, tools: list[dict[str, Any]]) -> str:
    messages: list[dict[str, Any]] = []
    for sys_text in example.system_blocks:
        messages.append({"role": "system", "content": sys_text})
    for msg in example.messages:
        messages.append({"role": msg.role, "content": msg.content})

    assistant_content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in example.assistant.content:
        if isinstance(block, Stage1TextBlock):
            assistant_content.append({"type": "text", "text": block.text})
        elif isinstance(block, Stage1ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        # arguments must be JSON-encoded string, not a dict:
                        # Qwen3/vLLM chat-template parser cannot round-trip raw dicts
                        "arguments": json.dumps(block.input),
                    },
                }
            )

    assistant_msg: dict[str, Any] = {"role": "assistant"}
    if assistant_content:
        assistant_msg["content"] = "".join(
            b["text"] for b in assistant_content if b["type"] == "text"
        )
    else:
        assistant_msg["content"] = ""
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages.append(assistant_msg)

    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
