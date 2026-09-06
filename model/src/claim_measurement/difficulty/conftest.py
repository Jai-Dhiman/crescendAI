"""#175: shared fixtures for the ensemble model-directory tests.

Lives in conftest.py rather than in the package so it can be shared across test
files WITHOUT being importable production code -- build_context.CODE_FILES
enumerates what ships in the image, and test scaffolding must never be on it.

The adapters written here are placeholder JSON, not real LoRA weights: the fast
suite exercises layout discovery and head arithmetic, and the embedder is tested
separately with an injected loader (test_audio_emb_extract.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.score_wav import (
    ADAPTER_SUBDIR,
    HEAD_FILENAME,
    write_ridge_head,
)


@pytest.fixture
def make_model_dir(tmp_path):
    def build(members: dict, layout: str = "members", root=None) -> Path:
        model_dir = Path(root) if root is not None else tmp_path / "model_dir"
        if layout == "legacy":
            if len(members) != 1:
                raise ValueError(
                    f"the legacy layout holds exactly one adapter, got "
                    f"{len(members)}")
            (head,) = members.values()
            _write_adapter(model_dir / ADAPTER_SUBDIR)
            write_ridge_head(model_dir / HEAD_FILENAME, head)
        elif layout == "members":
            for name, head in members.items():
                member = model_dir / "members" / name
                _write_adapter(member / ADAPTER_SUBDIR)
                write_ridge_head(member / HEAD_FILENAME, head)
        else:
            raise ValueError(f"unknown layout {layout!r}")
        return model_dir

    return build


def _write_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 4, "note": "placeholder -- no weights"}))
