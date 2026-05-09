import json
from pathlib import Path

import pytest

from teacher_model.stage1.negatives_loader import (
    NegativeLoadError,
    load_negatives,
)


def _valid_negative_dict(category: str = "chitchat") -> dict:
    return {
        "shape": "chat",
        "system_blocks": ["UNIFIED_TEACHER_SYSTEM"],
        "messages": [{"role": "user", "content": "thanks!"}],
        "assistant": {
            "role": "assistant",
            "content": [{"type": "text", "text": "You're welcome."}],
        },
        "category": category,
        "metadata": {"rationale": "social close"},
    }


def test_load_negatives_returns_valid_files(tmp_path: Path):
    (tmp_path / "neg_001.json").write_text(json.dumps(_valid_negative_dict("chitchat")))
    (tmp_path / "neg_002.json").write_text(json.dumps(_valid_negative_dict("premature")))
    loaded = load_negatives(tmp_path)
    assert len(loaded) == 2
    assert {n.category for n in loaded} == {"chitchat", "premature"}


def test_load_negatives_raises_with_filename_on_invalid(tmp_path: Path):
    bad = _valid_negative_dict()
    bad["category"] = "not_a_real_category"
    (tmp_path / "neg_bad.json").write_text(json.dumps(bad))

    with pytest.raises(NegativeLoadError) as exc_info:
        load_negatives(tmp_path)
    assert "neg_bad.json" in str(exc_info.value)
    assert "category" in str(exc_info.value)
