import json
from pathlib import Path

import pytest

from teacher_model.stage1.negatives_loader import (
    NegativeLoadError,
    load_negatives,
    load_pairs,
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


def _pair_dict(contrast_id: str, neg_contrast_id: str | None = None) -> dict:
    neg_id = neg_contrast_id if neg_contrast_id is not None else contrast_id
    return {
        "contrast_id": contrast_id,
        "positive": {
            "shape": "chat",
            "system_blocks": [],
            "messages": [{"role": "user", "content": "show me bars 5-8 of chopin.ballades.1"}],
            "assistant": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "score_highlight",
                        "input": {
                            "piece_id": "chopin.ballades.1",
                            "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
                        },
                    }
                ],
            },
            "metadata": {"contrast_id": contrast_id, "source": "hand"},
        },
        "negative": {
            "shape": "chat",
            "system_blocks": [],
            "messages": [{"role": "user", "content": "show me a Chopin piece sometime"}],
            "assistant": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Sure -- which one would you like to start with?"}],
            },
            "category": "ambiguous",
            "metadata": {"contrast_id": neg_id},
        },
    }


def test_load_pairs_returns_valid(tmp_path: Path):
    (tmp_path / "pair_001.json").write_text(json.dumps(_pair_dict("cp_001")))
    pairs = load_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].contrast_id == "cp_001"


def test_load_pairs_rejects_mismatched_contrast_id(tmp_path: Path):
    (tmp_path / "pair_bad.json").write_text(
        json.dumps(_pair_dict("cp_001", neg_contrast_id="cp_999"))
    )
    with pytest.raises(NegativeLoadError) as exc_info:
        load_pairs(tmp_path)
    assert "pair_bad.json" in str(exc_info.value)
    assert "contrast_id" in str(exc_info.value).lower()
