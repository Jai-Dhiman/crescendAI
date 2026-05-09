import json
from pathlib import Path

import pytest

from teacher_model.stage1.briefing_source import iter_synthesis_briefings


def test_iter_synthesis_briefings_yields_one_per_file(tmp_path: Path):
    briefing_a = {
        "briefing_id": "rec_aaa",
        "framing_text": "<session_data>{...A...}</session_data>",
        "composer": "Chopin",
        "skill_bucket": "intermediate",
    }
    briefing_b = {
        "briefing_id": "rec_bbb",
        "framing_text": "<session_data>{...B...}</session_data>",
        "composer": "Bach",
        "skill_bucket": "beginner",
    }
    (tmp_path / "rec_aaa.json").write_text(json.dumps(briefing_a))
    (tmp_path / "rec_bbb.json").write_text(json.dumps(briefing_b))

    yielded = sorted(iter_synthesis_briefings(tmp_path), key=lambda b: b.briefing_id)
    assert [b.briefing_id for b in yielded] == ["rec_aaa", "rec_bbb"]
    assert yielded[0].framing_text == briefing_a["framing_text"]
    assert yielded[0].composer == "Chopin"
    assert yielded[1].skill_bucket == "beginner"


def test_iter_synthesis_briefings_raises_on_missing_field(tmp_path: Path):
    (tmp_path / "bad.json").write_text(
        '{"briefing_id": "rec_bad", "framing_text": "x", "skill_bucket": "beginner"}'
        # "composer" is missing
    )
    with pytest.raises(KeyError):
        list(iter_synthesis_briefings(tmp_path))


from teacher_model.stage1.briefing_source import iter_chat_scenarios

_TEMPLATE = {
    "intents": [
        {"id": "show_bars", "user": "Can you show me bars {bar_start}-{bar_end} of {piece}?"},
        {"id": "find_piece", "user": "What's that {era} piece by {composer}?"},
    ],
    "fillers": {
        "bar_start": [1, 5, 12],
        "bar_end": [4, 8, 16],
        "piece": ["Chopin Ballade 1", "Bach Prelude in C"],
        "era": ["Romantic", "Baroque"],
        "composer": ["Chopin", "Bach"],
    },
}


def test_iter_chat_scenarios_n_and_determinism():
    a = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=42))
    b = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=42))
    c = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=99))

    assert len(a) == 10
    assert all(b_item.shape == "chat" for b_item in a)
    assert [x.briefing_id for x in a] == [x.briefing_id for x in b]
    assert [x.framing_text for x in a] == [x.framing_text for x in b]
    assert [x.framing_text for x in a] != [x.framing_text for x in c]
