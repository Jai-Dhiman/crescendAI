import json
from pathlib import Path

RUBRICS = Path(__file__).parent / "prompts" / "atomic_skill_rubrics.json"

EXPECTED_MOVE_IDS = {
    "voicing_diagnosis", "pedal_triage", "rubato_coaching",
    "phrasing_arc_analysis", "tempo_stability_triage",
    "dynamic_range_audit", "articulation_clarity_check", "exercise_proposal",
}


def _load():
    return json.loads(RUBRICS.read_text())


def test_eight_moves_present():
    data = _load()
    assert isinstance(data, list)
    assert len(data) == 8
    assert {entry["move_id"] for entry in data} == EXPECTED_MOVE_IDS


def test_each_move_has_five_criteria():
    data = _load()
    for entry in data:
        assert "criteria" in entry
        assert len(entry["criteria"]) == 5
        for crit in entry["criteria"]:
            assert "id" in crit
            assert "text" in crit
            assert isinstance(crit["text"], str)
            assert len(crit["text"]) > 10


def test_each_move_has_applies_when():
    data = _load()
    for entry in data:
        assert "applies_when" in entry
        assert isinstance(entry["applies_when"], str)
