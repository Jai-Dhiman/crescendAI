"""Offline unit tests for memory_seeder — no DB or live services required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def test_build_insert_rows_defaults_to_canary_facts():
    from memory_seeder import CANARY_FACTS, build_insert_rows

    rows = build_insert_rows(student_id="student-abc")

    assert len(rows) == len(CANARY_FACTS)
    for row in rows:
        assert row["student_id"] == "student-abc"
        assert row["fact_type"] in ("technical_observation", "repertoire_context", "student_goal")
        assert "valid_at" in row
        assert row["confidence"] in ("high", "medium", "low")
        assert row["evidence"]
        assert row["source_type"] == "eval_seed"


def test_fact_text_is_natural_prose_not_a_token():
    """fact_text must be natural prose (the teacher echoes meaning, not tokens)."""
    from memory_seeder import build_insert_rows

    rows = build_insert_rows("s1")
    texts = " ".join(r["fact_text"] for r in rows).lower()

    # No artificial CANARY_ token leaks into the seeded prose.
    assert "canary_" not in texts
    # The distinctive natural keywords ARE present so recall is assertable.
    assert "rachmaninoff" in texts
    assert "left hand" in texts


def test_keyword_groups_match_facts_and_are_findable_in_prose():
    from memory_seeder import CANARY_FACTS, CANARY_KEYWORD_GROUPS

    assert len(CANARY_KEYWORD_GROUPS) == len(CANARY_FACTS)
    # Each fact's first keyword appears (case-insensitively) in its own prose.
    for fact, group in zip(CANARY_FACTS, CANARY_KEYWORD_GROUPS):
        assert group, "every fact needs at least one assertion keyword"
        assert group[0].lower() in fact["fact_text"].lower()


def test_build_insert_rows_required_columns_present():
    from memory_seeder import build_insert_rows

    required = {
        "id", "student_id", "fact_text", "fact_type",
        "valid_at", "confidence", "evidence", "source_type",
    }
    rows = build_insert_rows("s1")
    assert required.issubset(rows[0].keys())


def test_build_insert_rows_unique_ids():
    from memory_seeder import build_insert_rows

    rows = build_insert_rows("s1")
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Each row must have a unique id"


def test_build_insert_rows_accepts_custom_facts():
    from memory_seeder import build_insert_rows

    custom = [{"fact_text": "natural prose fact", "fact_type": "student_goal", "keywords": ["prose"]}]
    rows = build_insert_rows("s1", custom)
    assert len(rows) == 1
    assert rows[0]["fact_text"] == "natural prose fact"
    assert rows[0]["source_type"] == "eval_seed"
