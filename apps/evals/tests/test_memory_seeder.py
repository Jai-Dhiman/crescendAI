"""Offline unit tests for memory_seeder — no DB or live services required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest


def test_build_insert_rows_returns_one_row_per_token():
    from memory_seeder import build_insert_rows

    rows = build_insert_rows(
        student_id="student-abc",
        tokens=["CANARY_RACHMANINOFF_ETUDE", "CANARY_LEFT_HAND_WEAKNESS"],
    )

    assert len(rows) == 2
    for row in rows:
        assert row["student_id"] == "student-abc"
        assert row["fact_text"].startswith("CANARY_")
        assert row["fact_type"] in ("technical_observation", "repertoire_context", "student_goal")
        assert "valid_at" in row
        assert row["confidence"] in ("high", "medium", "low")
        assert row["evidence"]
        assert row["source_type"]


def test_build_insert_rows_embeds_token_in_fact_text():
    from memory_seeder import build_insert_rows

    rows = build_insert_rows(
        student_id="s1",
        tokens=["CANARY_RACHMANINOFF_ETUDE", "CANARY_LEFT_HAND_WEAKNESS"],
    )

    texts = [r["fact_text"] for r in rows]
    assert any("CANARY_RACHMANINOFF_ETUDE" in t for t in texts)
    assert any("CANARY_LEFT_HAND_WEAKNESS" in t for t in texts)


def test_build_insert_rows_required_columns_present():
    from memory_seeder import build_insert_rows

    required = {
        "student_id", "fact_text", "fact_type",
        "valid_at", "confidence", "evidence", "source_type",
    }
    rows = build_insert_rows("s1", ["CANARY_X"])
    assert required.issubset(rows[0].keys())


def test_build_insert_rows_token_uniqueness():
    from memory_seeder import build_insert_rows

    tokens = ["CANARY_A", "CANARY_B", "CANARY_C"]
    rows = build_insert_rows("s1", tokens)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Each row must have a unique id"


def test_build_insert_rows_idempotent_tokens():
    """Calling build_insert_rows twice with same tokens embeds same token strings."""
    from memory_seeder import build_insert_rows

    tokens = ["CANARY_X", "CANARY_Y"]
    rows1 = build_insert_rows("s1", tokens)
    rows2 = build_insert_rows("s1", tokens)

    texts1 = [r["fact_text"] for r in rows1]
    texts2 = [r["fact_text"] for r in rows2]
    # Token strings must be present in both runs
    for token in tokens:
        assert any(token in t for t in texts1)
        assert any(token in t for t in texts2)
