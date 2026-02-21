"""Tests for quote bank construction."""

import pytest


def test_build_quote_bank_basic():
    from masterclass_experiments.quote_bank import build_quote_bank

    moments = [
        {
            "moment_id": "a",
            "feedback_summary": "Too loud in the left hand.",
            "transcript_text": "[10s] You need to balance the hands better.",
            "teacher": "Arie Vardi",
            "severity": "moderate",
            "feedback_type": "correction",
            "piece": "Chopin Ballade No. 1",
            "composer": "Chopin",
        },
        {
            "moment_id": "b",
            "feedback_summary": "Beautiful dynamics here.",
            "transcript_text": "[20s] That crescendo was wonderful.",
            "teacher": "Murray Perahia",
            "severity": "minor",
            "feedback_type": "praise",
            "piece": "Beethoven Sonata",
            "composer": "Beethoven",
        },
    ]
    # Both moments assigned to dimension "dynamics"
    assignments = {"a": "dynamics", "b": "dynamics"}

    bank = build_quote_bank(moments, assignments)

    assert "dynamics" in bank
    assert len(bank["dynamics"]) == 2
    assert bank["dynamics"][0]["teacher"] == "Arie Vardi"
    assert bank["dynamics"][0]["severity"] == "moderate"


def test_build_quote_bank_max_per_dimension():
    from masterclass_experiments.quote_bank import build_quote_bank

    moments = [
        {
            "moment_id": str(i),
            "feedback_summary": f"Feedback {i}",
            "transcript_text": f"[{i}s] Text {i}",
            "teacher": "Teacher",
            "severity": "minor",
            "feedback_type": "suggestion",
            "piece": None,
            "composer": None,
        }
        for i in range(20)
    ]
    assignments = {str(i): "dynamics" for i in range(20)}

    bank = build_quote_bank(moments, assignments, max_per_dim=10)

    assert len(bank["dynamics"]) == 10


def test_build_quote_bank_sorted_by_severity():
    from masterclass_experiments.quote_bank import build_quote_bank

    moments = [
        {
            "moment_id": "a",
            "feedback_summary": "Minor issue.",
            "transcript_text": "[1s] Minor.",
            "teacher": "T",
            "severity": "minor",
            "feedback_type": "suggestion",
            "piece": None,
            "composer": None,
        },
        {
            "moment_id": "b",
            "feedback_summary": "Critical issue.",
            "transcript_text": "[2s] Critical.",
            "teacher": "T",
            "severity": "critical",
            "feedback_type": "correction",
            "piece": None,
            "composer": None,
        },
    ]
    assignments = {"a": "dynamics", "b": "dynamics"}

    bank = build_quote_bank(moments, assignments)

    # Critical should come before minor
    assert bank["dynamics"][0]["severity"] == "critical"
    assert bank["dynamics"][1]["severity"] == "minor"
