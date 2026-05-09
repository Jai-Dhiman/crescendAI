# apps/evals/teacher_model/stage0/tests/test_aggregator.py
"""Aggregator: dossier shape, inconsistency flag, error-rate gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.stage0.aggregator import (
    DossierEmissionRefused,
    build_dossier,
)

_SONNET_BASELINE = {
    "dimensions": [
        {"name": "Audible-Specific Corrective Feedback", "mean_outcome": 1.387, "n": 504},
        {"name": "Concrete Artifact Provision", "mean_outcome": 2.164, "n": 511},
        {"name": "Specific Positive Praise", "mean_outcome": 2.834, "n": 513},
        {"name": "Autonomy-Supporting Motivation", "mean_outcome": 2.85, "n": 512},
        {"name": "Scaffolded Guided Discovery", "mean_outcome": 2.195, "n": 512},
        {"name": "Style-Consistent Musical Language", "mean_outcome": 3.0, "n": 513},
        {"name": "Appropriate Tone & Language", "mean_outcome": 3.0, "n": 513},
    ],
    "composite_mean": 2.483,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _synth_row(rid: str, scores: dict[str, int]) -> dict:
    return {
        "recording_id": rid,
        "judge_dimensions": [
            {"criterion": k, "process": v, "outcome": v, "score": v, "evidence": "", "reason": ""}
            for k, v in scores.items()
        ],
        "error": "",
        "routed_provider": "openai",
    }


_FULL_NINE = {
    "Audible-Specific Corrective Feedback": 1,
    "Concrete Artifact Provision": 2,
    "Specific Positive Praise": 3,
    "Autonomy-Supporting Motivation": 3,
    "Scaffolded Guided Discovery": 2,
    "Style-Consistent Musical Language": 3,
    "Appropriate Tone & Language": 3,
    "Taste Defensibility": 1,
    "Adaptation Specificity": 1,
}


def test_dossier_emits_seven_capability_rows(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(
        tool,
        [
            {"case_id": f"p{i}", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
            for i in range(20)
        ]
        + [
            {"case_id": f"n{i}", "expected_call": False, "called": False, "discipline_correct": True, "format_valid": None, "category": "chitchat", "error": ""}
            for i in range(20)
        ],
    )
    mcq.write_text(json.dumps({
        "accuracy": 0.7, "total": 50, "correct": 35,
        "by_topic": {"concepts": {"accuracy": 0.7, "total": 10, "correct": 7}}
    }))
    base.write_text(json.dumps(_SONNET_BASELINE))

    dossier = build_dossier(
        synthesis_jsonl=synth,
        tool_jsonl=tool,
        mcq_json=mcq,
        baseline_aggregate_json=base,
        out_dir=tmp_path / "out",
    )
    names = [c.name for c in dossier.capabilities]
    assert names == [
        "Judgment", "Taste", "Integration", "Voice",
        "Vocabulary", "Tool-calling", "Adaptation",
    ]
    md = (tmp_path / "out" / "capability_dossier.md").read_text()
    assert "Capability dossier" in md
    js = json.loads((tmp_path / "out" / "capability_dossier.json").read_text())
    assert len(js["capabilities"]) == 7


def test_error_rate_gate_refuses_emission_above_five_percent(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    rows = [_synth_row(f"r{i}", _FULL_NINE) for i in range(94)] + [
        {"recording_id": f"r{i}", "judge_dimensions": [], "error": "judge timeout", "routed_provider": "openai"}
        for i in range(94, 100)
    ]
    _write_jsonl(synth, rows)
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({"accuracy": 0.5, "total": 50, "correct": 25, "by_topic": {}}))
    base.write_text(json.dumps(_SONNET_BASELINE))

    with pytest.raises(DossierEmissionRefused, match="error rate"):
        build_dossier(
            synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
            baseline_aggregate_json=base, out_dir=tmp_path / "out",
        )


def test_inconsistency_flag_when_primary_and_corroborator_disagree(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"

    # Vocabulary primary (SCML) at-ceiling (3) but corroborating MCQ concepts =0 -> inconsistency.
    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({
        "accuracy": 0.0, "total": 50, "correct": 0,
        "by_topic": {"concepts": {"accuracy": 0.0, "total": 10, "correct": 0}}
    }))
    base.write_text(json.dumps(_SONNET_BASELINE))

    dossier = build_dossier(
        synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
        baseline_aggregate_json=base, out_dir=tmp_path / "out",
    )
    vocab = next(c for c in dossier.capabilities if c.name == "Vocabulary")
    assert vocab.inconsistency_flag is True


def test_continuation_metrics_appear_in_dossier_when_provided(tmp_path: Path) -> None:
    synth = tmp_path / "synth.jsonl"
    tool = tmp_path / "tool.jsonl"
    mcq = tmp_path / "mcq.json"
    base = tmp_path / "base.json"
    cont = tmp_path / "cont.jsonl"

    _write_jsonl(synth, [_synth_row(f"r{i}", _FULL_NINE) for i in range(20)])
    _write_jsonl(tool, [
        {"case_id": "p1", "expected_call": True, "called": True, "discipline_correct": True, "format_valid": True, "category": None, "error": ""}
    ])
    mcq.write_text(json.dumps({"accuracy": 0.5, "total": 50, "correct": 25, "by_topic": {}}))
    base.write_text(json.dumps(_SONNET_BASELINE))
    _write_jsonl(cont, [
        {"case_id": "p1", "category": "clean", "is_degenerate": False},
        {"case_id": "p2", "category": "refusal", "is_degenerate": True},
        {"case_id": "p3", "category": "repetition", "is_degenerate": True},
        {"case_id": "p4", "category": "clean", "is_degenerate": False},
    ])

    dossier = build_dossier(
        synthesis_jsonl=synth, tool_jsonl=tool, mcq_json=mcq,
        baseline_aggregate_json=base, out_dir=tmp_path / "out",
        continuation_jsonl=cont,
    )
    assert dossier.continuation_degeneracy_rate == 0.5
    assert dossier.continuation_by_category["refusal"] == 1
