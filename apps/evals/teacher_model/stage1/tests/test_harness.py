import json
from pathlib import Path

from teacher_model.stage1.harness import HarnessReport, run_harness


def _write_holdout(tmp_path: Path) -> Path:
    holdout = []
    # 4 held-out examples: 2 positives (one tool selection right, one wrong),
    # 2 negatives (one model correctly stays silent, one model wrongly tool-calls).
    holdout.append(
        {
            "id": "h1",
            "kind": "positive",
            "shape": "synthesis",
            "expected_tool": "create_exercise",
            "expected_input": {
                "source_passage": "bars 5-8",
                "target_skill": "voice balance",
                "exercises": [
                    {"title": "x", "instruction": "x", "focus_dimension": "dynamics"}
                ],
            },
            "rendered_input": "<rendered:positive_h1>",
        }
    )
    holdout.append(
        {
            "id": "h2",
            "kind": "positive",
            "shape": "chat",
            "expected_tool": "score_highlight",
            "expected_input": {
                "piece_id": "chopin.ballades.1",
                "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
            },
            "rendered_input": "<rendered:positive_h2>",
        }
    )
    holdout.append(
        {
            "id": "h3",
            "kind": "negative",
            "shape": "chat",
            "category": "chitchat",
            "rendered_input": "<rendered:negative_h3>",
        }
    )
    holdout.append(
        {
            "id": "h4",
            "kind": "negative",
            "shape": "chat",
            "category": "premature",
            "rendered_input": "<rendered:negative_h4>",
        }
    )
    path = tmp_path / "holdout.jsonl"
    path.write_text("\n".join(json.dumps(h) for h in holdout))
    return path


def _write_pin(tmp_path: Path) -> Path:
    pin = tmp_path / "tokenizer_pin.json"
    pin.write_text(json.dumps({"model_id": "stub", "files": [], "sha256": "stub"}))
    return pin


class _StubVLLM:
    """Returns canned tool_calls for each held-out id."""

    _RESPONSES = {
        "<rendered:positive_h1>": {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "create_exercise",
                        "arguments": json.dumps(
                            {
                                "source_passage": "bars 5-8",
                                "target_skill": "voice balance",
                                "exercises": [
                                    {
                                        "title": "x",
                                        "instruction": "x",
                                        "focus_dimension": "dynamics",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
            "text": "Nice work.",
        },
        "<rendered:positive_h2>": {
            # WRONG tool name: model emitted reference_browser instead of score_highlight
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "reference_browser",
                        "arguments": json.dumps({"description": "stub"}),
                    },
                }
            ],
            "text": "",
        },
        "<rendered:negative_h3>": {
            "tool_calls": [],  # correct: stayed silent
            "text": "You're welcome.",
        },
        "<rendered:negative_h4>": {
            # incorrect: emitted a tool when context didn't warrant
            "tool_calls": [
                {
                    "id": "c4",
                    "function": {
                        "name": "create_exercise",
                        "arguments": json.dumps(
                            {
                                "source_passage": "x",
                                "target_skill": "x",
                                "exercises": [
                                    {
                                        "title": "x",
                                        "instruction": "x",
                                        "focus_dimension": "dynamics",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
            "text": "",
        },
    }

    def complete(self, rendered_input: str) -> dict:
        return self._RESPONSES[rendered_input]


def test_run_harness_computes_seven_metrics(tmp_path: Path, monkeypatch):
    holdout = _write_holdout(tmp_path)
    pin = _write_pin(tmp_path)

    # Inject the stub via the harness's client factory hook:
    from teacher_model.stage1 import harness as harness_module

    monkeypatch.setattr(harness_module, "_make_client", lambda endpoint: _StubVLLM())

    # Bypass tokenizer pin verification (test pin is a stub):
    monkeypatch.setattr(harness_module, "verify_tokenizer_pin", lambda *a, **k: None)

    report: HarnessReport = run_harness(endpoint="http://stub", holdout_path=holdout, tokenizer_pin=pin)

    # All 7 metrics present
    metric_names = {m.name for m in report.metrics}
    assert metric_names == {
        "serving_runtime_parse_rate",
        "tool_selection_accuracy",
        "argument_pydantic_validity",
        "argument_semantic_accuracy",
        "negative_discrimination",
        "multi_tool_emission_distribution",
        "matched_contrast_pair_discrimination",
    }

    # 2 positive completions, both parsed -> 100% parse rate
    parse = next(m for m in report.metrics if m.name == "serving_runtime_parse_rate")
    assert parse.value == 1.0

    # h1 correct, h2 wrong tool -> 50% selection
    sel = next(m for m in report.metrics if m.name == "tool_selection_accuracy")
    assert sel.value == 0.5

    # both emitted tool inputs are schema-valid -> 100% pydantic validity
    pyd = next(m for m in report.metrics if m.name == "argument_pydantic_validity")
    assert pyd.value == 1.0

    # 2 negatives: h3 correct (no tool), h4 wrong (emitted tool) -> 50% discrimination
    neg = next(m for m in report.metrics if m.name == "negative_discrimination")
    assert neg.value == 0.5
