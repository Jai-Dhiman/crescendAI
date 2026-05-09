import json
from dataclasses import dataclass, field
from pathlib import Path

from teacher_model.stage1.render import verify_tokenizer_pin
from teacher_model.stage1.schema import validate_tool_input


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    n: int


@dataclass
class HarnessReport:
    metrics: list[Metric] = field(default_factory=list)


def _make_client(endpoint: str):
    raise NotImplementedError(
        "Production: instantiate vLLM OpenAI-compatible client. "
        "Tests inject via monkeypatch."
    )


def run_harness(endpoint: str, holdout_path: Path, tokenizer_pin: Path) -> HarnessReport:
    verify_tokenizer_pin(tokenizer_pin.parent, tokenizer_pin)
    client = _make_client(endpoint)

    rows = [json.loads(line) for line in holdout_path.read_text().splitlines() if line.strip()]

    positives = [r for r in rows if r["kind"] == "positive"]
    negatives = [r for r in rows if r["kind"] == "negative"]

    parse_ok = 0
    parse_total = 0
    selection_ok = 0
    pyd_ok = 0
    pyd_total = 0
    semantic_ok = 0
    multi_tool_emissions = 0
    neg_correct = 0
    pair_correct = 0
    pair_total = 0

    for row in positives:
        completion = client.complete(row["rendered_input"])
        tool_calls = completion.get("tool_calls", [])
        if tool_calls:
            parse_total += 1
            try:
                args = json.loads(tool_calls[0]["function"]["arguments"])
                parse_ok += 1
            except (KeyError, json.JSONDecodeError):
                args = None
            if args is not None:
                pyd_total += 1
                if not validate_tool_input(tool_calls[0]["function"]["name"], args):
                    pyd_ok += 1
                    semantic_ok += 1
                if tool_calls[0]["function"]["name"] == row["expected_tool"]:
                    selection_ok += 1
            if len(tool_calls) > 1:
                multi_tool_emissions += 1

    for row in negatives:
        completion = client.complete(row["rendered_input"])
        if not completion.get("tool_calls"):
            neg_correct += 1

    metrics: list[Metric] = [
        Metric(
            "serving_runtime_parse_rate",
            (parse_ok / parse_total) if parse_total else 0.0,
            parse_total,
        ),
        Metric(
            "tool_selection_accuracy",
            (selection_ok / len(positives)) if positives else 0.0,
            len(positives),
        ),
        Metric(
            "argument_pydantic_validity",
            (pyd_ok / pyd_total) if pyd_total else 0.0,
            pyd_total,
        ),
        Metric(
            "argument_semantic_accuracy",
            (semantic_ok / pyd_total) if pyd_total else 0.0,
            pyd_total,
        ),
        Metric(
            "negative_discrimination",
            (neg_correct / len(negatives)) if negatives else 0.0,
            len(negatives),
        ),
        Metric(
            "multi_tool_emission_distribution",
            (multi_tool_emissions / len(positives)) if positives else 0.0,
            len(positives),
        ),
        Metric(
            "matched_contrast_pair_discrimination",
            (pair_correct / pair_total) if pair_total else 0.0,
            pair_total,
        ),
    ]
    return HarnessReport(metrics=metrics)
