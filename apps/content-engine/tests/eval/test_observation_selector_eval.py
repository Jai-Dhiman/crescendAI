"""Eval gate: observation_selector dimension accuracy on golden set."""
import json
import os
from pathlib import Path
import pytest
from content_engine.adapters.llm_gateway import LlmGateway
from content_engine.adapters.model_runner import ModelOutput
from content_engine.agents.observation_selector import ObservationSelector, ClipMetadata


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EVAL") != "1",
    reason="eval-set tests run only when RUN_EVAL=1 (use real LLM)",
)


def test_selector_dimension_accuracy_at_least_70_percent():
    cases = json.loads((Path(__file__).parent / "golden_observations.json").read_text())
    gw = LlmGateway(
        cf_gateway_url=os.environ["CF_AI_GATEWAY_URL"],
        cf_token=os.environ["CF_API_TOKEN"],
        claude_bin=os.environ.get("CLAUDE_CODE_BIN", "/usr/local/bin/claude"),
    )
    selector = ObservationSelector(llm=gw)

    correct = 0
    for case in cases:
        mo_dict = case["model_output"]
        mo = ModelOutput(scores=mo_dict["scores"], duration_sec=mo_dict["duration_sec"], raw={})
        obs = selector.select(mo, ClipMetadata(duration_sec=mo.duration_sec))
        if obs.dimension == case["expected_dimension"]:
            correct += 1

    accuracy = correct / len(cases)
    assert accuracy >= 0.70, f"selector accuracy {accuracy:.2f} below 70% threshold"
