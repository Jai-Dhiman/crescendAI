"""Verifies ObservationSelector rejects time_range outside clip duration."""
import json
import pytest
from content_engine.adapters.llm_gateway import LlmResponse
from content_engine.adapters.model_runner import ModelOutput
from content_engine.agents.observation_selector import (
    ObservationSelector,
    ObservationSelectorError,
    ClipMetadata,
    VALID_DIMENSIONS,
)


class FakeLlm:
    def __init__(self, obj):
        self._obj = obj

    def complete(self, prompt, mode, schema=None):
        return LlmResponse(text=json.dumps(self._obj), parsed_json=self._obj)


def _model_output() -> ModelOutput:
    return ModelOutput(
        scores={d: [0.5] for d in VALID_DIMENSIONS},
        duration_sec=10.0,
        raw={},
    )


def test_time_range_beyond_duration_raises():
    fake = FakeLlm({"dimension": "phrasing", "time_range": [5.0, 12.0], "plain_english": "x"})
    selector = ObservationSelector(llm=fake)
    with pytest.raises(ObservationSelectorError, match="time_range"):
        selector.select(_model_output(), ClipMetadata(duration_sec=10.0))


def test_inverted_time_range_raises():
    fake = FakeLlm({"dimension": "phrasing", "time_range": [7.0, 5.0], "plain_english": "x"})
    selector = ObservationSelector(llm=fake)
    with pytest.raises(ObservationSelectorError, match="time_range"):
        selector.select(_model_output(), ClipMetadata(duration_sec=10.0))
