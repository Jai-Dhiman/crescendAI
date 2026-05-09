"""Verifies ObservationSelector returns valid Observation schema."""
import json
from content_engine.adapters.llm_gateway import LlmResponse, LlmMode
from content_engine.adapters.model_runner import ModelOutput
from content_engine.agents.observation_selector import (
    ObservationSelector,
    Observation,
    ClipMetadata,
    VALID_DIMENSIONS,
)


class FakeLlm:
    def __init__(self, response_obj):
        self._obj = response_obj
        self.calls = []

    def complete(self, prompt, mode, schema=None):
        self.calls.append((prompt, mode, schema))
        return LlmResponse(
            text=json.dumps(self._obj),
            parsed_json=self._obj,
        )


def _model_output(duration: float = 15.0) -> ModelOutput:
    return ModelOutput(
        scores={d: [0.5, 0.5] for d in VALID_DIMENSIONS},
        duration_sec=duration,
        raw={},
    )


def test_select_returns_valid_observation():
    fake = FakeLlm({
        "dimension": "phrasing",
        "time_range": [5.2, 7.1],
        "plain_english": "Phrasing peak arrives one beat early.",
    })
    selector = ObservationSelector(llm=fake)
    obs = selector.select(_model_output(15.0), ClipMetadata(duration_sec=15.0))

    assert isinstance(obs, Observation)
    assert obs.dimension in VALID_DIMENSIONS
    assert 0 <= obs.time_range[0] < obs.time_range[1] <= 15.0
    assert obs.plain_english != ""


def test_select_invokes_llm_in_selector_mode_with_schema():
    fake = FakeLlm({
        "dimension": "timing",
        "time_range": [1.0, 3.0],
        "plain_english": "Tempo dips on the dotted figure.",
    })
    selector = ObservationSelector(llm=fake)
    selector.select(_model_output(15.0), ClipMetadata(duration_sec=15.0))

    assert len(fake.calls) == 1
    _, mode, schema = fake.calls[0]
    assert mode == LlmMode.SELECTOR
    assert schema is not None
    assert "dimension" in schema["properties"]
