"""ObservationSelector: picks one shippable observation from ModelOutput."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
from content_engine.adapters.llm_gateway import LlmMode, LlmResponse
from content_engine.adapters.model_runner import ModelOutput


VALID_DIMENSIONS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]


@dataclass(frozen=True)
class ClipMetadata:
    duration_sec: float


@dataclass(frozen=True)
class Observation:
    dimension: str
    time_range: tuple[float, float]
    plain_english: str


class _LlmProtocol(Protocol):
    def complete(self, prompt: str, mode: LlmMode, schema: dict[str, Any] | None = None) -> LlmResponse: ...


_SCHEMA = {
    "type": "object",
    "required": ["dimension", "time_range", "plain_english"],
    "properties": {
        "dimension": {"type": "string", "enum": VALID_DIMENSIONS},
        "time_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "plain_english": {"type": "string", "minLength": 1},
    },
}


class ObservationSelectorError(Exception):
    pass


class ObservationSelector:
    def __init__(self, llm: _LlmProtocol):
        self._llm = llm

    def select(self, model_output: ModelOutput, metadata: ClipMetadata) -> Observation:
        prompt = self._build_prompt(model_output, metadata)
        resp = self._llm.complete(prompt=prompt, mode=LlmMode.SELECTOR, schema=_SCHEMA)
        if resp.parsed_json is None:
            raise ObservationSelectorError("LLM returned no parsed JSON")
        d = resp.parsed_json
        tr = (float(d["time_range"][0]), float(d["time_range"][1]))
        if not (0.0 <= tr[0] < tr[1] <= metadata.duration_sec):
            raise ObservationSelectorError(f"time_range out of clip bounds: {tr}")
        return Observation(
            dimension=d["dimension"],
            time_range=tr,
            plain_english=d["plain_english"],
        )

    @staticmethod
    def _build_prompt(model_output: ModelOutput, metadata: ClipMetadata) -> str:
        return (
            "You are picking one observation from crescendai's piano performance model output. "
            "Choose the single most concrete + audible observation a layperson could hear once it is pointed out. "
            "Return JSON with dimension, time_range [start_sec, end_sec], plain_english.\n\n"
            f"Clip duration: {metadata.duration_sec:.1f}s\n"
            f"Model scores per dimension (per time slice):\n{model_output.scores}\n"
        )
