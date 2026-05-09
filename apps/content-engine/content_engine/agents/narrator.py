"""Narrator: generates <=45-second voiceover scripts via Claude Code CLI."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
from content_engine.adapters.llm_gateway import LlmMode, LlmResponse
from content_engine.agents.observation_selector import Observation
from content_engine.render.templates import CtaTemplate


_MAX_WORDS = 120


@dataclass(frozen=True)
class ScriptText:
    text: str
    word_count: int


class _LlmProtocol(Protocol):
    def complete(self, prompt: str, mode: LlmMode, schema: dict[str, Any] | None = None) -> LlmResponse: ...


class Narrator:
    def __init__(self, llm: _LlmProtocol):
        self._llm = llm

    def write_script(
        self,
        observation: Observation,
        cta_template: CtaTemplate,
        style_examples: list[str],
    ) -> ScriptText:
        prompt = self._build_prompt(observation, cta_template, style_examples)
        resp = self._llm.complete(prompt=prompt, mode=LlmMode.NARRATOR)
        words = resp.text.split()
        truncated_words = words[:_MAX_WORDS]
        text = " ".join(truncated_words)
        return ScriptText(text=text, word_count=len(truncated_words))

    @staticmethod
    def _build_prompt(observation: Observation, cta_template: CtaTemplate, style_examples: list[str]) -> str:
        cta_section = (
            f"End the script with this spoken CTA verbatim: {cta_template.spoken_cta!r}."
            if cta_template.spoken_cta
            else "Do not include a spoken CTA — phase A has no spoken CTA."
        )
        examples_section = ""
        if style_examples:
            examples_section = "Style references (match tone, not content):\n" + "\n---\n".join(style_examples)
        return (
            "Write a YouTube Shorts voiceover script for crescendai. "
            f"Maximum {_MAX_WORDS} words (about 45 seconds spoken). "
            "Structure: hook in first 2 sentences, then the observation with audio-proof callout, then close. "
            f"\n\nObservation: {observation.plain_english}\n"
            f"Dimension: {observation.dimension}\n"
            f"Time range in clip: {observation.time_range[0]:.1f}s - {observation.time_range[1]:.1f}s\n\n"
            f"{cta_section}\n\n"
            f"{examples_section}"
        )
