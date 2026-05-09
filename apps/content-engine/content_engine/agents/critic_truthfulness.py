"""CriticTruthfulness: brand-safety binary kill/pass on observations.

LLM is a first filter; Jai's swipe-UI override is the final word per spec.
Default-pass is NEVER used -- infra failures must surface, not pass through.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any
from content_engine.adapters.llm_gateway import LlmMode, LlmResponse
from content_engine.agents.observation_selector import Observation


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


class CriticTruthfulnessError(Exception):
    pass


class _LlmProtocol(Protocol):
    def complete(self, prompt: str, mode: LlmMode, schema: dict[str, Any] | None = None) -> LlmResponse: ...


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|KILL)", re.IGNORECASE)


class CriticTruthfulness:
    def __init__(self, llm: _LlmProtocol):
        self._llm = llm

    def verify(self, clip_path: Path, observation: Observation) -> Verdict:
        prompt = self._build_prompt(clip_path, observation)
        resp = self._llm.complete(prompt=prompt, mode=LlmMode.CRITIC)
        match = _VERDICT_RE.search(resp.text)
        if not match:
            raise CriticTruthfulnessError(f"no VERDICT found in LLM response: {resp.text[:200]!r}")
        passed = match.group(1).upper() == "PASS"
        reason = self._extract_reason(resp.text)
        return Verdict(passed=passed, reason=reason)

    @staticmethod
    def _build_prompt(clip_path: Path, observation: Observation) -> str:
        return (
            "You are crescendai's truthfulness critic. Your sole job is to decide whether the following "
            "observation is genuinely audible in the cited time range of the clip.\n\n"
            f"Clip path: {clip_path}\n"
            f"Observation dimension: {observation.dimension}\n"
            f"Observation time range: {observation.time_range[0]:.2f}s - {observation.time_range[1]:.2f}s\n"
            f"Observation: {observation.plain_english}\n\n"
            "Reply in this exact format:\nVERDICT: PASS|KILL\nReason: <one sentence>\n"
        )

    @staticmethod
    def _extract_reason(text: str) -> str:
        for line in text.splitlines():
            if line.strip().lower().startswith("reason:"):
                return line.split(":", 1)[1].strip()
        return ""
