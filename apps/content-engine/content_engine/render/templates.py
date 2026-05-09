"""CTA template dataclasses for the three phased CTA strategies."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CtaTemplate:
    phase: str
    end_card_text: str
    spoken_cta: str
    landing_url: str
    watermark_enabled: bool

    @classmethod
    def for_phase(cls, phase: str) -> CtaTemplate:
        if phase == "A":
            return cls(
                phase="A",
                end_card_text="",
                spoken_cta="",
                landing_url="https://crescend.ai",
                watermark_enabled=True,
            )
        if phase == "B":
            return cls(
                phase="B",
                end_card_text="crescend.ai",
                spoken_cta="",
                landing_url="https://crescend.ai/shorts",
                watermark_enabled=True,
            )
        if phase == "C":
            return cls(
                phase="C",
                end_card_text="crescend.ai",
                spoken_cta="Want yours analyzed? crescend.ai/submit.",
                landing_url="https://crescend.ai/submit",
                watermark_enabled=True,
            )
        raise ValueError(f"unknown CTA phase: {phase!r}")
