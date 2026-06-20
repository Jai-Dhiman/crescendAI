from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VerdictResult:
    verdict: str            # SUPPORTED | REFUTED | UNVERIFIABLE
    reason_code: str | None
    measured_value: float   # d
    tau: float
    error_bar: float
    event_count: int
    units: str              # "percent", "fraction", "dB"
    substrate_versions: dict
    dimension: str
    location: dict | str    # bar range dict or "whole_piece"


class UnverifiableError(Exception):
    """Raised by LocationResolver and Measurers when verification cannot proceed."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
