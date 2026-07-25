"""Sampling-client boundary for the probe.

ProbeClient is the mockable seam: the probe driver only ever sees this
protocol. RecordedResponseClient is the offline implementation used by
every test and by re-scoring saved runs; the real Tinker implementation
lives in audio_teacher.tinker_client and is never imported by tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from audio_teacher.manifest import ContrastPair


@dataclass(frozen=True)
class ProbeResponse:
    pair_id: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ProbeClient(Protocol):
    def estimate_cost_usd(self, pair: ContrastPair) -> float: ...

    def ask(self, pair: ContrastPair) -> ProbeResponse: ...


class RecordedResponseClient:
    """Replays canned responses from a JSONL file keyed by pair_id.

    Record shape per line: {"pair_id": str, "text": str} (extra keys are
    ignored). A pair without a recording raises KeyError, and a duplicate
    pair_id raises ValueError -- an incomplete or corrupt fixture must fail
    loudly, never be skipped.
    """

    def __init__(self, responses_path: Path | str):
        self._responses: dict[str, str] = {}
        with Path(responses_path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["pair_id"] in self._responses:
                    raise ValueError(
                        f"duplicate recorded response for pair {rec['pair_id']!r} "
                        f"in {responses_path}"
                    )
                self._responses[rec["pair_id"]] = rec["text"]

    def estimate_cost_usd(self, pair: ContrastPair) -> float:
        return 0.0

    def ask(self, pair: ContrastPair) -> ProbeResponse:
        if pair.pair_id not in self._responses:
            raise KeyError(
                f"no recorded response for pair {pair.pair_id!r} -- the recorded "
                f"fixture is incomplete"
            )
        return ProbeResponse(
            pair_id=pair.pair_id,
            text=self._responses[pair.pair_id],
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
