"""crescendai HF inference endpoint adapter."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import httpx


class InferenceError(Exception):
    pass


@dataclass(frozen=True)
class ModelOutput:
    scores: dict[str, list[float]]
    duration_sec: float
    raw: dict[str, Any]


class ModelRunner:
    def __init__(self, url: str, token: str, timeout_s: float = 60.0):
        self._url = url
        self._token = token
        self._timeout = timeout_s

    def run(self, clip_path: Path) -> ModelOutput:
        audio_bytes = Path(clip_path).read_bytes()
        resp = httpx.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "audio/wav",
            },
            content=audio_bytes,
            timeout=self._timeout,
        )
        if resp.status_code >= 500:
            raise InferenceError(f"inference 5xx: {resp.status_code}")
        if resp.status_code >= 400:
            raise InferenceError(f"inference 4xx: {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        return ModelOutput(
            scores=body["scores"],
            duration_sec=float(body["duration_sec"]),
            raw=body,
        )
