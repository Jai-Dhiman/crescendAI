"""LLM gateway: single deep adapter for all LLM access (CLI + Workers AI)."""
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any
import httpx


class LlmMode(str, Enum):
    SELECTOR = "selector"
    NARRATOR = "narrator"
    CRITIC = "critic"


class LlmGatewayError(Exception):
    pass


@dataclass(frozen=True)
class LlmResponse:
    text: str
    parsed_json: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


_WORKERS_AI_MODEL = "@cf/google/gemma-4-26b-a4b-it"


class LlmGateway:
    def __init__(
        self,
        cf_gateway_url: str,
        cf_token: str,
        claude_bin: str,
        timeout_s: float = 60.0,
    ):
        self._cf_url = cf_gateway_url.rstrip("/")
        self._cf_token = cf_token
        self._claude_bin = claude_bin
        self._timeout = timeout_s

    def complete(self, prompt: str, mode: LlmMode, schema: dict[str, Any] | None = None) -> LlmResponse:
        if mode == LlmMode.SELECTOR:
            return self._workers_ai_complete(prompt, schema)
        if mode in (LlmMode.NARRATOR, LlmMode.CRITIC):
            return self._cli_complete(prompt)
        raise LlmGatewayError(f"unsupported mode: {mode}")

    def _workers_ai_complete(self, prompt: str, schema: dict[str, Any] | None) -> LlmResponse:
        url = f"{self._cf_url}/workers-ai/v1/chat/completions"
        body: dict[str, Any] = {
            "model": _WORKERS_AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": schema}
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {self._cf_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise LlmGatewayError(f"workers-ai {resp.status_code}: {resp.text[:200]}")
        body_json = resp.json()
        text = body_json.get("result", {}).get("response", "")
        return LlmResponse(text=text, raw=body_json)

    def _cli_complete(self, prompt: str) -> LlmResponse:
        result = subprocess.run(
            [self._claude_bin, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if result.returncode != 0:
            raise LlmGatewayError(f"claude cli exit {result.returncode}: {result.stderr.strip()}")
        return LlmResponse(text=result.stdout.strip(), raw=None)
