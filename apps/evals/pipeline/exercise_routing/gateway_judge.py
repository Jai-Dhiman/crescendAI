"""Workers-AI judge client over the authenticated CF AI Gateway.

The shared teaching_knowledge.LLMClient targets the legacy `crescendai-background`
gateway with only an Authorization header; that gateway now rejects with a 401
AiGatewayError. The production worker (api/src/harness/loop/gateway-client.ts)
instead uses the authenticated `crescendai` gateway, which requires BOTH:
  - cf-aig-authorization: Bearer <AI_GATEWAY_TOKEN>  (gateway auth)
  - Authorization:        Bearer <CLOUDFLARE_API_TOKEN>  (workers-ai upstream)

This client mirrors that exactly so the relevance judge reaches the same gateway
the product uses. It is self-contained (no LLMClient dependency) and satisfies
the relevance.JudgeClient protocol. Fails loud on missing creds / non-200.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_JUDGE_MODEL = "@cf/google/gemma-4-26b-a4b-it"
_DEV_VARS = Path(__file__).resolve().parents[3] / "api" / ".dev.vars"


def _dev_var(name: str) -> str | None:
    """Read a var from env, else apps/api/.dev.vars."""
    if os.environ.get(name):
        return os.environ[name]
    if _DEV_VARS.exists():
        prefix = f"{name}="
        for line in _DEV_VARS.read_text().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _require(name: str) -> str:
    value = _dev_var(name)
    if not value:
        raise RuntimeError(
            f"{name} not found in env or {_DEV_VARS}. The relevance judge needs the "
            "authenticated AI Gateway credentials."
        )
    return value


class WorkersAiGatewayJudge:
    """JudgeClient over the authenticated workers-ai gateway endpoint."""

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_JUDGE_MODEL
        endpoint = _require("AI_GATEWAY_ENDPOINT").rstrip("/")
        self._url = f"{endpoint}/workers-ai/v1/chat/completions"
        self._gateway_token = _require("AI_GATEWAY_TOKEN")
        self._cf_token = _require("CLOUDFLARE_API_TOKEN")

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = requests.post(
            self._url,
            headers={
                "cf-aig-authorization": f"Bearer {self._gateway_token}",
                "Authorization": f"Bearer {self._cf_token}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "max_tokens": max_tokens, "messages": messages},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"AI Gateway returned {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices in gateway response: {str(data)[:500]}")
        content = choices[0]["message"].get("content")
        if content is None:
            raise RuntimeError(f"Gateway returned null content: {str(data)[:500]}")
        return content
