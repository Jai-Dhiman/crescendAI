"""Workers-AI judge client over the unified CF AI control plane.

The production worker (api/src/harness/loop/gateway-client.ts) calls Workers AI
through the unified AI REST API on api.cloudflare.com, authenticated with the
standard Authorization header and routed to the `crescendai` gateway with
cf-aig-gateway-id. (cf-aig-authorization only applies to the provider-native
gateway.ai.cloudflare.com endpoints.)

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
    """JudgeClient over the unified AI REST endpoint."""

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_JUDGE_MODEL
        account_id = _require("CLOUDFLARE_ACCOUNT_ID")
        self._url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/v1/chat/completions"
        )
        self._gateway_id = _require("AI_GATEWAY_ID")
        self._cf_token = _require("CLOUDFLARE_API_TOKEN")

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._cf_token}",
                "cf-aig-gateway-id": self._gateway_id,
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
