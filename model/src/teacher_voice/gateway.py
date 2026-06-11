"""Anthropic access through the unified authenticated Cloudflare AI Gateway (BYOK).

The provider key is vaulted in the gateway; callers send only the gateway-auth
header. The Anthropic SDK always emits x-api-key, which BYOK rejects (the gateway
forwards it to Anthropic -> 401), so we strip it at the httpx transport layer.

AI_GATEWAY_ENDPOINT and AI_GATEWAY_TOKEN are read from the environment or, as a
fallback, apps/api/.dev.vars.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_var(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    # teacher_voice/ -> src/ -> model/ -> repo root, then apps/api/.dev.vars
    dev_vars = Path(__file__).resolve().parents[3] / "apps" / "api" / ".dev.vars"
    if dev_vars.exists():
        prefix = f"{name}="
        for line in dev_vars.read_text().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ[name] = value
                return value
    return None


def anthropic_client():
    """An `anthropic.Anthropic` client routed through the gateway (BYOK, no local key)."""
    import anthropic
    import httpx

    endpoint = _load_var("AI_GATEWAY_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AI_GATEWAY_ENDPOINT not found in env or apps/api/.dev.vars")
    token = _load_var("AI_GATEWAY_TOKEN")
    if not token:
        raise RuntimeError("AI_GATEWAY_TOKEN not found in env or apps/api/.dev.vars")
    endpoint = endpoint.rstrip("/")

    def _strip_provider_key(request: httpx.Request) -> None:
        request.headers.pop("x-api-key", None)

    return anthropic.Anthropic(
        api_key="byok-unused",
        base_url=f"{endpoint}/anthropic",
        default_headers={"cf-aig-authorization": f"Bearer {token}"},
        http_client=httpx.Client(event_hooks={"request": [_strip_provider_key]}),
    )
