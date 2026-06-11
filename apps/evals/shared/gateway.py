"""Provider access through the unified authenticated Cloudflare AI Gateway (BYOK).

Provider API keys (Anthropic, OpenRouter) are vaulted in the gateway's Provider
Keys store; callers send only the gateway-auth header (`cf-aig-authorization`).
No provider key lives on disk.

Empirically verified against the live gateway: the gateway forwards a request's
`x-api-key` to Anthropic verbatim, so BYOK injection only happens when that
header is ABSENT. The Anthropic SDK always emits `x-api-key`, so we strip it at
the httpx transport layer before the request leaves the process.

Env / config:
  AI_GATEWAY_ENDPOINT -- gateway base URL (…/<account>/<gateway>), provider path appended
  AI_GATEWAY_TOKEN    -- gateway-auth token (cf-aig-authorization: Bearer …)
Both are read from the environment or, as a fallback, apps/api/.dev.vars.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_var(name: str) -> str | None:
    """Return an env var, falling back to apps/api/.dev.vars (and caching to env)."""
    if os.environ.get(name):
        return os.environ[name]
    # shared/ -> evals/ -> apps/, then api/.dev.vars
    dev_vars = Path(__file__).resolve().parents[2] / "api" / ".dev.vars"
    if dev_vars.exists():
        prefix = f"{name}="
        for line in dev_vars.read_text().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ[name] = value
                return value
    return None


def gateway_endpoint() -> str:
    ep = _load_var("AI_GATEWAY_ENDPOINT")
    if not ep:
        raise RuntimeError(
            "AI_GATEWAY_ENDPOINT not found in env or apps/api/.dev.vars"
        )
    return ep.rstrip("/")


def gateway_token() -> str:
    tok = _load_var("AI_GATEWAY_TOKEN")
    if not tok:
        raise RuntimeError(
            "AI_GATEWAY_TOKEN not found in env or apps/api/.dev.vars"
        )
    return tok


def gateway_auth_header() -> dict[str, str]:
    """The single header that authenticates a request to the gateway."""
    return {"cf-aig-authorization": f"Bearer {gateway_token()}"}


def anthropic_client():
    """An `anthropic.Anthropic` client routed through the gateway (BYOK, no local key).

    Drop-in for `anthropic.Anthropic()` — all `.messages.create(...)` usage is
    unchanged. The `api_key` is a placeholder the gateway never sees: a request
    hook strips `x-api-key` so the vaulted key is injected instead.
    """
    import anthropic
    import httpx

    endpoint = gateway_endpoint()
    auth = gateway_auth_header()

    def _strip_provider_key(request: httpx.Request) -> None:
        request.headers.pop("x-api-key", None)

    return anthropic.Anthropic(
        api_key="byok-unused",
        base_url=f"{endpoint}/anthropic",
        default_headers=auth,
        http_client=httpx.Client(event_hooks={"request": [_strip_provider_key]}),
    )


def async_anthropic_client():
    """An `anthropic.AsyncAnthropic` client routed through the gateway (BYOK, no local key).

    Drop-in for `anthropic.AsyncAnthropic()` — all `await .messages.create(...)` usage
    is unchanged. The `api_key` is a placeholder the gateway never sees: a request
    hook strips `x-api-key` so the vaulted key is injected instead.
    """
    import anthropic
    import httpx

    endpoint = gateway_endpoint()
    auth = gateway_auth_header()

    async def _strip_provider_key(request: httpx.Request) -> None:
        request.headers.pop("x-api-key", None)

    return anthropic.AsyncAnthropic(
        api_key="byok-unused",
        base_url=f"{endpoint}/anthropic",
        default_headers=auth,
        http_client=httpx.AsyncClient(event_hooks={"request": [_strip_provider_key]}),
    )
