"""Unified LLM client for Workers AI (via CF AI Gateway), Anthropic, and OpenRouter.

Workers AI models are accessed through the Cloudflare AI Gateway using
OpenAI-compatible chat completions format. This avoids adding the openai
SDK as a dependency -- we use requests (already available).

Environment variables:
  CLOUDFLARE_API_TOKEN -- Cloudflare API token (read from env or apps/api/.dev.vars)
  CF_ACCOUNT_ID        -- Cloudflare account ID (defaults to wrangler.toml value)
  ANTHROPIC_API_KEY    -- Anthropic API key (only needed for --provider anthropic)
  OPENROUTER_API_KEY   -- OpenRouter API key (only needed for --provider openrouter)

Usage:
  client = LLMClient(provider="workers-ai")
  text = client.complete("You are a teacher.", "Analyze this transcript...", max_tokens=2000)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

# Cloudflare account ID from wrangler.toml
# (See apps/api/wrangler.toml for the canonical source)
DEFAULT_CF_ACCOUNT_ID = "5df63f40beeab277db407f1ecbd6e1ec"
DEFAULT_GATEWAY_ID = "crescendai-background"

# Model defaults per task type
MODELS = {
    "workers-ai": {
        "cheap": "@cf/google/gemma-4-26b-a4b-it",
        "quality": "@cf/openai/gpt-oss-120b",
        "judge": "@cf/google/gemma-4-26b-a4b-it",
        "default": "@cf/openai/gpt-oss-120b",
    },
    "anthropic": {
        "cheap": "claude-haiku-4-5-20251001",
        "quality": "claude-sonnet-4-6",
        "default": "claude-sonnet-4-6",
    },
    "openrouter": {
        "cheap": "openai/gpt-5.4-mini",
        "quality": "openai/gpt-5.4-mini",
        "judge": "openai/gpt-5.4-mini",
        "default": "openai/gpt-5.4-mini",
    },
}


def _load_dev_vars_key(var_name: str) -> str | None:
    """Load a key from apps/api/.dev.vars by variable name. Returns None if not found."""
    dev_vars = Path(__file__).resolve().parents[2] / "api" / ".dev.vars"
    if dev_vars.exists():
        prefix = f"{var_name}="
        for line in dev_vars.read_text().splitlines():
            line = line.strip()
            if line.startswith(prefix):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ[var_name] = value
                return value
    return None


def _load_cf_token() -> str:
    """Load CLOUDFLARE_API_TOKEN from env or apps/api/.dev.vars."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or _load_dev_vars_key("CLOUDFLARE_API_TOKEN")
    if token:
        return token
    raise RuntimeError(
        "CLOUDFLARE_API_TOKEN not found. Set it in env or apps/api/.dev.vars"
    )


def _load_anthropic_key() -> None:
    """Load ANTHROPIC_API_KEY from env or apps/api/.dev.vars."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if not _load_dev_vars_key("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Set it in env or apps/api/.dev.vars"
        )


def _load_openrouter_key() -> str:
    """Load OPENROUTER_API_KEY from env or apps/api/.dev.vars."""
    key = os.environ.get("OPENROUTER_API_KEY") or _load_dev_vars_key("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set it in env or apps/api/.dev.vars"
    )


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_openrouter_payload(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict:
    """Pure function that builds the OpenRouter chat-completions request body.

    Extracted from _openrouter_complete for unit testing without network I/O.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }


class LLMClient:
    """Unified client for Workers AI, Anthropic, and OpenRouter."""

    def __init__(
        self,
        provider: str = "workers-ai",
        model: str | None = None,
        tier: str = "default",
    ):
        self.provider = provider
        if model:
            self.model = model
        else:
            self.model = MODELS[provider][tier]

        if provider == "workers-ai":
            self._cf_token = _load_cf_token()
            self._account_id = os.environ.get("CF_ACCOUNT_ID", DEFAULT_CF_ACCOUNT_ID)
            self._gateway_id = os.environ.get("CF_GATEWAY_ID", DEFAULT_GATEWAY_ID)
            self._base_url = (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self._account_id}/{self._gateway_id}/workers-ai/v1/chat/completions"
            )
        elif provider == "anthropic":
            import anthropic

            _load_anthropic_key()
            self._anthropic = anthropic.Anthropic()
        elif provider == "openrouter":
            self._openrouter_key = _load_openrouter_key()

    def complete(
        self,
        user: str,
        max_tokens: int = 4000,
        system: str = "",
    ) -> str:
        """Send a chat completion and return the text response."""
        if self.provider == "workers-ai":
            return self._workers_ai_complete(system, user, max_tokens)
        elif self.provider == "anthropic":
            return self._anthropic_complete(system, user, max_tokens)
        elif self.provider == "openrouter":
            return self._openrouter_complete(system, user, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _workers_ai_complete(self, system: str, user: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        response = requests.post(
            self._base_url,
            headers={
                "Authorization": f"Bearer {self._cf_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Workers AI returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"No choices in Workers AI response: {json.dumps(data)[:500]}"
            )

        content = choices[0]["message"].get("content")
        if content is None:
            raise RuntimeError(
                f"Workers AI returned null content: {json.dumps(data)[:500]}"
            )

        return content

    def _anthropic_complete(self, system: str, user: str, max_tokens: int) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        response = self._anthropic.messages.create(**kwargs)
        return response.content[0].text

    def _openrouter_complete(self, system: str, user: str, max_tokens: int) -> str:
        payload = _build_openrouter_payload(self.model, system, user, max_tokens)
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://crescend.ai",
                "X-Title": "CrescendAI Evals",
            },
            json=payload,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"No choices in OpenRouter response: {json.dumps(data)[:500]}"
            )
        content = choices[0]["message"].get("content")
        if content is None:
            raise RuntimeError(
                f"OpenRouter returned null content: {json.dumps(data)[:500]}"
            )
        return content

    def complete_json(
        self,
        user: str,
        max_tokens: int = 4000,
        system: str = "",
    ) -> str:
        """Complete and strip markdown fences for JSON responses."""
        text = self.complete(user, max_tokens=max_tokens, system=system)
        return strip_json_fences(text)

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"


def strip_json_fences(text: str) -> str:
    """Strip markdown fences and preamble text from LLM JSON/YAML output.

    Handles: ```json ... ```, text preamble before ```, and trailing ```.
    """
    text = text.strip()
    # If there's a code fence anywhere, extract just the fenced content
    if "```" in text:
        # Find the first code fence
        fence_start = text.index("```")
        after_fence = text[fence_start + 3 :]
        # Skip the language tag line (e.g., "json\n")
        if "\n" in after_fence:
            after_fence = after_fence.split("\n", 1)[1]
        # Find closing fence
        if "```" in after_fence:
            after_fence = after_fence[: after_fence.rindex("```")]
        return after_fence.strip()
    return text
