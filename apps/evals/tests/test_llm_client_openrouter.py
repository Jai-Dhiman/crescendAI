from __future__ import annotations

import os

import pytest

from teaching_knowledge.llm_client import LLMClient, _build_openrouter_payload


def test_build_openrouter_payload_shape() -> None:
    payload = _build_openrouter_payload(
        model="openai/gpt-5.4-mini",
        system="You are a judge.",
        user="Evaluate this: hello world",
        max_tokens=2048,
    )
    assert payload["model"] == "openai/gpt-5.4-mini"
    assert payload["max_tokens"] == 2048
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "You are a judge."}
    assert payload["messages"][1] == {"role": "user", "content": "Evaluate this: hello world"}


def test_build_openrouter_payload_without_system() -> None:
    payload = _build_openrouter_payload(
        model="openai/gpt-5.4-mini",
        system="",
        user="hello",
        max_tokens=100,
    )
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


def test_openrouter_client_uses_judge_tier_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")
    client = LLMClient(provider="openrouter", tier="judge")
    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-5.4-mini"


def test_openrouter_client_accepts_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")
    client = LLMClient(provider="openrouter", model="anthropic/claude-sonnet-4-6")
    assert client.model == "anthropic/claude-sonnet-4-6"


def test_openrouter_client_needs_no_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # BYOK: the OpenRouter key is vaulted in the gateway, so constructing the
    # client requires no OPENROUTER_API_KEY on disk. Auth is the gateway token,
    # supplied per-request (see gateway_auth_header), not at construction.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient(provider="openrouter", tier="judge")
    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-5.4-mini"
