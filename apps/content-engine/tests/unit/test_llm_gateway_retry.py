"""Verifies LlmGateway retries once on transient 5xx in SELECTOR mode."""
import httpx
import pytest
from content_engine.adapters.llm_gateway import LlmGateway, LlmMode, LlmGatewayError


def test_retry_once_on_5xx_then_success(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={"result": {"response": '{"k":"v"}'}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/c")
    resp = gw.complete(prompt="x", mode=LlmMode.SELECTOR)
    assert calls["n"] == 2
    assert resp.text == '{"k":"v"}'


def test_two_consecutive_5xx_raises(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return httpx.Response(503, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/c")
    with pytest.raises(LlmGatewayError):
        gw.complete(prompt="x", mode=LlmMode.SELECTOR)
    assert calls["n"] == 2  # one retry then give up
