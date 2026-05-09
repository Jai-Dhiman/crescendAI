"""Verifies LlmGateway validates SELECTOR responses against JSON schema."""
import httpx
import pytest
from content_engine.adapters.llm_gateway import LlmGateway, LlmMode, LlmGatewayError


SCHEMA = {
    "type": "object",
    "required": ["dimension", "time_range", "plain_english"],
    "properties": {
        "dimension": {"type": "string", "enum": ["phrasing", "timing", "dynamics", "pedaling", "articulation", "interpretation"]},
        "time_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "plain_english": {"type": "string", "minLength": 1},
    },
}


def test_valid_response_passes_schema(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"dimension":"phrasing","time_range":[5.2,7.1],"plain_english":"rushed"}'}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/c")
    resp = gw.complete(prompt="x", mode=LlmMode.SELECTOR, schema=SCHEMA)
    assert resp.parsed_json == {"dimension": "phrasing", "time_range": [5.2, 7.1], "plain_english": "rushed"}


def test_invalid_response_raises(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"dimension":"BOGUS","time_range":[5.2,7.1],"plain_english":"x"}'}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/c")
    with pytest.raises(LlmGatewayError, match="schema"):
        gw.complete(prompt="x", mode=LlmMode.SELECTOR, schema=SCHEMA)
