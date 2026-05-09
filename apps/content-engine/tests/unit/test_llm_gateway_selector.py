"""Verifies LlmGateway routes SELECTOR mode to Workers AI HTTP."""
import httpx
from content_engine.adapters.llm_gateway import LlmGateway, LlmMode


def test_selector_mode_posts_to_workers_ai_gateway(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers", {})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"dimension":"phrasing","time_range":[5.2,7.1],"plain_english":"rushed peak"}'}}
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    gw = LlmGateway(
        cf_gateway_url="https://gateway.ai.cloudflare.com/v1/acct/crescendai-background",
        cf_token="cf_t",
        claude_bin="/usr/local/bin/claude",
    )
    response = gw.complete(prompt="pick the best obs", mode=LlmMode.SELECTOR)

    assert "workers-ai/v1/chat/completions" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer cf_t"
    assert response.text.startswith('{"dimension"')
