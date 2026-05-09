"""Verifies LlmGateway routes NARRATOR/CRITIC modes to Claude Code CLI subprocess."""
import subprocess
from content_engine.adapters.llm_gateway import LlmGateway, LlmMode


def test_narrator_mode_invokes_claude_cli_with_print_flag(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="Hook line. Observation. Close.", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    gw = LlmGateway(
        cf_gateway_url="https://gw.example/x",
        cf_token="t",
        claude_bin="/opt/claude",
    )
    resp = gw.complete(prompt="write a script for...", mode=LlmMode.NARRATOR)

    assert captured["cmd"][0] == "/opt/claude"
    assert "-p" in captured["cmd"]
    assert "write a script for..." in captured["cmd"]
    assert resp.text == "Hook line. Observation. Close."


def test_critic_mode_invokes_claude_cli(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="VERDICT: PASS\nReason: audible", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/opt/claude")
    resp = gw.complete(prompt="verify obs", mode=LlmMode.CRITIC)
    assert "VERDICT: PASS" in resp.text


def test_cli_nonzero_exit_raises(monkeypatch):
    import pytest
    from content_engine.adapters.llm_gateway import LlmGatewayError

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="auth required")

    monkeypatch.setattr(subprocess, "run", fake_run)
    gw = LlmGateway(cf_gateway_url="x", cf_token="t", claude_bin="/opt/claude")
    with pytest.raises(LlmGatewayError, match="auth required"):
        gw.complete(prompt="x", mode=LlmMode.NARRATOR)
