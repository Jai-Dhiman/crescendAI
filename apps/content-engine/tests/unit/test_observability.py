"""Verifies Sentry init reads DSN from env, no-op when unset."""
import sentry_sdk
from content_engine.observability import init_sentry


def test_init_sentry_with_dsn_sets_client(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN_CONTENT_ENGINE", "https://abc@sentry.io/1")
    init_sentry()
    client = sentry_sdk.get_client()
    assert client is not None
    assert client.dsn is not None


def test_init_sentry_without_dsn_is_noop(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN_CONTENT_ENGINE", raising=False)
    init_sentry()
    # No assertion needed — just verifying no exception.
