"""Sentry init for the content-engine app."""
from __future__ import annotations
import os
import sentry_sdk


def init_sentry() -> None:
    dsn = os.environ.get("SENTRY_DSN_CONTENT_ENGINE")
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        send_default_pii=False,
        environment=os.environ.get("CONTENT_ENGINE_ENV", "local"),
    )
