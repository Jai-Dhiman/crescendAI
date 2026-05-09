"""Verifies Scheduler adapter posts to Postiz across platforms."""
from datetime import datetime, timezone
from pathlib import Path
import httpx
from content_engine.adapters.scheduler import Scheduler


def test_schedule_posts_to_all_platforms_and_returns_per_platform_ids(tmp_path, monkeypatch):
    asset = tmp_path / "ep_001.mp4"
    asset.write_bytes(b"\x00\x00\x00\x18ftyp")

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers", {})
        return httpx.Response(
            200,
            json={
                "posts": [
                    {"platform": "youtube", "post_id": "yt_abc", "status": "scheduled"},
                    {"platform": "tiktok", "post_id": "tt_xyz", "status": "scheduled"},
                    {"platform": "instagram", "post_id": "ig_qrs", "status": "scheduled"},
                ],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    sched = Scheduler(url="https://postiz.example/api", token="postiz_t")
    when = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)
    results = sched.schedule(
        asset_path=asset,
        when=when,
        platforms=["youtube", "tiktok", "instagram"],
        caption="hook line",
        description_link="https://crescend.ai?utm_source=shorts",
    )

    assert captured["headers"]["Authorization"] == "Bearer postiz_t"
    assert {r.platform: r.post_id for r in results} == {
        "youtube": "yt_abc", "tiktok": "tt_xyz", "instagram": "ig_qrs",
    }


def test_schedule_records_partial_failure_per_platform(tmp_path, monkeypatch):
    asset = tmp_path / "ep_002.mp4"
    asset.write_bytes(b"x")

    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "posts": [
                    {"platform": "youtube", "post_id": "yt_ok", "status": "scheduled"},
                    {"platform": "tiktok", "post_id": None, "status": "rejected", "error": "auth"},
                ],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    sched = Scheduler(url="https://postiz.example/api", token="t")
    results = sched.schedule(
        asset_path=asset,
        when=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc),
        platforms=["youtube", "tiktok"],
        caption="hook",
        description_link="https://crescend.ai",
    )
    yt = next(r for r in results if r.platform == "youtube")
    tt = next(r for r in results if r.platform == "tiktok")
    assert yt.post_id == "yt_ok"
    assert tt.post_id is None
    assert tt.error == "auth"
