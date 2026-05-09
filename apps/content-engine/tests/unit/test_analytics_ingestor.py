"""Verifies AnalyticsIngestor returns per-platform metrics, tolerates missing."""
import httpx
from content_engine.adapters.analytics_ingestor import AnalyticsIngestor


def test_pull_returns_metrics_per_platform(monkeypatch):
    """YouTube Data API videos.list returns items[0].statistics.viewCount as a string.
    watch_time_sec and link_clicks are NOT available on the public Data API with an
    API key — they require the YouTube Analytics API + OAuth (out of MVP scope), so
    YouTube metrics intentionally surface views only and leave the other fields None.
    Postiz proxies TikTok + Instagram metrics with the flat schema we control."""

    def fake_get(url, **kwargs):
        if "googleapis.com/youtube" in url:
            return httpx.Response(
                200,
                json={"items": [{"statistics": {"viewCount": "1234", "likeCount": "12"}}]},
                request=httpx.Request("GET", url),
            )
        if "tiktok" in url:
            return httpx.Response(
                200,
                json={"views": 800, "watch_time_sec": 1200, "link_clicks": 3},
                request=httpx.Request("GET", url),
            )
        if "instagram" in url:
            return httpx.Response(
                200,
                json={"views": 400, "watch_time_sec": 600, "link_clicks": 1},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    ing = AnalyticsIngestor(
        youtube_api_key="yt_k",
        postiz_url="https://postiz.example/api",
        postiz_token="postiz_t",
    )
    metrics = ing.pull({"youtube": "yt_abc", "tiktok": "tt_xyz", "instagram": "ig_qrs"})
    by_platform = {m.platform: m for m in metrics}
    assert by_platform["youtube"].views == 1234
    assert by_platform["youtube"].watch_time_sec is None  # Data API key does not expose this
    assert by_platform["youtube"].link_clicks is None
    assert by_platform["tiktok"].link_clicks == 3
    assert by_platform["instagram"].watch_time_sec == 600


def test_pull_tolerates_missing_platform_data(monkeypatch):
    def fake_get(url, **kwargs):
        if "googleapis.com/youtube" in url:
            return httpx.Response(
                200,
                json={"items": [{"statistics": {"viewCount": "100"}}]},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    ing = AnalyticsIngestor(
        youtube_api_key="k",
        postiz_url="https://postiz.example/api",
        postiz_token="t",
    )
    metrics = ing.pull({"youtube": "yt_ok", "tiktok": "tt_dead"})
    by = {m.platform: m for m in metrics}
    assert by["youtube"].views == 100
    assert by["tiktok"].views is None  # missing data is acceptable


def test_pull_tolerates_empty_youtube_items(monkeypatch):
    """If a video is private/deleted, videos.list returns items=[]. Treat as missing."""
    def fake_get(url, **kwargs):
        return httpx.Response(200, json={"items": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    ing = AnalyticsIngestor(youtube_api_key="k", postiz_url="https://postiz.example/api", postiz_token="t")
    metrics = ing.pull({"youtube": "yt_dead"})
    assert metrics[0].views is None
