"""Analytics ingestion: pulls per-post metrics across YT/TikTok/IG."""
from __future__ import annotations
from dataclasses import dataclass
import httpx


@dataclass(frozen=True)
class PostMetrics:
    platform: str
    post_id: str
    views: int | None
    watch_time_sec: int | None
    link_clicks: int | None


class AnalyticsIngestor:
    def __init__(self, youtube_api_key: str, postiz_url: str, postiz_token: str, timeout_s: float = 30.0):
        self._yt_key = youtube_api_key
        self._postiz_url = postiz_url.rstrip("/")
        self._postiz_token = postiz_token
        self._timeout = timeout_s

    def pull(self, post_ids: dict[str, str]) -> list[PostMetrics]:
        results: list[PostMetrics] = []
        for platform, pid in post_ids.items():
            metrics = self._pull_one(platform, pid)
            results.append(metrics)
        return results

    def _pull_one(self, platform: str, post_id: str) -> PostMetrics:
        if platform == "youtube":
            return self._pull_youtube(post_id)
        return self._pull_postiz(platform, post_id)

    def _pull_youtube(self, post_id: str) -> PostMetrics:
        # Public YouTube Data API: videos.list returns
        #   {"items": [{"statistics": {"viewCount": "<str>", ...}}]}
        # watch_time and link_clicks are NOT exposed here — they require the
        # YouTube Analytics API + OAuth, which is out of MVP scope. We surface
        # views only and leave the other fields None so FeedbackScorer's signal
        # is honest about what we can measure.
        url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?id={post_id}&key={self._yt_key}&part=statistics"
        )
        resp = httpx.get(url, timeout=self._timeout)
        empty = PostMetrics(platform="youtube", post_id=post_id, views=None, watch_time_sec=None, link_clicks=None)
        if resp.status_code != 200:
            return empty
        items = resp.json().get("items") or []
        if not items:
            return empty
        stats = items[0].get("statistics") or {}
        view_str = stats.get("viewCount")
        views = int(view_str) if view_str is not None else None
        return PostMetrics(
            platform="youtube",
            post_id=post_id,
            views=views,
            watch_time_sec=None,
            link_clicks=None,
        )

    def _pull_postiz(self, platform: str, post_id: str) -> PostMetrics:
        url = f"{self._postiz_url}/posts/{platform}/{post_id}/metrics"
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {self._postiz_token}"},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            return PostMetrics(platform=platform, post_id=post_id, views=None, watch_time_sec=None, link_clicks=None)
        body = resp.json()
        return PostMetrics(
            platform=platform,
            post_id=post_id,
            views=body.get("views"),
            watch_time_sec=body.get("watch_time_sec"),
            link_clicks=body.get("link_clicks"),
        )
