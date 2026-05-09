"""Postiz scheduler adapter: cross-posts an asset to YT Shorts + TikTok + Reels."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import base64
import httpx


@dataclass(frozen=True)
class PostResult:
    platform: str
    post_id: str | None
    status: str
    error: str | None = None


class SchedulerError(Exception):
    pass


class Scheduler:
    def __init__(self, url: str, token: str, timeout_s: float = 30.0):
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    def schedule(
        self,
        asset_path: Path,
        when: datetime,
        platforms: list[str],
        caption: str,
        description_link: str,
    ) -> list[PostResult]:
        asset_b64 = base64.b64encode(Path(asset_path).read_bytes()).decode("ascii")
        resp = httpx.post(
            f"{self._url}/posts",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json={
                "asset_b64": asset_b64,
                "scheduled_at": when.isoformat(),
                "platforms": platforms,
                "caption": caption,
                "description_link": description_link,
            },
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise SchedulerError(f"postiz {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        return [
            PostResult(
                platform=p["platform"],
                post_id=p.get("post_id"),
                status=p["status"],
                error=p.get("error"),
            )
            for p in body["posts"]
        ]
