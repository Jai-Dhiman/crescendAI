"""Episode dataclass: durable record of one content pipeline run."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any
from content_engine.pipeline.states import State


@dataclass(frozen=False)
class Episode:
    id: str
    candidate_url: str
    source_type: str
    state: State
    config_versions: dict[str, int]
    created_at: datetime
    updated_at: datetime
    model_output: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    script_text: str | None = None
    voiceover_path: str | None = None
    render_path: str | None = None
    posts: dict[str, str] | None = None
    analytics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        return cls(
            id=d["id"],
            candidate_url=d["candidate_url"],
            source_type=d["source_type"],
            state=State(d["state"]),
            config_versions=d["config_versions"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            model_output=d.get("model_output"),
            observation=d.get("observation"),
            script_text=d.get("script_text"),
            voiceover_path=d.get("voiceover_path"),
            render_path=d.get("render_path"),
            posts=d.get("posts"),
            analytics=d.get("analytics"),
        )
