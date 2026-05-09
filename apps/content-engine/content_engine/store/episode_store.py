"""SQLite-backed Episode persistence with state-machine transition validation."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State, is_valid_transition


_SCHEMA_PATH = Path(__file__).parent / "schema" / "001_init.sql"


class InvalidTransitionError(Exception):
    pass


class EpisodeStore:
    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_PATH.read_text())

    def save(self, ep: Episode) -> None:
        d = ep.to_dict()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episode (
                id, candidate_url, source_type, state, config_versions,
                model_output, observation, script_text, voiceover_path,
                render_path, posts, analytics, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d["id"], d["candidate_url"], d["source_type"], d["state"],
                json.dumps(d["config_versions"]),
                json.dumps(d["model_output"]) if d["model_output"] is not None else None,
                json.dumps(d["observation"]) if d["observation"] is not None else None,
                d["script_text"],
                d["voiceover_path"],
                d["render_path"],
                json.dumps(d["posts"]) if d["posts"] is not None else None,
                json.dumps(d["analytics"]) if d["analytics"] is not None else None,
                d["created_at"], d["updated_at"],
            ),
        )

    def get(self, episode_id: str) -> Episode | None:
        row = self._conn.execute(
            "SELECT * FROM episode WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    def transition(self, episode_id: str, new_state: State) -> Episode:
        with self._conn:
            ep = self.get(episode_id)
            if ep is None:
                raise KeyError(f"episode not found: {episode_id}")
            if not is_valid_transition(ep.state, new_state):
                raise InvalidTransitionError(
                    f"cannot transition {ep.state.value} -> {new_state.value}"
                )
            ep.state = new_state
            ep.updated_at = datetime.now(timezone.utc)
            self.save(ep)
        return ep

    def list_by_state(self, state: State) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episode WHERE state = ? ORDER BY created_at",
            (state.value,),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        d = dict(row)
        d["config_versions"] = json.loads(d["config_versions"])
        for k in ("model_output", "observation", "posts", "analytics"):
            d[k] = json.loads(d[k]) if d[k] is not None else None
        return Episode.from_dict(d)
