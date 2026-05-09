"""Versioned config storage. Versions are immutable once written."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA_PATH = Path(__file__).parent / "schema" / "002_config.sql"


@dataclass(frozen=True)
class ConfigRow:
    key: str
    version: int
    value: dict[str, Any]


class ConfigStore:
    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_PATH.read_text())

    def create_version(self, key: str, value: dict[str, Any]) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS m FROM config_version WHERE key = ?", (key,)
        ).fetchone()
        next_v = (row["m"] or 0) + 1
        self._conn.execute(
            "INSERT INTO config_version (key, version, value, created_at) VALUES (?, ?, ?, ?)",
            (key, next_v, json.dumps(value), datetime.now(timezone.utc).isoformat()),
        )
        return next_v

    def get(self, key: str, version: int | None = None) -> ConfigRow | None:
        if version is None:
            row = self._conn.execute(
                "SELECT key, version, value FROM config_version WHERE key = ? "
                "ORDER BY version DESC LIMIT 1",
                (key,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT key, version, value FROM config_version WHERE key = ? AND version = ?",
                (key, version),
            ).fetchone()
        if row is None:
            return None
        return ConfigRow(key=row["key"], version=row["version"], value=json.loads(row["value"]))
