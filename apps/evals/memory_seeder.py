"""Seed and clean up canary synthesized_facts for the debug user.

Interface:
    seed_canary_facts(student_id, db_dsn) -> CanarySeed
    cleanup_canary_facts(student_id, db_dsn) -> int  (rows deleted)
    build_insert_rows(student_id, tokens) -> list[dict]  (offline-testable)

Hides: psycopg2 connection lifecycle, INSERT SQL, DELETE-on-prefix cleanup,
       student_id resolution via /api/auth/debug.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

CANARY_PREFIX = "CANARY_"

CANARY_TOKENS = [
    "CANARY_RACHMANINOFF_ETUDE",
    "CANARY_LEFT_HAND_WEAKNESS",
]

_TOKEN_TEMPLATES: dict[str, tuple[str, str]] = {
    "CANARY_RACHMANINOFF_ETUDE": (
        "CANARY_RACHMANINOFF_ETUDE: student is currently preparing this passage "
        "and struggles with maintaining tempo under pressure.",
        "repertoire_context",
    ),
    "CANARY_LEFT_HAND_WEAKNESS": (
        "CANARY_LEFT_HAND_WEAKNESS: left hand consistently trails right by "
        "30-50ms in fast passages across three sessions.",
        "technical_observation",
    ),
}

_DEFAULT_TEMPLATE = (
    "{token}: a canary marker for automated recall verification.",
    "student_goal",
)


@dataclass
class CanarySeed:
    """Outcome of a successful seed operation."""
    student_id: str
    tokens: list[str] = field(default_factory=list)
    rows_inserted: int = 0


def build_insert_rows(student_id: str, tokens: list[str]) -> list[dict[str, Any]]:
    """Build INSERT row dicts for synthesized_facts — no DB required.

    Each row contains all required non-null columns. The token string is
    embedded verbatim in fact_text so keyword-search assertions work.

    Args:
        student_id: The student UUID to seed facts for.
        tokens: List of canary token strings (must contain the token literally).

    Returns:
        List of dicts ready for psycopg2 executemany (column -> value).
    """
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for token in tokens:
        if token in _TOKEN_TEMPLATES:
            fact_text, fact_type = _TOKEN_TEMPLATES[token]
        else:
            template, fact_type = _DEFAULT_TEMPLATE
            fact_text = template.format(token=token)

        rows.append({
            "id": str(uuid.uuid4()),
            "student_id": student_id,
            "fact_text": fact_text,
            "fact_type": fact_type,
            "valid_at": now,
            "confidence": "high",
            "evidence": f"Automated canary seed for e2e recall verification ({token})",
            "source_type": "eval_seed",
        })

    return rows


def seed_canary_facts(
    student_id: str,
    db_dsn: str,
    tokens: list[str] | None = None,
) -> CanarySeed:
    """Insert canary synthesized_facts into crescendai_dev for student_id.

    Removes any previous canary rows (fact_text LIKE 'CANARY_%') for this
    student first to avoid accumulating stale rows across runs.

    Args:
        student_id: Student UUID (stable for debug@crescend.ai across restarts).
        db_dsn: PostgreSQL DSN, e.g. postgresql://jdhiman:postgres@localhost:5432/crescendai_dev
        tokens: Canary token strings to seed (defaults to CANARY_TOKENS).

    Returns:
        CanarySeed with student_id, tokens, and rows_inserted count.

    Raises:
        RuntimeError: If psycopg2 connection or INSERT fails.
    """
    import psycopg2  # type: ignore[import]

    effective_tokens = tokens if tokens is not None else CANARY_TOKENS

    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to DB ({type(exc).__name__}): {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM synthesized_facts "
                    "WHERE student_id = %s AND fact_text LIKE %s",
                    (student_id, f"{CANARY_PREFIX}%"),
                )

                rows = build_insert_rows(student_id, effective_tokens)
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO synthesized_facts
                            (id, student_id, fact_text, fact_type,
                             valid_at, confidence, evidence, source_type)
                        VALUES
                            (%(id)s, %(student_id)s, %(fact_text)s, %(fact_type)s,
                             %(valid_at)s, %(confidence)s, %(evidence)s, %(source_type)s)
                        """,
                        row,
                    )
    except Exception as exc:
        raise RuntimeError(f"Failed to seed canary facts: {exc}") from exc
    finally:
        conn.close()

    return CanarySeed(
        student_id=student_id,
        tokens=effective_tokens,
        rows_inserted=len(effective_tokens),
    )


def cleanup_canary_facts(student_id: str, db_dsn: str) -> int:
    """Delete all canary rows for student_id. Returns count deleted.

    Raises:
        RuntimeError: If DB connection or DELETE fails.
    """
    import psycopg2  # type: ignore[import]

    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to DB ({type(exc).__name__}): {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM synthesized_facts "
                    "WHERE student_id = %s AND fact_text LIKE %s",
                    (student_id, f"{CANARY_PREFIX}%"),
                )
                return cur.rowcount
    except Exception as exc:
        raise RuntimeError(f"Failed to cleanup canary facts: {exc}") from exc
    finally:
        conn.close()


def get_debug_student_id(wrangler_url: str) -> str:
    """Fetch the debug user's student_id from the API.

    Calls POST /api/auth/debug and reads the studentId from the response body.

    Raises:
        RuntimeError: If the API is unreachable or returns a non-200 status,
                      or if studentId is absent from the response.
    """
    import requests

    try:
        resp = requests.post(f"{wrangler_url}/api/auth/debug", timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"API not reachable at {wrangler_url}. Run `just dev` first."
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"POST /api/auth/debug returned {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    student_id = data.get("studentId")
    if not student_id:
        raise RuntimeError(
            f"/api/auth/debug response missing 'studentId' "
            f"(got keys {list(data.keys())}): {data}"
        )
    return student_id
