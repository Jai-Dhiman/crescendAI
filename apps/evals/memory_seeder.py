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

# source_type marker used as the stable deletion key, decoupled from fact_text
# (which must read as natural prose the teacher paraphrases back, never an
# artificial token it would refuse to echo verbatim).
CANARY_SOURCE_TYPE = "eval_seed"

# Each canary fact: natural prose the teacher echoes in its own words, plus the
# distinctive natural keywords we assert on. The teacher recalls the MEANING, so
# assertions match keywords case-insensitively, not literal token strings.
CANARY_FACTS: list[dict[str, Any]] = [
    {
        "fact_text": (
            "The student is currently preparing Rachmaninoff's Etude-Tableau in "
            "E-flat minor and struggles to maintain tempo under pressure."
        ),
        "fact_type": "repertoire_context",
        "keywords": ["rachmaninoff"],
    },
    {
        "fact_text": (
            "The student's left hand consistently trails the right hand by 30-50ms "
            "in fast passages across three sessions."
        ),
        "fact_type": "technical_observation",
        "keywords": ["left hand", "left-hand"],
    },
]

# Per-fact keyword groups for the recall assertion: recall passes when, for EVERY
# group, at least one keyword appears (case-insensitively) in the teacher's reply.
CANARY_KEYWORD_GROUPS: list[list[str]] = [f["keywords"] for f in CANARY_FACTS]


@dataclass
class CanarySeed:
    """Outcome of a successful seed operation."""
    student_id: str
    keyword_groups: list[list[str]] = field(default_factory=list)
    rows_inserted: int = 0


def build_insert_rows(
    student_id: str, facts: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Build INSERT row dicts for synthesized_facts — no DB required.

    fact_text is natural prose (no artificial token); source_type is the stable
    deletion marker (CANARY_SOURCE_TYPE).

    Args:
        student_id: The student UUID to seed facts for.
        facts: Canary fact specs (defaults to CANARY_FACTS); each needs
            fact_text + fact_type.

    Returns:
        List of dicts ready for psycopg2 (column -> value).
    """
    effective = facts if facts is not None else CANARY_FACTS
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for fact in effective:
        rows.append({
            "id": str(uuid.uuid4()),
            "student_id": student_id,
            "fact_text": fact["fact_text"],
            "fact_type": fact["fact_type"],
            "valid_at": now,
            "confidence": "high",
            "evidence": "Automated canary seed for e2e recall verification",
            "source_type": CANARY_SOURCE_TYPE,
        })

    return rows


def seed_canary_facts(
    student_id: str,
    db_dsn: str,
    facts: list[dict[str, Any]] | None = None,
) -> CanarySeed:
    """Insert canary synthesized_facts into crescendai_dev for student_id.

    Removes any previous canary rows (source_type='eval_seed', plus legacy
    'canary%' fact_text) for this student first to avoid stale accumulation.

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

    effective = facts if facts is not None else CANARY_FACTS

    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to DB ({type(exc).__name__}): {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM synthesized_facts "
                    "WHERE student_id = %s "
                    "AND (source_type = %s OR fact_text ILIKE %s)",
                    (student_id, CANARY_SOURCE_TYPE, "canary%"),
                )

                rows = build_insert_rows(student_id, effective)
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
        keyword_groups=[f["keywords"] for f in effective],
        rows_inserted=len(effective),
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
                    "WHERE student_id = %s "
                    "AND (source_type = %s OR fact_text ILIKE %s)",
                    (student_id, CANARY_SOURCE_TYPE, "canary%"),
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
