"""In-memory SQLite replica of D1 memory tables for eval.

Mirrors the schemas from 0005_observations.sql and 0006_memory_system.sql,
and replicates the 4 retrieval queries from memory.rs exactly.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass


@dataclass
class SynthesizedFact:
    """Mirrors the Rust SynthesizedFact struct."""
    id: str
    fact_text: str
    fact_type: str
    dimension: str | None
    piece_context: str | None
    valid_at: str
    trend: str | None
    confidence: str
    source_type: str


@dataclass
class RecentObservationWithEngagement:
    """Mirrors the Rust RecentObservationWithEngagement struct."""
    dimension: str
    observation_text: str
    framing: str
    created_at: str
    engaged: bool


class MemoryDB:
    """In-memory SQLite database replicating D1 memory tables."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()

        # From 0005_observations.sql
        cur.execute("""
            CREATE TABLE observations (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                chunk_index INTEGER,
                dimension TEXT NOT NULL,
                observation_text TEXT NOT NULL,
                elaboration_text TEXT,
                reasoning_trace TEXT,
                framing TEXT,
                dimension_score REAL,
                student_baseline REAL,
                piece_context TEXT,
                learning_arc TEXT,
                is_fallback BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX idx_observations_student ON observations(student_id, created_at)")
        cur.execute("CREATE INDEX idx_observations_session ON observations(session_id)")

        # From 0006_memory_system.sql
        cur.execute("""
            CREATE TABLE synthesized_facts (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                dimension TEXT,
                piece_context TEXT,
                valid_at TEXT NOT NULL,
                invalid_at TEXT,
                trend TEXT,
                confidence TEXT NOT NULL,
                evidence TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expired_at TEXT
            )
        """)
        cur.execute("CREATE INDEX idx_synthesized_facts_student ON synthesized_facts(student_id)")
        cur.execute(
            "CREATE INDEX idx_synthesized_facts_active "
            "ON synthesized_facts(student_id, invalid_at, expired_at)"
        )

        cur.execute("""
            CREATE TABLE teaching_approaches (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                framing TEXT NOT NULL,
                approach_summary TEXT NOT NULL,
                engaged INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX idx_teaching_approaches_student ON teaching_approaches(student_id)")
        cur.execute("CREATE INDEX idx_teaching_approaches_observation ON teaching_approaches(observation_id)")

        cur.execute("""
            CREATE TABLE student_memory_meta (
                student_id TEXT PRIMARY KEY,
                last_synthesis_at TEXT,
                total_observations INTEGER DEFAULT 0,
                total_facts INTEGER DEFAULT 0
            )
        """)

        # Students table (minimal, for baselines)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                baseline_dynamics REAL,
                baseline_timing REAL,
                baseline_pedaling REAL,
                baseline_articulation REAL,
                baseline_phrasing REAL,
                baseline_interpretation REAL
            )
        """)

        self.conn.commit()

    # -- Insert helpers --

    def insert_observation(
        self,
        id: str,
        student_id: str,
        session_id: str,
        dimension: str,
        observation_text: str,
        framing: str = "correction",
        dimension_score: float | None = None,
        student_baseline: float | None = None,
        reasoning_trace: str = "",
        piece_context: str | None = None,
        created_at: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO observations
               (id, student_id, session_id, dimension, observation_text,
                framing, dimension_score, student_baseline, reasoning_trace,
                piece_context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, student_id, session_id, dimension, observation_text,
             framing, dimension_score, student_baseline, reasoning_trace,
             piece_context, created_at),
        )
        self.conn.commit()

    def insert_fact(
        self,
        id: str,
        student_id: str,
        fact_text: str,
        fact_type: str,
        dimension: str | None = None,
        piece_context: str | None = None,
        valid_at: str = "",
        trend: str | None = None,
        confidence: str = "medium",
        evidence: str = "[]",
        source_type: str = "synthesized",
        created_at: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO synthesized_facts
               (id, student_id, fact_text, fact_type, dimension, piece_context,
                valid_at, invalid_at, trend, confidence, evidence, source_type, created_at, expired_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL)""",
            (id, student_id, fact_text, fact_type, dimension, piece_context,
             valid_at, trend, confidence, evidence, source_type, created_at),
        )
        self.conn.commit()

    def insert_teaching_approach(
        self,
        student_id: str,
        observation_id: str,
        dimension: str,
        framing: str,
        approach_summary: str,
        engaged: bool = False,
        created_at: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO teaching_approaches
               (id, student_id, observation_id, dimension, framing,
                approach_summary, engaged, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), student_id, observation_id, dimension,
             framing, approach_summary, 1 if engaged else 0, created_at),
        )
        self.conn.commit()

    def insert_student(self, student_id: str, baselines: dict[str, float]) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO students
               (student_id, baseline_dynamics, baseline_timing, baseline_pedaling,
                baseline_articulation, baseline_phrasing, baseline_interpretation)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (student_id,
             baselines.get("dynamics"), baselines.get("timing"),
             baselines.get("pedaling"), baselines.get("articulation"),
             baselines.get("phrasing"), baselines.get("interpretation")),
        )
        self.conn.commit()

    def invalidate_fact(self, fact_id: str, student_id: str, invalid_at: str, expired_at: str) -> None:
        self.conn.execute(
            "UPDATE synthesized_facts SET invalid_at = ?, expired_at = ? WHERE id = ? AND student_id = ?",
            (invalid_at, expired_at, fact_id, student_id),
        )
        self.conn.commit()

    # -- Retrieval queries (mirror memory.rs exactly) --

    def query_active_facts(self, student_id: str) -> list[SynthesizedFact]:
        """Query 1: Active synthesized facts. Mirrors memory.rs query_active_facts."""
        rows = self.conn.execute(
            """SELECT id, fact_text, fact_type, dimension, piece_context, valid_at,
                      trend, confidence, source_type
               FROM synthesized_facts
               WHERE student_id = ? AND invalid_at IS NULL AND expired_at IS NULL
               AND source_type != 'student_reported'
               ORDER BY fact_type, valid_at DESC
               LIMIT 12""",
            (student_id,),
        ).fetchall()
        return [
            SynthesizedFact(
                id=r["id"], fact_text=r["fact_text"], fact_type=r["fact_type"],
                dimension=r["dimension"], piece_context=r["piece_context"],
                valid_at=r["valid_at"], trend=r["trend"],
                confidence=r["confidence"] or "medium",
                source_type=r["source_type"] or "synthesized",
            )
            for r in rows
        ]

    def query_recent_observations_with_engagement(
        self, student_id: str,
    ) -> list[RecentObservationWithEngagement]:
        """Query 2: Recent observations with engagement. Mirrors memory.rs."""
        rows = self.conn.execute(
            """SELECT o.dimension, o.observation_text, o.framing, o.created_at,
                      COALESCE(ta.engaged, 0) AS engaged
               FROM observations o
               LEFT JOIN teaching_approaches ta ON ta.observation_id = o.id
               WHERE o.student_id = ?
               ORDER BY o.created_at DESC
               LIMIT 5""",
            (student_id,),
        ).fetchall()
        return [
            RecentObservationWithEngagement(
                dimension=r["dimension"],
                observation_text=r["observation_text"],
                framing=r["framing"] or "correction",
                created_at=r["created_at"],
                engaged=r["engaged"] == 1,
            )
            for r in rows
        ]

    def query_piece_facts(self, student_id: str, piece_title: str) -> list[SynthesizedFact]:
        """Query 4: Piece-specific facts. Mirrors memory.rs query_piece_facts."""
        rows = self.conn.execute(
            """SELECT id, fact_text, fact_type, dimension, piece_context, valid_at,
                      trend, confidence, source_type
               FROM synthesized_facts
               WHERE student_id = ?
                 AND piece_context IS NOT NULL
                 AND json_extract(piece_context, '$.title') = ?
                 AND invalid_at IS NULL AND expired_at IS NULL""",
            (student_id, piece_title),
        ).fetchall()
        return [
            SynthesizedFact(
                id=r["id"], fact_text=r["fact_text"], fact_type=r["fact_type"],
                dimension=r["dimension"], piece_context=r["piece_context"],
                valid_at=r["valid_at"], trend=r["trend"],
                confidence=r["confidence"] or "medium",
                source_type=r["source_type"] or "synthesized",
            )
            for r in rows
        ]

    def query_student_reported_facts(
        self, student_id: str, today: str,
    ) -> list[SynthesizedFact]:
        """Query student-reported facts (chat-derived). Mirrors memory.rs."""
        rows = self.conn.execute(
            """SELECT id, fact_text, fact_type, dimension, piece_context, valid_at,
                      trend, confidence, source_type
               FROM synthesized_facts
               WHERE student_id = ? AND source_type = 'student_reported'
               AND (invalid_at IS NULL OR invalid_at > ?) AND expired_at IS NULL
               ORDER BY created_at DESC
               LIMIT 10""",
            (student_id, today),
        ).fetchall()
        return [
            SynthesizedFact(
                id=r["id"], fact_text=r["fact_text"], fact_type=r["fact_type"],
                dimension=r["dimension"], piece_context=r["piece_context"],
                valid_at=r["valid_at"], trend=r["trend"],
                confidence=r["confidence"] or "medium",
                source_type=r["source_type"] or "student_reported",
            )
            for r in rows
        ]

    def insert_student_reported_fact(
        self,
        id: str,
        student_id: str,
        fact_text: str,
        category: str,
        valid_at: str = "",
        invalid_at: str | None = None,
        created_at: str = "",
    ) -> None:
        """Insert a student_reported fact."""
        self.conn.execute(
            """INSERT INTO synthesized_facts
               (id, student_id, fact_text, fact_type, dimension, piece_context,
                valid_at, invalid_at, trend, confidence, evidence, source_type, created_at, expired_at)
               VALUES (?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, '[]', ?, ?, NULL)""",
            (id, student_id, fact_text, "student_reported", category,
             valid_at, invalid_at, "high", "student_reported", created_at),
        )
        self.conn.commit()

    def format_student_reported_context(
        self, student_id: str, today: str,
    ) -> str:
        """Format student-reported facts as bullets. Mirrors memory.rs."""
        facts = self.query_student_reported_facts(student_id, today)
        if not facts:
            return ""
        out = ""
        for fact in facts:
            category = fact.dimension or "general"
            out += f"- [{category}] {fact.fact_text}\n"
        return out

    def build_memory_context(
        self, student_id: str, piece_title: str | None = None, today: str = "",
    ) -> dict:
        """Build full memory context, mirroring memory.rs build_memory_context."""
        active_facts = self.query_active_facts(student_id)
        recent_obs = self.query_recent_observations_with_engagement(student_id)

        piece_facts: list[SynthesizedFact] = []
        if piece_title:
            raw_piece_facts = self.query_piece_facts(student_id, piece_title)
            active_ids = {f.id for f in active_facts}
            piece_facts = [f for f in raw_piece_facts if f.id not in active_ids]

        student_facts: list[SynthesizedFact] = []
        if today:
            student_facts = self.query_student_reported_facts(student_id, today)

        return {
            "active_facts": active_facts,
            "recent_observations": recent_obs,
            "piece_facts": piece_facts,
            "student_facts": student_facts,
        }

    def format_memory_context(
        self, student_id: str, piece_title: str | None = None, today: str = "",
    ) -> str:
        """Format memory context as plain text. Mirrors memory.rs format_memory_context."""
        ctx = self.build_memory_context(student_id, piece_title, today=today)
        active_facts: list[SynthesizedFact] = ctx["active_facts"]
        recent_obs: list[RecentObservationWithEngagement] = ctx["recent_observations"]
        piece_facts: list[SynthesizedFact] = ctx["piece_facts"]
        student_facts: list[SynthesizedFact] = ctx.get("student_facts", [])

        if not active_facts and not recent_obs and not student_facts:
            return ""

        out = "## Student Memory\n\n"

        if student_facts:
            out += "### About Student\n"
            for fact in student_facts:
                category = fact.dimension or "general"
                out += f"- [{category}] {fact.fact_text}\n"
            out += "\n"

        if active_facts:
            out += "### Active Patterns\n"
            for fact in active_facts:
                dim_label = f"{fact.dimension}/" if fact.dimension else ""
                trend_label = f", {fact.trend}" if fact.trend else ""
                out += (
                    f"- [{fact.fact_type}{dim_label}{trend_label}, "
                    f"{fact.confidence} confidence] "
                    f"{fact.fact_text} (since {fact.valid_at})\n"
                )
            out += "\n"

        if recent_obs:
            out += "### Recent Feedback\n"
            for obs in recent_obs:
                engaged_label = ", student asked for elaboration" if obs.engaged else ""
                out += (
                    f"- [{obs.created_at}] {obs.dimension}: "
                    f"\"{obs.observation_text}\" "
                    f"(framing: {obs.framing}{engaged_label})\n"
                )
            out += "\n"

        if piece_facts:
            out += "### Current Piece History\n"
            for fact in piece_facts:
                out += f"- {fact.fact_text} (since {fact.valid_at})\n"
            out += "\n"

        return out

    # -- Helpers for synthesis eval --

    def get_new_observations_since(
        self, student_id: str, since: str,
    ) -> list[dict]:
        """Get observations since a timestamp, as dicts (for synthesis prompt)."""
        rows = self.conn.execute(
            """SELECT id, dimension, observation_text, framing, dimension_score,
                      student_baseline, reasoning_trace, piece_context, created_at
               FROM observations
               WHERE student_id = ? AND created_at > ?
               ORDER BY created_at ASC""",
            (student_id, since),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_teaching_approaches_since(
        self, student_id: str, since: str,
    ) -> list[dict]:
        """Get teaching approaches since a timestamp, as dicts."""
        rows = self.conn.execute(
            """SELECT dimension, framing, approach_summary, engaged
               FROM teaching_approaches
               WHERE student_id = ? AND created_at > ?""",
            (student_id, since),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_baselines(self, student_id: str) -> dict:
        """Get student baselines as a dict."""
        row = self.conn.execute(
            """SELECT baseline_dynamics, baseline_timing, baseline_pedaling,
                      baseline_articulation, baseline_phrasing, baseline_interpretation
               FROM students WHERE student_id = ?""",
            (student_id,),
        ).fetchone()
        if row:
            return dict(row)
        return {}

    def reset(self) -> None:
        """Clear all data (but keep schema)."""
        for table in ["observations", "synthesized_facts", "teaching_approaches",
                       "student_memory_meta", "students"]:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
