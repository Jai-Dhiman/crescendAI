"""Core dataclasses for memory evaluation scenarios."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


DIMENSIONS = frozenset({
    "dynamics", "timing", "pedaling",
    "articulation", "phrasing", "interpretation",
})

FACT_TYPES = frozenset({
    "dimension", "approach", "arc", "student_reported",
})

TRENDS = frozenset({
    "improving", "stable", "declining", "new", "resolved",
})

CONFIDENCES = frozenset({"high", "medium", "low"})


@dataclass
class Observation:
    """A single teacher observation fed into the memory system."""
    id: str
    dimension: str
    observation_text: str
    framing: str = "correction"
    dimension_score: float | None = None
    student_baseline: float | None = None
    reasoning_trace: str = ""
    piece_context: str | None = None  # JSON string: {"composer": "...", "title": "..."}
    session_id: str = ""
    session_date: str = ""  # ISO date
    engaged: bool = False


@dataclass
class ExpectedFact:
    """A gold-standard fact that synthesis should produce."""
    id: str = ""
    fact_text_pattern: str = ""  # regex pattern to match against fact_text
    fact_type: str = "dimension"
    dimension: str | None = None
    trend: str | None = None
    confidence: str = "medium"
    valid_at: str = ""  # ISO date when this fact should become active
    gold_fact_text: str = ""  # natural-language sentence for downstream eval


@dataclass
class SynthesisCheckpoint:
    """Defines when to run synthesis and what to expect."""
    after_observation_index: int = 0  # run synthesis after this many observations
    expected_new_facts: list[str] = field(default_factory=list)  # ExpectedFact IDs
    expected_invalidations: list[str] = field(default_factory=list)  # fact IDs to invalidate


@dataclass
class TemporalAssertion:
    """Assert whether a fact should be active at a given time."""
    query_time: str = ""  # ISO date
    fact_pattern: str = ""  # regex to match fact_text
    should_be_active: bool = True
    category: str = ""  # extraction|multi_session|temporal|knowledge_update|abstention


@dataclass
class RetrievalQuery:
    """A query to test retrieval correctness."""
    id: str = ""
    query_type: str = "active_facts"  # active_facts|recent_observations|piece_facts
    piece_title: str | None = None  # for piece_facts queries
    expected_fact_ids: list[str] = field(default_factory=list)
    expected_absent_ids: list[str] = field(default_factory=list)


@dataclass
class MemoryEvalScenario:
    """A complete evaluation scenario with observations, checkpoints, and assertions."""
    id: str
    name: str
    category: str  # single_dim|multi_dim|piece_lifecycle|temporal|engagement
    student_id: str = "test-student-001"
    baselines: dict = field(default_factory=lambda: {
        "dynamics": 3.0, "timing": 3.0, "pedaling": 3.0,
        "articulation": 3.0, "phrasing": 3.0, "interpretation": 3.0,
    })
    observations: list[Observation] = field(default_factory=list)
    checkpoints: list[SynthesisCheckpoint] = field(default_factory=list)
    expected_facts: list[ExpectedFact] = field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)
    temporal_assertions: list[TemporalAssertion] = field(default_factory=list)
    version: str = "1.0"

    def validate(self) -> list[str]:
        errors = []
        if not self.id:
            errors.append("id is required")
        if not self.name:
            errors.append("name is required")
        if not self.observations:
            errors.append("at least one observation required")
        if not self.checkpoints:
            errors.append("at least one checkpoint required")
        if not self.retrieval_queries:
            errors.append("at least one retrieval query required")
        for obs in self.observations:
            if obs.dimension not in DIMENSIONS:
                errors.append(f"observation {obs.id}: invalid dimension '{obs.dimension}'")
        for ef in self.expected_facts:
            if ef.fact_type not in FACT_TYPES:
                errors.append(f"expected fact {ef.id}: invalid fact_type '{ef.fact_type}'")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MemoryEvalScenario:
        observations = [Observation(**o) for o in d.pop("observations", [])]
        checkpoints = [SynthesisCheckpoint(**c) for c in d.pop("checkpoints", [])]
        expected_facts = [ExpectedFact(**f) for f in d.pop("expected_facts", [])]
        retrieval_queries = [RetrievalQuery(**q) for q in d.pop("retrieval_queries", [])]
        temporal_assertions = [TemporalAssertion(**a) for a in d.pop("temporal_assertions", [])]
        return cls(
            observations=observations,
            checkpoints=checkpoints,
            expected_facts=expected_facts,
            retrieval_queries=retrieval_queries,
            temporal_assertions=temporal_assertions,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )


@dataclass
class ChatExchange:
    id: str
    user_message: str
    assistant_response: str
    session_date: str  # ISO date, used as "today" for extraction


@dataclass
class ExpectedChatFact:
    id: str
    fact_text_pattern: str  # regex
    gold_fact_text: str     # natural language for downstream eval
    category: str           # identity|background|goals|preferences|repertoire|events
    operation: str          # "add" or "update"
    supersedes_fact_id: str | None = None  # for UPDATE ops
    permanent: bool = True
    invalid_at: str | None = None  # for temporal facts


@dataclass
class ChatExtractionScenario:
    id: str
    name: str
    category: str  # per_category|update|temporal|selectivity|e2e
    student_id: str = "chat-test-student-001"
    pre_existing_facts: list[ExpectedChatFact] = field(default_factory=list)
    exchanges: list[ChatExchange] = field(default_factory=list)
    expected_facts: list[ExpectedChatFact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ChatExtractionScenario:
        pre_existing = [ExpectedChatFact(**f) for f in d.pop("pre_existing_facts", [])]
        exchanges = [ChatExchange(**e) for e in d.pop("exchanges", [])]
        expected_facts = [ExpectedChatFact(**f) for f in d.pop("expected_facts", [])]
        return cls(
            pre_existing_facts=pre_existing,
            exchanges=exchanges,
            expected_facts=expected_facts,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )


def save_chat_scenarios(scenarios: list[ChatExtractionScenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_chat_scenarios(path: Path) -> list[ChatExtractionScenario]:
    scenarios = []
    with open(path) as f:
        for line in f:
            scenarios.append(ChatExtractionScenario.from_dict(json.loads(line)))
    return scenarios


def save_scenarios(scenarios: list[MemoryEvalScenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_scenarios(path: Path) -> list[MemoryEvalScenario]:
    scenarios = []
    with open(path) as f:
        for line in f:
            scenarios.append(MemoryEvalScenario.from_dict(json.loads(line)))
    return scenarios


SCENARIOS_PATH = Path(__file__).parents[1] / "data" / "scenarios.jsonl"
REALISTIC_SCENARIOS_PATH = Path(__file__).parents[1] / "data" / "realistic_scenarios.jsonl"


def load_all_scenarios(include_temporal: bool = False) -> list[MemoryEvalScenario]:
    """Load both synthetic and realistic scenarios.

    Args:
        include_temporal: If True, filter realistic scenarios to those with
            temporal_assertions instead of expected_facts.  Use this for the
            temporal eval.  If False (default), keep scenarios that have
            expected_facts annotated (for synthesis eval).
    """
    scenarios = load_scenarios(SCENARIOS_PATH)
    if REALISTIC_SCENARIOS_PATH.exists():
        realistic = load_scenarios(REALISTIC_SCENARIOS_PATH)
        if include_temporal:
            annotated = [s for s in realistic if s.temporal_assertions]
        else:
            annotated = [s for s in realistic if s.expected_facts]
        scenarios.extend(annotated)
    return scenarios
