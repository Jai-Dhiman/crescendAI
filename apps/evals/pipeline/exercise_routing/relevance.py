"""Relevance@1 LLM-judge for exercise selection.

The routing eval's `dimension_match` axis is pure label-equality: it can verify
the teacher targeted the right *dimension*, but it is structurally blind to WHICH
drill the serving layer then picked within that dimension. `selectPrimitive`
(corpus-drill.ts) returns a constant per dimension, so a finger warm-up can be
prescribed for a pedaling weakness and `dimension_match` still passes.

This module closes that gap. Given the diagnosed weakness (dimension + the
teaching-moment context) and the *chosen drill's* metadata (title + technique
tags + the dimensions it can address), an LLM judge rates whether the drill is
PEDAGOGICALLY APPROPRIATE. The headline metric is relevance@1: the fraction of
prescriptions the judge deems appropriate.

Pure + client-injected (mirrors shared/judge_atomic.py) so the scoring logic is
unit-testable with a fake client and no network. The concrete client is
teaching_knowledge.llm_client.LLMClient, reached through the CF AI Gateway.

Judge-family note: the V6 teacher is glm@WorkersAI. The default judge here is a
different family (Workers-AI gemma), honouring the same-family-judging-forbidden
principle from the teaching-knowledge eval. The selection itself is deterministic
code, not the LLM, so cross-family bias is weak regardless.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# relevance@1 threshold on the 0-3 appropriateness scale. >=2 == "appropriate":
# the drill plausibly trains the diagnosed weakness. 0/1 == off-target or actively
# wrong (e.g. a finger drill for a pedaling problem).
APPROPRIATE_THRESHOLD = 2


class JudgeClient(Protocol):
    """Minimal completion interface (satisfied by LLMClient.complete)."""

    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class DrillInfo:
    """The chosen drill's manifest metadata -- what the judge sees was picked."""

    primitive_id: str
    title: str
    source: str
    dimensions: list[str]
    techniques: list[str]


@dataclass(frozen=True)
class RelevanceCase:
    """One prescription to judge: a diagnosed weakness + the drill chosen for it."""

    weakness_dimension: str
    weakness_context: str  # the teaching-moment text describing what went wrong
    bar_range: tuple[int, int] | None
    drill: DrillInfo


@dataclass(frozen=True)
class RelevanceVerdict:
    appropriate: bool
    score: int  # 0-3 appropriateness
    rationale: str
    raw_response: str


@dataclass(frozen=True)
class RelevanceAggregate:
    relevance_at_1: float  # fraction appropriate (score >= APPROPRIATE_THRESHOLD)
    mean_score: float  # mean 0-3 appropriateness
    n_judged: int


SYSTEM = (
    "You are an expert piano pedagogue evaluating whether a prescribed practice "
    "drill is appropriate for a diagnosed performance weakness. Judge ONLY "
    "pedagogical fit: does practising this specific drill plausibly remediate the "
    "diagnosed weakness? A drill is appropriate when the mechanism it trains "
    "(from its title and technique tags) targets the weak dimension -- e.g. a "
    "pedal-isolation prelude for a pedaling weakness, a velocity etude for a "
    "timing/evenness weakness. A drill is inappropriate when it trains an "
    "unrelated mechanism -- e.g. a finger-independence warm-up prescribed for a "
    "pedaling or dynamics weakness -- even if it is a fine exercise in isolation. "
    "Do not reward mere dimension-label overlap; weigh whether the actual "
    "technique addresses the actual problem. Output strict JSON, no prose."
)


def build_relevance_user(case: RelevanceCase) -> str:
    """Render one case into the judge user message. Pure."""
    bar = (
        f"bars {case.bar_range[0]}-{case.bar_range[1]}"
        if case.bar_range is not None
        else "no specific bars (session-scoped)"
    )
    drill = case.drill
    parts = [
        "<diagnosed_weakness>",
        f"dimension: {case.weakness_dimension}",
        f"location: {bar}",
        f"context: {case.weakness_context or '(no additional context)'}",
        "</diagnosed_weakness>",
        "<prescribed_drill>",
        f"title: {drill.title}",
        f"source: {drill.source}",
        f"addresses_dimensions: {', '.join(drill.dimensions) or '(none)'}",
        f"technique_tags: {', '.join(drill.techniques) or '(none)'}",
        "</prescribed_drill>",
        "Rate appropriateness on 0-3:",
        "  3 = directly trains the diagnosed weakness",
        "  2 = plausibly helps the diagnosed weakness",
        "  1 = weakly/tangentially related",
        "  0 = trains an unrelated mechanism (wrong drill for this weakness)",
        'Output JSON exactly: {"score": <0-3 int>, "rationale": "<one sentence>"}',
    ]
    return "\n".join(parts)


def _parse_verdict(raw: str) -> tuple[int, str]:
    """Parse the judge JSON; fail loud on malformed output (no silent default)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"relevance judge returned non-JSON: {exc}; raw={raw[:200]!r}"
        ) from exc
    if "score" not in data:
        raise ValueError(f"relevance judge response missing 'score': {raw[:200]!r}")
    score = int(data["score"])
    if not 0 <= score <= 3:
        raise ValueError(f"relevance score out of range [0,3]: {score}")
    return score, str(data.get("rationale", ""))


def judge_relevance(
    case: RelevanceCase, client: JudgeClient, max_tokens: int = 512
) -> RelevanceVerdict:
    """Judge one prescription's pedagogical relevance. Client-injected; no I/O here
    beyond the injected client call."""
    user = build_relevance_user(case)
    raw = client.complete(user=user, system=SYSTEM, max_tokens=max_tokens)
    score, rationale = _parse_verdict(raw)
    return RelevanceVerdict(
        appropriate=score >= APPROPRIATE_THRESHOLD,
        score=score,
        rationale=rationale,
        raw_response=raw,
    )


def aggregate_relevance(verdicts: list[RelevanceVerdict]) -> RelevanceAggregate:
    """Aggregate per-case verdicts into relevance@1 + mean score. Pure."""
    n = len(verdicts)
    if n == 0:
        return RelevanceAggregate(relevance_at_1=0.0, mean_score=0.0, n_judged=0)
    return RelevanceAggregate(
        relevance_at_1=sum(1 for v in verdicts if v.appropriate) / n,
        mean_score=sum(v.score for v in verdicts) / n,
        n_judged=n,
    )


def load_drill_info(primitive_id: str, manifest_path: Path) -> DrillInfo:
    """Look up a selected primitive's metadata from the API manifest.

    Raises KeyError if the primitive is absent -- a selection that points at a
    drill not in the manifest is a bug, not a judging edge case.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if primitive_id not in manifest:
        raise KeyError(
            f"primitive {primitive_id!r} not in manifest {manifest_path}"
        )
    entry = manifest[primitive_id]
    return DrillInfo(
        primitive_id=primitive_id,
        title=entry["title"],
        source=entry["source"],
        dimensions=entry["dimensions"],
        techniques=entry["techniques"],
    )
