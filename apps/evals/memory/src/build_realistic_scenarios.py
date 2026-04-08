"""LLM-powered generator for realistic (messy) memory eval scenarios.

Produces 25 total scenarios:
  Synthesis (17):
    - incomplete (3): Student plays 2-3 chunks then stops
    - piece_switch (3): Student switches pieces mid-practice
    - multi_session_arc (4): Same piece over 3 sessions (struggles -> improves -> plateaus)
    - vague_engagement (2): Ambiguous responses ("ok", "thanks")
    - contradictory (3): Observations contradict across sessions
    - sparse (2): Only 1-2 observations, insufficient data

  Temporal (8):
    - delayed_creation (2): Pattern only clear by session 3
    - cross_session_invalidation (3): Contradictory evidence 2+ weeks later
    - abstention (3): Insufficient evidence, system should NOT create a fact

Outputs to data/realistic_scenarios.jsonl
Expected facts are EMPTY placeholders -- human annotates later.

Run as:
    cd apps/evals/memory && CF_API_TOKEN=... uv run python -m src.build_realistic_scenarios
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .scenarios import (
    MemoryEvalScenario,
    Observation,
    RetrievalQuery,
    SynthesisCheckpoint,
    TemporalAssertion,
    save_scenarios,
)

DATA_DIR = Path(__file__).parents[1] / "data"
_DEV_VARS_PATH = Path(__file__).parents[3] / "api" / ".dev.vars"
DEFAULT_CF_ACCOUNT_ID = "5df63f40beeab277db407f1ecbd6e1ec"
DEFAULT_GATEWAY_ID = "crescendai-background"
# GPT-OSS-120B for quality scenario generation (no thinking mode quirks)
_WORKERS_AI_MODEL = "@cf/openai/gpt-oss-120b"


def _load_cf_token() -> str:
    token = os.environ.get("CF_API_TOKEN")
    if token:
        return token
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("CF_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("CF_API_TOKEN not found in env or apps/api/.dev.vars")


def _workers_ai_complete(messages: list[dict], max_tokens: int = 4096) -> str:
    """Call Workers AI via CF AI Gateway with a messages list."""
    token = _load_cf_token()
    account_id = os.environ.get("CF_ACCOUNT_ID", DEFAULT_CF_ACCOUNT_ID)
    gateway_id = os.environ.get("CF_GATEWAY_ID", DEFAULT_GATEWAY_ID)
    url = (
        f"https://gateway.ai.cloudflare.com/v1/"
        f"{account_id}/{gateway_id}/workers-ai/v1/chat/completions"
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": _WORKERS_AI_MODEL, "max_tokens": max_tokens, "messages": messages},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    if content is None:
        raise RuntimeError(f"Workers AI returned null content: {json.dumps(data)[:500]}")
    return content

DIMENSIONS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]

PIECES = [
    {"composer": "Bach", "title": "Prelude in C Major, WTC I"},
    {"composer": "Bach", "title": "Invention No. 1 in C Major"},
    {"composer": "Chopin", "title": "Nocturne in E-flat Major, Op. 9 No. 2"},
    {"composer": "Chopin", "title": "Waltz in A minor, Op. posth."},
    {"composer": "Chopin", "title": "Etude Op. 10 No. 3"},
    {"composer": "Debussy", "title": "Clair de Lune"},
    {"composer": "Debussy", "title": "Arabesque No. 1"},
    {"composer": "Beethoven", "title": "Fur Elise"},
    {"composer": "Beethoven", "title": "Moonlight Sonata, Op. 27 No. 2, Mvt. 1"},
    {"composer": "Rachmaninoff", "title": "Prelude in C-sharp Minor, Op. 3 No. 2"},
    {"composer": "Mozart", "title": "Sonata in C Major, K. 545, Mvt. 1"},
    {"composer": "Schubert", "title": "Moment Musical No. 3 in F minor"},
    {"composer": "Brahms", "title": "Intermezzo in A Major, Op. 118 No. 2"},
    {"composer": "Liszt", "title": "Consolation No. 3 in D-flat Major"},
]

PERSONAS = [
    {"name": "adult beginner", "level": "beginner", "style": "cautious, asks many questions"},
    {"name": "teenage student", "level": "intermediate", "style": "easily distracted, rushes through"},
    {"name": "retired professional", "level": "advanced beginner", "style": "methodical, self-critical"},
    {"name": "college student", "level": "intermediate", "style": "sporadic practice, bursts of effort"},
    {"name": "motivated hobbyist", "level": "intermediate-advanced", "style": "dedicated but overambitious"},
    {"name": "young child with parent", "level": "beginner", "style": "short attention span, needs encouragement"},
    {"name": "returning pianist", "level": "intermediate", "style": "frustrated by regression, impatient"},
]


@dataclass
class ScenarioSpec:
    id: str
    name: str
    subtype: str  # e.g. "incomplete", "piece_switch", etc.
    category: str  # "synthesis" or "temporal"
    persona: dict
    piece: dict
    second_piece: dict | None  # for piece_switch scenarios
    num_sessions: int
    date_start: str
    description: str  # what the LLM should generate


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_specs() -> list[ScenarioSpec]:
    base = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    specs: list[ScenarioSpec] = []

    # --- Synthesis: incomplete (3) ---
    for i in range(3):
        persona = PERSONAS[i % len(PERSONAS)]
        piece = PIECES[i % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-incomplete-{i+1:02d}",
            name=f"Incomplete practice: {persona['name']} stops after {2 + i % 2} chunks",
            subtype="incomplete",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=1,
            date_start=_iso(base + timedelta(days=i * 3)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Generate {2 + i % 2} teacher observations within a single session. "
                f"The session ends abruptly -- the student stops. "
                f"Observations should be varied across dimensions but not conclusive enough to form a definitive fact. "
                f"Mix framings: some corrections, one encouragement. Include dimension scores (1-5). "
                f"The student is NOT particularly engaged (engaged: false)."
            ),
        ))

    # --- Synthesis: piece_switch (3) ---
    for i in range(3):
        persona = PERSONAS[(i + 2) % len(PERSONAS)]
        piece = PIECES[(i + 2) % len(PIECES)]
        second_piece = PIECES[(i + 5) % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-piece-switch-{i+1:02d}",
            name=f"Piece switch: {persona['name']} moves from {piece['title']} to {second_piece['title']}",
            subtype="piece_switch",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=second_piece,
            num_sessions=2,
            date_start=_iso(base + timedelta(days=10 + i * 4)),
            description=(
                f"A {persona['name']} ({persona['style']}) starts with {piece['composer']} {piece['title']}, "
                f"gets 2-3 observations, then abruptly switches to {second_piece['composer']} {second_piece['title']} "
                f"for 2-3 more observations. "
                f"The observations for the two pieces should be about different dimensions or show different patterns. "
                f"Include piece_context JSON for each observation. "
                f"Make the switch feel organic (student says something like 'let me try something else')."
            ),
        ))

    # --- Synthesis: multi_session_arc (4) ---
    arcs = [
        ("struggles -> slow improvement", "dynamics"),
        ("struggles -> improves significantly", "timing"),
        ("improves -> plateaus", "phrasing"),
        ("struggles -> improves -> slight regression", "pedaling"),
    ]
    for i, (arc_desc, primary_dim) in enumerate(arcs):
        persona = PERSONAS[(i + 1) % len(PERSONAS)]
        piece = PIECES[(i + 7) % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-arc-{i+1:02d}",
            name=f"Multi-session arc: {arc_desc} ({primary_dim})",
            subtype="multi_session_arc",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=3,
            date_start=_iso(base + timedelta(days=25 + i * 7)),
            description=(
                f"A {persona['name']} ({persona['style']}) practices {piece['composer']} {piece['title']} "
                f"across 3 sessions (each 1 week apart). The arc is: {arc_desc}. "
                f"Primary dimension: {primary_dim}. Generate 3-4 observations per session (9-12 total). "
                f"Session 1: student struggling. Session 2: showing some improvement. "
                f"Session 3: {arc_desc.split('->')[-1].strip()}. "
                f"Include realistic dimension scores that reflect the arc. "
                f"Mix framings naturally. Student is engaged (engaged: true) in sessions 2 and 3."
            ),
        ))

    # --- Synthesis: vague_engagement (2) ---
    for i in range(2):
        persona = PERSONAS[(i + 4) % len(PERSONAS)]
        piece = PIECES[(i + 3) % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-vague-{i+1:02d}",
            name=f"Vague engagement: student gives minimal responses",
            subtype="vague_engagement",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=2,
            date_start=_iso(base + timedelta(days=55 + i * 5)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Generate 4-5 observations across 2 sessions. "
                f"The student's responses are vague and non-committal -- they say things like 'ok', 'thanks', "
                f"'I'll try'. The teacher's observations are specific but the student doesn't engage deeply. "
                f"Include some observations where engaged=false mixed with engaged=true to show inconsistency. "
                f"Avoid strong positive or negative patterns -- keep it ambiguous."
            ),
        ))

    # --- Synthesis: contradictory (3) ---
    for i in range(3):
        persona = PERSONAS[(i + 3) % len(PERSONAS)]
        piece = PIECES[(i + 9) % len(PIECES)]
        primary_dim = DIMENSIONS[i % len(DIMENSIONS)]
        specs.append(ScenarioSpec(
            id=f"rs-contradictory-{i+1:02d}",
            name=f"Contradictory observations: {primary_dim} across sessions",
            subtype="contradictory",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=3,
            date_start=_iso(base + timedelta(days=70 + i * 6)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Generate 2-3 observations per session (6-9 total) across 3 sessions. "
                f"The observations are contradictory -- {primary_dim} seems good in session 1, "
                f"poor in session 2, good again in session 3. "
                f"This could reflect inconsistent practice, performance anxiety, or day-to-day variation. "
                f"Scores should zig-zag: e.g. 3.8 -> 2.1 -> 3.5 for {primary_dim}. "
                f"The contradiction should make it hard for the memory system to synthesize a clear fact."
            ),
        ))

    # --- Synthesis: sparse (2) ---
    for i in range(2):
        persona = PERSONAS[(i + 6) % len(PERSONAS)]
        piece = PIECES[(i + 1) % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-sparse-{i+1:02d}",
            name=f"Sparse data: only {1 + i} observation(s)",
            subtype="sparse",
            category="synthesis",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=1,
            date_start=_iso(base + timedelta(days=90 + i * 3)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Generate only {1 + i} observation(s) -- a single brief session. "
                f"The observation(s) should be generic and not specific enough to form a fact. "
                f"Include a dimension score but no student baseline. "
                f"This tests whether the system correctly abstains from fact synthesis with insufficient data."
            ),
        ))

    # --- Temporal: delayed_creation (2) ---
    for i in range(2):
        persona = PERSONAS[i % len(PERSONAS)]
        piece = PIECES[(i + 4) % len(PIECES)]
        primary_dim = DIMENSIONS[(i + 1) % len(DIMENSIONS)]
        specs.append(ScenarioSpec(
            id=f"rs-delayed-{i+1:02d}",
            name=f"Delayed fact creation: {primary_dim} pattern emerges by session 3",
            subtype="delayed_creation",
            category="temporal",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=3,
            date_start=_iso(base + timedelta(days=100 + i * 14)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']} "
                f"across 3 sessions (2 weeks apart). "
                f"Sessions 1-2: only 1-2 observations about {primary_dim}, inconclusive. "
                f"Session 3: 3-4 clear observations about {primary_dim} that confirm a pattern. "
                f"The fact should NOT be creatable from sessions 1-2 alone but SHOULD be clear after session 3. "
                f"Include consistent dimension scores in session 3 (e.g., all below 2.5 or all above 4.0)."
            ),
        ))

    # --- Temporal: cross_session_invalidation (3) ---
    for i in range(3):
        persona = PERSONAS[(i + 2) % len(PERSONAS)]
        piece = PIECES[(i + 6) % len(PIECES)]
        primary_dim = DIMENSIONS[(i + 2) % len(DIMENSIONS)]
        specs.append(ScenarioSpec(
            id=f"rs-invalidation-{i+1:02d}",
            name=f"Cross-session invalidation: {primary_dim} improves significantly",
            subtype="cross_session_invalidation",
            category="temporal",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=4,
            date_start=_iso(base + timedelta(days=130 + i * 21)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Sessions 1-2 (close together): 3-4 observations show a clear weakness in {primary_dim}. "
                f"Gap of 2+ weeks. "
                f"Sessions 3-4: 3-4 observations clearly show {primary_dim} has improved significantly "
                f"(student practiced, took a workshop, or just had a breakthrough). "
                f"The earlier 'weakness' fact should now be invalidated. "
                f"Scores: sessions 1-2 around 2.0-2.5, sessions 3-4 around 3.8-4.5 for {primary_dim}."
            ),
        ))

    # --- Temporal: abstention (3) ---
    for i in range(3):
        persona = PERSONAS[(i + 4) % len(PERSONAS)]
        piece = PIECES[(i + 11) % len(PIECES)]
        specs.append(ScenarioSpec(
            id=f"rs-abstention-{i+1:02d}",
            name=f"Abstention: insufficient evidence to create a fact",
            subtype="abstention",
            category="temporal",
            persona=persona,
            piece=piece,
            second_piece=None,
            num_sessions=2,
            date_start=_iso(base + timedelta(days=190 + i * 7)),
            description=(
                f"A {persona['name']} ({persona['style']}) plays {piece['composer']} {piece['title']}. "
                f"Generate 3-4 observations across 2 sessions. "
                f"The observations are mixed signals -- some dimensions look ok, others look problematic, "
                f"but no single pattern is strong enough or repeated enough to warrant a fact. "
                f"Some observations lack dimension scores. Some sessions have only one observation. "
                f"The system should correctly abstain from creating any fact. "
                f"This is a true negative test -- the expected_facts list will remain empty."
            ),
        ))

    return specs


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _build_obs_prompt(spec: ScenarioSpec) -> str:
    session_dates = []
    base_dt = datetime.fromisoformat(spec.date_start.replace("Z", "+00:00"))
    for s in range(spec.num_sessions):
        session_dates.append(_iso(base_dt + timedelta(weeks=s * 2 if spec.num_sessions > 1 else 0)))

    piece_context = json.dumps(spec.piece)
    second_piece_ctx = json.dumps(spec.second_piece) if spec.second_piece else "null"

    return f"""You are generating realistic piano teacher observation sequences for an evaluation dataset.

Scenario: {spec.name}
Student persona: {spec.persona['name']} -- {spec.persona['style']}
Primary piece: {spec.piece['composer']} -- {spec.piece['title']}
Second piece (if applicable): {second_piece_ctx}
Sessions: {spec.num_sessions}
Session dates: {json.dumps(session_dates)}

Instructions:
{spec.description}

Valid dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation
Valid framings: correction, recognition, encouragement, question
Dimension scores and student_baseline: float 1.0-5.0 or null

Return a JSON array of observation objects. Each object must have exactly these fields:
{{
  "id": "obs-<scenario_id>-<three_digit_number>",
  "dimension": "<one of the valid dimensions>",
  "observation_text": "<specific, realistic teacher observation, 1-3 sentences>",
  "framing": "<one of the valid framings>",
  "dimension_score": <float 1.0-5.0 or null>,
  "student_baseline": <float 1.0-5.0 or null>,
  "piece_context": {piece_context},
  "session_id": "<sess-<scenario_id>-01 through -0N>",
  "session_date": "<ISO datetime from the session_dates list above>",
  "engaged": <true or false>
}}

For piece_switch scenarios, use the second piece context for observations after the switch.
Observation text should sound like a real piano teacher talking to a student -- specific, warm, actionable.
Do NOT add any explanation. Return ONLY the JSON array."""


def _generate_observations(spec: ScenarioSpec) -> list[dict[str, Any]]:
    prompt = _build_obs_prompt(spec)

    raw = _workers_ai_complete([{"role": "user", "content": prompt}], max_tokens=4096)
    cleaned = _strip_code_fences(raw)

    try:
        observations = json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        # Retry once with an explicit repair prompt
        repair_prompt = (
            f"The following text is supposed to be a JSON array but has a syntax error. "
            f"Return ONLY valid JSON, no explanation:\n\n{cleaned}"
        )
        retry_raw = _workers_ai_complete([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_prompt},
        ], max_tokens=4096)
        retry_cleaned = _strip_code_fences(retry_raw)
        try:
            observations = json.loads(retry_cleaned)
        except json.JSONDecodeError as second_err:
            raise ValueError(
                f"Failed to parse LLM output as JSON for scenario {spec.id} "
                f"after retry. First error: {first_err}. Second error: {second_err}. "
                f"Raw output:\n{retry_cleaned}"
            ) from second_err

    if not isinstance(observations, list):
        raise ValueError(
            f"Expected JSON array for scenario {spec.id}, got {type(observations).__name__}"
        )

    return observations


def _build_checkpoints(spec: ScenarioSpec, num_obs: int) -> list[SynthesisCheckpoint]:
    """Build appropriate checkpoints based on scenario subtype."""
    if spec.subtype == "incomplete":
        return [SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[])]

    if spec.subtype == "piece_switch":
        mid = max(2, num_obs // 2)
        return [
            SynthesisCheckpoint(after_observation_index=mid, expected_new_facts=[]),
            SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[]),
        ]

    if spec.subtype == "multi_session_arc":
        per_session = max(1, num_obs // 3)
        return [
            SynthesisCheckpoint(after_observation_index=per_session, expected_new_facts=[]),
            SynthesisCheckpoint(after_observation_index=per_session * 2, expected_new_facts=[]),
            SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[]),
        ]

    if spec.subtype == "vague_engagement":
        return [SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[])]

    if spec.subtype == "contradictory":
        per_session = max(1, num_obs // 3)
        return [
            SynthesisCheckpoint(after_observation_index=per_session, expected_new_facts=[]),
            SynthesisCheckpoint(after_observation_index=per_session * 2, expected_new_facts=[]),
            SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[]),
        ]

    if spec.subtype == "sparse":
        return [SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[])]

    if spec.subtype == "delayed_creation":
        per_session = max(1, num_obs // 3)
        return [
            SynthesisCheckpoint(
                after_observation_index=per_session,
                expected_new_facts=[],  # no fact yet after session 1
            ),
            SynthesisCheckpoint(
                after_observation_index=per_session * 2,
                expected_new_facts=[],  # still no fact after session 2
            ),
            SynthesisCheckpoint(
                after_observation_index=num_obs,
                expected_new_facts=[],  # fact should appear -- human annotates ID
            ),
        ]

    if spec.subtype == "cross_session_invalidation":
        per_session = max(1, num_obs // 4)
        return [
            SynthesisCheckpoint(
                after_observation_index=per_session * 2,
                expected_new_facts=[],  # weakness fact created after sessions 1-2
            ),
            SynthesisCheckpoint(
                after_observation_index=num_obs,
                expected_new_facts=[],
                expected_invalidations=[],  # weakness fact invalidated -- human annotates
            ),
        ]

    if spec.subtype == "abstention":
        return [SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[])]

    # Fallback
    return [SynthesisCheckpoint(after_observation_index=num_obs, expected_new_facts=[])]


def _build_temporal_assertions(spec: ScenarioSpec, session_dates: list[str]) -> list[TemporalAssertion]:
    """Build temporal assertions appropriate to the scenario subtype."""
    if spec.subtype == "delayed_creation":
        if len(session_dates) >= 3:
            return [
                TemporalAssertion(
                    query_time=session_dates[1],
                    fact_pattern=r".*",  # human fills in
                    should_be_active=False,
                    category="temporal",
                ),
                TemporalAssertion(
                    query_time=session_dates[-1],
                    fact_pattern=r".*",  # human fills in
                    should_be_active=True,
                    category="temporal",
                ),
            ]

    if spec.subtype == "cross_session_invalidation":
        if len(session_dates) >= 3:
            return [
                TemporalAssertion(
                    query_time=session_dates[1],
                    fact_pattern=r".*",  # human fills in (weakness fact)
                    should_be_active=True,
                    category="temporal",
                ),
                TemporalAssertion(
                    query_time=session_dates[-1],
                    fact_pattern=r".*",  # human fills in (weakness fact, now invalidated)
                    should_be_active=False,
                    category="temporal",
                ),
            ]

    if spec.subtype == "abstention":
        if session_dates:
            return [
                TemporalAssertion(
                    query_time=session_dates[-1],
                    fact_pattern=r".*",  # any fact pattern -- should NOT be active
                    should_be_active=False,
                    category="abstention",
                ),
            ]

    return []


def _build_scenario(spec: ScenarioSpec, raw_obs: list[dict[str, Any]]) -> MemoryEvalScenario:
    observations: list[Observation] = []
    for o in raw_obs:
        piece_ctx = o.get("piece_context")
        if piece_ctx is not None and not isinstance(piece_ctx, str):
            piece_ctx = json.dumps(piece_ctx)

        observations.append(Observation(
            id=o.get("id", f"obs-{spec.id}-{len(observations)+1:03d}"),
            dimension=o.get("dimension", "dynamics"),
            observation_text=o.get("observation_text", ""),
            framing=o.get("framing", "correction"),
            dimension_score=o.get("dimension_score"),
            student_baseline=o.get("student_baseline"),
            piece_context=piece_ctx,
            session_id=o.get("session_id", f"sess-{spec.id}-01"),
            session_date=o.get("session_date", spec.date_start),
            engaged=bool(o.get("engaged", False)),
        ))

    num_obs = len(observations)

    # Extract unique session dates for temporal assertions
    seen: list[str] = []
    for obs in observations:
        if obs.session_date and obs.session_date not in seen:
            seen.append(obs.session_date)
    session_dates = sorted(seen)

    checkpoints = _build_checkpoints(spec, num_obs)
    temporal_assertions = _build_temporal_assertions(spec, session_dates)

    return MemoryEvalScenario(
        id=spec.id,
        name=spec.name,
        category=f"realistic_{spec.subtype}",
        student_id=f"student-{spec.id}",
        observations=observations,
        checkpoints=checkpoints,
        expected_facts=[],  # human annotates later
        retrieval_queries=[
            RetrievalQuery(
                id=f"rq-{spec.id}-1",
                query_type="active_facts",
                expected_fact_ids=[],  # human annotates later
            )
        ],
        temporal_assertions=temporal_assertions,
        version="2.0",
    )


def generate_all_scenarios() -> list[MemoryEvalScenario]:
    _load_cf_token()  # validate token exists before generating
    specs = _build_specs()

    print(f"Generating {len(specs)} scenarios...")
    scenarios: list[MemoryEvalScenario] = []

    for i, spec in enumerate(specs):
        print(f"  [{i+1}/{len(specs)}] {spec.id}: {spec.name[:60]}...", flush=True)
        raw_obs = _generate_observations(spec)
        scenario = _build_scenario(spec, raw_obs)
        scenarios.append(scenario)
        print(f"         -> {len(scenario.observations)} observations generated")

    return scenarios


def main() -> None:
    scenarios = generate_all_scenarios()

    out_path = DATA_DIR / "realistic_scenarios.jsonl"
    save_scenarios(scenarios, out_path)
    print(f"\nSaved {len(scenarios)} scenarios to {out_path}")

    temporal_count = sum(1 for s in scenarios if "delayed" in s.category or "invalidation" in s.category or "abstention" in s.category)
    print(f"Synthesis scenarios: {len(scenarios) - temporal_count}")
    print(f"Temporal scenarios:  {temporal_count}")


if __name__ == "__main__":
    main()
