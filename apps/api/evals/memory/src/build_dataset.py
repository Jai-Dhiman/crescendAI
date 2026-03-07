"""Build 30 gold-standard memory evaluation scenarios.

5 categories x 6 scenarios each:
  1. Single-dimension progression (fact creation from repeated obs in one dim)
  2. Multi-dimension interaction (cross-dim patterns)
  3. Piece lifecycle (new -> mid-learning -> polishing)
  4. Temporal evolution (creation -> invalidation -> supersession)
  5. Approach/engagement (framing engagement patterns)
"""

from __future__ import annotations

import json
from pathlib import Path

from .scenarios import (
    ExpectedFact,
    MemoryEvalScenario,
    Observation,
    RetrievalQuery,
    SynthesisCheckpoint,
    TemporalAssertion,
    save_scenarios,
)

DATA_DIR = Path(__file__).parents[1] / "data"


def _make_obs(
    idx: int,
    dimension: str,
    text: str,
    framing: str = "correction",
    score: float | None = None,
    baseline: float | None = None,
    piece: dict | None = None,
    session_date: str = "",
    engaged: bool = False,
    session_id: str = "",
) -> Observation:
    return Observation(
        id=f"obs-{idx:03d}",
        dimension=dimension,
        observation_text=text,
        framing=framing,
        dimension_score=score,
        student_baseline=baseline,
        piece_context=json.dumps(piece) if piece else None,
        session_id=session_id or f"sess-{idx:03d}",
        session_date=session_date,
        engaged=engaged,
    )


# ---------------------------------------------------------------------------
# Category 1: Single-dimension progression
# ---------------------------------------------------------------------------

def _single_dim_scenarios() -> list[MemoryEvalScenario]:
    return [
        MemoryEvalScenario(
            id="sd-01",
            name="Dynamics: repeated flat dynamics across sessions",
            category="single_dim",
            observations=[
                _make_obs(1, "dynamics", "Dynamics remain uniformly mezzo-forte throughout the passage", "correction", 2.5, 3.5, session_date="2026-02-01T10:00:00Z"),
                _make_obs(2, "dynamics", "Very little dynamic contrast between forte and piano sections", "correction", 2.3, 3.5, session_date="2026-02-03T10:00:00Z"),
                _make_obs(3, "dynamics", "The crescendo in bars 12-16 lacks gradual build", "suggestion", 2.6, 3.5, session_date="2026-02-05T10:00:00Z"),
                _make_obs(4, "dynamics", "Still playing at a single dynamic level despite marked contrasts", "correction", 2.4, 3.5, session_date="2026-02-07T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd01-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd01-1", fact_text_pattern=r"(?i)(flat|uniform|limited|lack).*(dynamic|contrast)", fact_type="dimension", dimension="dynamics", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd01-1", query_type="active_facts", expected_fact_ids=["ef-sd01-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)dynamic", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="sd-02",
            name="Timing: rushing in fast passages",
            category="single_dim",
            observations=[
                _make_obs(5, "timing", "Noticeable rushing in the sixteenth-note runs", "correction", 2.1, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(6, "timing", "Tempo accelerates through technically demanding sections", "correction", 2.2, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(7, "timing", "Rushing returns in the coda where density increases", "suggestion", 2.0, 3.0, session_date="2026-02-06T10:00:00Z"),
                _make_obs(8, "timing", "Fast passages still tend to accelerate beyond intended tempo", "correction", 2.3, 3.0, session_date="2026-02-08T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd02-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd02-1", fact_text_pattern=r"(?i)(rush|accelerat|tempo).*(fast|passage|dense|technic)", fact_type="dimension", dimension="timing", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd02-1", query_type="active_facts", expected_fact_ids=["ef-sd02-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-09T00:00:00Z", fact_pattern=r"(?i)rush", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="sd-03",
            name="Pedaling: blurry harmonies from over-pedaling",
            category="single_dim",
            observations=[
                _make_obs(9, "pedaling", "Pedaling holds through harmony changes, creating muddy sonority", "correction", 2.0, 3.0, session_date="2026-02-02T10:00:00Z"),
                _make_obs(10, "pedaling", "Sustained pedal blurs the bass line into the treble melody", "correction", 2.1, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(11, "pedaling", "Try lifting the pedal at each harmony change for clarity", "suggestion", 2.2, 3.0, session_date="2026-02-06T10:00:00Z"),
                _make_obs(12, "pedaling", "Still over-pedaling through chord transitions", "correction", 2.0, 3.0, session_date="2026-02-09T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd03-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd03-1", fact_text_pattern=r"(?i)(over-?pedal|muddy|blur|sustain).*(harmony|chord|change)", fact_type="dimension", dimension="pedaling", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd03-1", query_type="active_facts", expected_fact_ids=["ef-sd03-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)pedal", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="sd-04",
            name="Articulation: legato improving over sessions",
            category="single_dim",
            observations=[
                _make_obs(13, "articulation", "Legato line is choppy with gaps between notes", "correction", 2.0, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(14, "articulation", "Some improvement in connecting notes but still breaks in longer phrases", "encouragement", 2.5, 3.0, session_date="2026-02-03T10:00:00Z"),
                _make_obs(15, "articulation", "Legato is more consistent, only breaking at wide interval leaps", "encouragement", 3.0, 3.0, session_date="2026-02-06T10:00:00Z"),
                _make_obs(16, "articulation", "Smooth legato maintained throughout, nice improvement", "encouragement", 3.4, 3.0, session_date="2026-02-09T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd04-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd04-1", fact_text_pattern=r"(?i)legato.*(improv|better|progress)", fact_type="dimension", dimension="articulation", trend="improving", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd04-1", query_type="active_facts", expected_fact_ids=["ef-sd04-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)legato", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="sd-05",
            name="Phrasing: breath points missing consistently",
            category="single_dim",
            observations=[
                _make_obs(17, "phrasing", "Phrases run together without breathing points", "correction", 2.2, 3.0, session_date="2026-02-02T10:00:00Z"),
                _make_obs(18, "phrasing", "No lift between phrase endings and beginnings", "correction", 2.1, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(19, "phrasing", "Musical sentences lack punctuation -- everything flows without pause", "suggestion", 2.3, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(20, "phrasing", "Phrasing still continuous without natural breath marks", "correction", 2.0, 3.0, session_date="2026-02-10T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd05-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd05-1", fact_text_pattern=r"(?i)(phrase|breath|lift|pause|punctuat)", fact_type="dimension", dimension="phrasing", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd05-1", query_type="active_facts", expected_fact_ids=["ef-sd05-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)phras", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="sd-06",
            name="Interpretation: growing voicing awareness",
            category="single_dim",
            observations=[
                _make_obs(21, "interpretation", "Melody and accompaniment at the same volume", "correction", 2.0, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(22, "interpretation", "Starting to bring out the top voice but inconsistently", "encouragement", 2.5, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(23, "interpretation", "Better melody projection, accompaniment still too prominent in LH", "suggestion", 2.8, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(24, "interpretation", "Voicing balance much improved, melody sings above texture", "encouragement", 3.3, 3.0, session_date="2026-02-10T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-sd06-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-sd06-1", fact_text_pattern=r"(?i)(voic|melody|balanc).*(improv|progress|better|grow)", fact_type="dimension", dimension="interpretation", trend="improving", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-sd06-1", query_type="active_facts", expected_fact_ids=["ef-sd06-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)voic", should_be_active=True, category="extraction"),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 2: Multi-dimension interaction
# ---------------------------------------------------------------------------

def _multi_dim_scenarios() -> list[MemoryEvalScenario]:
    return [
        MemoryEvalScenario(
            id="md-01",
            name="Pedaling improves but dynamics worsen",
            category="multi_dim",
            observations=[
                _make_obs(25, "pedaling", "Over-pedaling through cadences", "correction", 2.0, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(26, "dynamics", "Good dynamic shaping in opening theme", "encouragement", 3.5, 3.0, session_date="2026-02-01T10:05:00Z"),
                _make_obs(27, "pedaling", "Cleaner pedal changes at cadence points", "encouragement", 3.0, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(28, "dynamics", "Dynamic range has narrowed, everything mezzo-forte", "correction", 2.3, 3.0, session_date="2026-02-04T10:05:00Z"),
                _make_obs(29, "pedaling", "Pedaling now clear and well-timed", "encouragement", 3.5, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(30, "dynamics", "Still playing without much dynamic contrast", "correction", 2.2, 3.0, session_date="2026-02-07T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md01-1", "ef-md01-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md01-1", fact_text_pattern=r"(?i)pedal.*(improv|clear|better)", fact_type="dimension", dimension="pedaling", trend="improving"),
                ExpectedFact(id="ef-md01-2", fact_text_pattern=r"(?i)dynamic.*(declin|worsen|narrow|flat)", fact_type="dimension", dimension="dynamics", trend="declining"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md01-1", query_type="active_facts", expected_fact_ids=["ef-md01-1", "ef-md01-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)pedal.*improv", should_be_active=True, category="multi_session"),
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)dynamic.*declin", should_be_active=True, category="multi_session"),
            ],
        ),
        MemoryEvalScenario(
            id="md-02",
            name="Timing and articulation both decline",
            category="multi_dim",
            observations=[
                _make_obs(31, "timing", "Rushing through the development section", "correction", 2.2, 3.0, session_date="2026-02-02T10:00:00Z"),
                _make_obs(32, "articulation", "Staccato passages played too connected", "correction", 2.3, 3.0, session_date="2026-02-02T10:05:00Z"),
                _make_obs(33, "timing", "Tempo instability in the recapitulation", "correction", 2.0, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(34, "articulation", "Accents missing on downbeats", "correction", 2.1, 3.0, session_date="2026-02-05T10:05:00Z"),
                _make_obs(35, "timing", "Still rushing when note density increases", "correction", 2.1, 3.0, session_date="2026-02-08T10:00:00Z"),
                _make_obs(36, "articulation", "Touch remains undifferentiated between legato and staccato", "correction", 2.0, 3.0, session_date="2026-02-08T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md02-1", "ef-md02-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md02-1", fact_text_pattern=r"(?i)(rush|tempo|timing).*(persist|consist|stable)", fact_type="dimension", dimension="timing", trend="stable"),
                ExpectedFact(id="ef-md02-2", fact_text_pattern=r"(?i)(articulat|staccato|legato|touch).*(persist|undifferent|weak)", fact_type="dimension", dimension="articulation", trend="stable"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md02-1", query_type="active_facts", expected_fact_ids=["ef-md02-1", "ef-md02-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-09T00:00:00Z", fact_pattern=r"(?i)timing|rush", should_be_active=True, category="multi_session"),
            ],
        ),
        MemoryEvalScenario(
            id="md-03",
            name="Phrasing improves, interpretation follows",
            category="multi_dim",
            observations=[
                _make_obs(37, "phrasing", "Phrases lack direction and shape", "correction", 2.0, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(38, "interpretation", "Playing is technically correct but musically flat", "suggestion", 2.0, 3.0, session_date="2026-02-01T10:05:00Z"),
                _make_obs(39, "phrasing", "Better phrase shaping with clearer rise and fall", "encouragement", 2.8, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(40, "interpretation", "More musical expression emerging", "encouragement", 2.5, 3.0, session_date="2026-02-05T10:05:00Z"),
                _make_obs(41, "phrasing", "Phrases now have clear direction and arrival points", "encouragement", 3.5, 3.0, session_date="2026-02-09T10:00:00Z"),
                _make_obs(42, "interpretation", "Musical personality starting to show through", "encouragement", 3.2, 3.0, session_date="2026-02-09T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md03-1", "ef-md03-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md03-1", fact_text_pattern=r"(?i)phras.*(improv|better|progress)", fact_type="dimension", dimension="phrasing", trend="improving"),
                ExpectedFact(id="ef-md03-2", fact_text_pattern=r"(?i)(interpret|express|music).*(improv|emerg|develop)", fact_type="dimension", dimension="interpretation", trend="improving"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md03-1", query_type="active_facts", expected_fact_ids=["ef-md03-1", "ef-md03-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)phras.*improv", should_be_active=True, category="multi_session"),
            ],
        ),
        MemoryEvalScenario(
            id="md-04",
            name="Dynamics strong, pedaling weak -- no conflation",
            category="multi_dim",
            observations=[
                _make_obs(43, "dynamics", "Excellent dynamic range and contrast", "encouragement", 4.0, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(44, "pedaling", "Pedal changes lag behind harmony changes", "correction", 2.0, 3.0, session_date="2026-02-01T10:05:00Z"),
                _make_obs(45, "dynamics", "Dynamic shading beautifully handled in the second theme", "encouragement", 4.2, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(46, "pedaling", "Pedal still held too long at key transitions", "correction", 2.1, 3.0, session_date="2026-02-04T10:05:00Z"),
                _make_obs(47, "dynamics", "Continues to show strong dynamic control", "encouragement", 4.1, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(48, "pedaling", "Muddy texture from sustained pedal in chromatic passages", "correction", 1.9, 3.0, session_date="2026-02-07T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md04-1", "ef-md04-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md04-1", fact_text_pattern=r"(?i)dynamic.*(strong|excell|strength)", fact_type="dimension", dimension="dynamics", trend="stable"),
                ExpectedFact(id="ef-md04-2", fact_text_pattern=r"(?i)pedal.*(weak|persist|problem|issue)", fact_type="dimension", dimension="pedaling", trend="stable"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md04-1", query_type="active_facts", expected_fact_ids=["ef-md04-1", "ef-md04-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)dynamic.*strong", should_be_active=True, category="multi_session"),
            ],
        ),
        MemoryEvalScenario(
            id="md-05",
            name="All dimensions improving -- general progress fact",
            category="multi_dim",
            observations=[
                _make_obs(49, "dynamics", "Much better dynamic contrast today", "encouragement", 3.5, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(50, "timing", "Steady tempo throughout, good control", "encouragement", 3.3, 3.0, session_date="2026-02-01T10:05:00Z"),
                _make_obs(51, "phrasing", "Phrases are well-shaped with clear direction", "encouragement", 3.4, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(52, "articulation", "Clean articulation in both legato and staccato", "encouragement", 3.6, 3.0, session_date="2026-02-04T10:05:00Z"),
                _make_obs(53, "pedaling", "Appropriate pedaling throughout", "encouragement", 3.2, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(54, "interpretation", "Playing with more musical conviction and personality", "encouragement", 3.5, 3.0, session_date="2026-02-07T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md05-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md05-1", fact_text_pattern=r"(?i)(overall|general|broad|across).*(improv|progress|growth)", fact_type="arc", trend="improving"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md05-1", query_type="active_facts", expected_fact_ids=["ef-md05-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)improv", should_be_active=True, category="multi_session"),
            ],
        ),
        MemoryEvalScenario(
            id="md-06",
            name="Interpretation regresses while timing improves",
            category="multi_dim",
            observations=[
                _make_obs(55, "interpretation", "Expressive playing with rubato and color changes", "encouragement", 3.8, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(56, "timing", "Some tempo instabilities in transitions", "correction", 2.5, 3.0, session_date="2026-02-01T10:05:00Z"),
                _make_obs(57, "interpretation", "Less expressive today, playing more mechanically", "correction", 2.8, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(58, "timing", "Smoother transitions with better tempo control", "encouragement", 3.2, 3.0, session_date="2026-02-05T10:05:00Z"),
                _make_obs(59, "interpretation", "Playing has become quite flat and routine", "correction", 2.3, 3.0, session_date="2026-02-09T10:00:00Z"),
                _make_obs(60, "timing", "Solid rhythmic foundation throughout", "encouragement", 3.5, 3.0, session_date="2026-02-09T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-md06-1", "ef-md06-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-md06-1", fact_text_pattern=r"(?i)(interpret|express).*(declin|regress|less|worsen)", fact_type="dimension", dimension="interpretation", trend="declining"),
                ExpectedFact(id="ef-md06-2", fact_text_pattern=r"(?i)(timing|rhythm|tempo).*(improv|better|solid)", fact_type="dimension", dimension="timing", trend="improving"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-md06-1", query_type="active_facts", expected_fact_ids=["ef-md06-1", "ef-md06-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)interpret.*declin", should_be_active=True, category="multi_session"),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 3: Piece lifecycle
# ---------------------------------------------------------------------------

def _piece_lifecycle_scenarios() -> list[MemoryEvalScenario]:
    chopin = {"composer": "Chopin", "title": "Nocturne Op.9 No.2"}
    beethoven = {"composer": "Beethoven", "title": "Sonata Op.13 Pathetique"}
    bach = {"composer": "Bach", "title": "Prelude in C Major BWV 846"}
    debussy = {"composer": "Debussy", "title": "Clair de Lune"}
    mozart = {"composer": "Mozart", "title": "Sonata K.545"}
    schubert = {"composer": "Schubert", "title": "Impromptu Op.90 No.3"}

    return [
        MemoryEvalScenario(
            id="pl-01",
            name="Chopin Nocturne: new -> mid-learning",
            category="piece_lifecycle",
            observations=[
                _make_obs(61, "dynamics", "Exploring the dynamic palette of this nocturne", "suggestion", 2.0, 3.0, piece=chopin, session_date="2026-02-01T10:00:00Z"),
                _make_obs(62, "pedaling", "Getting familiar with the pedaling requirements", "suggestion", 2.0, 3.0, piece=chopin, session_date="2026-02-01T10:05:00Z"),
                _make_obs(63, "dynamics", "More confident dynamic shaping in the A section", "encouragement", 2.8, 3.0, piece=chopin, session_date="2026-02-05T10:00:00Z"),
                _make_obs(64, "pedaling", "Pedaling becoming cleaner at cadences", "encouragement", 2.7, 3.0, piece=chopin, session_date="2026-02-05T10:05:00Z"),
                _make_obs(65, "phrasing", "Still working on long-line phrasing in the B section", "suggestion", 2.5, 3.0, piece=chopin, session_date="2026-02-09T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=5, expected_new_facts=["ef-pl01-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl01-1", fact_text_pattern=r"(?i)(nocturne|chopin).*(progress|learning|develop|mid)", fact_type="arc", dimension=None, confidence="medium"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl01-1", query_type="piece_facts", piece_title="Nocturne Op.9 No.2"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)nocturne|chopin", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="pl-02",
            name="Beethoven Pathetique: from struggle to polishing",
            category="piece_lifecycle",
            observations=[
                _make_obs(66, "timing", "Struggling with the tempo changes between sections", "correction", 1.8, 3.0, piece=beethoven, session_date="2026-01-15T10:00:00Z"),
                _make_obs(67, "dynamics", "Forte sections lack weight and projection", "correction", 2.0, 3.0, piece=beethoven, session_date="2026-01-20T10:00:00Z"),
                _make_obs(68, "timing", "Tempo transitions getting smoother", "encouragement", 2.8, 3.0, piece=beethoven, session_date="2026-02-01T10:00:00Z"),
                _make_obs(69, "dynamics", "Better dynamic contrast between sections", "encouragement", 3.2, 3.0, piece=beethoven, session_date="2026-02-05T10:00:00Z"),
                _make_obs(70, "interpretation", "Now refining the character of each section", "encouragement", 3.5, 3.0, piece=beethoven, session_date="2026-02-10T10:00:00Z"),
                _make_obs(71, "phrasing", "Polishing phrase endings in the slow movement", "suggestion", 3.3, 3.0, piece=beethoven, session_date="2026-02-12T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-pl02-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl02-1", fact_text_pattern=r"(?i)(beethoven|pathetique|sonata).*(polish|refin|advanc)", fact_type="arc", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl02-1", query_type="piece_facts", piece_title="Sonata Op.13 Pathetique"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-13T00:00:00Z", fact_pattern=r"(?i)pathetique|beethoven", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="pl-03",
            name="Bach Prelude: piece-specific pedaling lesson",
            category="piece_lifecycle",
            observations=[
                _make_obs(72, "pedaling", "No pedal needed here -- keep it clean and dry", "correction", 2.0, 3.0, piece=bach, session_date="2026-02-01T10:00:00Z"),
                _make_obs(73, "articulation", "Even finger articulation needed for the arpeggiated patterns", "suggestion", 2.5, 3.0, piece=bach, session_date="2026-02-03T10:00:00Z"),
                _make_obs(74, "pedaling", "Good -- minimal pedal, letting the counterpoint breathe", "encouragement", 3.2, 3.0, piece=bach, session_date="2026-02-06T10:00:00Z"),
                _make_obs(75, "timing", "Keep the pulse absolutely steady -- this is a meditation", "suggestion", 2.8, 3.0, piece=bach, session_date="2026-02-08T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-pl03-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl03-1", fact_text_pattern=r"(?i)(bach|prelude).*(minimal|dry|clean|no).*(pedal)", fact_type="dimension", dimension="pedaling"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl03-1", query_type="piece_facts", piece_title="Prelude in C Major BWV 846"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-09T00:00:00Z", fact_pattern=r"(?i)bach|prelude", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="pl-04",
            name="Debussy Clair de Lune: pedaling as color tool",
            category="piece_lifecycle",
            observations=[
                _make_obs(76, "pedaling", "Pedal is the color palette here -- use half-pedaling for shimmer", "suggestion", 2.5, 3.0, piece=debussy, session_date="2026-02-02T10:00:00Z"),
                _make_obs(77, "dynamics", "This piece lives in pianissimo -- explore the softest range", "suggestion", 2.3, 3.0, piece=debussy, session_date="2026-02-02T10:05:00Z"),
                _make_obs(78, "pedaling", "Half-pedaling attempts are improving the color", "encouragement", 3.0, 3.0, piece=debussy, session_date="2026-02-06T10:00:00Z"),
                _make_obs(79, "interpretation", "Capturing the impressionistic quality better now", "encouragement", 3.2, 3.0, piece=debussy, session_date="2026-02-10T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-pl04-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl04-1", fact_text_pattern=r"(?i)(debussy|clair).*(pedal|half-pedal|color)", fact_type="dimension", dimension="pedaling"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl04-1", query_type="piece_facts", piece_title="Clair de Lune"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)debussy|clair", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="pl-05",
            name="Mozart Sonata: contrasting piece-specific vs general facts",
            category="piece_lifecycle",
            observations=[
                _make_obs(80, "articulation", "Mozart requires lighter, more detached touch than your Chopin", "suggestion", 2.5, 3.0, piece=mozart, session_date="2026-02-01T10:00:00Z"),
                _make_obs(81, "timing", "Classical phrasing needs stricter rhythmic discipline", "correction", 2.3, 3.0, piece=mozart, session_date="2026-02-03T10:00:00Z"),
                _make_obs(82, "articulation", "Good progress on the lighter touch for Alberti bass", "encouragement", 3.0, 3.0, piece=mozart, session_date="2026-02-06T10:00:00Z"),
                _make_obs(83, "interpretation", "Keeping it elegant and balanced -- very Classical", "encouragement", 3.2, 3.0, piece=mozart, session_date="2026-02-09T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-pl05-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl05-1", fact_text_pattern=r"(?i)(mozart|classical).*(lighter|detach|touch|style)", fact_type="dimension", dimension="articulation"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl05-1", query_type="piece_facts", piece_title="Sonata K.545"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)mozart", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="pl-06",
            name="Schubert Impromptu: long learning arc",
            category="piece_lifecycle",
            observations=[
                _make_obs(84, "phrasing", "The long melodic lines need more breath and direction", "correction", 2.0, 3.0, piece=schubert, session_date="2026-01-10T10:00:00Z"),
                _make_obs(85, "dynamics", "Accompaniment overpowers the melody throughout", "correction", 2.0, 3.0, piece=schubert, session_date="2026-01-15T10:00:00Z"),
                _make_obs(86, "phrasing", "Phrase direction improving but still needs more arc", "encouragement", 2.5, 3.0, piece=schubert, session_date="2026-02-01T10:00:00Z"),
                _make_obs(87, "dynamics", "Better balance -- melody now audible above the triplets", "encouragement", 3.0, 3.0, piece=schubert, session_date="2026-02-05T10:00:00Z"),
                _make_obs(88, "phrasing", "Beautiful long-line phrasing in the A section now", "encouragement", 3.5, 3.0, piece=schubert, session_date="2026-02-10T10:00:00Z"),
                _make_obs(89, "interpretation", "The Schubertian singing quality is coming through", "encouragement", 3.4, 3.0, piece=schubert, session_date="2026-02-12T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-pl06-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-pl06-1", fact_text_pattern=r"(?i)(schubert|impromptu).*(progress|improv|polish)", fact_type="arc", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-pl06-1", query_type="piece_facts", piece_title="Impromptu Op.90 No.3"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-13T00:00:00Z", fact_pattern=r"(?i)schubert|impromptu", should_be_active=True, category="extraction"),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 4: Temporal evolution
# ---------------------------------------------------------------------------

def _temporal_scenarios() -> list[MemoryEvalScenario]:
    return [
        MemoryEvalScenario(
            id="te-01",
            name="Dynamics weakness resolves over time",
            category="temporal",
            observations=[
                _make_obs(90, "dynamics", "Flat dynamics throughout", "correction", 2.0, 3.0, session_date="2026-01-15T10:00:00Z"),
                _make_obs(91, "dynamics", "Still very limited dynamic range", "correction", 2.2, 3.0, session_date="2026-01-20T10:00:00Z"),
                _make_obs(92, "dynamics", "Some dynamic contrast emerging", "encouragement", 2.8, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(93, "dynamics", "Good dynamic shaping now consistent", "encouragement", 3.5, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(94, "dynamics", "Dynamic control strong -- applying contrast naturally", "encouragement", 3.8, 3.0, session_date="2026-02-10T10:00:00Z"),
                _make_obs(95, "dynamics", "Excellent dynamic range maintained across repertoire", "encouragement", 4.0, 3.0, session_date="2026-02-15T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=3, expected_new_facts=["ef-te01-1"]),
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-te01-2"], expected_invalidations=["ef-te01-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-te01-1", fact_text_pattern=r"(?i)(flat|limited|weak).*(dynamic)", fact_type="dimension", dimension="dynamics", trend="stable", confidence="high"),
                ExpectedFact(id="ef-te01-2", fact_text_pattern=r"(?i)dynamic.*(resolv|improv|strong|master)", fact_type="dimension", dimension="dynamics", trend="resolved"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-te01-1", query_type="active_facts", expected_fact_ids=["ef-te01-2"], expected_absent_ids=["ef-te01-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-01-25T00:00:00Z", fact_pattern=r"(?i)(flat|limited|weak).*dynamic", should_be_active=True, category="temporal"),
                TemporalAssertion(query_time="2026-02-16T00:00:00Z", fact_pattern=r"(?i)(flat|limited|weak).*dynamic", should_be_active=False, category="knowledge_update"),
                TemporalAssertion(query_time="2026-02-16T00:00:00Z", fact_pattern=r"(?i)dynamic.*(resolv|improv|strong)", should_be_active=True, category="knowledge_update"),
            ],
        ),
        MemoryEvalScenario(
            id="te-02",
            name="Fact supersession: pedaling issue changes character",
            category="temporal",
            observations=[
                _make_obs(96, "pedaling", "Over-pedaling causing muddy sound", "correction", 2.0, 3.0, session_date="2026-01-20T10:00:00Z"),
                _make_obs(97, "pedaling", "Still holding pedal too long", "correction", 2.1, 3.0, session_date="2026-01-25T10:00:00Z"),
                _make_obs(98, "pedaling", "Pedal changes improved but now too dry and choppy", "correction", 2.5, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(99, "pedaling", "Under-pedaling -- needs more sustain for legato", "correction", 2.3, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(100, "pedaling", "Still too little pedal, losing warmth", "correction", 2.2, 3.0, session_date="2026-02-10T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=2, expected_new_facts=["ef-te02-1"]),
                SynthesisCheckpoint(after_observation_index=5, expected_new_facts=["ef-te02-2"], expected_invalidations=["ef-te02-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-te02-1", fact_text_pattern=r"(?i)over-?pedal", fact_type="dimension", dimension="pedaling", trend="stable"),
                ExpectedFact(id="ef-te02-2", fact_text_pattern=r"(?i)(under-?pedal|too little|too dry|insufficient)", fact_type="dimension", dimension="pedaling", trend="new"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-te02-1", query_type="active_facts", expected_fact_ids=["ef-te02-2"], expected_absent_ids=["ef-te02-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-01-26T00:00:00Z", fact_pattern=r"(?i)over-?pedal", should_be_active=True, category="temporal"),
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)over-?pedal", should_be_active=False, category="knowledge_update"),
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)(under-?pedal|too little|too dry)", should_be_active=True, category="knowledge_update"),
            ],
        ),
        MemoryEvalScenario(
            id="te-03",
            name="Abstention: insufficient evidence for a fact",
            category="temporal",
            observations=[
                _make_obs(101, "timing", "Slight rushing in the coda", "correction", 2.8, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(102, "dynamics", "Good dynamic range", "encouragement", 3.2, 3.0, session_date="2026-02-03T10:00:00Z"),
                _make_obs(103, "articulation", "Clean articulation overall", "encouragement", 3.0, 3.0, session_date="2026-02-05T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=3, expected_new_facts=[]),
            ],
            expected_facts=[],
            retrieval_queries=[
                RetrievalQuery(id="rq-te03-1", query_type="active_facts", expected_fact_ids=[], expected_absent_ids=[]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-06T00:00:00Z", fact_pattern=r"(?i)timing.*rush", should_be_active=False, category="abstention"),
            ],
        ),
        MemoryEvalScenario(
            id="te-04",
            name="Stale student-reported goal should be flagged",
            category="temporal",
            observations=[
                _make_obs(104, "dynamics", "Working on dynamic contrast as requested", "suggestion", 2.5, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(105, "timing", "Good tempo control today", "encouragement", 3.2, 3.0, session_date="2026-02-03T10:00:00Z"),
                _make_obs(106, "phrasing", "Phrasing is coming along nicely", "encouragement", 3.0, 3.0, session_date="2026-02-05T10:00:00Z"),
            ],
            baselines={"dynamics": 3.0, "timing": 3.0, "pedaling": 3.0, "articulation": 3.0, "phrasing": 3.0, "interpretation": 3.0},
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=3, expected_invalidations=["stale-goal-1"]),
            ],
            expected_facts=[],
            retrieval_queries=[
                RetrievalQuery(id="rq-te04-1", query_type="active_facts", expected_absent_ids=["stale-goal-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-06T00:00:00Z", fact_pattern=r"(?i)want.*learn.*jazz", should_be_active=False, category="abstention"),
            ],
        ),
        MemoryEvalScenario(
            id="te-05",
            name="Fact confidence escalation: low -> medium -> high",
            category="temporal",
            observations=[
                _make_obs(107, "articulation", "Staccato is a bit heavy", "correction", 2.5, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(108, "articulation", "Staccato still lacks crispness", "correction", 2.4, 3.0, session_date="2026-02-04T10:00:00Z"),
                _make_obs(109, "articulation", "Heavy staccato persists across repertoire", "correction", 2.3, 3.0, session_date="2026-02-07T10:00:00Z"),
                _make_obs(110, "articulation", "Staccato consistently too heavy and connected", "correction", 2.2, 3.0, session_date="2026-02-10T10:00:00Z"),
                _make_obs(111, "articulation", "Heavy staccato remains the primary articulation issue", "correction", 2.1, 3.0, session_date="2026-02-13T10:00:00Z"),
                _make_obs(112, "articulation", "Fifth session with same staccato issue noted", "correction", 2.0, 3.0, session_date="2026-02-16T10:00:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=3, expected_new_facts=["ef-te05-1"]),
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-te05-2"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-te05-1", fact_text_pattern=r"(?i)staccato.*(heavy|crisp|weak)", fact_type="dimension", dimension="articulation", trend="stable", confidence="medium"),
                ExpectedFact(id="ef-te05-2", fact_text_pattern=r"(?i)staccato.*(heavy|persist|consist)", fact_type="dimension", dimension="articulation", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-te05-1", query_type="active_facts", expected_fact_ids=["ef-te05-2"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)staccato", should_be_active=True, category="temporal"),
                TemporalAssertion(query_time="2026-02-17T00:00:00Z", fact_pattern=r"(?i)staccato", should_be_active=True, category="temporal"),
            ],
        ),
        MemoryEvalScenario(
            id="te-06",
            name="Multi-session cross-reference: timing issue spans 3 sessions",
            category="temporal",
            observations=[
                _make_obs(113, "timing", "Rubato excessive in the slow section", "correction", 2.3, 3.0, session_date="2026-02-01T10:00:00Z"),
                _make_obs(114, "timing", "Rubato again pulling phrases out of shape", "correction", 2.2, 3.0, session_date="2026-02-05T10:00:00Z"),
                _make_obs(115, "timing", "Excessive rubato continues to distort phrase structure", "correction", 2.1, 3.0, session_date="2026-02-09T10:00:00Z"),
                _make_obs(116, "phrasing", "Phrasing suffers when rubato gets too free", "correction", 2.3, 3.0, session_date="2026-02-09T10:05:00Z"),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=["ef-te06-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-te06-1", fact_text_pattern=r"(?i)(rubato|excessive).*(timing|phrase|distort)", fact_type="dimension", dimension="timing", trend="stable", confidence="high"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-te06-1", query_type="active_facts", expected_fact_ids=["ef-te06-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-10T00:00:00Z", fact_pattern=r"(?i)rubato", should_be_active=True, category="multi_session"),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Category 5: Approach/engagement
# ---------------------------------------------------------------------------

def _engagement_scenarios() -> list[MemoryEvalScenario]:
    return [
        MemoryEvalScenario(
            id="en-01",
            name="Student engages with correction framing, ignores suggestion",
            category="engagement",
            observations=[
                _make_obs(117, "dynamics", "Your dynamic range is too narrow", "correction", 2.5, 3.0, session_date="2026-02-01T10:00:00Z", engaged=True),
                _make_obs(118, "dynamics", "Try varying volume between phrases", "suggestion", 2.5, 3.0, session_date="2026-02-03T10:00:00Z", engaged=False),
                _make_obs(119, "timing", "Rushing through the transition", "correction", 2.3, 3.0, session_date="2026-02-05T10:00:00Z", engaged=True),
                _make_obs(120, "timing", "You might try counting through the bridge", "suggestion", 2.3, 3.0, session_date="2026-02-07T10:00:00Z", engaged=False),
                _make_obs(121, "pedaling", "Pedal is held too long here", "correction", 2.0, 3.0, session_date="2026-02-09T10:00:00Z", engaged=True),
                _make_obs(122, "pedaling", "Consider using half-pedal technique", "suggestion", 2.0, 3.0, session_date="2026-02-11T10:00:00Z", engaged=False),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-en01-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-en01-1", fact_text_pattern=r"(?i)(engag|respond|prefer).*(correction|direct)", fact_type="approach"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-en01-1", query_type="active_facts", expected_fact_ids=["ef-en01-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-12T00:00:00Z", fact_pattern=r"(?i)engag.*correction", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="en-02",
            name="Student prefers encouragement framing",
            category="engagement",
            observations=[
                _make_obs(123, "dynamics", "Good effort on dynamic contrast today", "encouragement", 2.8, 3.0, session_date="2026-02-01T10:00:00Z", engaged=True),
                _make_obs(124, "dynamics", "Dynamic range still needs work", "correction", 2.5, 3.0, session_date="2026-02-03T10:00:00Z", engaged=False),
                _make_obs(125, "timing", "Nice improvement in your tempo stability", "encouragement", 2.9, 3.0, session_date="2026-02-05T10:00:00Z", engaged=True),
                _make_obs(126, "timing", "Tempo still inconsistent in places", "correction", 2.5, 3.0, session_date="2026-02-07T10:00:00Z", engaged=False),
                _make_obs(127, "articulation", "Your staccato is getting crisper", "encouragement", 3.0, 3.0, session_date="2026-02-09T10:00:00Z", engaged=True),
                _make_obs(128, "articulation", "Staccato needs to be shorter", "correction", 2.5, 3.0, session_date="2026-02-11T10:00:00Z", engaged=False),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-en02-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-en02-1", fact_text_pattern=r"(?i)(engag|respond|prefer).*(encouragement|positive|praise)", fact_type="approach"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-en02-1", query_type="active_facts", expected_fact_ids=["ef-en02-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-12T00:00:00Z", fact_pattern=r"(?i)engag.*encouragement", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="en-03",
            name="Engagement shifts from correction to question over time",
            category="engagement",
            observations=[
                _make_obs(129, "dynamics", "Your dynamics are flat", "correction", 2.5, 3.0, session_date="2026-01-15T10:00:00Z", engaged=True),
                _make_obs(130, "timing", "Rushing in fast passages", "correction", 2.3, 3.0, session_date="2026-01-20T10:00:00Z", engaged=True),
                _make_obs(131, "dynamics", "What do you think the composer intended dynamically here?", "question", 2.5, 3.0, session_date="2026-02-01T10:00:00Z", engaged=True),
                _make_obs(132, "timing", "Tempo is unsteady", "correction", 2.3, 3.0, session_date="2026-02-05T10:00:00Z", engaged=False),
                _make_obs(133, "phrasing", "Where do you hear the phrase climax?", "question", 2.5, 3.0, session_date="2026-02-08T10:00:00Z", engaged=True),
                _make_obs(134, "dynamics", "Dynamic contrast insufficient", "correction", 2.5, 3.0, session_date="2026-02-10T10:00:00Z", engaged=False),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-en03-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-en03-1", fact_text_pattern=r"(?i)(engag|respond|prefer).*(question|inquiry|socratic)", fact_type="approach"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-en03-1", query_type="active_facts", expected_fact_ids=["ef-en03-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-11T00:00:00Z", fact_pattern=r"(?i)engag.*question", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="en-04",
            name="No clear engagement pattern -- abstention expected",
            category="engagement",
            observations=[
                _make_obs(135, "dynamics", "Flat dynamics", "correction", 2.5, 3.0, session_date="2026-02-01T10:00:00Z", engaged=True),
                _make_obs(136, "timing", "Try slowing down here", "suggestion", 2.5, 3.0, session_date="2026-02-03T10:00:00Z", engaged=True),
                _make_obs(137, "pedaling", "Nice pedaling improvement", "encouragement", 3.0, 3.0, session_date="2026-02-05T10:00:00Z", engaged=False),
                _make_obs(138, "articulation", "What articulation works here?", "question", 2.5, 3.0, session_date="2026-02-07T10:00:00Z", engaged=True),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=4, expected_new_facts=[]),
            ],
            expected_facts=[],
            retrieval_queries=[
                RetrievalQuery(id="rq-en04-1", query_type="active_facts"),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)engag.*(prefer|pattern)", should_be_active=False, category="abstention"),
            ],
        ),
        MemoryEvalScenario(
            id="en-05",
            name="Student engages with dimension-specific feedback (dynamics)",
            category="engagement",
            observations=[
                _make_obs(139, "dynamics", "Work on your crescendo buildup", "correction", 2.5, 3.0, session_date="2026-02-01T10:00:00Z", engaged=True),
                _make_obs(140, "timing", "Rushing in the allegro", "correction", 2.3, 3.0, session_date="2026-02-01T10:05:00Z", engaged=False),
                _make_obs(141, "dynamics", "Try more contrast in the recapitulation", "suggestion", 2.5, 3.0, session_date="2026-02-04T10:00:00Z", engaged=True),
                _make_obs(142, "pedaling", "Pedal is muddy in the bridge", "correction", 2.0, 3.0, session_date="2026-02-04T10:05:00Z", engaged=False),
                _make_obs(143, "dynamics", "The fortissimo needs more weight", "correction", 2.5, 3.0, session_date="2026-02-07T10:00:00Z", engaged=True),
                _make_obs(144, "articulation", "Staccato too heavy", "correction", 2.5, 3.0, session_date="2026-02-07T10:05:00Z", engaged=False),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-en05-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-en05-1", fact_text_pattern=r"(?i)(engag|respond|interest).*(dynamic)", fact_type="approach"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-en05-1", query_type="active_facts", expected_fact_ids=["ef-en05-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-08T00:00:00Z", fact_pattern=r"(?i)engag.*dynamic", should_be_active=True, category="extraction"),
            ],
        ),
        MemoryEvalScenario(
            id="en-06",
            name="Engagement pattern changes when engagement with elaboration stops",
            category="engagement",
            observations=[
                _make_obs(145, "dynamics", "Try a bigger crescendo", "suggestion", 2.5, 3.0, session_date="2026-01-20T10:00:00Z", engaged=True),
                _make_obs(146, "dynamics", "Build more intensity here", "suggestion", 2.5, 3.0, session_date="2026-01-25T10:00:00Z", engaged=True),
                _make_obs(147, "dynamics", "More dynamic contrast needed", "suggestion", 2.5, 3.0, session_date="2026-02-01T10:00:00Z", engaged=False),
                _make_obs(148, "dynamics", "Consider experimenting with subito piano", "suggestion", 2.5, 3.0, session_date="2026-02-05T10:00:00Z", engaged=False),
                _make_obs(149, "dynamics", "Dynamic range remains limited", "correction", 2.5, 3.0, session_date="2026-02-08T10:00:00Z", engaged=True),
                _make_obs(150, "dynamics", "Dynamics flat again today", "correction", 2.5, 3.0, session_date="2026-02-11T10:00:00Z", engaged=True),
            ],
            checkpoints=[
                SynthesisCheckpoint(after_observation_index=6, expected_new_facts=["ef-en06-1"]),
            ],
            expected_facts=[
                ExpectedFact(id="ef-en06-1", fact_text_pattern=r"(?i)(shift|chang|prefer).*(correction|direct)", fact_type="approach"),
            ],
            retrieval_queries=[
                RetrievalQuery(id="rq-en06-1", query_type="active_facts", expected_fact_ids=["ef-en06-1"]),
            ],
            temporal_assertions=[
                TemporalAssertion(query_time="2026-02-12T00:00:00Z", fact_pattern=r"(?i)(shift|prefer).*correction", should_be_active=True, category="extraction"),
            ],
        ),
    ]


def build_all_scenarios() -> list[MemoryEvalScenario]:
    scenarios = []
    scenarios.extend(_single_dim_scenarios())
    scenarios.extend(_multi_dim_scenarios())
    scenarios.extend(_piece_lifecycle_scenarios())
    scenarios.extend(_temporal_scenarios())
    scenarios.extend(_engagement_scenarios())
    return scenarios


def main() -> None:
    scenarios = build_all_scenarios()

    # Validate all
    all_errors = []
    for s in scenarios:
        errors = s.validate()
        if errors:
            all_errors.append((s.id, errors))

    if all_errors:
        print("Validation errors:")
        for sid, errors in all_errors:
            for e in errors:
                print(f"  {sid}: {e}")
        raise SystemExit(1)

    output_path = DATA_DIR / "scenarios.jsonl"
    save_scenarios(scenarios, output_path)
    print(f"Saved {len(scenarios)} scenarios to {output_path}")

    # Summary
    from collections import Counter
    cats = Counter(s.category for s in scenarios)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Verify each has >= 1 checkpoint and >= 1 retrieval query
    for s in scenarios:
        assert len(s.checkpoints) >= 1, f"{s.id} missing checkpoint"
        assert len(s.retrieval_queries) >= 1, f"{s.id} missing retrieval query"

    print("All scenarios validated.")


if __name__ == "__main__":
    main()
