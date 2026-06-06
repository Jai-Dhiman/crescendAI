"""Tests for briefing.py -- the matcher + transform + memory loop.

build_briefing ties slice B (match) and slice C (transforms) together: given a
diagnosed weakness and a query embedding, it emits an ExerciseBriefing that maps
onto the api-side ExerciseArtifact contract, choosing a dimension-appropriate
exercise type + deterministic transform, and respecting a 3-day cooldown so the
same weakness is not re-prescribed back-to-back.

Synthetic-catalog tests cover the planning logic without Aria weights; one
end-to-end test runs the whole loop against the real 22-MIDI catalog and
actually executes the planned transform to prove the plan is realizable.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from partitura.score import Note

from exercise_corpus import Primitive
from exercise_corpus.briefing import (
    CooldownError,
    Diagnosis,
    ExerciseBriefing,
    PrescriptionRecord,
    build_briefing,
    should_prescribe,
)
from exercise_corpus.catalog import write_primitives
from exercise_corpus.transforms import load_primitive, scale_tempo

DAY = 86_400
REAL_DB = Path("data/exercise_primitives.db")


def _make_catalog(tmp_path: Path, vectors: dict[str, np.ndarray]) -> Path:
    primitives, embeddings = [], {}
    for i, (pid, vec) in enumerate(vectors.items(), start=1):
        source = pid.split("_")[0]
        primitives.append(
            Primitive(
                primitive_id=pid,
                source=source,
                source_exercise_number=i,
                title=f"{source} {i}",
                musicxml_path=tmp_path / f"{pid}.xml",
                midi_path=tmp_path / f"{pid}.mid",
                n_notes=100 + i,
            )
        )
        embeddings[pid] = torch.from_numpy(vec.astype(np.float32))
    db = tmp_path / "cat.db"
    write_primitives(primitives, embeddings, db)
    return db


def _diagnosis(dimension="timing", severity="moderate", bars=(5, 8)) -> Diagnosis:
    return Diagnosis(
        dimension=dimension,
        severity=severity,
        bar_range=bars,
        piece_id="fur_elise",
    )


def test_build_briefing_emits_briefing_for_a_weakness(tmp_path: Path):
    rng = np.random.default_rng(0)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 6)}
    db = _make_catalog(tmp_path, vectors)

    briefing = build_briefing(
        _diagnosis(), query_embedding=vectors["hanon_003"], db_path=db, history=[], now=0
    )

    assert isinstance(briefing, ExerciseBriefing)
    assert briefing.matched_primitive_id == "hanon_003"  # self-match ranks #1
    assert briefing.target_dimension == "timing"
    assert briefing.exercise_type == "segment_loop"
    assert briefing.bar_range == (5, 8)  # student's piece bars pass through
    assert "5" in briefing.instruction and "8" in briefing.instruction
    assert briefing.estimated_minutes == 5  # moderate
    assert len(briefing.candidates) >= 1


def test_dimension_selects_appropriate_transform(tmp_path: Path):
    rng = np.random.default_rng(1)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 4)}
    db = _make_catalog(tmp_path, vectors)
    q = vectors["hanon_001"]

    phrasing = build_briefing(
        _diagnosis("phrasing", "significant"), query_embedding=q, db_path=db, history=[], now=0
    )
    assert phrasing.exercise_type == "slow_practice"
    assert phrasing.transform == "tempo"
    assert phrasing.estimated_minutes == 8  # significant

    dynamics = build_briefing(
        _diagnosis("dynamics"), query_embedding=q, db_path=db, history=[], now=0
    )
    assert dynamics.exercise_type == "dynamic_exaggeration"
    # no deterministic symbolic transform exists for dynamics in slice C
    assert dynamics.transform is None


def test_minor_severity_rejected(tmp_path: Path):
    rng = np.random.default_rng(2)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 4)}
    db = _make_catalog(tmp_path, vectors)
    with pytest.raises(ValueError, match="severity"):
        build_briefing(
            _diagnosis(severity="minor"),
            query_embedding=vectors["hanon_001"],
            db_path=db,
            history=[],
            now=0,
        )


def test_cooldown_blocks_recent_repeat(tmp_path: Path):
    rng = np.random.default_rng(3)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 4)}
    db = _make_catalog(tmp_path, vectors)
    q = vectors["hanon_001"]
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 1 * DAY,  # yesterday, overlaps bars 5-8
        )
    ]
    with pytest.raises(CooldownError):
        build_briefing(_diagnosis(bars=(6, 9)), query_embedding=q, db_path=db, history=history, now=now)


def test_cooldown_expired_allows_represcribe(tmp_path: Path):
    rng = np.random.default_rng(4)
    vectors = {f"hanon_{i:03d}": rng.standard_normal(512) for i in range(1, 4)}
    db = _make_catalog(tmp_path, vectors)
    q = vectors["hanon_001"]
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 5 * DAY,  # 5 days ago, outside 3-day window
        )
    ]
    briefing = build_briefing(_diagnosis(bars=(5, 8)), query_embedding=q, db_path=db, history=history, now=now)
    assert isinstance(briefing, ExerciseBriefing)


def test_should_prescribe_predicate():
    now = 10 * DAY
    rec = PrescriptionRecord("hanon_001", "timing", (5, 8), now - 1 * DAY)
    # same dimension, overlapping bars, within window -> blocked
    assert should_prescribe("timing", (6, 9), [rec], now) is False
    # different dimension -> allowed
    assert should_prescribe("dynamics", (6, 9), [rec], now) is True
    # non-overlapping bars -> allowed
    assert should_prescribe("timing", (20, 24), [rec], now) is True
    # outside window -> allowed
    assert should_prescribe("timing", (6, 9), [rec], now + 5 * DAY) is True


def test_end_to_end_briefing_transform_is_realizable():
    """Full loop on the real catalog: match -> plan -> execute the transform."""
    if not REAL_DB.exists():
        pytest.skip("real catalog DB not present")
    idx_q = None
    # Use a real primitive's embedding as the query (deterministic self-match).
    from exercise_corpus.match import load_index

    idx = load_index(REAL_DB)
    idx_q = next(r.embedding for r in idx.rows if r.primitive_id == "hanon_005")

    briefing = build_briefing(
        _diagnosis("phrasing", "significant"),
        query_embedding=idx_q,
        db_path=REAL_DB,
        history=[],
        now=0,
    )
    # Execute the planned transform against the matched primitive's MIDI.
    part = load_primitive(Path("data/midi/exercise_primitives") / f"{briefing.matched_primitive_id}.mid")
    assert briefing.transform == "tempo"
    variant = scale_tempo(part, briefing.transform_params["factor"])
    assert len(list(variant.part.iter_all(Note))) == len(list(part.iter_all(Note)))
