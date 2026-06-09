"""Tests for briefing.py -- the matcher + transform + memory loop.

build_briefing ties dimension-tag retrieval (slice B, match_by_dimension) and
slice C (transforms) together: given a diagnosed weakness and a tag map, it emits
an ExerciseBriefing that maps onto the api-side ExerciseArtifact contract,
choosing a dimension-appropriate exercise type + deterministic transform, and
respecting a 3-day cooldown so the same weakness is not re-prescribed back-to-back.

Synthetic-catalog tests cover the planning logic without Aria weights; one
end-to-end test runs the whole loop against the real catalog (skipped when the
gitignored catalog DB is absent) and executes the planned transform.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from partitura.score import Note

import exercise_corpus
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
from exercise_corpus.match import NoPrimitiveForDimensionError, load_index
from exercise_corpus.tags import TagSet, load_tags
from exercise_corpus.transforms import load_primitive, scale_tempo

DAY = 86_400
REAL_DB = Path("data/exercise_primitives.db")
SHIPPED_TAGS = Path(exercise_corpus.__file__).resolve().parent / "technique_tags.toml"


def _make_catalog(tmp_path: Path, primitive_ids: list[str]) -> Path:
    rng = np.random.default_rng(0)
    primitives, embeddings = [], {}
    for i, pid in enumerate(primitive_ids, start=1):
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
        embeddings[pid] = torch.from_numpy(rng.standard_normal(512).astype(np.float32))
    db = tmp_path / "cat.db"
    write_primitives(primitives, embeddings, db)
    return db


def _tags(mapping: dict[str, list[str]]) -> dict[str, TagSet]:
    return {pid: TagSet(frozenset(dims), frozenset(), key="C") for pid, dims in mapping.items()}


def _diagnosis(dimension="timing", severity="moderate", bars=(5, 8)) -> Diagnosis:
    return Diagnosis(
        dimension=dimension,
        severity=severity,
        bar_range=bars,
        piece_id="bach.prelude.bwv_846",
    )


def test_build_briefing_emits_briefing_for_a_weakness(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002", "hanon_003"])
    tags = _tags(
        {pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002", "hanon_003"]}
    )

    briefing = build_briefing(_diagnosis(), tags, history=[], now=0, db_path=db)

    assert isinstance(briefing, ExerciseBriefing)
    # Deterministic ranking -> lowest (source_exercise_number, primitive_id) first.
    assert briefing.matched_primitive_id == "hanon_001"
    assert briefing.matched_primitive_id in {"hanon_001", "hanon_002", "hanon_003"}
    assert briefing.target_dimension == "timing"
    assert briefing.exercise_type == "segment_loop"
    assert briefing.bar_range == (5, 8)
    assert "5" in briefing.instruction and "8" in briefing.instruction
    assert briefing.estimated_minutes == 5  # moderate
    assert len(briefing.candidates) >= 1


def test_phrasing_selects_slow_practice_tempo(tmp_path: Path):
    db = _make_catalog(tmp_path, ["burgmuller_001"])
    tags = _tags({"burgmuller_001": ["phrasing", "interpretation"]})

    phrasing = build_briefing(
        _diagnosis("phrasing", "significant"), tags, history=[], now=0, db_path=db
    )
    assert phrasing.exercise_type == "slow_practice"
    assert phrasing.transform == "tempo"
    assert phrasing.estimated_minutes == 8  # significant


def test_minor_severity_rejected(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    with pytest.raises(ValueError, match="severity"):
        build_briefing(_diagnosis(severity="minor"), tags, history=[], now=0, db_path=db)


def test_cooldown_blocks_recent_repeat(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 1 * DAY,
        )
    ]
    with pytest.raises(CooldownError):
        build_briefing(_diagnosis(bars=(6, 9)), tags, history=history, now=now, db_path=db)


def test_cooldown_expired_allows_represcribe(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    now = 10 * DAY
    history = [
        PrescriptionRecord(
            primitive_id="hanon_002",
            dimension="timing",
            bar_range=(5, 8),
            prescribed_at=now - 5 * DAY,
        )
    ]
    briefing = build_briefing(_diagnosis(bars=(5, 8)), tags, history=history, now=now, db_path=db)
    assert isinstance(briefing, ExerciseBriefing)


def test_should_prescribe_predicate():
    now = 10 * DAY
    rec = PrescriptionRecord("hanon_001", "timing", (5, 8), now - 1 * DAY)
    assert should_prescribe("timing", (6, 9), [rec], now) is False
    assert should_prescribe("dynamics", (6, 9), [rec], now) is True
    assert should_prescribe("timing", (20, 24), [rec], now) is True
    assert should_prescribe("timing", (6, 9), [rec], now + 5 * DAY) is True


def test_untagged_dimension_raises(tmp_path: Path):
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    # dynamics is a valid teacher dimension (passes the _DIMENSION_PLAN check) but
    # nothing in this catalog is tagged for it -> honest failure at retrieval.
    with pytest.raises(NoPrimitiveForDimensionError):
        build_briefing(_diagnosis("dynamics"), tags, history=[], now=0, db_path=db)


FIXTURES_SCORES = Path(__file__).resolve().parent / "fixtures" / "scores"


def test_build_briefing_transpose_semitones_eb(tmp_path: Path):
    """C-major drill transposed +3 when passage is in Eb."""
    db = _make_catalog(tmp_path, ["hanon_001", "hanon_002"])
    tags = _tags({pid: ["timing", "articulation"] for pid in ["hanon_001", "hanon_002"]})
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
    )
    assert briefing.transpose_semitones == 3
    assert briefing.target_key == "Eb"


def test_build_briefing_transpose_none_when_key_absent(tmp_path: Path):
    """transpose_semitones and target_key are None when piece JSON has null key_signature."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    # Write a fixture with null key_signature into tmp_path
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(3, 6),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transpose_semitones is None
    assert briefing.target_key is None


def test_build_briefing_excerpt_end_bar_from_bar_range_8_bars(tmp_path: Path):
    """excerpt transform_params["end_bar"] equals bar_range length (8 bars)."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 12),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transform == "excerpt"
    assert briefing.transform_params["start_bar"] == 1
    assert briefing.transform_params["end_bar"] == 8  # 12 - 5 + 1


def test_build_briefing_excerpt_end_bar_from_bar_range_4_bars(tmp_path: Path):
    """excerpt transform_params["end_bar"] equals bar_range length (4 bars)."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    null_key_json = tmp_path / "no_key_piece.json"
    null_key_json.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="no_key_piece",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=tmp_path
    )
    assert briefing.transform == "excerpt"
    assert briefing.transform_params["end_bar"] == 4  # 8 - 5 + 1


def test_build_briefing_target_key_in_instruction(tmp_path: Path):
    """When target_key is set, instruction text contains the key name."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    diag = Diagnosis(
        dimension="timing",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    briefing = build_briefing(
        diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
    )
    assert "Eb" in briefing.instruction


def test_untagged_dimension_raises_before_key_resolution(tmp_path: Path):
    """NoPrimitiveForDimensionError is raised before any key resolution attempt."""
    db = _make_catalog(tmp_path, ["hanon_001"])
    tags = _tags({"hanon_001": ["timing", "articulation"]})
    diag = Diagnosis(
        dimension="dynamics",
        severity="moderate",
        bar_range=(5, 8),
        piece_id="test_piece_eb",
    )
    with pytest.raises(NoPrimitiveForDimensionError):
        build_briefing(
            diag, tags, history=[], now=0, db_path=db, scores_dir=FIXTURES_SCORES
        )


def test_end_to_end_briefing_transform_is_realizable():
    """Full loop on the real catalog: tag-match -> plan -> execute the transform."""
    if not REAL_DB.exists():
        pytest.skip("real catalog DB not present")
    idx = load_index(REAL_DB)
    tags = load_tags(
        SHIPPED_TAGS, known_primitive_ids={r.primitive_id for r in idx.rows}
    )

    briefing = build_briefing(
        _diagnosis("phrasing", "significant"), tags, history=[], now=0, index=idx
    )
    part = load_primitive(
        Path("data/midi/exercise_primitives") / f"{briefing.matched_primitive_id}.mid"
    )
    assert briefing.transform == "tempo"
    variant = scale_tempo(part, briefing.transform_params["factor"])
    assert len(list(variant.part.iter_all(Note))) == len(list(part.iter_all(Note)))
