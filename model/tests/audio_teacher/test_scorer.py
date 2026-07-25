"""Scorer behavior: partitioned cells, ex-ante verdicts, determinism.

The #21 scar: synthetic-vs-real pooling produced a fatally misleading
number once. Any pooled figure in this report is a test failure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audio_teacher.manifest import ContrastPair, ProbeManifest
from audio_teacher.scorer import ProbeIncompleteError, score_responses


def _pair(pair_id: str, axis: str = "pedaling", population: str = "real",
          degraded: str = "a") -> ContrastPair:
    return ContrastPair(
        pair_id=pair_id,
        axis=axis,
        population=population,
        clip_a=Path(f"clips/{pair_id}_a.wav"),
        clip_b=Path(f"clips/{pair_id}_b.wav"),
        degraded=degraded,
        description="",
    )


def _manifest(pairs) -> ProbeManifest:
    return ProbeManifest(sample_rate=16000, pairs=tuple(pairs))


def test_cells_are_partitioned_by_population_and_never_pooled():
    manifest = _manifest(
        [
            _pair("r1", population="real", degraded="a"),
            _pair("r2", population="real", degraded="b"),
            _pair("s1", population="synthetic", degraded="a"),
        ]
    )
    responses = {
        "r1": "blurred.\nANSWER: A",   # correct
        "r2": "muddy.\nANSWER: A",     # wrong (degraded is b)
        "s1": "ANSWER: A",             # correct
    }
    report = score_responses(manifest, responses)
    assert report["cells"]["pedaling/real"] == {
        "n": 2, "correct": 1, "unparseable": 0,
        "accuracy": 0.5, "unparseable_rate": 0.0,
    }
    assert report["cells"]["pedaling/synthetic"] == {
        "n": 1, "correct": 1, "unparseable": 0,
        "accuracy": 1.0, "unparseable_rate": 0.0,
    }
    # No pooled number anywhere: only axis/population cell keys, no
    # bare-axis cell, no overall/pooled key in the report.
    assert set(report["cells"]) == {"pedaling/real", "pedaling/synthetic"}
    assert "overall" not in report and "accuracy" not in report

    with pytest.raises(ProbeIncompleteError):
        score_responses(manifest, {"r1": "ANSWER: A"})  # missing r2, s1
    with pytest.raises(ProbeIncompleteError):
        score_responses(manifest, {**responses, "ghost": "ANSWER: B"})  # extra
