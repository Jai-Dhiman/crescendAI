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


def _bulk(axis: str, population: str, n_correct: int, n_wrong: int,
          n_unparseable: int, start: int = 0):
    """Build (pairs, responses) for one cell. degraded='a'; correct answers
    say A, wrong say B, unparseable give no ANSWER line."""
    pairs, responses = [], {}
    i = start
    for count, text in (
        (n_correct, "ANSWER: A"),
        (n_wrong, "ANSWER: B"),
        (n_unparseable, "cannot tell"),
    ):
        for _ in range(count):
            pid = f"{axis[:3]}_{population[:3]}_{i}"
            pairs.append(_pair(pid, axis=axis, population=population, degraded="a"))
            responses[pid] = text
            i += 1
    return pairs, responses


@pytest.mark.parametrize(
    "cell_specs,expected_verdict,expected_reason_fragment",
    [
        # 20/20 real correct + synthetic all wrong: synthetic never gates.
        ([("pedaling", "real", 20, 0, 0), ("pedaling", "synthetic", 0, 5, 0)],
         "PASS", None),
        # 10/20 real correct: below the 0.70 kill threshold.
        ([("pedaling", "real", 10, 10, 0)], "FAIL", "below"),
        # Only 5 real pairs: insufficient evidence, gate stays closed.
        ([("pedaling", "real", 5, 0, 0)], "FAIL", "only 5 pairs"),
        # Synthetic-only axis: never opens the gate.
        ([("dynamics", "synthetic", 20, 0, 0)], "FAIL", "no real-population"),
        # 17 correct + 3 unparseable of 20: accuracy 0.85 but ambiguity 0.15.
        ([("pedaling", "real", 17, 0, 3)], "FAIL", "unparseable"),
    ],
    ids=["pass_synthetic_never_gates", "fail_accuracy", "fail_insufficient",
         "fail_synthetic_only", "fail_ambiguous"],
)
def test_verdict_applies_ex_ante_kill_rules(
    cell_specs, expected_verdict, expected_reason_fragment
):
    pairs, responses = [], {}
    start = 0
    for axis, population, n_correct, n_wrong, n_unparseable in cell_specs:
        p, r = _bulk(axis, population, n_correct, n_wrong, n_unparseable, start)
        pairs += p
        responses.update(r)
        start += len(p)
    report = score_responses(_manifest(pairs), responses)
    assert report["verdict"] == expected_verdict
    if expected_reason_fragment is None:
        assert report["verdict_reasons"] == []
    else:
        assert any(expected_reason_fragment in r for r in report["verdict_reasons"])


def test_same_inputs_render_byte_identical_reports():
    from audio_teacher.scorer import render_report

    pairs, responses = _bulk("pedaling", "real", 20, 0, 0)
    manifest = _manifest(pairs)
    reversed_responses = dict(reversed(list(responses.items())))
    r1 = render_report(score_responses(manifest, responses))
    r2 = render_report(score_responses(manifest, reversed_responses))
    assert r1 == r2
    assert r1.endswith("\n")
    assert "generated_at" not in r1  # volatile metadata lives in run_meta.json
