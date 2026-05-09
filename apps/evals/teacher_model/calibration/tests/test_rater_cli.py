from __future__ import annotations

import json

from teacher_model.calibration.rater_cli import redact_for_rater


def test_redacted_view_strips_all_judge_score_fields():
    leak_sentinel = "LEAK_DETECTOR_42"
    row = {
        "synth_id": "p__r__3",
        "piece_slug": "p",
        "recording_id": "r",
        "skill_bucket": 3,
        "composer": "Chopin",
        "title": "Test piece",
        "synthesis_text": "Some teacher feedback here.",
        "muq_means": {"dynamics": 0.5},
        "judge_dimensions": [
            {"criterion": "Audible-Specific Corrective Feedback",
             "process": 2, "outcome": 1, "score": 1,
             "evidence": leak_sentinel, "reason": leak_sentinel},
        ],
        "judge_latency_ms": 1234.5,
        "judge_model": "claude-sonnet-4-6",
    }

    redacted = redact_for_rater(row)
    serialized = json.dumps(redacted)

    assert leak_sentinel not in serialized
    assert "judge_dimensions" not in redacted
    assert "judge_latency_ms" not in redacted
    assert "judge_model" not in redacted
    assert not any("judge" in k.lower() for k in redacted.keys())


def test_redacted_view_keeps_what_rater_needs():
    row = {
        "synth_id": "p__r__3",
        "piece_slug": "p",
        "recording_id": "r",
        "skill_bucket": 3,
        "composer": "Chopin",
        "title": "Test piece",
        "synthesis_text": "Some teacher feedback here.",
        "muq_means": {"dynamics": 0.5},
        "judge_dimensions": [],
    }
    redacted = redact_for_rater(row)
    assert redacted["synth_id"] == "p__r__3"
    assert redacted["composer"] == "Chopin"
    assert redacted["title"] == "Test piece"
    assert redacted["skill_bucket"] == 3
    assert redacted["synthesis_text"] == "Some teacher feedback here."
    # muq_means is allowed because the constrained-rubric dims (Praise outcome,
    # Concrete Artifact outcome) use it as ground truth.
    assert redacted["muq_means"] == {"dynamics": 0.5}


import json
from pathlib import Path
from teacher_model.calibration.rater_cli import (
    capture_synthesis_ratings,
    PHASE_1_SUB_SCORES,
)


def test_capture_writes_one_event_per_sub_score(tmp_path: Path):
    redacted = {
        "synth_id": "p__r__3",
        "synthesis_text": "x",
        "title": "Piece",
        "composer": "Chopin",
        "skill_bucket": 3,
    }
    output = tmp_path / "ratings.jsonl"

    inputs = iter([
        (2, "ev_ascf_p", "rs_ascf_p"),
        (3, "ev_ca_p",   "rs_ca_p"),
        (3, "ev_pr_p",   "rs_pr_p"),
        (3, "ev_au_p",   "rs_au_p"),
        (2, "ev_sc_p",   "rs_sc_p"),
        (3, "ev_st_p",   "rs_st_p"),
        (3, "ev_to_p",   "rs_to_p"),
        (3, "ev_au_o",   "rs_au_o"),
        (3, "ev_to_o",   "rs_to_o"),
        (2, "ev_ca_o",   "rs_ca_o"),
        (2, "ev_pr_o",   "rs_pr_o"),
    ])

    def provider(_redacted: dict, sub_score: str) -> tuple[int, str, str]:
        return next(inputs)

    n = capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id="S001",
        session_idx_start=1,
        output_path=output,
        input_provider=provider,
    )
    assert n == 11
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(events) == 11
    assert all(e["event_type"] == "rating" for e in events)
    assert all(e["synth_id"] == "p__r__3" for e in events)
    assert all(e["session_id"] == "S001" for e in events)
    assert {e["sub_score"] for e in events} == set(PHASE_1_SUB_SCORES)
    assert events[0]["value"] == 2 and events[0]["evidence"] == "ev_ascf_p"


def test_phase_1_sub_scores_has_exactly_eleven():
    assert len(PHASE_1_SUB_SCORES) == 11
    assert "ascf_outcome" not in PHASE_1_SUB_SCORES
    assert "scaffolded_outcome" not in PHASE_1_SUB_SCORES
    assert "style_outcome" not in PHASE_1_SUB_SCORES
