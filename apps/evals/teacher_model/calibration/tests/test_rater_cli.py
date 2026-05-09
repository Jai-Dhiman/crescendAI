from __future__ import annotations

import json
import re

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
