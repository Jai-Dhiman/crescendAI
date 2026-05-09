from __future__ import annotations

import json
from pathlib import Path

import pytest

from teacher_model.calibration.rater_cli import (
    PHASE_1_SUB_SCORES,
    SessionCapExceeded,
    capture_synthesis_ratings,
    compute_resume_state,
    redact_for_rater,
)


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

    def provider(_redacted: dict, _sub_score: str) -> tuple[int, str, str]:
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
    assert [e["session_idx"] for e in events] == list(range(1, 12))


def test_phase_1_sub_scores_has_exactly_eleven():
    assert len(PHASE_1_SUB_SCORES) == 11
    assert "ascf_outcome" not in PHASE_1_SUB_SCORES
    assert "scaffolded_outcome" not in PHASE_1_SUB_SCORES
    assert "style_outcome" not in PHASE_1_SUB_SCORES


def test_session_cap_blocks_when_remaining_slots_insufficient(tmp_path: Path):
    # Cap is 15. Start at session_idx_start=10 — only 6 slots remain, but
    # capturing one synthesis writes 11 rating events, so this must raise
    # BEFORE writing anything.
    redacted = {"synth_id": "p__r__3", "synthesis_text": "x"}
    output = tmp_path / "ratings.jsonl"

    def provider(_, __):
        return (3, "e", "r")

    with pytest.raises(SessionCapExceeded):
        capture_synthesis_ratings(
            redacted_row=redacted,
            sub_scores=PHASE_1_SUB_SCORES,
            session_id="S001",
            session_idx_start=10,
            output_path=output,
            input_provider=provider,
        )

    # Output must be empty or non-existent — no partial writes
    if output.exists():
        assert output.read_text() == ""


def test_session_cap_allows_when_slots_sufficient(tmp_path: Path):
    redacted = {"synth_id": "p__r__3", "synthesis_text": "x"}
    output = tmp_path / "ratings.jsonl"

    def provider(_, __):
        return (3, "e", "r")

    n = capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id="S001",
        session_idx_start=1,
        output_path=output,
        input_provider=provider,
    )
    assert n == 11


def test_session_cap_exact_boundary(tmp_path: Path):
    # session_idx_start=5, 11 sub-scores → last_idx = 5+11-1 = 15 = MAX_RATINGS_PER_SESSION.
    # Exactly at cap must be ALLOWED; one slot over (start=6, last=16) must RAISE.
    redacted = {"synth_id": "p__r__3", "synthesis_text": "x"}

    def provider(_, __):
        return (3, "e", "r")

    # At cap (last_idx == 15): allowed
    n = capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id="S001",
        session_idx_start=5,
        output_path=tmp_path / "at_cap.jsonl",
        input_provider=provider,
    )
    assert n == 11

    # One over cap (last_idx == 16): must raise
    with pytest.raises(SessionCapExceeded):
        capture_synthesis_ratings(
            redacted_row=redacted,
            sub_scores=PHASE_1_SUB_SCORES,
            session_id="S001",
            session_idx_start=6,
            output_path=tmp_path / "over_cap.jsonl",
            input_provider=provider,
        )


def _ratings_for(synth_id: str, sub_scores: list[str]) -> list[dict]:
    return [
        {
            "event_type": "rating",
            "synth_id": synth_id,
            "anchor_origin_id": None,
            "sub_score": sub,
            "value": 2,
            "evidence": "", "reason": "",
            "session_id": "S001", "session_idx": i + 1,
            "ts": "x",
        }
        for i, sub in enumerate(sub_scores)
    ]


def test_resume_state_picks_next_unstarted_synthesis(tmp_path: Path):
    manifest = {
        "main": [
            {"synth_id": "S0", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
            {"synth_id": "S1", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
            {"synth_id": "S2", "band": "high", "era": "Romantic",
             "skill_bucket": 3, "is_anchor_seed": False, "anchor_position": None},
        ],
        "anchors": [],
    }

    ratings_path = tmp_path / "ratings.jsonl"
    # S0 fully rated; S1 partial (only 5 sub-scores); S2 untouched.
    full = _ratings_for("S0", PHASE_1_SUB_SCORES)
    partial = _ratings_for("S1", PHASE_1_SUB_SCORES[:5])
    with ratings_path.open("w") as f:
        for r in full + partial:
            f.write(json.dumps(r) + "\n")

    state = compute_resume_state(manifest=manifest, ratings_path=ratings_path)
    # Resume index is the first synthesis that is NOT fully rated; S1 is
    # partial, so we resume there (the partial events get flagged).
    assert state["next_main_index"] == 1
    assert state["partially_rated"] == ["S1"]
    assert state["fully_rated"] == ["S0"]


def test_resume_state_returns_zero_when_no_ratings(tmp_path: Path):
    manifest = {
        "main": [{"synth_id": "S0", "band": "high", "era": "Romantic",
                  "skill_bucket": 3, "is_anchor_seed": False,
                  "anchor_position": None}],
        "anchors": [],
    }
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text("")

    state = compute_resume_state(manifest=manifest, ratings_path=ratings_path)
    assert state["next_main_index"] == 0
    assert state["fully_rated"] == []
    assert state["partially_rated"] == []
