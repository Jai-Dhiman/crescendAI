# apps/evals/tests/test_run_eval_style_injection.py
from __future__ import annotations

from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_bach_injects_baroque_guidance() -> None:
    meta = {
        "piece_slug": "bach_wtc",
        "title": "WTC Prelude 1",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"articulation": 0.5},
        duration_seconds=60.0,
        meta=meta,
    )
    assert "<style_guidance" in msg
    assert "Baroque" in msg
    assert "articulation" in msg.lower()


def test_style_guidance_between_session_data_and_task() -> None:
    meta = {
        "piece_slug": "chopin_ballade",
        "title": "Ballade 1",
        "composer": "Chopin",
        "skill_bucket": 5,
    }
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.7},
        duration_seconds=90.0,
        meta=meta,
    )
    sess_end = msg.index("</session_data>")
    guidance_idx = msg.index("<style_guidance")
    task_idx = msg.index("<task>")
    assert sess_end < guidance_idx < task_idx


def test_unknown_composer_omits_style_block() -> None:
    meta = {
        "piece_slug": "unk",
        "title": "Unknown",
        "composer": "Nobody",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.5},
        duration_seconds=30.0,
        meta=meta,
    )
    assert "<style_guidance" not in msg


def test_existing_session_data_still_present() -> None:
    meta = {
        "piece_slug": "bach",
        "title": "WTC",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    msg = build_synthesis_user_msg(
        muq_means={"articulation": 0.5},
        duration_seconds=60.0,
        meta=meta,
    )
    assert "<session_data>" in msg
    assert "WTC" in msg
    assert "<task>" in msg
