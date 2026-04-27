from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_user_msg_contains_teacher_voice_and_also_consider():
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.8, "timing": 0.3, "pedaling": 0.5,
                   "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5},
        duration_seconds=900,
        meta={"piece_slug": "x", "title": "Prelude", "composer": "Chopin", "skill_bucket": 3},
    )
    assert "<teacher_voice" in msg
    assert "<also_consider" in msg


def test_user_msg_blocks_appear_between_style_and_task():
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.8, "timing": 0.3, "pedaling": 0.5,
                   "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5},
        duration_seconds=900,
        meta={"piece_slug": "x", "title": "Prelude", "composer": "Chopin", "skill_bucket": 3},
    )
    style_idx = msg.find("<style_guidance")
    voice_idx = msg.find("<teacher_voice")
    task_idx = msg.find("<task>")
    assert -1 < style_idx < voice_idx < task_idx
