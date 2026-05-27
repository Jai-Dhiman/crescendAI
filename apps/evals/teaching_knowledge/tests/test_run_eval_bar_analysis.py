from __future__ import annotations

import json

from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_bar_analysis_appears_on_top_moment_when_chunks_provided() -> None:
    muq_means = {
        "dynamics": 0.50, "timing": 0.20, "pedaling": 0.50,
        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50,
    }
    meta = {"piece_slug": "chopin_ballade_1", "title": "Ballade", "composer": "Chopin", "skill_bucket": 3}
    chunks = [{
        "chunk_index": 0,
        "predictions": muq_means,
        "midi_notes": [
            {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
            {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        ],
        "pedal_events": [],
    }]
    out = build_synthesis_user_msg(muq_means, 60.0, meta, chunks=chunks)
    assert "bar_analysis" in out
    # The JSON inside <session_data> must parse and contain the field
    start = out.index("<session_data>") + len("<session_data>")
    end = out.index("</session_data>")
    payload = json.loads(out[start:end].strip())
    top = payload["top_moments"]
    has_bar = [m for m in top if "bar_analysis" in m]
    assert len(has_bar) >= 1


def test_no_bar_analysis_when_chunks_none() -> None:
    muq_means = {
        "dynamics": 0.50, "timing": 0.20, "pedaling": 0.50,
        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50,
    }
    meta = {"piece_slug": "clair_de_lune", "title": "Clair", "composer": "Debussy", "skill_bucket": 3}
    out = build_synthesis_user_msg(muq_means, 60.0, meta, chunks=None)
    assert "bar_analysis" not in out
