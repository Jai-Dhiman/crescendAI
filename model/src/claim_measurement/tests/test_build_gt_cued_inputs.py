"""GT-label-cued teacher inputs (#101 FRONT 8c) -- pure, no LLM."""
from __future__ import annotations

from claim_measurement.dynamics_supply.build_gt_cued_inputs import build_input, build_inputs


def _bundle(vid, gt_vel, median=58.2):
    return {"video_id": vid, "gt_mean_velocity": gt_vel, "gt_corpus_median": median,
            "notes": [{"velocity": 60, "onset": 0.0, "offset": 0.5, "pitch": 60}]}


def test_build_input_cue_from_ground_truth_label():
    loud = build_input(_bundle("s_loud", 80.0))    # +21.8 -> loud
    soft = build_input(_bundle("s_soft", 40.0))    # -18.2 -> soft
    bal = build_input(_bundle("s_bal", 60.0))      # +1.8 -> balanced
    assert "LOUDER" in loud["dynamics_level_cue"]
    assert "SOFTER" in soft["dynamics_level_cue"]
    assert "balanced" in bal["dynamics_level_cue"]
    assert loud["dynamics_level_signal"]["label"] == "loud"
    assert soft["dynamics_level_signal"]["source"] == "ground_truth_midi"


def test_build_input_carries_stub_muq_and_recording_id():
    x = build_input(_bundle("seg42", 70.0))
    assert x["recording_id"] == "seg42"
    assert set(x["muq_means"]) == {"dynamics", "timing", "pedaling",
                                   "articulation", "phrasing", "interpretation"}
    assert x["dynamics_level_signal"]["gt_d"] == round(70.0 - 58.2, 2)


def test_build_inputs_preserves_order_and_count():
    bundles = [_bundle("a", 80.0), _bundle("b", 40.0), _bundle("c", 60.0)]
    out = build_inputs(bundles)
    assert [x["recording_id"] for x in out] == ["a", "b", "c"]
    assert [x["dynamics_level_signal"]["label"] for x in out] == ["loud", "soft", "balanced"]
