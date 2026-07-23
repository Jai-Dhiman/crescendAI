"""Two-arm dynamics level-cue input builder (#101 / #67) -- pure, no LLM, no AMT."""
from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.build_teacher_inputs import (
    REFERENCE_VELOCITY,
    TAU,
    build_arms,
    build_signal,
    cue_text,
    level_label,
    mean_velocity,
)


def _notes(vels: list[int]) -> list[dict]:
    return [{"pitch": 60, "onset": float(i), "offset": float(i) + 0.5, "velocity": v}
            for i, v in enumerate(vels)]


def test_mean_velocity_is_plain_mean():
    assert mean_velocity(_notes([40, 60])) == pytest.approx(50.0)


def test_mean_velocity_empty_raises():
    with pytest.raises(ValueError):
        mean_velocity([])


def test_level_label_bands_use_tau():
    assert level_label(TAU + 1.0) == "loud"
    assert level_label(-(TAU + 1.0)) == "soft"
    assert level_label(TAU) == "balanced"          # boundary is inclusive-neutral
    assert level_label(-TAU) == "balanced"
    assert level_label(0.0) == "balanced"


def test_build_signal_is_measured_and_signed():
    # mean 62 -> d = +10.5 vs 51.5 -> loud
    sig = build_signal(_notes([60, 64]))
    assert sig["label"] == "loud"
    assert sig["mean_velocity"] == pytest.approx(62.0)
    assert sig["d"] == pytest.approx(62.0 - REFERENCE_VELOCITY)
    assert sig["measured"] is True
    assert "signal-fidelity" in sig["caveat"]


def test_cue_text_reflects_direction():
    assert "LOUDER" in cue_text(build_signal(_notes([70, 74])))
    assert "SOFTER" in cue_text(build_signal(_notes([30, 34])))
    assert "balanced" in cue_text(build_signal(_notes([51, 52])))


def test_build_arms_pairs_and_only_arm_b_is_cued():
    bundles = [
        {"video_id": "loudperf", "notes": _notes([70, 74])},
        {"video_id": "softperf", "notes": _notes([30, 34])},
        {"video_id": "no_muq", "notes": _notes([50, 52])},  # dropped: no muq_means
    ]
    muq = {"loudperf": {"dynamics": 0.6}, "softperf": {"dynamics": 0.4}}
    arm_a, arm_b = build_arms(bundles, muq, "chopin_ballade_1")

    assert len(arm_a) == len(arm_b) == 2  # no_muq skipped
    assert [x["recording_id"] for x in arm_a] == ["loudperf", "softperf"]
    # ARM A carries NO cue (isolates the one variable)
    assert all("dynamics_level_cue" not in x for x in arm_a)
    assert all("dynamics_level_signal" not in x for x in arm_a)
    # ARM B carries the measured cue + ground-truth signal for sign-fidelity scoring
    assert all("dynamics_level_cue" in x for x in arm_b)
    assert arm_b[0]["dynamics_level_signal"]["label"] == "loud"
    assert arm_b[1]["dynamics_level_signal"]["label"] == "soft"
    # both arms share identical base input (same muq_means, same order)
    assert arm_a[0]["muq_means"] == arm_b[0]["muq_means"]
