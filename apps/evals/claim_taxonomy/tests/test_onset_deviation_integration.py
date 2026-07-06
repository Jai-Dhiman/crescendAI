"""FRONT 7b: end-to-end verify() over the score-relative onset-deviation measurer (#101).

Proves the full stack -- LocationResolver -> OnsetDeviationMeasurer -> FROZEN
route_verdict -- adjudicates a ms-unit statistic correctly. Uses an INLINE taxonomy
(a timing-like dimension pointed at the new measurement key) so the shipped
claim_taxonomy.json and its dimension-count tests are untouched; the shipped-taxonomy
repoint waits for the offline pipeline slice.

route_verdict polarity contract (frozen): '-' SUPPORTED iff d<0 & |d|>tau (rush);
'+' SUPPORTED iff d>0 & |d|>tau (drag); 'neutral' SUPPORTED iff |d|<=tau (steady).
"""
from __future__ import annotations

from claim_taxonomy.verifier.orchestrator import verify
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

TAU_MS = 20.0

_TAXONOMY = {
    "dimensions": {
        "timing_sr": {
            "status": "active",
            "reference": "score",
            "check": "signed_onset_deviation_vs_score",
            "measurement": "amt_score_relative_onset_deviation",
            "tolerance": {"name": "signed_onset_deviation", "provisional": TAU_MS,
                          "unit": "ms", "locked": False},
            "reliability_tier": 1,
            "minimum_events": 8,
        }
    },
    "localization_granularity": {"coverage_gate": {"threshold": 0.0}},
}


def _aligned_bundle(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(perf_onset_sec, detrended_score_onset_sec), ...]."""
    notes = [{"onset": p, "offset": p + 0.1, "pitch": 60, "velocity": 80, "score_onset": s}
             for (p, s) in pairs]
    dur = max(p for p, _ in pairs)
    return {
        "notes": notes, "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0},
                          {"bar_number": 2, "start_sec": dur}],
        "anchors": {"perf_audio_sec": [0.0, dur], "score_audio_sec": [0.0, dur]},
        "substrate_versions": {"amt": "aria-amt", "align": "parangonar"},
    }


def _claim(polarity: str) -> dict:
    return {"dimension": "timing_sr", "location": "whole_piece",
            "polarity": polarity, "proposition": "timing", "magnitude": None}


def _verify(bundle, polarity):
    return verify(_claim(polarity), bundle, _TAXONOMY, SubstrateErrorEngine(seed=42))


def test_rush_claim_on_rushing_perf_supported() -> None:
    # perf 40ms early -> d ~ -40 < -tau; a "you rushed" claim is polarity "-"
    bundle = _aligned_bundle([(i * 0.5 - 0.04, i * 0.5) for i in range(20)])
    r = _verify(bundle, "-")
    assert r.verdict == "SUPPORTED", (r.verdict, r.measured_value)
    assert r.measured_value < 0 and r.units == "ms"


def test_drag_claim_on_rushing_perf_refuted() -> None:
    # same rushing perf, but the claim says dragging (polarity "+") -> REFUTED
    bundle = _aligned_bundle([(i * 0.5 - 0.04, i * 0.5) for i in range(20)])
    r = _verify(bundle, "+")
    assert r.verdict == "REFUTED", (r.verdict, r.measured_value)


def test_drag_claim_on_dragging_perf_supported() -> None:
    bundle = _aligned_bundle([(i * 0.5 + 0.05, i * 0.5) for i in range(20)])
    r = _verify(bundle, "+")
    assert r.verdict == "SUPPORTED", (r.verdict, r.measured_value)
    assert r.measured_value > 0


def test_neutral_claim_on_in_time_perf_supported() -> None:
    # within tau -> steady/well-paced affirmed
    bundle = _aligned_bundle([(i * 0.5 + 0.003, i * 0.5) for i in range(20)])
    r = _verify(bundle, "neutral")
    assert r.verdict == "SUPPORTED", (r.verdict, r.measured_value)


def test_rush_claim_below_tau_refuted() -> None:
    # perf only 5ms early -> |d| < tau -> a "you rushed" claim is not supported
    bundle = _aligned_bundle([(i * 0.5 - 0.005, i * 0.5) for i in range(20)])
    r = _verify(bundle, "-")
    assert r.verdict == "REFUTED", (r.verdict, r.measured_value)


def test_unaligned_bundle_is_unverifiable_not_refuted() -> None:
    # a performance-side-only bundle (no score_onset) must ABSTAIN, not be REFUTED --
    # legible abstention is the paper's design (timing without a score is unverifiable).
    notes = [{"onset": i * 0.5, "offset": i * 0.5 + 0.1, "pitch": 60, "velocity": 80}
             for i in range(20)]
    bundle = {"notes": notes, "pedal_events": [],
              "measure_table": [{"bar_number": 1, "start_sec": 0.0},
                                {"bar_number": 2, "start_sec": 9.5}],
              "anchors": {"perf_audio_sec": [0.0, 9.5], "score_audio_sec": [0.0, 9.5]},
              "substrate_versions": {}}
    r = _verify(bundle, "-")
    assert r.verdict == "UNVERIFIABLE"
    assert r.reason_code == "substrate_failure"
