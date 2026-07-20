"""Behavioral tests for the follower confidence-calibration measurement (#119)."""
from __future__ import annotations

from follower_bench.calibration import CalibrationStats, calibration_stats
from follower_bench.clip_generator import SynthClip
from follower_bench.follower import MatchedNote
from follower_bench.trajectory import TrueTrajectory


def _clip(true_anchors):
    return SynthClip(
        asap_piece="synthetic", pathology_type="clean", seed=0, notes=(),
        true_trajectory=TrueTrajectory(anchors=tuple(true_anchors)), event_labels=(),
    )


def test_calibration_stats_rewards_confident_low_error_and_reports_risk_coverage() -> None:
    # Truth: position == time over [0, 10].
    clip = _clip([(0.0, 0.0), (10.0, 10.0)])
    # Estimate: perfectly tracks truth in [0,5) with HIGH confidence; in [5,10)
    # it is off by 4.0 with LOW confidence. Matches are (perf_time, score_position).
    matches = tuple(
        MatchedNote(perf_index=k, score_index=k, perf_time=float(k),
                    score_position=(float(k) if k < 5 else float(k) + 4.0),
                    confidence=(0.95 if k < 5 else 0.1))
        for k in range(11)
    )
    stats = calibration_stats(matches, clip)
    assert isinstance(stats, CalibrationStats)
    # higher confidence => lower error: rho(confidence, -|error|) is strongly positive
    assert stats.spearman_rho > 0.5
    # risk-coverage: the top-20%-confident head has lower median error than the whole pool
    head_cov, head_risk = stats.risk_coverage[0]
    assert head_cov <= 0.2
    assert head_risk < stats.overall_median_error
