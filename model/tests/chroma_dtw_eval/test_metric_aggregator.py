"""Seconds-tolerance primary, G4 = consecutive-chunk continuity, G2 scaled threshold."""
from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, Metrics, aggregate,
)


def _baseline() -> Baseline:
    return Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=0.0, g5=100.0),
    )


def test_guardset_has_g4_continuity_field() -> None:
    g = GuardSet(g1=0.0, g2=0.5, g3=0.0, g4=100.0, g5=0.0)
    assert g.g4 == 100.0


def test_primary_counts_practice_within_seconds_tolerance() -> None:
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.5, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=1.4, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=30.0, error_seconds=1.6, cost=0.2),  # fail
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=45.0,
                    predicted_score_sec=45.0, error_seconds=10.0, cost=0.3),  # fail
    ]
    m = aggregate(results, baseline=_baseline(), tolerance_s=1.5)
    assert m.primary == 50.0  # 2 of 4 pass


def test_g4_continuity_consecutive_chunks() -> None:
    # Three consecutive chunks at 0, 15, 30s on the same clip. Pair 0->1
    # is continuous (delta_pred = 15 = delta_audio); pair 1->2 jumps
    # 30s in predicted-score (delta 30 - delta_audio 15 = 15 > 5) -> not continuous.
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=60.0, error_seconds=0.0, cost=0.1),
    ]
    m = aggregate(results, baseline=_baseline(), tolerance_s=1.5)
    # 1 continuous pair out of 2 -> 50%
    assert m.guards.g4 == 50.0


def test_g4_regression_drop_over_5pp() -> None:
    # Baseline g4 = 100; measured g4 = 50 -> regression.
    bl = Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=100.0, g5=100.0),
    )
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=0.0,
                    predicted_score_sec=0.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=15.0,
                    predicted_score_sec=15.0, error_seconds=0.0, cost=0.1),
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=30.0,
                    predicted_score_sec=60.0, error_seconds=0.0, cost=0.1),
    ]
    m = aggregate(results, baseline=bl, tolerance_s=1.5)
    assert "g4" in m.regressed


def test_g2_threshold_scales_with_chunk_count() -> None:
    # With small n the G2 regression threshold widens. Baseline g2 = 0.9;
    # measured 0.85 with n=4 chunks should NOT regress (scaled tol).
    bl = Baseline(
        primary=0.0,
        guards=GuardSet(g1=100.0, g2=0.9, g3=100.0, g4=0.0, g5=100.0),
    )
    results = [
        ChunkResult(kind="practice", piece="p", video_id="v", start_audio_sec=float(i),
                    predicted_score_sec=float(i), error_seconds=0.5 if i < 2 else 2.0,
                    cost=0.1 + 0.1 * i)
        for i in range(4)
    ]
    m = aggregate(results, baseline=bl, tolerance_s=1.5)
    # The exact AUC value here is implementation-dependent; the test asserts
    # that g2 IS NOT marked regressed at n=4 even when below baseline by a
    # small amount (sqrt(50/4) ~ 3.5 -> threshold scaled 3.5x).
    assert "g2" not in m.regressed or m.guards.g2 < 0.9 - 0.02 * 3.5
