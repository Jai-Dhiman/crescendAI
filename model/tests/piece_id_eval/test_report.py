"""Integration test: EvalReport.run on a toy 3-piece catalog produces a verdict."""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
from piece_id_eval.query_set import LabeledQueryWindow
from piece_id_eval.report import EvalReport, MatcherResult, ReportResult
from piece_id_eval.score_chroma import build_score_chroma


def _make_toy_catalog() -> dict[str, np.ndarray]:
    catalog: dict[str, np.ndarray] = {}
    for i, pc in enumerate([0, 4, 7]):
        notes = [{"pitch": 60 + pc, "onset_seconds": 0.0, "duration_seconds": 4.0}]
        catalog[f"piece_{i}"] = build_score_chroma(notes, frame_rate_hz=10.0)
    return catalog


def _make_query_windows(catalog: dict[str, np.ndarray], holdout_ids: set[str]) -> list[LabeledQueryWindow]:
    windows: list[LabeledQueryWindow] = []
    for piece_id, chroma in catalog.items():
        # Use the full score chroma as query (circularity sanity)
        windows.append(LabeledQueryWindow(
            query_id=f"{piece_id}/q0",
            slug=piece_id,
            video_id="synthetic",
            piece_id=piece_id,
            is_in_catalog=(piece_id not in holdout_ids),
            chroma=chroma,
        ))
    return windows


def test_report_run_returns_report_result() -> None:
    catalog = _make_toy_catalog()
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=2),
        TwoDFTMatcher(catalog, oti=False),
    ]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, matchers, thresholds=thresholds)
    assert isinstance(result, ReportResult)
    assert result.verdict in ("KILL", "TUNE", "PROCEED")


def test_report_matcher_results_present_for_all_matchers() -> None:
    catalog = _make_toy_catalog()
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=2),
        TwoDFTMatcher(catalog, oti=False),
    ]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, matchers, thresholds=thresholds)
    assert len(result.matcher_results) == 3
    for mr in result.matcher_results:
        assert isinstance(mr, MatcherResult)
        assert 0.0 <= mr.recall_at_1 <= 1.0
        assert 0.0 <= mr.recall_at_10 <= 1.0
        assert 0.0 <= mr.mrr <= 1.0


def test_report_circularity_gives_perfect_recall_dtw() -> None:
    # Query = own score chroma -> DTW ceiling should rank own piece #1 -> recall@1 = 1.0
    catalog = _make_toy_catalog()
    matchers = [DtwCeilingMatcher(catalog, oti=False)]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, matchers, thresholds=thresholds)
    dtw_result = result.matcher_results[0]
    assert dtw_result.recall_at_1 == 1.0
    assert dtw_result.recall_at_10 == 1.0


def test_report_raises_on_multiple_dtw_ceiling_matchers() -> None:
    catalog = _make_toy_catalog()
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        DtwCeilingMatcher(catalog, oti=True),
    ]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    import pytest
    with pytest.raises(ValueError, match="at most one DTW ceiling matcher"):
        EvalReport.run(windows, matchers, thresholds=thresholds)


def test_report_open_set_curve_behavioral() -> None:
    # Use DtwCeilingMatcher + a holdout window so open-set scoring actually executes.
    # piece_2 is held out (is_in_catalog=False) — its query is the out-of-catalog probe.
    # The remaining 2 pieces are in-catalog queries.
    # With circularity (query = own score chroma) DTW scores are near-perfect for
    # in-catalog and lower for the out-of-catalog holdout -> real scores flow into the curve.
    catalog = _make_toy_catalog()
    holdout_ids = {"piece_2"}
    matchers = [DtwCeilingMatcher(catalog, oti=False)]
    windows = _make_query_windows(catalog, holdout_ids=holdout_ids)
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, matchers, thresholds=thresholds)
    # Curve arrays must match threshold shape
    assert result.open_set_fa.shape == thresholds.shape
    assert result.open_set_ta.shape == thresholds.shape
    # At threshold=0 every query is accepted: ta[0] == 1.0 and fa[0] == 1.0
    # This can ONLY pass if open-set scoring actually executed (non-zero in/out score lists).
    assert result.open_set_ta[0] == 1.0, (
        f"ta at threshold=0 should be 1.0 (all in-catalog accepted), got {result.open_set_ta[0]}"
    )
    assert result.open_set_fa[0] == 1.0, (
        f"fa at threshold=0 should be 1.0 (all out-of-catalog accepted), got {result.open_set_fa[0]}"
    )
