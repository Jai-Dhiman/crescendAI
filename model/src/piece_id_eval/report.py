"""Orchestration: run all matchers, aggregate metrics, apply decision rule.

EvalReport.run is the single composition root that the CLI delegates to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piece_id_eval.decision import decide
from piece_id_eval.matchers.base import Matcher
from piece_id_eval.metrics import (
    mrr as compute_mrr,
    open_set_curve,
    open_set_ok,
    recall_at_k,
)
from piece_id_eval.query_set import LabeledQueryWindow

_OPEN_SET_MAX_FA = 0.10
_OPEN_SET_MIN_TA = 0.75


@dataclass
class MatcherResult:
    matcher_name: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


@dataclass
class ReportResult:
    matcher_results: list[MatcherResult]
    open_set_fa: np.ndarray       # false-accept rate curve
    open_set_ta: np.ndarray       # true-accept rate curve
    open_set_ok_flag: bool
    verdict: str                  # "KILL" | "TUNE" | "PROCEED"


class EvalReport:
    @staticmethod
    def run(
        query_windows: list[LabeledQueryWindow],
        matchers: list[Matcher],
        thresholds: np.ndarray,
    ) -> ReportResult:
        """Run all matchers over in/out-of-catalog windows, aggregate metrics.

        Args:
            query_windows: labeled windows from QuerySet.load.
            matchers: list of Matcher implementations to evaluate.
            thresholds: array of confidence thresholds for the open-set sweep.

        Returns:
            ReportResult with per-matcher metrics, open-set curve, and verdict.

        Raises:
            ValueError: if query_windows is empty.
        """
        if not query_windows:
            raise ValueError("query_windows is empty; nothing to evaluate")

        in_windows = [w for w in query_windows if w.is_in_catalog]
        out_windows = [w for w in query_windows if not w.is_in_catalog]

        matcher_results: list[MatcherResult] = []
        dtw_recall10 = 0.0
        best_indexable_recall10 = 0.0

        # Collect best-match scores for open-set curve (using DTW ceiling matcher)
        in_best_scores: list[float] = []
        out_best_scores: list[float] = []

        for matcher in matchers:
            # Build rankings for in-catalog windows only
            rankings = []
            in_ranked_cache = []  # reused below for DTW open-set scoring
            for w in in_windows:
                ranked = matcher.rank(w.chroma)
                rankings.append((w.piece_id, ranked))
                in_ranked_cache.append(ranked)

            r1 = recall_at_k(rankings, k=1) if rankings else 0.0
            r5 = recall_at_k(rankings, k=5) if rankings else 0.0
            r10 = recall_at_k(rankings, k=10) if rankings else 0.0
            m = compute_mrr(rankings) if rankings else 0.0

            mr = MatcherResult(
                matcher_name=matcher.name,
                recall_at_1=r1,
                recall_at_5=r5,
                recall_at_10=r10,
                mrr=m,
            )
            matcher_results.append(mr)

            # Track DTW ceiling and best indexable; collect open-set scores from DTW
            if "dtw_ceiling" in matcher.name:
                dtw_recall10 = r10
                # Reuse cached in-catalog rankings (already computed above)
                for ranked in in_ranked_cache:
                    in_best_scores.append(ranked[0].score if ranked else 0.0)
                for w in out_windows:
                    ranked = matcher.rank(w.chroma)
                    out_best_scores.append(ranked[0].score if ranked else 0.0)
            else:
                best_indexable_recall10 = max(best_indexable_recall10, r10)

        # Open-set curve from best-match scores
        if in_best_scores and out_best_scores:
            fa, ta = open_set_curve(
                np.array(in_best_scores), np.array(out_best_scores), thresholds
            )
            os_ok = open_set_ok(fa, ta, max_fa=_OPEN_SET_MAX_FA, min_ta=_OPEN_SET_MIN_TA)
        elif in_best_scores:
            # No out-of-catalog windows: assume open-set condition not met
            fa = np.zeros_like(thresholds)
            ta = np.ones_like(thresholds)
            os_ok = False
        else:
            fa = np.zeros_like(thresholds)
            ta = np.zeros_like(thresholds)
            os_ok = False

        verdict = decide(dtw_recall10, best_indexable_recall10, os_ok)

        return ReportResult(
            matcher_results=matcher_results,
            open_set_fa=fa,
            open_set_ta=ta,
            open_set_ok_flag=os_ok,
            verdict=verdict,
        )
