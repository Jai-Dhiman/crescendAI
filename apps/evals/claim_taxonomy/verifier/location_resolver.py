from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


@dataclass
class ResolvedRegion:
    audio_start_sec: float
    audio_end_sec: float
    alignment_uncertainty_sec: float
    location_span_bars: float  # math.inf for whole_piece


class LocationResolver:
    """Maps claim location to audio time range with MC alignment uncertainty."""

    def __init__(self, bundle: dict, engine: SubstrateErrorEngine) -> None:
        self._measure_table: dict[int, dict] = {
            int(row["bar_number"]): row
            for row in bundle["measure_table"]
        }
        self._sorted_bars = sorted(self._measure_table.keys())
        perf = bundle["anchors"]["perf_audio_sec"]
        score = bundle["anchors"]["score_audio_sec"]
        self._perf_audio_sec = np.asarray(perf, dtype=np.float64)
        self._score_audio_sec = np.asarray(score, dtype=np.float64)
        self._engine = engine

    def _score_sec_to_audio_sec(self, score_sec: float) -> float:
        if self._perf_audio_sec.size < 2:
            raise UnverifiableError(
                "unlocalizable",
                f"fewer than 2 alignment anchors; cannot interpolate score_sec={score_sec}",
            )
        return float(np.interp(score_sec, self._score_audio_sec, self._perf_audio_sec))

    def _bar_start_score_sec(self, bar_number: int) -> float:
        if bar_number not in self._measure_table:
            raise UnverifiableError(
                "unlocalizable",
                f"bar {bar_number} not in measure_table (available: {self._sorted_bars})",
            )
        return float(self._measure_table[bar_number]["start_sec"])

    def _bar_duration_sec(self, bar_number: int) -> float:
        """Infer bar duration from adjacent measure_table entry."""
        idx = self._sorted_bars.index(bar_number)
        if idx + 1 < len(self._sorted_bars):
            next_bar = self._sorted_bars[idx + 1]
            return float(
                self._measure_table[next_bar]["start_sec"]
                - self._measure_table[bar_number]["start_sec"]
            )
        if idx > 0:
            prev_bar = self._sorted_bars[idx - 1]
            return float(
                self._measure_table[bar_number]["start_sec"]
                - self._measure_table[prev_bar]["start_sec"]
            )
        return 2.0  # fallback: 2 seconds per bar

    def resolve(self, location: dict | str) -> ResolvedRegion:
        """Map location to audio time range.

        Raises UnverifiableError("unlocalizable") if bar not found,
        < 2 anchors, or alignment uncertainty >= location span.
        """
        if location == "whole_piece":
            if self._perf_audio_sec.size < 2:
                raise UnverifiableError(
                    "unlocalizable",
                    "fewer than 2 alignment anchors for whole_piece resolution",
                )
            audio_start = float(self._perf_audio_sec.min())
            audio_end = float(self._perf_audio_sec.max())
            uncertainty = self._engine.alignment_uncertainty_sec(
                self._perf_audio_sec,
                self._score_audio_sec,
                float(self._score_audio_sec[0]),
            )
            return ResolvedRegion(
                audio_start_sec=audio_start,
                audio_end_sec=audio_end,
                alignment_uncertainty_sec=uncertainty,
                location_span_bars=math.inf,
            )

        bar_start = int(location["bar_start"])
        bar_end = int(location["bar_end"])
        location_span_bars = bar_end - bar_start + 1

        start_score_sec = self._bar_start_score_sec(bar_start)
        end_bar = bar_end + 1
        if end_bar in self._measure_table:
            end_score_sec = float(self._measure_table[end_bar]["start_sec"])
        else:
            end_score_sec = start_score_sec + self._bar_duration_sec(bar_end) * location_span_bars

        audio_start = self._score_sec_to_audio_sec(start_score_sec)
        audio_end = self._score_sec_to_audio_sec(end_score_sec)

        uncertainty = self._engine.alignment_uncertainty_sec(
            self._perf_audio_sec,
            self._score_audio_sec,
            start_score_sec,
        )

        bar_dur = self._bar_duration_sec(bar_start)
        if bar_dur > 0:
            uncertainty_bars = uncertainty / bar_dur
        else:
            uncertainty_bars = math.inf

        if uncertainty_bars >= location_span_bars:
            raise UnverifiableError(
                "unlocalizable",
                f"alignment uncertainty {uncertainty:.3f}s >= location span "
                f"{location_span_bars} bars ({location_span_bars * bar_dur:.3f}s); "
                f"bar_dur={bar_dur:.3f}s",
            )

        return ResolvedRegion(
            audio_start_sec=audio_start,
            audio_end_sec=audio_end,
            alignment_uncertainty_sec=uncertainty,
            location_span_bars=float(location_span_bars),
        )
