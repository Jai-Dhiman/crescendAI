from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

MINIMUM_EVENTS = 2          # minimum sustain-on events in region
SUSTAIN_THRESHOLD = 64      # CC64 value >= 64 = sustain on


class PedalingMeasurer:
    """Measure pedal presence density for pedaling claims.

    Sign convention:
    - d < 0: region has lower pedal density than self_density (sparse)
    - d > 0: region has higher pedal density than self_density (over-pedaled)
    - Whole-piece: d = pedal-bar fraction (presence statistic, 0.0 to 1.0)
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        pedal_events = bundle.get("pedal_events") or []
        sustain_on_times = np.array(
            [float(e["time"]) for e in pedal_events if int(e["value"]) >= SUSTAIN_THRESHOLD],
            dtype=np.float64,
        )
        measure_table = bundle.get("measure_table") or []
        if not measure_table:
            raise UnverifiableError("substrate_failure", "measure_table is empty")

        bars = sorted(measure_table, key=lambda r: r["bar_number"])

        if location == "whole_piece":
            return self._measure_whole_piece(sustain_on_times, bars, engine)

        bar_start = int(location["bar_start"])
        bar_end = int(location["bar_end"])

        global_event_count = int(sustain_on_times.size)
        if global_event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"bundle has only {global_event_count} sustain-on events total; "
                f"need >= {MINIMUM_EVENTS}",
            )

        region_events = sustain_on_times[
            (sustain_on_times >= region.audio_start_sec)
            & (sustain_on_times < region.audio_end_sec)
        ]
        event_count = int(region_events.size)

        region_bar_rows = [r for r in bars if bar_start <= r["bar_number"] <= bar_end]
        region_fraction = self._pedal_bar_fraction(region_bar_rows, sustain_on_times, bars)

        self_density = self._pedal_bar_fraction(bars, sustain_on_times, bars)

        d = region_fraction - self_density

        per_bar_presence = np.array([
            1.0 if self._bar_has_pedal(r, sustain_on_times, bars) else 0.0
            for r in region_bar_rows
        ])
        bootstrapped = engine.bootstrap_d(per_bar_presence, np.mean)
        sampling_var = float(np.var(bootstrapped - self_density))

        threshold_jitters = engine.pedal_threshold_jitter()
        threshold_samples = np.empty(len(threshold_jitters))
        for i, jitter in enumerate(threshold_jitters):
            threshold = SUSTAIN_THRESHOLD + jitter
            region_ev_j = np.array(
                [float(e["time"]) for e in pedal_events if int(e["value"]) >= threshold],
                dtype=np.float64,
            )
            region_ev_j = region_ev_j[
                (region_ev_j >= region.audio_start_sec)
                & (region_ev_j < region.audio_end_sec)
            ]
            frac_j = self._pedal_bar_fraction_from_times(region_bar_rows, region_ev_j, bars)
            threshold_samples[i] = frac_j - self_density

        substrate_var = float(np.var(threshold_samples))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _bar_start_end_sec(self, bar_row: dict, all_bars: list[dict]) -> tuple[float, float]:
        start = float(bar_row["start_sec"])
        bar_num = int(bar_row["bar_number"])
        next_bars = [r for r in all_bars if int(r["bar_number"]) == bar_num + 1]
        if next_bars:
            end = float(next_bars[0]["start_sec"])
        else:
            prev_bars = [r for r in all_bars if int(r["bar_number"]) == bar_num - 1]
            if prev_bars:
                dur = start - float(prev_bars[0]["start_sec"])
            else:
                dur = 2.0
            end = start + dur
        return start, end

    def _bar_has_pedal(
        self, bar_row: dict, sustain_times: np.ndarray, all_bars: list[dict]
    ) -> bool:
        start, end = self._bar_start_end_sec(bar_row, all_bars)
        return bool(np.any((sustain_times >= start) & (sustain_times < end)))

    def _pedal_bar_fraction(
        self, bar_rows: list[dict], sustain_times: np.ndarray, all_bars: list[dict]
    ) -> float:
        if not bar_rows:
            return 0.0
        count = sum(1 for r in bar_rows if self._bar_has_pedal(r, sustain_times, all_bars))
        return count / len(bar_rows)

    def _pedal_bar_fraction_from_times(
        self, bar_rows: list[dict], sustain_times: np.ndarray, all_bars: list[dict]
    ) -> float:
        return self._pedal_bar_fraction(bar_rows, sustain_times, all_bars)

    def _measure_whole_piece(
        self,
        sustain_on_times: np.ndarray,
        bars: list[dict],
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        event_count = int(sustain_on_times.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} sustain-on events; need >= {MINIMUM_EVENTS}",
            )
        d = self._pedal_bar_fraction(bars, sustain_on_times, bars)
        per_bar = np.array([
            1.0 if self._bar_has_pedal(r, sustain_on_times, bars) else 0.0
            for r in bars
        ])
        bootstrapped = engine.bootstrap_d(per_bar, np.mean)
        sampling_var = float(np.var(bootstrapped))
        substrate_var = 0.0
        error_bar = math.sqrt(sampling_var + substrate_var)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
