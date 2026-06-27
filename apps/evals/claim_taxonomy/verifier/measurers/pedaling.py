from __future__ import annotations

import math

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

# Reference pedal usage = corpus-median AMT sustain time-on-fraction over fixed-gain
# piano-rendered PercePiano (#101 front-3, G-A neutral renders, n=30). The signed
# whole_piece statistic is the performance on-fraction minus this neutral anchor.
# AMT-derived (NOT MIDI-native): aria-amt inflates low pedal on-fraction (positive
# floor bias) and compresses the range, so a MIDI-native median would mis-zero the
# AMT-substrate measurement. locked:false -- front 4 / G-C recalibrate.
REFERENCE_FRACTION = 0.4623

# CC64 value >= 64 = sustain on (PerfPedalEvent uses 0/127; aria-amt emits clean
# 0/127 on-off pairs, transcription.py:182-224).
SUSTAIN_THRESHOLD = 64

# AMT pedal-boundary onset jitter (s). Each on-span has two boundaries (on, off);
# both perturb the on-time numerator. Placeholder substrate noise until G-C measures
# the real re-transcription pedal-boundary churn (the dynamics VELOCITY_QUANT_STEP analog).
BOUNDARY_JITTER_SEC = 0.010


class PedalingMeasurer:
    """Measure sustain-pedal time-on-fraction (the GATE-2-validated statistic) for
    pedaling claims.

    Substrate: aria-amt-transcribed ``pedal_events`` (CC64 on/off). The statistic is
    the fraction of duration the sustain pedal is down -- the EXACT statistic validated
    against PercePiano perceived pedaling at partial-rho 0.478 (n=1202, GATE 2 /
    ``gate2_expert_anchor.json``, measure ``pedal_frac`` = ``pedaling_on_fraction``).
    The previously-shipped pedal-BAR-fraction (fraction of bars with >=1 event) was an
    unvalidated cousin; switching to time-on-fraction makes the verifier check the
    statistic G-B measured, so 0.478 is inherited verbatim (frac - const is monotone
    in frac). Units: time-on-fraction in [0,1] for BOTH location tiers (one tau).

    Sign convention (signed d vs reference, consumed by the frozen router):
    - whole_piece: d = on_fraction - REFERENCE_FRACTION
        d > 0 wetter / more sustain than a neutral performance (over-pedaled);
        d < 0 drier / less sustain (under-pedaled). A genuinely dry performance
        (zero pedal) measures d = -REFERENCE_FRACTION -- it is an informative signed
        measurement, NOT an abstention (this is what makes "-" claims adjudicable).
    - region: d = region_on_fraction - whole_piece_on_fraction
        d > 0 region wetter than the piece; d < 0 drier. Within-clip, gain-free.
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        pedal_events = bundle.get("pedal_events") or []
        notes = bundle.get("notes") or []
        total_dur = self._total_duration(notes, pedal_events)
        if total_dur <= 0.0:
            raise UnverifiableError(
                "substrate_failure",
                "bundle has no measurable duration (no notes or pedal events)",
            )

        spans = self._sustain_spans(pedal_events, total_dur)

        if location == "whole_piece":
            return self._measure_whole_piece(spans, total_dur, engine)

        return self._measure_region(spans, region, total_dur, engine)

    def _total_duration(self, notes: list[dict], pedal_events: list[dict]) -> float:
        note_end = max((float(n["offset"]) for n in notes), default=0.0)
        pedal_end = max((float(e["time"]) for e in pedal_events), default=0.0)
        return max(note_end, pedal_end)

    def _sustain_spans(
        self, pedal_events: list[dict], total_dur: float
    ) -> list[tuple[float, float]]:
        """Reconstruct (on, off) sustain spans from the CC64 on/off event stream.

        An unterminated final 'on' is closed at total_dur (pedal held to the end).
        """
        evs = sorted(
            ((float(e["time"]), int(e["value"])) for e in pedal_events),
            key=lambda e: e[0],
        )
        spans: list[tuple[float, float]] = []
        on: float | None = None
        for t, v in evs:
            if v >= SUSTAIN_THRESHOLD and on is None:
                on = t
            elif v < SUSTAIN_THRESHOLD and on is not None:
                if t > on:
                    spans.append((on, t))
                on = None
        if on is not None and total_dur > on:
            spans.append((on, total_dur))
        return spans

    def _on_fraction(
        self, spans: list[tuple[float, float]], start: float, end: float, dur: float
    ) -> float:
        if dur <= 0.0:
            return 0.0
        on = 0.0
        for a, b in spans:
            lo = max(a, start)
            hi = min(b, end)
            if hi > lo:
                on += hi - lo
        return float(min(on / dur, 1.0))

    def _measure_whole_piece(
        self,
        spans: list[tuple[float, float]],
        total_dur: float,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        frac = self._on_fraction(spans, 0.0, total_dur, total_dur)
        d = frac - REFERENCE_FRACTION
        durations = np.array([b - a for a, b in spans], dtype=np.float64)
        error_bar = self._error_bar(durations, total_dur, engine)
        return Measurement(
            d=d, error_bar=error_bar, event_count=len(spans), substrate_failure=False
        )

    def _measure_region(
        self,
        spans: list[tuple[float, float]],
        region: ResolvedRegion,
        total_dur: float,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        r_start = region.audio_start_sec
        r_end = region.audio_end_sec
        region_dur = r_end - r_start
        if region_dur <= 0.0:
            raise UnverifiableError(
                "region_too_short", f"region duration {region_dur:.3f}s <= 0"
            )

        whole_frac = self._on_fraction(spans, 0.0, total_dur, total_dur)
        region_frac = self._on_fraction(spans, r_start, r_end, region_dur)
        d = region_frac - whole_frac

        region_durations = np.array(
            [
                min(b, r_end) - max(a, r_start)
                for a, b in spans
                if min(b, r_end) > max(a, r_start)
            ],
            dtype=np.float64,
        )
        error_bar = self._error_bar(region_durations, region_dur, engine)
        return Measurement(
            d=d,
            error_bar=error_bar,
            event_count=int(region_durations.size),
            substrate_failure=False,
        )

    def _error_bar(
        self, durations: np.ndarray, dur: float, engine: SubstrateErrorEngine
    ) -> float:
        """Error bar on the on-fraction: bootstrap-sampling variance over spans plus
        an assumed boundary-jitter substrate term. front 5 / G-C replaces the
        substrate term with the measured re-transcription churn."""
        n = int(durations.size)
        if n == 0 or dur <= 0.0:
            return 0.0
        bootstrapped = engine.bootstrap_d(durations, lambda x: float(np.sum(x)) / dur)
        sampling_var = float(np.var(bootstrapped))
        # two boundaries per span, each ~ N(0, BOUNDARY_JITTER_SEC); var of on-time
        # = 2n*sigma^2, scaled into fraction units by dur^2.
        substrate_var = (2.0 * n * BOUNDARY_JITTER_SEC ** 2) / (dur ** 2)
        return math.sqrt(sampling_var + substrate_var)
