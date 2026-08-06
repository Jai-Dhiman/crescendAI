from __future__ import annotations

import math

import numpy as np
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

# Minimum inter-onset interval (sec) for a note to contribute a duration/IOI ratio.
# Notes inside a CHORD are near-simultaneous, so an unfloored IOI drives the ratio to
# infinity: at the FRONT 9 probe's 1ms floor the measured |AMT - GT| statistic error
# reaches 13.4 ratio units against a corpus MAD of 0.37. 50ms is the calibrated floor
# (#101 FRONT 10): it minimises substrate error while still retaining 54% of note pairs,
# and it coincides with the AMT onset-match tolerance -- below it, two piano onsets are
# not reliably ordered by the transcriber anyway.
IOI_FLOOR_SEC = 0.05

# Corpus-median AMT articulation ratio over Transkun-transcribed MAESTRO-test real audio
# (n=188 27s windows, #101 FRONT 10). The signed whole_piece statistic is the window
# ratio minus this neutral anchor. It is the AMT median, NOT the ground-truth median
# (0.8796): d is computed from the AMT statistic, so a GT-anchored reference would add
# Transkun's systematic release bias (-0.093 ratio units) to every measurement -- FRONT
# 8d Cause 1, calibration debt, reborn. locked:false -- recalibrate per substrate AND
# per corpus (FRONT 9 Finding 2: the analogous dynamics reference swung the rate 0.236
# -> 0.979 on this one line). Enforced by RES-001.
REFERENCE_RATIO = 0.8159

# Substrate error of the articulation statistic, measured DIRECTLY against ground-truth
# MIDI offsets on real audio (#101 FRONT 10) rather than against re-transcription churn
# -- a strictly stronger reference than the dynamics G-C constants. Same two-term
# structure as dynamics:
#   - per-note (independent): offset noise on single notes, averaging as sigma/sqrt(N)
#   - correlated FLOOR: Transkun's release model is biased per performance (mean
#     signed error -0.10 ratio units), so the statistic error does NOT vanish with N.
SUBSTRATE_RATIO_SIGMA = (
    0.0926  # ratio units, sd of the signed per-window AMT-vs-GT error
)
SUBSTRATE_STATISTIC_FLOOR = (
    0.0583  # ratio units, p68 (1 sigma) of |AMT - GT| per window
)

# A bar seldom carries 20 floored pairs, which is how the taxonomy's "single-bar
# articulation claims are too fragile" rule is enforced mechanically rather than by
# prose.
MINIMUM_PAIRS = 20


def _duration_ioi_ratios(notes: list[dict]) -> np.ndarray:
    """Per-note duration/IOI for onset-sorted notes whose IOI clears IOI_FLOOR_SEC."""
    ns = sorted(notes, key=lambda n: float(n["onset"]))
    ratios = []
    for i in range(len(ns) - 1):
        ioi = float(ns[i + 1]["onset"]) - float(ns[i]["onset"])
        if ioi > IOI_FLOOR_SEC:
            ratios.append((float(ns[i]["offset"]) - float(ns[i]["onset"])) / ioi)
    return np.array(ratios, dtype=np.float64)


class ArticulationMeasurer:
    """Measure the note-offset-derived legato/staccato ratio for articulation claims.

    Statistic: the per-window MEDIAN of note_duration / inter-onset-interval over notes
    whose IOI clears IOI_FLOOR_SEC. Ratio > 1 means notes overlap their successors
    (legato); < 1 means they release early (detached / staccato). The median, not the
    mean, is load-bearing: Transkun's offset error has a median of 9.4ms but a p90 of
    90ms, and a median absorbs that tail (the analytic single-note propagation of the
    p90 tail is ~1.3 ratio units, while the measured per-window statistic error is 0.163
    at p90 -- 8x smaller).

    Substrate: Transkun note offsets. Validated on real MAESTRO-test audio against
    ground-truth MIDI at corr 0.930 (#101 FRONT 10; FRONT 9's 0.876 was the unfloored,
    chord-pathological variant of the same statistic). Aria-amt could not measure this
    dimension at all (offset F1 ~0.37, #125), which is why articulation was
    `gated_on_measurement` until Transkun.

    HONEST STANDING: this is the least-clean of the verifiable dimensions. tau (0.163)
    is 0.97x the corpus MAD of the statistic, so the substrate's own offset tail is
    about as wide as the between-performance spread -- only performances a MAD or more
    from the corpus median are adjudicable, and everything else abstains as
    near_threshold BY DESIGN.

    Sign convention (signed d vs reference, consumed by the frozen router):
    - whole_piece: d = median_ratio - REFERENCE_RATIO
        d > 0 more legato / over-held than a neutral performance; d < 0 more detached.
    - region: d = median_ratio(region) - median_ratio(all)
        d > 0 region more legato than the piece. Within-clip, so free of any corpus
        reference.
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        notes = bundle.get("notes") or []
        all_ratios = _duration_ioi_ratios(notes)

        if location == "whole_piece":
            return self._measure_whole_piece(all_ratios, engine)

        if all_ratios.size < MINIMUM_PAIRS:
            raise UnverifiableError(
                "region_too_short",
                f"bundle has only {all_ratios.size} notes with IOI > {IOI_FLOOR_SEC}s; "
                f"need >= {MINIMUM_PAIRS}",
            )

        in_region = [
            n
            for n in notes
            if region.audio_start_sec <= float(n["onset"]) < region.audio_end_sec
        ]
        region_ratios = _duration_ioi_ratios(in_region)
        event_count = int(region_ratios.size)
        if event_count < MINIMUM_PAIRS:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} notes with IOI > {IOI_FLOOR_SEC}s in region "
                f"[{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]; "
                f"need >= {MINIMUM_PAIRS}",
            )

        baseline = float(np.median(all_ratios))
        d = float(np.median(region_ratios) - baseline)
        error_bar = self._error_bar(region_ratios, engine, baseline=baseline)
        return Measurement(
            d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False
        )

    def _measure_whole_piece(
        self, all_ratios: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(all_ratios.size)
        if event_count < MINIMUM_PAIRS:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} notes with IOI over "
                f"{IOI_FLOOR_SEC}s; "
                f""
                f""
                f"need >= {MINIMUM_PAIRS}",
            )
        d = float(np.median(all_ratios) - REFERENCE_RATIO)
        error_bar = self._error_bar(all_ratios, engine, baseline=REFERENCE_RATIO)
        return Measurement(
            d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False
        )

    def _error_bar(
        self, ratios: np.ndarray, engine: SubstrateErrorEngine, baseline: float
    ) -> float:
        # sampling variance of the median (bootstrap), mirroring the dynamics
        # convention.
        bootstrapped = engine.bootstrap_d(ratios, np.median)
        sampling_var = float(np.var(bootstrapped - baseline))
        # substrate: max of the shrinking per-note term and the flat correlated floor,
        # both measured against ground truth on real audio (FRONT 10). The floor is what
        # keeps the near-threshold dead-band honest at large note counts.
        substrate_var = max(
            (SUBSTRATE_RATIO_SIGMA**2) / max(int(ratios.size), 1),
            SUBSTRATE_STATISTIC_FLOOR**2,
        )
        return math.sqrt(sampling_var + substrate_var)
