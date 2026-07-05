from __future__ import annotations

import math

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

# Reference loudness = corpus-median AMT note velocity over fixed-gain piano-rendered
# PercePiano (n=180, #101 G-B). The signed whole_piece statistic is mean-velocity minus
# this neutral anchor. locked:false -- recalibrate per substrate (front 4 / G-C).
REFERENCE_VELOCITY = 51.5

# AMT velocity quantization step (transcription.py velocity_quantization step=5). Retained
# for reference; superseded for the error bar by the G-C empirical churn constants below.
VELOCITY_QUANT_STEP = 5.0

# Empirical substrate error of the mean-velocity statistic (G-C, #101; n=12 PercePiano
# clips, model/src/claim_measurement/gc_error_bars/). Measured by re-transcribing the SAME
# performance under perceptually neutral recording nuisances (sub-JND +-0.5 dB gain jitter +
# 40 dB-SNR additive noise); aria-amt decodes greedily, so identical audio is a no-op and
# the churn is nuisance-driven (verified per clip). It has TWO physically distinct parts:
#   - per-note (independent): quantization + local noise, averages out as sigma_note/sqrt(N)
#   - correlated FLOOR: a global gain shift moves every note together, so the statistic
#     sigma does NOT shrink with N -> a flat floor the error bar can never drop below.
# The prior VELOCITY_QUANT_STEP/sqrt(12)/sqrt(N) placeholder had only the shrinking term and
# under-covered the measured statistic churn ~5x at typical N (0.14 vs 0.68 median). Wired as
# the pooled p90 (conservative "covers 90% of clips"); max observed statistic churn 2.39.
SUBSTRATE_VELOCITY_SIGMA = 2.69   # velocity units, measured per-note churn p90
SUBSTRATE_STATISTIC_FLOOR = 1.39  # velocity units, measured correlated re-capture floor p90

MINIMUM_NOTES = 20


class DynamicsMeasurer:
    """Measure mean AMT note-velocity (perceived-loudness proxy) for dynamics claims.

    Substrate: AMT-transcribed note velocities from the bundle (``notes``), NOT librosa
    RMS. Validated against PercePiano perceived dynamics at partial-rho 0.544 (n=180,
    #101 G-B) -- statistically indistinguishable from ground-truth MIDI velocity (0.525)
    and gain-robust. (Frame RMS fails: it conflates strike velocity with note density,
    partial-rho ~0.16; and absolute audio level is recording-gain-bound.) Units: MIDI
    velocity (0-127); the dimension tau is in velocity units for BOTH location tiers.

    Sign convention (signed d vs reference, consumed by the frozen router):
    - whole_piece: d = mean(all note velocities) - REFERENCE_VELOCITY
        d > 0 louder/more projected than a neutral performance; d < 0 softer/flatter.
    - region: d = mean(region note velocities) - mean(all note velocities)
        d > 0 region louder than the piece; d < 0 softer. Within-clip, so gain-free.
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        notes = bundle.get("notes") or []
        all_vel = np.array([float(n["velocity"]) for n in notes], dtype=np.float64)

        if location == "whole_piece":
            return self._measure_whole_piece(all_vel, engine)

        if all_vel.size < MINIMUM_NOTES:
            raise UnverifiableError(
                "region_too_short",
                f"bundle has only {all_vel.size} notes; need >= {MINIMUM_NOTES}",
            )

        onsets = np.array([float(n["onset"]) for n in notes], dtype=np.float64)
        mask = (onsets >= region.audio_start_sec) & (onsets < region.audio_end_sec)
        region_vel = all_vel[mask]
        event_count = int(region_vel.size)
        if event_count < MINIMUM_NOTES:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} notes in region "
                f"[{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]; "
                f"need >= {MINIMUM_NOTES}",
            )

        d = float(np.mean(region_vel) - np.mean(all_vel))
        error_bar = self._error_bar(region_vel, engine, baseline=float(np.mean(all_vel)))
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _measure_whole_piece(
        self, all_vel: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(all_vel.size)
        if event_count < MINIMUM_NOTES:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} notes; need >= {MINIMUM_NOTES}",
            )
        d = float(np.mean(all_vel) - REFERENCE_VELOCITY)
        error_bar = self._error_bar(all_vel, engine, baseline=REFERENCE_VELOCITY)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _error_bar(
        self, vel: np.ndarray, engine: SubstrateErrorEngine, baseline: float
    ) -> float:
        # sampling variance of the mean (bootstrap); subtracting a constant baseline
        # leaves variance unchanged but mirrors the signed-d convention.
        bootstrapped = engine.bootstrap_d(vel, np.mean)
        sampling_var = float(np.var(bootstrapped - baseline))
        # substrate: max of the (shrinking) per-note term and the (flat) correlated floor,
        # both measured empirically (G-C, #101). The floor guarantees the near-threshold
        # dead-band (verdict_dispatch: abs(abs(d)-tau) <= error_bar) covers the measured
        # re-capture 1-sigma at every note count, not just small ones.
        substrate_var = max(
            (SUBSTRATE_VELOCITY_SIGMA ** 2) / max(int(vel.size), 1),
            SUBSTRATE_STATISTIC_FLOOR ** 2,
        )
        return math.sqrt(sampling_var + substrate_var)
